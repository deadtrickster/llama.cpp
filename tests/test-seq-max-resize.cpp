// [seq-max] llama_set_n_seq_max(): the sequence ceiling of a live context can be raised and lowered
// without disturbing the sequences it already holds.
//
// A context is created with a ceiling of 1 and generates on sequence 0. It must then be able to take
// a sequence 1 - which the ceiling refuses before the raise - and both sequences must continue exactly
// as a context created with a ceiling of 2 from the start does: same tokens, interleaved, greedy. A
// shrink back to 1 must be refused while sequence 1 holds state and accepted once it is removed, and
// sequence 0 must again continue identically. The recurrent module is what this is for (its per-id
// state rows are reallocated and copied); a pure-attention model exercises the unified-KV path, where
// the ceiling is a number.
//
// Two failure shapes are driven as well, because both killed or would have killed a server:
//
//   - the side-by-side allocation a resize tries first is a PROBE with a fallback, and the backends must
//     not report its failure as an error (the live log carried `cudaMalloc failed: out of memory` on every
//     ceiling change, and anyone grepping for OOM read a healthy server as dying);
//   - a resize whose every allocation is refused - the new size next to the old, the new size in the old
//     one's place, the old size back - used to throw out of llama_set_n_seq_max() with the state in a
//     staging vector. It must return false, keep the ceiling, and leave every sequence continuing exactly
//     as before. The refusals are injected (ggml_backend_alloc_fail_next); on CPU the host fallback lands
//     on the same buffer type, so what this proves is the control flow and the state, not the placement.
//
//   test-seq-max-resize <model.gguf>
//
// CPU only, so it can run next to a busy GPU.

#include "llama.h"
#include "common.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// log capture: how many lines at each level since the last reset
static int g_log_n_error = 0;
static void count_log(enum ggml_log_level level, const char * text, void * user_data) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        g_log_n_error++;
    }
    fputs(text, stderr);
    (void) user_data;
}

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); return 1; } } while (0)

static llama_context * make_ctx(llama_model * model, uint32_t n_seq_max) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 512;
    cp.n_batch    = 64;
    cp.n_ubatch   = 64;
    cp.n_seq_max  = n_seq_max;
    cp.kv_unified = true;
    cp.n_threads  = 4;
    cp.n_threads_batch = 4;
    return llama_init_from_model(model, cp);
}

static int32_t decode_tokens(llama_context * ctx, const std::vector<llama_token> & toks, llama_seq_id seq, llama_pos pos0) {
    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        common_batch_add(batch, toks[i], pos0 + (llama_pos) i, { seq }, i + 1 == toks.size());
    }
    const int32_t ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return ret;
}

static llama_token greedy(llama_context * ctx, int32_t idx) {
    const float * logits = llama_get_logits_ith(ctx, idx);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    int best = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

// generate n tokens for each of the given sequences in lockstep (one batch per step, one token per seq)
static int generate(llama_context * ctx, std::vector<llama_seq_id> seqs, std::vector<llama_token> last, std::vector<llama_pos> pos, int n,
                    std::vector<std::vector<llama_token>> & out) {
    out.assign(seqs.size(), {});
    for (int step = 0; step < n; ++step) {
        llama_batch batch = llama_batch_init((int32_t) seqs.size(), 0, 1);
        for (size_t s = 0; s < seqs.size(); ++s) {
            common_batch_add(batch, last[s], pos[s], { seqs[s] }, true);
        }
        const int32_t ret = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (ret != 0) {
            fprintf(stderr, "decode failed at step %d: %d\n", step, ret);
            return ret;
        }
        for (size_t s = 0; s < seqs.size(); ++s) {
            last[s] = greedy(ctx, (int32_t) s);
            pos[s]++;
            out[s].push_back(last[s]);
        }
    }
    return 0;
}

static std::string show(const std::vector<llama_token> & v) {
    std::string s;
    for (auto t : v) { s += std::to_string(t) + " "; }
    return s;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }

    llama_backend_init();

    // ---- an allocation that is expected to fail must not be reported as an error ----
    {
        llama_log_set(count_log, nullptr);
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        const size_t huge = ((size_t) 1) << 60; // an exabyte: no host refuses to refuse this

        g_log_n_error = 0;
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(buft, huge);
        CHECK(b == nullptr, "an exabyte allocated");
        CHECK(g_log_n_error > 0, "a genuine allocation failure logged no error (the guard below would be vacuous)");
        const int n_genuine = g_log_n_error;

        g_log_n_error = 0;
        ggml_backend_alloc_probe_begin();
        b = ggml_backend_buft_alloc_buffer(buft, huge);
        ggml_backend_alloc_probe_end();
        CHECK(b == nullptr, "an exabyte allocated under a probe");
        CHECK(g_log_n_error == 0, "a PROBE allocation failure logged %d error line(s) (%d outside a probe)", g_log_n_error, n_genuine);
        CHECK(!ggml_backend_alloc_is_probe(), "probe depth did not return to zero");
        printf("probe: %d error line(s) for a genuine failure, 0 for a probe\n", n_genuine);
        llama_log_set(nullptr, nullptr);
    }

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    CHECK(model, "failed to load %s", argv[1]);

    // fixed token ids rather than text: the generated test models (test-llama-archs) carry a vocab but no
    // tokenizer, and the test is about state rows, not about words
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    CHECK(n_vocab > 64, "vocab of %d tokens is too small", n_vocab);
    auto fixed = [&](int seed, int n) {
        std::vector<llama_token> v;
        for (int i = 0; i < n; ++i) {
            v.push_back(3 + (seed * 7919 + i * 104729) % (n_vocab - 3));
        }
        return v;
    };
    const std::vector<llama_token> P = fixed(1, 9);
    const std::vector<llama_token> Q = fixed(2, 8);

    const int N1 = 12, N2 = 12, N3 = 8, N4 = 6;

    // ---- reference: a context sized for two sequences from the start ----
    std::vector<llama_token> R0, R0b, R1, R0c, R0d, R1d;
    {
        llama_context * ref = make_ctx(model, 2);
        CHECK(ref, "reference context");
        CHECK(decode_tokens(ref, P, 0, 0) == 0, "ref: decode P");
        std::vector<std::vector<llama_token>> g;
        CHECK(generate(ref, {0}, {greedy(ref, -1)}, {(llama_pos) P.size()}, N1, g) == 0, "ref: gen 0");
        R0 = g[0];
        CHECK(decode_tokens(ref, Q, 1, 0) == 0, "ref: decode Q on seq 1");
        const llama_token q_first = greedy(ref, -1);
        // interleave: seq 0 continues from its last token, seq 1 starts from Q's
        CHECK(generate(ref, {0, 1}, {R0.back(), q_first}, {(llama_pos) (P.size() + N1), (llama_pos) Q.size()}, N2, g) == 0, "ref: gen 0+1");
        R0b = g[0];
        R1  = g[1];
        R1.insert(R1.begin(), q_first);
        // a few more on both: the context under test compares these after a resize that failed every allocation
        CHECK(generate(ref, {0, 1}, {R0b.back(), R1.back()}, {(llama_pos) (P.size() + N1 + N2), (llama_pos) (Q.size() + N2)}, N4, g) == 0, "ref: gen 0+1 more");
        R0d = g[0];
        R1d = g[1];
        CHECK(llama_memory_seq_rm(llama_get_memory(ref), 1, -1, -1), "ref: seq_rm 1");
        CHECK(generate(ref, {0}, {R0d.back()}, {(llama_pos) (P.size() + N1 + N2 + N4)}, N3, g) == 0, "ref: gen 0 alone");
        R0c = g[0];
        llama_free(ref);
    }

    // ---- the context under test: ceiling 1, raised and lowered while sequence 0 lives ----
    llama_context * ctx = make_ctx(model, 1);
    CHECK(ctx, "context");
    CHECK(llama_n_seq_max(ctx) == 1, "n_seq_max should start at 1");

    CHECK(decode_tokens(ctx, P, 0, 0) == 0, "decode P");
    std::vector<std::vector<llama_token>> g;
    CHECK(generate(ctx, {0}, {greedy(ctx, -1)}, {(llama_pos) P.size()}, N1, g) == 0, "gen 0");
    CHECK(g[0] == R0, "seq 0 at ceiling 1 differs from the reference:\n  got %s\n  ref %s", show(g[0]).c_str(), show(R0).c_str());
    const llama_token last0 = g[0].back();

    // BEFORE the raise: a second sequence is refused by the ceiling of 1 - by the recurrent memory, which has
    // one state cell per id. A unified attention cache accepts any id up to the library maximum whatever the
    // ceiling says (the batch allocator validates against LLAMA_MAX_SEQ under kv_unified), so on a pure
    // attention model the ceiling is an accounting number and this decode goes through; drop what it left.
    {
        const bool has_rs = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
        const int32_t ret = decode_tokens(ctx, Q, 1, 0);
        if (has_rs) {
            CHECK(ret != 0, "a decode on seq 1 succeeded at a ceiling of 1 on a model with recurrent state");
        } else if (ret == 0) {
            printf("note: pure-attention model, the unified cache took seq 1 at a ceiling of 1\n");
            CHECK(llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1), "seq_rm 1 after the pre-raise decode");
        }
    }

    // what the raise costs, asked of the model
    {
        ggml_backend_buffer_type_t bufts[16];
        size_t sizes[16];
        const int32_t n = llama_seq_max_cost(ctx, 2, bufts, sizes, 16);
        CHECK(n >= 0, "llama_seq_max_cost refused a unified context");
        size_t total = 0;
        for (int32_t i = 0; i < n && i < 16; ++i) {
            printf("cost of a 2nd sequence on %s: %.3f MiB\n", ggml_backend_buft_name(bufts[i]), sizes[i] / (1024.0 * 1024.0));
            total += sizes[i];
        }
        printf("cost of a 2nd sequence, total: %.3f MiB (%s)\n", total / (1024.0 * 1024.0), argv[1]);
    }

    // the raise
    CHECK(llama_set_n_seq_max(ctx, 2), "llama_set_n_seq_max(2) refused");
    CHECK(llama_n_seq_max(ctx) == 2, "n_seq_max is %u after the raise", llama_n_seq_max(ctx));

    // the raise must not have touched sequence 0, and sequence 1 must now work: both as in the reference
    CHECK(decode_tokens(ctx, Q, 1, 0) == 0, "decode Q on seq 1 after the raise");
    const llama_token q_first = greedy(ctx, -1);
    CHECK(generate(ctx, {0, 1}, {last0, q_first}, {(llama_pos) (P.size() + N1), (llama_pos) Q.size()}, N2, g) == 0, "gen 0+1 after the raise");
    std::vector<llama_token> A1 = g[1];
    A1.insert(A1.begin(), q_first);
    CHECK(g[0] == R0b, "seq 0 after the raise diverged:\n  got %s\n  ref %s", show(g[0]).c_str(), show(R0b).c_str());
    CHECK(A1 == R1,    "seq 1 after the raise diverged:\n  got %s\n  ref %s", show(A1).c_str(), show(R1).c_str());

    // ---- a raise whose every allocation is refused: refused, not thrown, and both sequences intact ----
    // Three refusals cover the side-by-side probe, the in-place retry and the rebuild of the old size where
    // it was; the fourth attempt (the old size in host memory) is allowed through.
    {
        g_log_n_error = 0;
        llama_log_set(count_log, nullptr);
        ggml_backend_alloc_fail_next(3);
        bool threw = false;
        bool ok    = true;
        std::string what;
        try {
            ok = llama_set_n_seq_max(ctx, 3);
        } catch (const std::exception & e) {
            threw = true;
            what  = e.what();
        }
        ggml_backend_alloc_fail_next(0);
        llama_log_set(nullptr, nullptr);
        CHECK(!threw, "llama_set_n_seq_max(3) THREW with every allocation refused: %s", what.c_str());
        CHECK(!ok, "llama_set_n_seq_max(3) succeeded with every allocation refused");
        CHECK(llama_n_seq_max(ctx) == 2, "a refused raise left n_seq_max at %u", llama_n_seq_max(ctx));
        printf("refused raise: returned false, %d error line(s) said why\n", g_log_n_error);
    }

    CHECK(generate(ctx, {0, 1}, {R0b.back(), A1.back()}, {(llama_pos) (P.size() + N1 + N2), (llama_pos) (Q.size() + N2)}, N4, g) == 0, "gen 0+1 after the refused raise");
    CHECK(g[0] == R0d, "seq 0 after the refused raise diverged:\n  got %s\n  ref %s", show(g[0]).c_str(), show(R0d).c_str());
    CHECK(g[1] == R1d, "seq 1 after the refused raise diverged:\n  got %s\n  ref %s", show(g[1]).c_str(), show(R1d).c_str());
    const llama_token last0b = g[0].back();

    // the cost query at the new ceiling: the compute term can now be measured (ceiling >= 2)
    {
        ggml_backend_buffer_type_t bufts[16];
        size_t sizes[16];
        const int32_t n = llama_seq_max_cost(ctx, 3, bufts, sizes, 16);
        CHECK(n >= 0, "llama_seq_max_cost(3) refused");
        for (int32_t i = 0; i < n && i < 16; ++i) {
            printf("cost of a 3rd sequence on %s: %.3f MiB\n", ggml_backend_buft_name(bufts[i]), sizes[i] / (1024.0 * 1024.0));
        }
    }

    // a shrink is refused while sequence 1 holds state, and accepted once it does not
    CHECK(!llama_set_n_seq_max(ctx, 1), "shrink to 1 accepted while seq 1 holds state");
    CHECK(llama_n_seq_max(ctx) == 2, "a refused shrink changed n_seq_max");
    CHECK(llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1), "seq_rm 1");
    CHECK(llama_set_n_seq_max(ctx, 1), "shrink to 1 refused with seq 1 removed");
    CHECK(llama_n_seq_max(ctx) == 1, "n_seq_max is %u after the shrink", llama_n_seq_max(ctx));

    CHECK(generate(ctx, {0}, {last0b}, {(llama_pos) (P.size() + N1 + N2 + N4)}, N3, g) == 0, "gen 0 after the shrink");
    CHECK(g[0] == R0c, "seq 0 after the shrink diverged:\n  got %s\n  ref %s", show(g[0]).c_str(), show(R0c).c_str());

    // out-of-range requests are refused without effect
    CHECK(!llama_set_n_seq_max(ctx, 0), "n_seq_max 0 accepted");
    CHECK(!llama_set_n_seq_max(ctx, (uint32_t) llama_max_parallel_sequences() + 1), "n_seq_max above the library maximum accepted");
    CHECK(llama_n_seq_max(ctx) == 1, "a refused request changed n_seq_max");

    // and a bigger jump works too
    CHECK(llama_set_n_seq_max(ctx, 4), "raise to 4 refused");
    CHECK(decode_tokens(ctx, Q, 3, 0) == 0, "decode on seq 3 after the raise to 4");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("OK\n");
    return 0;
}

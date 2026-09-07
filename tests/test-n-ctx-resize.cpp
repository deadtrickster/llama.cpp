// [pool] llama_set_n_ctx(): the KV cell count of a live context can be grown and shrunk without
// disturbing the sequences it holds.
//
// A context is created with 512 cells and generates on sequence 0. Grown to 1024 it must take a
// 300-token sequence 1 and both must continue exactly as a context created with 2048 cells from the
// start does: same tokens, interleaved, greedy. A shrink that would cut live cells off is refused; a
// shrink that fits them PACKS them toward the front (sequence 2 is placed above cell 256 and the pool
// is cut to 256 under it) and the sequence continues identically. A grow whose every allocation is
// refused (ggml_backend_alloc_fail_next) returns false without a throw, leaves the size where it was
// and every sequence continuing as before - the shape of failure that kills a server otherwise.
//
// The cost query is exercised at each size: the KV rows are exact and the compute term is measured on
// the context's own graphs, so the numbers are printed rather than asserted, except that a shrink
// costs nothing and a grow costs something.
//
//   test-n-ctx-resize <model.gguf>
//
// CPU only, so it can run next to a busy GPU. A pure-attention model exercises the plain cache, a
// hybrid one the cache next to recurrent state, a GLM-shaped one the DSA indexer cache in lockstep.

#include "llama.h"
#include "common.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_log_n_error = 0;
static void count_log(enum ggml_log_level level, const char * text, void * user_data) {
    if (level == GGML_LOG_LEVEL_ERROR) {
        g_log_n_error++;
    }
    fputs(text, stderr);
    (void) user_data;
}

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); return 1; } } while (0)

static const int N_BATCH = 64;

static llama_context * make_ctx(llama_model * model, uint32_t n_ctx) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = n_ctx;
    cp.n_batch    = N_BATCH;
    cp.n_ubatch   = N_BATCH;
    cp.n_seq_max  = 4;
    cp.kv_unified = true;
    cp.n_threads  = 4;
    cp.n_threads_batch = 4;
    return llama_init_from_model(model, cp);
}

// decode toks on seq from pos0, in batches of N_BATCH; logits for the last token only
static int32_t decode_tokens(llama_context * ctx, const std::vector<llama_token> & toks, llama_seq_id seq, llama_pos pos0) {
    for (size_t i0 = 0; i0 < toks.size(); i0 += N_BATCH) {
        const size_t n = std::min<size_t>(N_BATCH, toks.size() - i0);
        llama_batch batch = llama_batch_init((int32_t) n, 0, 1);
        for (size_t i = 0; i < n; ++i) {
            common_batch_add(batch, toks[i0 + i], pos0 + (llama_pos) (i0 + i), { seq }, i0 + i + 1 == toks.size());
        }
        const int32_t ret = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (ret != 0) {
            return ret;
        }
    }
    return 0;
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

static int32_t print_cost(llama_context * ctx, uint32_t n_ctx, const char * tag) {
    ggml_backend_buffer_type_t bufts[16];
    size_t sizes[16];
    const int32_t n = llama_n_ctx_cost(ctx, n_ctx, bufts, sizes, 16);
    size_t total = 0;
    for (int32_t i = 0; i < n && i < 16; ++i) {
        printf("cost of %u -> %u cells on %s: %.3f MiB\n", llama_n_ctx(ctx), n_ctx, ggml_backend_buft_name(bufts[i]), sizes[i] / (1024.0 * 1024.0));
        total += sizes[i];
    }
    printf("cost of %u -> %u cells, total: %.3f MiB (%s)\n", llama_n_ctx(ctx), n_ctx, total / (1024.0 * 1024.0), tag);
    return n;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    CHECK(model, "failed to load %s", argv[1]);

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    CHECK(n_vocab > 64, "vocab of %d tokens is too small", n_vocab);
    auto fixed = [&](int seed, int n) {
        std::vector<llama_token> v;
        for (int i = 0; i < n; ++i) {
            v.push_back(3 + (seed * 7919 + i * 104729) % (n_vocab - 3));
        }
        return v;
    };
    const std::vector<llama_token> P  = fixed(1, 9);
    const std::vector<llama_token> Q  = fixed(2, 300); // lands above cell 256 in the context under test
    const std::vector<llama_token> Q2 = fixed(3, 8);

    const int N1 = 12, N2 = 12, N4 = 6, N5 = 6, N6 = 8;

    const llama_pos p0_1 = (llama_pos) P.size() + N1;               // seq 0 after N1
    const llama_pos p0_2 = p0_1 + N2;                                // after N2
    const llama_pos p1_1 = (llama_pos) Q.size() + N2;               // seq 1 after N2
    const llama_pos p1_2 = p1_1 + N4;
    const llama_pos p1_3 = p1_2 + N5;
    const llama_pos p2_1 = (llama_pos) Q2.size() + N5;              // seq 2 after N5

    // ---- reference: 2048 cells from the start, the same operations in the same order ----
    std::vector<llama_token> R0, R0b, R1, R0d, R1d, R1e, R2, R2c;
    {
        llama_context * ref = make_ctx(model, 2048);
        CHECK(ref, "reference context");
        std::vector<std::vector<llama_token>> g;
        CHECK(decode_tokens(ref, P, 0, 0) == 0, "ref: decode P");
        CHECK(generate(ref, {0}, {greedy(ref, -1)}, {(llama_pos) P.size()}, N1, g) == 0, "ref: gen 0");
        R0 = g[0];
        CHECK(decode_tokens(ref, Q, 1, 0) == 0, "ref: decode Q on seq 1");
        const llama_token q_first = greedy(ref, -1);
        CHECK(generate(ref, {0, 1}, {R0.back(), q_first}, {p0_1, (llama_pos) Q.size()}, N2, g) == 0, "ref: gen 0+1");
        R0b = g[0];
        R1  = g[1];
        R1.insert(R1.begin(), q_first);
        CHECK(generate(ref, {0, 1}, {R0b.back(), R1.back()}, {p0_2, p1_1}, N4, g) == 0, "ref: gen 0+1 more");
        R0d = g[0];
        R1d = g[1];
        CHECK(llama_memory_seq_rm(llama_get_memory(ref), 0, -1, -1), "ref: seq_rm 0");
        CHECK(decode_tokens(ref, Q2, 2, 0) == 0, "ref: decode Q2 on seq 2");
        const llama_token q2_first = greedy(ref, -1);
        CHECK(generate(ref, {1, 2}, {R1d.back(), q2_first}, {p1_2, (llama_pos) Q2.size()}, N5, g) == 0, "ref: gen 1+2");
        R1e = g[0];
        R2  = g[1];
        R2.insert(R2.begin(), q2_first);
        CHECK(llama_memory_seq_rm(llama_get_memory(ref), 1, -1, -1), "ref: seq_rm 1");
        CHECK(generate(ref, {2}, {R2.back()}, {p2_1}, N6, g) == 0, "ref: gen 2 alone");
        R2c = g[0];
        llama_free(ref);
    }

    // ---- the context under test: 512 cells, grown and shrunk while sequences live ----
    llama_context * ctx = make_ctx(model, 512);
    CHECK(ctx, "context");
    CHECK(llama_n_ctx(ctx) == 512, "n_ctx should start at 512, is %u", llama_n_ctx(ctx));

    std::vector<std::vector<llama_token>> g;
    CHECK(decode_tokens(ctx, P, 0, 0) == 0, "decode P");
    CHECK(generate(ctx, {0}, {greedy(ctx, -1)}, {(llama_pos) P.size()}, N1, g) == 0, "gen 0");
    CHECK(g[0] == R0, "seq 0 at 512 cells differs from the reference:\n  got %s\n  ref %s", show(g[0]).c_str(), show(R0).c_str());
    const llama_token last0 = g[0].back();

    // 300 tokens do not fit next to sequence 0 in 512 cells: the pool is grown first
    CHECK(print_cost(ctx, 1024, argv[1]) >= 0, "llama_n_ctx_cost(1024) refused a unified context");
    CHECK(llama_set_n_ctx(ctx, 1024), "llama_set_n_ctx(1024) refused");
    CHECK(llama_n_ctx(ctx) == 1024, "n_ctx is %u after the grow", llama_n_ctx(ctx));

    CHECK(decode_tokens(ctx, Q, 1, 0) == 0, "decode Q on seq 1 after the grow");
    const llama_token q_first = greedy(ctx, -1);
    CHECK(generate(ctx, {0, 1}, {last0, q_first}, {p0_1, (llama_pos) Q.size()}, N2, g) == 0, "gen 0+1 after the grow");
    std::vector<llama_token> A1 = g[1];
    A1.insert(A1.begin(), q_first);
    CHECK(g[0] == R0b, "seq 0 after the grow diverged:\n  got %s\n  ref %s", show(g[0]).c_str(), show(R0b).c_str());
    CHECK(A1 == R1,    "seq 1 after the grow diverged:\n  got %s\n  ref %s", show(A1).c_str(), show(R1).c_str());

    // ---- a grow whose every allocation is refused: refused, not thrown, and both sequences intact ----
    {
        g_log_n_error = 0;
        llama_log_set(count_log, nullptr);
        ggml_backend_alloc_fail_next(3);
        bool threw = false;
        bool ok    = true;
        std::string what;
        try {
            ok = llama_set_n_ctx(ctx, 4096);
        } catch (const std::exception & e) {
            threw = true;
            what  = e.what();
        }
        ggml_backend_alloc_fail_next(0);
        llama_log_set(nullptr, nullptr);
        CHECK(!threw, "llama_set_n_ctx(4096) THREW with every allocation refused: %s", what.c_str());
        CHECK(!ok, "llama_set_n_ctx(4096) succeeded with every allocation refused");
        CHECK(llama_n_ctx(ctx) == 1024, "a refused grow left n_ctx at %u", llama_n_ctx(ctx));
        printf("refused grow: returned false, %d error line(s) said why\n", g_log_n_error);
    }

    CHECK(generate(ctx, {0, 1}, {R0b.back(), A1.back()}, {p0_2, p1_1}, N4, g) == 0, "gen 0+1 after the refused grow");
    CHECK(g[0] == R0d, "seq 0 after the refused grow diverged:\n  got %s\n  ref %s", show(g[0]).c_str(), show(R0d).c_str());
    CHECK(g[1] == R1d, "seq 1 after the refused grow diverged:\n  got %s\n  ref %s", show(g[1]).c_str(), show(R1d).c_str());

    // a shrink under the live cells is refused without effect
    CHECK(!llama_set_n_ctx(ctx, 256), "shrink to 256 accepted with ~360 live cells");
    CHECK(llama_n_ctx(ctx) == 1024, "a refused shrink changed n_ctx to %u", llama_n_ctx(ctx));
    {
        ggml_backend_buffer_type_t bufts[16];
        size_t sizes[16];
        CHECK(llama_n_ctx_cost(ctx, 256, bufts, sizes, 16) == 0, "a shrink reported a cost");
    }

    // ---- sequence 2 lands above cell 256; the pool is cut to 256 under it once it is alone ----
    CHECK(llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1), "seq_rm 0");
    CHECK(print_cost(ctx, 2048, argv[1]) >= 0, "llama_n_ctx_cost(2048) refused");
    CHECK(llama_set_n_ctx(ctx, 2048), "grow to 2048 refused");
    CHECK(llama_n_ctx(ctx) == 2048, "n_ctx is %u after the second grow", llama_n_ctx(ctx));

    CHECK(decode_tokens(ctx, Q2, 2, 0) == 0, "decode Q2 on seq 2");
    const llama_token q2_first = greedy(ctx, -1);
    CHECK(generate(ctx, {1, 2}, {R1d.back(), q2_first}, {p1_2, (llama_pos) Q2.size()}, N5, g) == 0, "gen 1+2");
    std::vector<llama_token> A2 = g[1];
    A2.insert(A2.begin(), q2_first);
    CHECK(g[0] == R1e, "seq 1 at 2048 cells diverged:\n  got %s\n  ref %s", show(g[0]).c_str(), show(R1e).c_str());
    CHECK(A2 == R2,    "seq 2 at 2048 cells diverged:\n  got %s\n  ref %s", show(A2).c_str(), show(R2).c_str());

    CHECK(llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1), "seq_rm 1");
    CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx), 2) == p2_1 - 1, "seq 2 lost positions before the shrink");

    // sequence 2 sits at cells ~345..359: a shrink to 256 must pack it, not refuse it
    CHECK(llama_set_n_ctx(ctx, 256), "shrink to 256 refused with only %d live cells", (int) p2_1);
    CHECK(llama_n_ctx(ctx) == 256, "n_ctx is %u after the shrink", llama_n_ctx(ctx));
    CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx), 2) == p2_1 - 1, "seq 2 lost positions in the shrink");

    CHECK(generate(ctx, {2}, {A2.back()}, {p2_1}, N6, g) == 0, "gen 2 after the shrink");
    CHECK(g[0] == R2c, "seq 2 after the packing shrink diverged:\n  got %s\n  ref %s", show(g[0]).c_str(), show(R2c).c_str());

    // out-of-range requests are refused without effect
    CHECK(!llama_set_n_ctx(ctx, 0), "n_ctx 0 accepted");
    CHECK(llama_n_ctx(ctx) == 256, "a refused request changed n_ctx to %u", llama_n_ctx(ctx));

    // and a bigger jump works too, with the sequence still there
    CHECK(llama_set_n_ctx(ctx, 8192), "grow to 8192 refused");
    CHECK(llama_n_ctx(ctx) == 8192, "n_ctx is %u after the grow to 8192", llama_n_ctx(ctx));
    CHECK(decode_tokens(ctx, Q, 3, 0) == 0, "decode 300 tokens on seq 3 after the grow to 8192");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("OK\n");
    return 0;
}

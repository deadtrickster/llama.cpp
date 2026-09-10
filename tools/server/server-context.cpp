#include <deque>
#include <list>
#include <map>
#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <exception>
#include <memory>
#include <filesystem>
#include <random>
#include <utility>
#include <fstream>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

constexpr int HTTP_POLLING_SECONDS = 1;

static common_speculative_output_limits server_output_limits(const common_params & params) {
    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return { params.n_batch, 1 };
    }

    // sized by the sequence ceiling, not the slot count: llama_context::output_reserve(n_seq_max)
    // asserts n_seq_max <= n_outputs_max (the two are equal unless --seq-max is set)
    auto result = common_speculative_get_output_limits(
            params.n_batch, params.n_seq_max_eff(), common_speculative_n_max(&params.speculative));

    result.total   = std::max<int32_t>(1, result.total);
    result.per_seq = std::max<int32_t>(1, result.per_seq);
    return result;
}

// synthetic draft verification for benchmarking - accept draft tokens at random instead of by match with the target
// on replay the draft was already accepted before a context checkpoint restore, so repeat the same decisions
static std::vector<llama_token> server_sample_and_accept_synth(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft,
        const std::vector<double> & synth_probs,
        std::mt19937 & rng,
        bool is_replay) {
    GGML_ASSERT(idxs.size() == draft.size() + 1);
    GGML_ASSERT(synth_probs.size() >= draft.size());

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < draft.size(); ++i) {
        const llama_token id = common_sampler_sample(smpl, ctx, idxs[i]);
        const bool accept = is_replay || dist(rng) < synth_probs[i];
        // do not accept a drafted EOG token - it would end the generation early
        // on replay the last token is from the target and can be EOG, so skip this check
        if (accept && (is_replay || !llama_vocab_is_eog(vocab, draft[i]))) {
            // synthetic draft tokens do not advance grammar or reasoning state
            // the last replay token is from the target and must advance both
            const bool is_replay_target = is_replay && i + 1 == draft.size();
            common_sampler_accept(smpl, draft[i], is_replay_target);
            result.push_back(draft[i]);
            continue;
        }

        common_sampler_accept(smpl, id, true);
        result.push_back(id);
        return result;
    }

    const llama_token id = common_sampler_sample(smpl, ctx, idxs[draft.size()]);
    common_sampler_accept(smpl, id, true);
    result.push_back(id);

    return result;
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot; // forward declaration

struct server_batch {
    llama_batch batch;
    bool batch_rendered = false;

    struct token {
        llama_seq_id seq_id;
        llama_token token;
        llama_pos pos;
        bool output;
        bool is_prompt; // for stats tracking
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;
    int32_t n_embd = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    // in embd mode, we temporarily swap out the tokens arr and restore it on clear()
    bool has_embd = false;
    llama_token * tokens_ptr = nullptr;
    std::vector<float> embd;

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    server_batch() {
        batch.pos = nullptr; // sentinel: uninitialized batch
    }

    ~server_batch() {
        if (batch.pos != nullptr) {
            clear();
            llama_batch_free(batch);
        }
    }

    void init(int32_t n_tokens_alloc, int32_t n_embd) {
        this->n_tokens_alloc = n_tokens_alloc;
        this->n_embd = n_embd;
        batch = llama_batch_init(n_tokens_alloc, 0, 1);
        tokens_ptr = batch.token;
        tokens.reserve(n_tokens_alloc);
    }

    bool add(llama_seq_id seq_id, llama_token token, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(!has_embd); // cannot mix tokens + embd in same batch
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ seq_id, token, pos, output, is_prompt });
        return true;
    }

    bool add(llama_seq_id seq_id, const std::vector<float> & embd_in, llama_pos pos, bool output, bool is_prompt) {
        GGML_ASSERT(batch.pos != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ seq_id, LLAMA_TOKEN_NULL, pos, output, is_prompt });
        has_embd = true;
        embd.insert(embd.end(), embd_in.begin(), embd_in.end());
        return true;
    }

    void clear() {
        tokens.clear();
        embd.clear();
        common_batch_clear(batch);
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
        batch_rendered    = false;
        has_embd          = false;
        if (batch.token == nullptr) {
            batch.token = tokens_ptr;
            batch.embd  = nullptr;
        }
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void render() {
        GGML_ASSERT(!batch_rendered);
        GGML_ASSERT(batch.pos != nullptr);
        common_batch_clear(batch);
        for (int32_t i = 0; i < size(); i++) {
            const auto & t = tokens[i];
            common_batch_add(batch, t.token, t.pos, { t.seq_id }, t.output);
        }
        if (has_embd) {
            batch.token = nullptr; // will be restored on clear()
            batch.embd  = embd.data();
        }
        batch_rendered = true;
    }

    // remove the tokens of a slot from index `from` on, then re-render
    // returns old index -> new index, -1 for removed tokens, so the caller can remap slot batch indices
    std::vector<int32_t> remove_from(llama_seq_id seq_id, int32_t from) {
        GGML_ASSERT(batch_rendered);

        std::vector<int32_t> map(tokens.size(), -1);
        std::vector<token>   kept;
        std::vector<float>   embd_kept;

        kept.reserve(tokens.size());
        for (int32_t i = 0; i < size(); i++) {
            if (i >= from && tokens[i].seq_id == seq_id) {
                continue;
            }
            map[i] = (int32_t) kept.size();
            kept.push_back(tokens[i]);
            if (has_embd) {
                embd_kept.insert(embd_kept.end(), embd.begin() + i*n_embd, embd.begin() + (i + 1)*n_embd);
            }
        }

        tokens = std::move(kept);
        embd   = std::move(embd_kept);

        // same as clear(): render() writes through batch.token
        if (batch.token == nullptr) {
            batch.token = tokens_ptr;
            batch.embd  = nullptr;
        }
        batch_rendered = false;
        render();

        return map;
    }

    llama_batch get_view(int32_t off, int32_t n_tokens) const {
        GGML_ASSERT(batch.pos != nullptr);
        GGML_ASSERT(batch_rendered);
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        auto * token = batch.token ? batch.token + off          : nullptr;
        auto * embd  = batch.embd  ? batch.embd  + off * n_embd : nullptr;

        llama_batch view = {
            n_tokens,
            token,
            embd,
            batch.pos      + off,
            batch.n_seq_id + off,
            batch.seq_id   + off,
            batch.logits   + off,
        };

        return view;
    }
};

// [slot-aux] Sidecar file holding the checkpoints and draft KV that
// llama_state_seq_save_file does not store. Without them a restored slot takes
// the do_reset path and reprocesses the whole prompt. Missing sidecar is not an
// error - the restore degrades to the old behaviour.
static const uint32_t SLOT_AUX_MAGIC   = 0x58554153; // "SAUX"
static const uint32_t SLOT_AUX_VERSION = 1;

template <typename V>
static void slot_aux_write_vec(std::ofstream & f, const V & v) {
    const uint64_t n = v.size();
    f.write(reinterpret_cast<const char *>(&n), sizeof(n));
    if (n > 0) {
        f.write(reinterpret_cast<const char *>(v.data()), n);
    }
}

template <typename V>
static bool slot_aux_read_vec(std::ifstream & f, V & v) {
    uint64_t n = 0;
    if (!f.read(reinterpret_cast<char *>(&n), sizeof(n))) {
        return false;
    }
    v.resize(n);
    if (n > 0 && !f.read(reinterpret_cast<char *>(v.data()), n)) {
        return false;
    }
    return true;
}

static bool slot_aux_save(
        const std::string & filepath,
        llama_context * ctx_dft,
        llama_seq_id seq_id,
        const std::list<common_prompt_checkpoint> & ckpts) {
    std::ofstream f(filepath + ".aux", std::ios::binary);
    if (!f) {
        return false;
    }

    f.write(reinterpret_cast<const char *>(&SLOT_AUX_MAGIC),   sizeof(SLOT_AUX_MAGIC));
    f.write(reinterpret_cast<const char *>(&SLOT_AUX_VERSION), sizeof(SLOT_AUX_VERSION));

    std::vector<uint8_t> dft;
    if (ctx_dft != nullptr) {
        const size_t n = llama_state_seq_get_size_ext(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
        dft.resize(n);
        if (n > 0) {
            llama_state_seq_get_data_ext(ctx_dft, dft.data(), n, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }
    }
    slot_aux_write_vec(f, dft);

    const uint32_t n_ckpt = (uint32_t) ckpts.size();
    f.write(reinterpret_cast<const char *>(&n_ckpt), sizeof(n_ckpt));

    for (const auto & c : ckpts) {
        f.write(reinterpret_cast<const char *>(&c.n_tokens), sizeof(c.n_tokens));
        f.write(reinterpret_cast<const char *>(&c.id_task),  sizeof(c.id_task));
        f.write(reinterpret_cast<const char *>(&c.pos_min),  sizeof(c.pos_min));
        f.write(reinterpret_cast<const char *>(&c.pos_max),  sizeof(c.pos_max));
        slot_aux_write_vec(f, c.data_tgt);
        slot_aux_write_vec(f, c.data_dft);
        slot_aux_write_vec(f, c.data_spec);
    }

    return f.good();
}

static bool slot_aux_load(
        const std::string & filepath,
        llama_context * ctx_dft,
        llama_seq_id seq_id,
        std::list<common_prompt_checkpoint> & ckpts) {
    std::ifstream f(filepath + ".aux", std::ios::binary);
    if (!f) {
        return false;
    }

    uint32_t magic = 0, version = 0;
    if (!f.read(reinterpret_cast<char *>(&magic), sizeof(magic)) ||
        !f.read(reinterpret_cast<char *>(&version), sizeof(version))) {
        return false;
    }
    if (magic != SLOT_AUX_MAGIC || version != SLOT_AUX_VERSION) {
        return false;
    }

    std::vector<uint8_t> dft;
    if (!slot_aux_read_vec(f, dft)) {
        return false;
    }
    if (ctx_dft != nullptr && !dft.empty()) {
        llama_state_seq_set_data_ext(ctx_dft, dft.data(), dft.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    }

    uint32_t n_ckpt = 0;
    if (!f.read(reinterpret_cast<char *>(&n_ckpt), sizeof(n_ckpt))) {
        return false;
    }

    std::list<common_prompt_checkpoint> out;
    for (uint32_t i = 0; i < n_ckpt; ++i) {
        common_prompt_checkpoint c;
        if (!f.read(reinterpret_cast<char *>(&c.n_tokens), sizeof(c.n_tokens)) ||
            !f.read(reinterpret_cast<char *>(&c.id_task),  sizeof(c.id_task))  ||
            !f.read(reinterpret_cast<char *>(&c.pos_min),  sizeof(c.pos_min))  ||
            !f.read(reinterpret_cast<char *>(&c.pos_max),  sizeof(c.pos_max))) {
            return false;
        }
        if (!slot_aux_read_vec(f, c.data_tgt)  ||
            !slot_aux_read_vec(f, c.data_dft)  ||
            !slot_aux_read_vec(f, c.data_spec)) {
            return false;
        }
        out.push_back(std::move(c));
    }

    ckpts = std::move(out);
    return true;
}

// save a sequence's state into the prompt cache, straight from the contexts. false = nothing to save,
// already cached, or refused (state over the cache limit); the caller tells those apart with contains()
static bool prompt_state_save(server_prompt_cache & prompt_cache, llama_context * ctx_tgt, llama_context * ctx_dft, llama_seq_id seq_id, const server_prompt & prompt) {
    if (prompt.tokens.size() == 0) {
        return false;
    }

    const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

    const size_t cur_size = cur_size_tgt + cur_size_dft;

    SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
            (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

    auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
    if (cur == nullptr) {
        return false;
    }

    llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (ctx_dft) {
        llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    }

    return true;
}

struct server_slot;

// [seq] One record per live sequence (T2.3). A sequence is a llama_seq_id and the KV cells it holds; a
// seat (server_slot) is a batch position that drives one sequence at a time. The record is what outlives
// the seat:
//
//   RUNNING    seated and the seat is processing          cells + a seat
//   RESIDENT   holds cells under seq_id, nobody decoding    cells, no seat needed - seated-idle (the seat
//              that last drove it still points at it) or unseated (the seat moved on)
//   OFFLOADED  no cells, no id: a yielded generation whose id was needed by someone else, held in
//              data_tgt/data_dft until an id frees up (T2.7 routes this through the prompt cache)
//
// The state is derived from the facts (seq_id, slot, task), never stored.
//
// While a sequence is SEATED its prompt and generation state live in the slot, as they always did; they
// move here when the seat is released and back when one is acquired. The moves are cheap (vectors); the
// KV never moves while the sequence keeps its id - that is the whole point.
//
// Everything reset() clears and seat_acquire() does not restore is silent corruption of a resumed
// generation, visible only as a subtly different completion, so the mid-flight fields mirror reset() exactly.
struct server_sequence {
    llama_seq_id  seq_id = -1;      // -1: OFFLOADED, no cells
    server_slot * slot   = nullptr; // the seat, while seated (idle or running)

    // LRU for eviction: last launch, release or re-seat
    int64_t t_last_used = 0;

    // valid while UNSEATED; a seated sequence's prompt is slot->prompt
    server_prompt prompt;

    // mid-flight generation: non-null while a yielded task waits for a seat
    std::unique_ptr<const server_task> task;

    // OFFLOADED only: the whole KV state, target and draft
    std::vector<uint8_t> data_tgt;
    std::vector<uint8_t> data_dft;

    // sampler: repetition penalties, grammar position, RNG
    common_sampler_ptr smpl;

    // generation progress - mirrors reset()
    size_t       last_nl_pos = 0;
    std::string  generated_text;
    bool         has_new_line = false;
    bool         truncated    = false;
    stop_type    stop         = STOP_TYPE_NONE;
    std::string  stopping_word;
    size_t       n_sent_text  = 0;
    llama_tokens generated_tokens;
    std::vector<completion_token_output> generated_token_probs;
    json         json_schema;
    llama_token  sampled = 0;
    int32_t      n_predict_max = -1;
    int32_t      alora_invocation_start = -1;

    // speculative decoding
    bool                     spec_is_replay = false;
    llama_tokens             spec_draft;
    llama_tokens             spec_prompt;
    std::vector<int32_t>     spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    std::mt19937             spec_synth_rng;

    server_slot_stats     stats;
    std::vector<uint64_t> n_accepted_per_pos;

    std::vector<common_adapter_lora_info> lora;

    int64_t t_suspended_ms = 0;
    int     n_decoded_at_suspend = 0;

    uint64_t next_yield_at = 0;

    // how many times a restore of the offloaded state has failed. The resume pass retries
    // once and then answers the task instead of dropping it.
    int n_resume_failures = 0;

    bool seated()     const { return slot != nullptr; }
    bool mid_flight() const { return task != nullptr; }
    bool offloaded()  const { return seq_id < 0; }

    // forget the task (cancelled or aborted): what is left is a finished conversation
    void drop_mid_flight() {
        task.reset();
        smpl.reset();
        generated_text.clear();
        generated_tokens.clear();
        generated_token_probs.clear();
        spec_draft.clear();
        spec_prompt.clear();
        spec_i_batch.clear();
        spec_ckpt.clear();
        stats = {};
        n_accepted_per_pos.clear();
    }
};

struct server_slot {
    int id;

    // the llama_seq_id this slot drives in ctx_tgt/ctx_dft (batch, memory, sampler, state, checkpoints, spec).
    // -1 while the seat holds no sequence; always == seq->seq_id otherwise
    llama_seq_id seq_id = -1;

    // [seq] the registry record this seat drives, idle or running; nullptr when it holds none
    server_sequence * seq = nullptr;

    bool bound() const { return seq != nullptr; }

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    common_memory mem;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    // speculative decoding
    common_speculative * spec;

    llama_tokens spec_draft;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;
    bool spec_is_replay = false;
    std::mt19937 spec_synth_rng;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // [deadline] when the current occupant took the seat (launch or resume): the one that has had compute the
    // longest is the one a sequence past its suspension deadline takes the seat from
    int64_t t_seated_us = 0;

    // generation props
    int32_t n_ctx   = 0;  // context size per slot
    int32_t n_keep  = 0;
    int32_t i_batch = -1;

    // effective generation limit for the current task, -1 means unlimited
    int32_t n_predict_max = -1;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;
    size_t n_sent_text = 0; // number of sent text character (i.e. handle partial UTF-8 on streaming)

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        return prompt_state_save(prompt_cache, ctx_tgt, ctx_dft, seq_id, prompt);
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, seq_id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear() {
        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        // an unbound seat holds no cells, and seq_rm(-1) would remove everyone's
        if (seq_id >= 0) {
            mem.seq_rm(seq_id, -1, -1);
        }

        prompt.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // for TTS models, this is the embd generated from prev step, decode this to generate next hidden state
    // corresponding to one token position (size = n_embd)
    std::vector<float> inp_embd;

    server_slot_stats stats;

    // accepted tokens per draft position
    // not in server_slot_stats to avoid copying to every task result
    std::vector<uint64_t> n_accepted_per_pos;

    std::function<void(int /* id_slot */)>   callback_on_release;
    std::function<void(const server_slot &)> callback_on_reset; // called before reset()

    // this is for printing timings with slot progress, not part of metrics
    int64_t t_print_last = 0;
    int32_t n_gen_last = 0;

    // [preempt] n_gen at which the --slot-quantum trigger next considers
    // yielding. A threshold, not `n_gen % quantum == 0`: a speculative step
    // advances n_gen by 1 + n_accepted, which steps over multiples of the
    // quantum. 0 = not yet armed for this generation.
    uint64_t next_yield_at = 0;

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        spec_is_replay = false;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        task_prev = std::move(task);
        task.reset();

        // note: callback_on_reset() must have run before this, see release()
        stats = {};
        n_accepted_per_pos.clear();
        next_yield_at = 0;

        n_predict_max = -1;

        if (seq_id >= 0) {
            llama_set_sampler(ctx_tgt, seq_id, nullptr);
        }

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();
    }

    // Bind the backend sampler for `t`, or unbind it. reset() nulls the binding,
    // so both a fresh launch and resume() have to redo it - a resumed slot that
    // skipped this silently fell back to CPU sampling under --backend-sampling.
    void sampler_bind(const server_task & t) const {
        llama_sampler * backend = nullptr;

        if (t.need_sampling() && smpl) {
            const bool need_pre_sample_logits = t.params.sampling.n_probs > 0 && !t.params.post_sampling_probs;

            bool use_backend_sampling = t.params.sampling.backend_sampling;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            use_backend_sampling &= !need_pre_sample_logits;

            if (use_backend_sampling) {
                backend = common_sampler_get(smpl.get());
            }
        }

        llama_set_sampler(ctx_tgt, seq_id, backend);
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd();
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type
            && inp_embd.size() == other_slot.inp_embd.size()
            && are_lora_equal(lora, other_slot.lora);
    }

    // returns -1 if the generation is limitless
    int32_t n_remaining() const {
        return n_predict_max == -1 ? -1 : n_predict_max - (int32_t) stats.n_gen;
    }

    bool has_budget() const {
        return n_predict_max == -1 || n_remaining() > 0;
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return !!spec;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining() > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining() - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            if (!inp_embd.empty()) {
                add_ok &= batch.add(seq_id, inp_embd, prompt.tokens.pos_next(), true, false);
            } else {
                add_ok &= batch.add(seq_id, sampled, prompt.tokens.pos_next(), true, false);
            }

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(seq_id, sampled, pos0++, true, false);
            for (auto token : spec_draft) {
                add_ok &= batch.add(seq_id, token, pos0++, true, false);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    // [seq] Hand the seat back and leave the sequence where it is: the cells stay under seq_id, the record
    // keeps the id, nothing is copied (T2.4). An idle seat moves its prompt into the record; a generating
    // seat moves the mid-flight state too, so a later seat_acquire() produces the identical continuation -
    // which is why the field list mirrors reset() exactly. Never during prefill: the prompt is only half in
    // the KV, so that returns false and changes nothing.
    bool seat_release() {
        GGML_ASSERT(seq && seq->slot == this);

        if (is_processing() && (state != SLOT_STATE_GENERATING || !task)) {
            return false;
        }

        server_sequence & s = *seq;

        const bool mid_flight = is_processing();

        // the backend sampler binding belongs to the id; seat_acquire() or the next launch on this id re-binds it
        llama_set_sampler(ctx_tgt, seq_id, nullptr);

        s.prompt = std::move(prompt);   // move-only: server_tokens deletes copy-assign
        prompt.clear();
        prompt.tokens.has_mtmd = mctx != nullptr;

        if (mid_flight) {
            // a clone, not a move: llama_set_sampler() initializes a backend sampler chain and asserts on a second
            // init, and seat_acquire() binds again. The clone starts uninitialized and carries the sampling state
            // (repetition penalties, grammar position, RNG) - without it the continuation diverges on identical KV
            s.smpl.reset(common_sampler_clone(smpl.get()));
            s.last_nl_pos            = last_nl_pos;
            s.generated_text         = generated_text;
            s.has_new_line           = has_new_line;
            s.truncated              = truncated;
            s.stop                   = stop;
            s.stopping_word          = stopping_word;
            s.n_sent_text            = n_sent_text;
            s.generated_tokens       = generated_tokens;
            s.generated_token_probs  = generated_token_probs;
            s.json_schema            = json_schema;
            s.sampled                = sampled;
            s.n_predict_max          = n_predict_max;
            s.alora_invocation_start = alora_invocation_start;
            s.spec_is_replay         = spec_is_replay;
            s.spec_draft             = spec_draft;
            s.spec_prompt            = spec_prompt;
            s.spec_i_batch           = spec_i_batch;
            s.spec_ckpt              = spec_ckpt;
            s.spec_synth_rng         = spec_synth_rng;
            s.stats                  = stats;
            s.n_accepted_per_pos     = n_accepted_per_pos;
            s.lora                   = lora;
            s.n_decoded_at_suspend   = (int) stats.n_gen;
            s.next_yield_at          = next_yield_at;
            s.t_suspended_ms         = ggml_time_ms();
            s.task                   = std::move(task);
        }

        s.slot        = nullptr;
        s.t_last_used = ggml_time_us();

        seq    = nullptr;
        seq_id = -1;

        if (mid_flight) {
            SLT_INF(*this, "suspended after %d generated tokens (sequence %d stays resident, zero copy)\n",
                    s.n_decoded_at_suspend, s.seq_id);

            // hand the seat back WITHOUT release(): task has already moved out, and callback_on_reset() must
            // NOT run - it publishes a FINISHED generation's stats. seat_acquire() restores `stats`; the final
            // release() counts them once.
            state = SLOT_STATE_IDLE;
            t_last_used = ggml_time_us();
            reset();
            callback_on_release(id);
        }

        return true;
    }

    // [seq] Drive `s` from this seat, the inverse of seat_release(). The seat must be idle and unbound, `s`
    // unseated and holding an id. A finished record is just re-seated (its next task goes through the normal
    // launch); a mid-flight one resumes generating where it stopped. No KV moves.
    void seat_acquire(server_sequence & s) {
        GGML_ASSERT(!seq && !is_processing());
        GGML_ASSERT(!s.seated() && !s.offloaded());

        seq    = &s;
        seq_id = s.seq_id;

        s.slot        = this;
        s.t_last_used = ggml_time_us();

        prompt = std::move(s.prompt);
        s.prompt.clear();
        prompt.tokens.has_mtmd = mctx != nullptr;

        if (!s.mid_flight()) {
            return;
        }

        last_nl_pos            = s.last_nl_pos;
        generated_text         = s.generated_text;
        has_new_line           = s.has_new_line;
        truncated              = s.truncated;
        stop                   = s.stop;
        stopping_word          = s.stopping_word;
        n_sent_text            = s.n_sent_text;
        generated_tokens       = s.generated_tokens;
        generated_token_probs  = s.generated_token_probs;
        json_schema            = s.json_schema;
        sampled                = s.sampled;
        n_predict_max          = s.n_predict_max;
        alora_invocation_start = s.alora_invocation_start;
        spec_is_replay         = s.spec_is_replay;
        spec_draft             = s.spec_draft;
        spec_prompt            = s.spec_prompt;
        spec_i_batch           = s.spec_i_batch;
        spec_ckpt              = s.spec_ckpt;
        spec_synth_rng         = s.spec_synth_rng;
        stats                  = s.stats;
        n_accepted_per_pos     = s.n_accepted_per_pos;
        lora                   = s.lora;
        next_yield_at          = s.next_yield_at;

        smpl = std::move(s.smpl);
        task = std::move(s.task);

        // the seat's reset() unbound the backend sampler; bind the restored one
        sampler_bind(*task);

        has_next_token = true;
        state          = SLOT_STATE_GENERATING;
        t_seated_us    = ggml_time_us();

        SLT_INF(*this, "resumed at %d generated tokens after %" PRId64 " ms suspended (sequence %d)\n",
                s.n_decoded_at_suspend, ggml_time_ms() - s.t_suspended_ms, seq_id);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used = ggml_time_us();

            state = SLOT_STATE_IDLE;

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear();
            }

            callback_on_reset(*this);

            reset();

            callback_on_release(id);
        }
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (stats.n_gen < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = stats.n_gen_tps();
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (stats.n_gen - n_gen_last);

        t_print_last = t_now;
        n_gen_last = stats.n_gen;

        SLT_INF(*this, "n_gen = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", (int) stats.n_gen, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double t_prompt_total = stats.t_prompt_ms();

        if (t_prompt_total < 3000.0) {
            return;
        }

        const double n_prompt_second = stats.n_prompt_tps();
        const double f_progress = task->n_tokens() > 0 ? (double) prompt.n_tokens() / task->n_tokens() : 0.0;

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                (int) stats.n_prompt_processed, f_progress, t_prompt_total / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt_total = stats.t_prompt_ms();
        const double t_gen_total    = stats.t_gen_ms();

        const double t_prompt        = stats.t_prompt_per_token_ms();
        const double n_prompt_second = stats.n_prompt_tps();

        const double t_gen        = stats.t_gen_per_token_ms();
        const double n_gen_second = stats.n_gen_tps();

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_total, (int) stats.n_prompt_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_gen_total, (int) stats.n_gen, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_total + t_gen_total, (int) (stats.n_prompt_processed + stats.n_gen));

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        const int32_t n_draft_total       = stats.n_draft_tokens;
        const int32_t n_draft_accepted    = stats.n_draft_accepted;
        const int32_t n_draft_verif_steps = stats.n_draft_verif_steps;

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec);
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = stats.n_prompt_processed;
            res["n_prompt_tokens_cache"]     = stats.n_prompt_cached;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = json::array({
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining()},
                    {"n_decoded",      stats.n_gen},
                }
            });

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        mem.seq_rm(other.seq_id,     -1, -1);
        mem.seq_cp(seq_id, other.seq_id, -1, -1);

        other.i_batch = i_batch;

        other.stats = stats;

        other.prompt = prompt.clone();
        other.init_sampler();
    }
};

// returns 0 on success
// caller need to update prompt.tokens after a successful call to keep track of the processing progress
// note: this is not a member of server_slot because we want to run it inside yield_to_queue
//       slot is passed as const to avoid accidental modification of the slot state
//       some pointers are allowed to be used, they are not used by to_json()
static int process_mtmd_chunk(const server_slot & slot, mtmd::batch_ptr & mbatch, size_t idx, size_t & n_tokens_out) {
    GGML_ASSERT(slot.mctx);
    const auto & mctx = slot.mctx;
    const auto & input_tokens = slot.task->tokens;
    const auto & chunk = input_tokens.find_chunk(idx);
    int32_t res = 0;

    auto try_decode = [&]() -> int32_t {
        if (mbatch) {
            float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
            if (embd) {
                void * cb_data = slot.spec;
                static auto cb = [](llama_batch batch, void * user_data) {
                    common_speculative * spec = static_cast<common_speculative *>(user_data);
                    if (!common_speculative_process(spec, batch)) {
                        return 1;
                    }
                    return 0;
                };

                llama_pos new_n_past; // unused for now
                res = mtmd_helper_decode_image_chunk(
                    mctx,
                    slot.ctx_tgt,
                    chunk.get(),
                    embd,
                    slot.prompt.tokens.pos_next(),
                    slot.seq_id,
                    llama_n_batch(slot.ctx_tgt),
                    &new_n_past,
                    cb,
                    cb_data
                );
                if (res != 0) {
                    SLT_ERR(slot, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                    return -1;
                }
                n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                return 0; // success
            }
        }
        return 1; // (non-error) need to create & encode batch
    };

    // if the batch is already exist, try searching & encode
    res = try_decode();
    if (res == 0) {
        return 0;
    }
    if (res < 0) {
        // fatal error
        return res;
    }

    // otherwise, the batch is either uninitialized or is used up
    // we need to create & encode a new batch
    mbatch.reset(mtmd_batch_init(mctx));
    res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
    GGML_ASSERT(res == 0); // we should never have an empty batch

    // try batching as much as possible
    int n_added = 1;
    size_t idx_cur = idx;
    while (res == 0) {
        auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
        if (next_chunk == nullptr) {
            break;
        }
        res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
        n_added += (res == 0 ? 1 : 0);
        idx_cur = next_idx;
        SLT_DBG(slot, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
        // if res != 0, batch is full or chunk is not compatible -> this loop breaks
    }

    // TODO @ngxson : move this log line to debug when it become more stable
    SLT_TRC(slot, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

    res = mtmd_batch_encode(mbatch.get());
    if (res != 0) {
        SLT_ERR(slot, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
        return -1;
    }

    return try_decode();
}

//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    // note: video_params.ffmpeg_bin_dir points into params_base, which outlives this struct
    mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

    server_metrics get_metrics() const {
        return metrics;
    }

    void reset_metrics_bucket() {
        metrics.reset_bucket();
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    server_batch batch;

    llama_model   * model_dft = nullptr;
    llama_context * ctx_dft   = nullptr;

    common_speculative_init_result_ptr spec_init;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // the target memory holds state that is a function of the token history rather than
    // per-token entries (a recurrent or hybrid model). such state cannot be shifted along
    // with the KV: after an edit it is only valid up to the point the old and new prompts share
    bool ctx_tgt_has_recurrent_state = false;

    // [seats] the batch positions. Their number FOLLOWS residency (seat_add / seat_shrink): --parallel is the floor,
    // never the governing number. A std::deque because sequences point at their seat and a vector would move every
    // seat on growth.
    std::deque<server_slot> slots;

    // [seq] the sequence registry (T2.3): one record per live sequence in any state. Walked by everything
    // that used to walk `slots` and miss what was not in one: flush, sleep, shutdown, cancel, abort,
    // metrics, eviction. A std::list because seats point into it.
    std::list<server_sequence> seqs;

    // seq_rm/seq_cp on both contexts for records that have no seat
    common_memory seq_mem;

    int trace = 0;        // env: LLAMA_TRACE
    int slots_debug = 0;  // env: LLAMA_SERVER_SLOTS_DEBUG
    int slots_n_diff = 0; // env: LLAMA_SERVER_SLOTS_N_DIFF

    int n_empty_consecutive = 0;

    // [ratio] decode-only batches run since the last prefill batch, while both kinds of work exist
    int n_decode_only = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    // queued prompt stats - llama_decode() is async, so the timing is only valid after a sync
    // note: kept out of server_metrics, which is copied as-is into the task result
    int64_t  t_decode_start  = 0; // start of the last submitted decode
    int64_t  t_prompt_start  = 0; // start of the oldest queued prompt decode
    uint64_t n_prompt_queued = 0;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    int64_t t_last_load_progress_ms = 0;

    void destroy() {
        // [l2-persist] this runs on the sleep path too, where the model is unloaded
        // and reloaded around a swap. Without this the whole cache is discarded and
        // every conversation pays a full re-prefill on its next turn.
        if (prompt_cache) {
            prompt_cache->spill_all();
        }

        // [seq] the context goes and every record holding cells with it; offloaded ones live in RAM and stay.
        // Seats are rebuilt by load_model(); one still processing at exit is unbound like any other.
        for (auto & slot : slots) {
            slot.seq    = nullptr;
            slot.seq_id = -1;
        }
        seqs.remove_if([](const server_sequence & s) { return !s.offloaded(); });

        spec.reset();
        spec_init.reset();

        ctx_dft   = nullptr;
        model_dft = nullptr;

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;
    }

    // [l2-spill] save what the SLOTS are holding into the prompt cache, then
    // spill the cache to disk. spill_all() writes only what the cache already
    // contains, and a slot's conversation is copied into the cache lazily - on
    // a later get_available_slot() pass. A conversation that has had no
    // successor is therefore still only in its slot, and a spill without this
    // walk correctly writes nothing and loses it. Runs on both the exit path
    // (clean_up) and the sleep path (handle_sleeping_state), before destroy()
    // frees the context the slots' state lives in.
    void flush_prompt_cache() {
        if (!prompt_cache) {
            return;
        }

        // [seq] one walk over the registry, whatever state each sequence is in. An offloaded generation's
        // state is already a byte blob in exactly the shape the cache stores and needs no context; the rest
        // is read out of the contexts, so it is skipped when the context is gone (a shutdown while sleeping:
        // destroy() freed it, and the sleep path flushed before the free). The records stay: on the exit path
        // nothing reads them again, on the sleep path a waiting task is not this function's to drop.
        int n_live       = 0;
        int n_saved      = 0;
        int n_cached     = 0;
        int n_offl       = 0;
        int n_offl_saved = 0;

        for (auto & s : seqs) {
            if (s.offloaded()) {
                n_offl++;
                auto * cur = prompt_cache->alloc(s.prompt, s.data_tgt.size(), s.data_dft.size());
                if (cur == nullptr) {
                    if (prompt_cache->contains(s.prompt)) {
                        n_cached++;
                    } else {
                        SRV_WRN("flush: suspended task %d holds %d tokens but the cache refused it (state over the cache size limit?)\n",
                                s.task->id, s.prompt.n_tokens());
                    }
                    continue;
                }
                std::memcpy(cur->data.main.data(), s.data_tgt.data(), s.data_tgt.size());
                if (!s.data_dft.empty()) {
                    std::memcpy(cur->data.drft.data(), s.data_dft.data(), s.data_dft.size());
                }
                n_offl_saved++;
                continue;
            }

            if (ctx_tgt == nullptr) {
                continue;
            }

            const auto & prompt = seq_prompt(s);

            const int nt = prompt.n_tokens();
            if (nt == 0) {
                continue;
            }

            n_live++;
            if (prompt_state_save(*prompt_cache, ctx_tgt, ctx_dft, s.seq_id, prompt)) {
                n_saved++;
            } else if (prompt_cache->contains(prompt)) {
                // the normal case: this conversation is already in the cache, so declining
                // is not a refusal and warning about it sends readers looking for a size
                // limit that never applied
                n_cached++;
                SRV_DBG("flush: sequence %d holds %d tokens, already in the cache\n", s.seq_id, nt);
            } else {
                SRV_WRN("flush: sequence %d holds %d tokens but prompt_save() refused it (state over the cache size limit?)\n", s.seq_id, nt);
            }
        }

        if (ctx_tgt == nullptr) {
            SRV_INF("flush: context is gone (%s), only offloaded state could be saved\n", sleeping ? "sleeping" : "not loaded");
        }
        SRV_INF("flush: %d sequence(s) with tokens, %d saved, %d already cached, %d offloaded (%d saved), cache now %zu entries / %.1f MiB\n",
                n_live, n_saved, n_cached, n_offl, n_offl_saved, prompt_cache->states.size(),
                prompt_cache->size() / 1048576.0);
        if (n_saved > 0 || n_offl_saved > 0) {
            prompt_cache->update();
        }

        prompt_cache->spill_all();
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            if (callback_state) {
                callback_state(SERVER_STATE_SLEEPING, {});
                // note: for sleeping == false, event is emitted by load_model()
            }
            SRV_INF("%s", "server is entering sleeping state\n");
            // the slots are rebuilt empty on reload, so anything still only in
            // a slot has to reach the cache now or it is gone
            flush_prompt_cache();
            // a yielded generation's cells die with the context; copy them out so it can resume after the wake
            for (auto * s : seq_mid_flight(/*resident_only*/ true)) {
                seq_offload(*s);
            }
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // [l2-name] Key identifying the KV state this server produces, used to name the
    // prompt cache's spill files. Two servers sharing a --slot-save-path must never
    // read each other's entries: a foreign KV blob passes the size checks in
    // llama_state_seq_set_data_ext and then decodes as garbage, with no error.
    //
    // The existing /slots save path is no help here - it takes its filename straight
    // from the request body, so the <alias>-<hash> convention seen in practice lives
    // in the caller, not in the server. This builds its own key instead, and hashes
    // everything that changes the shape or the meaning of a sequence state buffer:
    // the model's architecture, parameter count, quantization and geometry (all of
    // which llama_model_desc summarizes, with n_embd/n_layer/n_vocab spelled out),
    // the file it came from, the KV cache element types, the context size, and the
    // draft model if one is attached, since data.drft belongs to that context.
    std::string build_prompt_cache_model_key() const {
        std::string desc;
        {
            char buf[256] = {0};
            llama_model_desc(model_tgt, buf, sizeof(buf));
            desc = buf;
        }

        std::string fp;
        fp += desc;
        fp += "|path="   + params_base.model.path;
        fp += "|n_embd=" + std::to_string(llama_model_n_embd(model_tgt));
        fp += "|n_layer=" + std::to_string(llama_model_n_layer(model_tgt));
        fp += "|n_vocab=" + std::to_string(llama_vocab_n_tokens(llama_model_get_vocab(model_tgt)));
        fp += "|n_params=" + std::to_string(llama_model_n_params(model_tgt));
        fp += "|n_ctx="  + std::to_string(n_ctx);
        fp += "|type_k=" + std::to_string((int) params_base.cache_type_k);
        fp += "|type_v=" + std::to_string((int) params_base.cache_type_v);
        fp += "|kvu="    + std::to_string((int) params_base.kv_unified);

        if (model_dft) {
            char buf[256] = {0};
            llama_model_desc(model_dft, buf, sizeof(buf));
            fp += "|dft=";
            fp += buf;
        }

        // FNV-1a, same as the spill-name hash
        uint64_t h = 0xcbf29ce484222325ull;
        for (unsigned char c : fp) {
            h ^= c;
            h *= 0x100000001b3ull;
        }

        char hex[17];
        snprintf(hex, sizeof(hex), "%016llx", (unsigned long long) h);

        // the alias reaches this from the command line, and the key goes into a
        // path - keep only characters that cannot mean anything to a filesystem
        std::string tag;
        for (char c : model_name) {
            if (tag.size() >= 48) {
                break;
            }
            const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                            (c >= '0' && c <= '9') || c == '.' || c == '_';
            tag += ok ? c : '_';
        }
        if (tag.empty() || tag.find_first_not_of('.') == std::string::npos) {
            tag = "model";
        }

        return tag + "-" + hex;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;

        params_base = params;
        const auto output_limits = server_output_limits(params_base);
        params_base.n_outputs_max = output_limits.total;
        params_base.n_outputs_max_per_seq = output_limits.per_seq;

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        const bool has_spec = has_draft || spec_mtp;

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        std::string & mmproj_path = params_base.mmproj.path;
        mtmd_context_params mparams = mtmd_context_params_default();
        if (has_mmproj) {
            mparams.use_gpu          = params_base.mmproj_use_gpu;
            mparams.device           = params_base.mmproj_device;
            mparams.print_timings    = false;
            mparams.n_threads        = params_base.cpuparams.n_threads;
            mparams.flash_attn_type  = params_base.flash_attn_type;
            mparams.warmup           = params_base.warmup;
            mparams.image_min_tokens = params_base.image_min_tokens;
            mparams.image_max_tokens = params_base.image_max_tokens;
            mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
            mparams.media_marker     = get_media_marker();
            // progress callback
            mparams.progress_callback           = load_progress_callback;
            mparams.progress_callback_user_data = &load_progress_mmproj;
        }

        // optionally get the memory usage of mmproj
        if (has_mmproj && params_base.fit_params) {
            int64_t t_start = ggml_time_us();
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams);
            int64_t t_elapsed = ggml_time_us() - t_start;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        // note: the draft / MTP context is fitted together with the target model, see common_fit_extra_model

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        llama_init = common_init_from_params(params_base);

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        if (ctx_tgt == nullptr) {
            SRV_ERR("failed to create_context with model '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_spec) {
            // spec_mtp doesn't use load a model internally, so we report 0.0 and 1.0 manually
            load_progress_callback(0.0f, &load_progress_spec);
            load_progress_spec.t_last_load_progress_ms = 0;  // reset so internal cbs aren't delayed

            {
                common_params params_dft = common_base_params_to_speculative(params_base);

                // progress callback
                params_dft.load_progress_callback           = load_progress_callback;
                params_dft.load_progress_callback_user_data = &load_progress_spec;

                spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
                model_dft = spec_init->model();
                ctx_dft   = spec_init->context();

                if (has_draft && model_dft == nullptr) {
                    SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                    return false;
                }

                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create MTP context\n");
                    return false;
                }

                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft;
            }

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }
            SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());

            init_opt.video_params.fps_target = params_base.video_fps;
            init_opt.video_params.timestamp_interval_ms = params_base.video_timestamp_interval_ms;
            init_opt.video_params.ffmpeg_bin_dir = params_base.video_ffmpeg_bin_dir.empty()
                                ? nullptr : params_base.video_ffmpeg_bin_dir.c_str();

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }
        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        ctx_tgt_has_recurrent_state = llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);

        if (ctx_tgt_has_recurrent_state && params_base.n_cache_reuse > 0) {
            SRV_INF("cache_reuse = %d on a model with recurrent state: a chunk that matches past an edit is not shifted into place, "
                    "because the recurrent state that goes with it was computed over the old history and cannot be moved; "
                    "the prompt is rebuilt from the edit instead, which regenerates the attention KV the shift would have kept\n",
                    params_base.n_cache_reuse);
        }

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        {
            // note: the capping itself is done in n_ctx_slot(), here we only report it
            const int n_ctx_seq = llama_n_ctx_seq(ctx_tgt);

            if (params_base.kv_unified_per_slot > 0) {
                if (n_ctx_seq > params_base.kv_unified_per_slot) {
                    SRV_INF("capping per-slot context (%d) to --kv-unified-per-slot (%d)\n",
                            n_ctx_seq, params_base.kv_unified_per_slot);
                } else if (params_base.kv_unified_per_slot > n_ctx_seq) {
                    // cap is above the per-slot pool capacity, so it can never bind
                    SRV_WRN(
                        "--kv-unified-per-slot (%d) exceeds the per-slot pool capacity (%d) - cap has no effect, "
                        "slots are limited to %d (raise the KV pool with -c, or unset -c to size it to "
                        "n_parallel * kv_unified_per_slot)\n",
                        params_base.kv_unified_per_slot, n_ctx_seq, n_ctx_seq);
                }
            }

            const int n_ctx_capped = params_base.kv_unified_per_slot > 0 ?
                std::min(n_ctx_seq, params_base.kv_unified_per_slot) : n_ctx_seq;

            if (n_ctx_capped > n_ctx_train) {
                SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n",
                        n_ctx_capped, n_ctx_train);
            }
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        // the number of sequence ids the context was allocated for (--seq-max, default n_parallel). Seats are
        // bound to ids as sequences need them; the ceiling moves at runtime and the seat count follows it
        const uint32_t n_seq_max = llama_n_seq_max(ctx_tgt);
        GGML_ASSERT(n_seq_max >= (uint32_t) params_base.n_parallel);

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_seq_max = %u, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_seq_max, n_ctx_slot(), params_base.kv_unified ? "true" : "false");

        if (pool_elastic()) {
            SRV_INF("[pool] elastic: starts at %u cells, one conversation always has room for one id and %u cells (%s)\n",
                    pool_size(), pool_min_ctx(), params_base.pool_min_ctx > 0 ? "--pool-min-ctx" : "derived from n_batch");
            if (pool_selftest().on) {
                SRV_WRN("[pool] SELF-TEST: buffer types without a device are priced at %.2f MiB with %.2f MiB per id (LLAMA_SERVER_POOL_SELFTEST) - never in production\n",
                        pool_selftest().budget / 1048576.0, pool_selftest().id_bytes / 1048576.0);
            }
        } else if (params_base.kv_unified) {
            SRV_INF("[pool] static: %u cells for the life of the server (%s)\n", pool_size(),
                    params_base.pool_static ? "--pool-static" : "the draft shares the target's KV tensors");
        }

        // try speculative decoding
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, n_seq_max));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
                if (params_base.speculative.has_synth()) {
                    return false;
                }
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            spec_init.reset();
            ctx_dft   = nullptr;
            model_dft = nullptr;
        }

        if (!spec && params_base.speculative.has_synth()) {
            SRV_ERR("%s", "synthetic acceptance requires an initialized speculative decoding context\n");
            return false;
        }

        seq_mem.init(ctx_tgt, ctx_dft);

        // [seats] the floor: --parallel seats exist from the start. Everything above follows residency
        for (int i = 0; i < params_base.n_parallel; i++) {
            if (seat_add("the floor") == nullptr) {
                SRV_ERR("could not create the %d floor slots (cap %u)\n", params_base.n_parallel, seat_cap());
                return false;
            }
        }

        SRV_INF("seats: floor %u, cap %u (ceiling cap %u, %d per seat in a batch of %d)\n",
                seat_floor(), seat_cap(), seq_ceiling_cap(), seat_tokens_per_batch(), llama_n_batch(ctx_tgt));

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_N_DIFF = getenv("LLAMA_SERVER_SLOTS_N_DIFF");
            slots_n_diff = LLAMA_SERVER_SLOTS_N_DIFF ? atoi(LLAMA_SERVER_SLOTS_N_DIFF) : 0;

            if (slots_n_diff) {
                SRV_WRN("LLAMA_SERVER_SLOTS_N_DIFF = %d\n", slots_n_diff);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch tokens, plus what every seat adds for
        // its generation step (seat_cap() is bounded so that this fits one n_batch view - see seat_cap)
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            const int32_t n_embd  = llama_model_n_embd_inp(model_tgt);
            batch.init(std::max(n_batch, (int32_t) seat_cap() * seat_tokens_per_batch()), n_embd);
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx, params_base.slot_save_path, params_base.cache_disk_mib);
            prompt_cache->model_key = build_prompt_cache_model_key();
            prompt_cache->has_mtmd  = mctx != nullptr;
            prompt_cache->sweep_orphans();

            // [l2-persist] adopt whatever the previous run (or the pre-sleep cache)
            // left behind for this exact model. Only headers are read here; the KV
            // bulk stays on disk until an entry actually wins a prefix match.
            prompt_cache->index_disk();
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task, bool is_yielding) {
            return process_single_task(std::move(task), is_yielding);
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;
            bool enable_thinking = false;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

                // thinking is enabled if:
                // 1. It's not explicitly disabled via --reasoning off
                // 2. The chat template supports it
                const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
                enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
                SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);
            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* allow_video           */ mctx ? mtmd_helper_support_video(mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool specified = params_base.preserve_reasoning_specified;
                // note: the kwarg is enabled by default if not specified explicitly, so check the value
                bool enabled = it != params_base.default_template_kwargs.end() && it->second == "true";
                if (supported) {
                    SRV_TRC("preserve_reasoning kwarg: %s\n",
                            it == params_base.default_template_kwargs.end() ? "unset (template default)" : it->second.c_str());
                } else {
                    SRV_TRC("%s", "preserve_reasoning kwarg: not supported by template\n");
                }
                if (supported && !specified) {
                    SRV_WRN("%s", "chat template supports preserving reasoning, it is enabled by default (may use more tokens, disable via --no-reasoning-preserve)\n");
                }
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && specified && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // [seats] a pinned id names a seat by index, and seats above the floor come and go. Rather than wrap onto
        // a different seat (a save on 3 restored onto 0), an id below the cap gets its seats back
        while (id_slot >= 0 && (size_t) id_slot >= slots.size() && seat_add("a pinned slot id") != nullptr) {
        }

        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    // the slot currently driving a sequence, or nullptr. A batch token carries a seq id, not a slot index,
    // and the seat can have let the sequence go (finished, cancelled, yielded) before the metrics see the batch
    server_slot * get_slot_by_seq_id(llama_seq_id seq_id) {
        for (server_slot & slot : slots) {
            if (slot.bound() && slot.seq_id == seq_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    //
    // [seq] the registry: sequences outlive seats (T2.3), and the ceiling of ids moves with them (seq-max)
    //

    bool seq_running(const server_sequence & s) const {
        return s.slot != nullptr && s.slot->is_processing();
    }

    const server_prompt & seq_prompt(const server_sequence & s) const {
        return s.slot != nullptr ? s.slot->prompt : s.prompt;
    }

    // what the context says right now, never a cached copy: it moves
    uint32_t seq_ceiling() const {
        return llama_n_seq_max(ctx_tgt);
    }

    // how far the ceiling may be raised: --seq-max when given, else the library maximum
    uint32_t seq_ceiling_cap() const {
        return params_base.n_seq_max > 0 ? (uint32_t) params_base.n_seq_max : (uint32_t) llama_max_parallel_sequences();
    }

    server_sequence * seq_by_id(llama_seq_id id) {
        for (auto & s : seqs) {
            if (s.seq_id == id) {
                return &s;
            }
        }
        return nullptr;
    }

    llama_seq_id seq_id_free() {
        const uint32_t n = seq_ceiling();
        for (uint32_t id = 0; id < n; ++id) {
            if (seq_by_id(id) == nullptr) {
                return id;
            }
        }
        return -1;
    }

    llama_seq_id seq_id_highest() const {
        llama_seq_id res = -1;
        for (const auto & s : seqs) {
            res = std::max(res, s.seq_id);
        }
        return res;
    }

    server_sequence & seq_create(llama_seq_id id) {
        GGML_ASSERT(id >= 0 && seq_by_id(id) == nullptr);

        seqs.emplace_back();
        auto & s = seqs.back();

        s.seq_id      = id;
        s.t_last_used = ggml_time_us();
        s.prompt.tokens.has_mtmd = mctx != nullptr;

        return s;
    }

    // drop a record: unseat it, remove its cells (unless the context is already gone), forget it
    void seq_erase(server_sequence & s, bool rm_cells = true) {
        if (s.slot != nullptr) {
            auto & slot = *s.slot;
            GGML_ASSERT(!slot.is_processing());
            slot.prompt.clear();
            slot.prompt.tokens.has_mtmd = mctx != nullptr;
            slot.seq    = nullptr;
            slot.seq_id = -1;
        }

        if (rm_cells && s.seq_id >= 0 && ctx_tgt != nullptr) {
            seq_mem.seq_rm(s.seq_id, -1, -1);
        }

        for (auto it = seqs.begin(); it != seqs.end(); ++it) {
            if (&*it == &s) {
                seqs.erase(it);
                return;
            }
        }

        GGML_ABORT("sequence is not in the registry");
    }

    // [evict] what an eviction is FOR decides who goes. A resident sequence has two costs and only one scales
    // with depth: a FIXED cost per id (its state rows and compute-buffer share, which llama_seq_max_cost prices
    // per device - ~437 MiB on GLM, the same at 1 token or 250k) and its cells, one per token. Rebuild is linear
    // in tokens (prefill), so per MiB HELD a deep context is worth ~10x a shallow one: the shallow one pays the
    // whole fixed cost for a few cells. Hence two orders, by which resource is short:
    //   SEQ_NEED_ID     every eviction frees the same fixed cost; the only question is who is needed next: LRU
    //   SEQ_NEED_CELLS  cells are short, the fixed cost is not; the shallowest go first, as many as it takes,
    //                   rather than one deep victim that costs the most to rebuild
    // Empty sequences go first either way: they hold an id and nothing else. The inputs are logged at the
    // eviction (seq_evict) so the choice can be read back and argued with.
    enum seq_need {
        SEQ_NEED_ID,
        SEQ_NEED_CELLS,
    };

    static const char * seq_need_str(seq_need need) {
        return need == SEQ_NEED_ID ? "id-bound" : "token-bound";
    }

    // finished sequences nobody is decoding, cheapest to give up first for `need`
    std::vector<server_sequence *> seq_evictable(seq_need need = SEQ_NEED_ID) {
        std::vector<server_sequence *> res;
        for (auto & s : seqs) {
            if (s.offloaded() || s.mid_flight() || seq_running(s)) {
                continue;
            }
            res.push_back(&s);
        }
        std::sort(res.begin(), res.end(), [&](const server_sequence * a, const server_sequence * b) {
            const size_t na = seq_prompt(*a).tokens.size();
            const size_t nb = seq_prompt(*b).tokens.size();
            if ((na == 0) != (nb == 0)) {
                return na == 0;
            }
            if (need == SEQ_NEED_CELLS && na != nb) {
                return na < nb;
            }
            return a->t_last_used < b->t_last_used;
        });
        return res;
    }

    // the fixed cost of one more id, summed over devices, as the loaded model prices it right now (arithmetic on
    // a cached measurement, no allocation). -1: the context cannot say
    double seq_fixed_cost_mib() const {
        ggml_backend_buffer_type_t bufts[16];
        size_t sizes[16];
        const int32_t n = llama_seq_max_cost(ctx_tgt, seq_ceiling() + 1, bufts, sizes, 16);
        if (n < 0) {
            return -1.0;
        }
        size_t total = 0;
        for (int32_t i = 0; i < std::min<int32_t>(n, 16); ++i) {
            total += sizes[i];
        }
        return total / 1048576.0;
    }

    // yielded generations waiting for a seat, oldest suspension first
    std::vector<server_sequence *> seq_mid_flight(bool resident_only) {
        std::vector<server_sequence *> res;
        for (auto & s : seqs) {
            if (!s.mid_flight() || s.seated()) {
                continue;
            }
            if (resident_only && s.offloaded()) {
                continue;
            }
            res.push_back(&s);
        }
        std::sort(res.begin(), res.end(), [](const server_sequence * a, const server_sequence * b) {
            return a->t_suspended_ms < b->t_suspended_ms;
        });
        return res;
    }

    // [seq] a finished conversation leaves the KV: into the prompt cache when there is one and it takes it,
    // then its cells go and the record with them. A refused save (state over the cache limit) keeps the
    // sequence unless `force` - the caller decided losing it beats the alternative. No cache at all means
    // the conversation is lost, as it always was, and the log says so.
    bool seq_evict(server_sequence & s, bool force, const char * why, seq_need need = SEQ_NEED_ID) {
        GGML_ASSERT(!s.offloaded() && !s.mid_flight() && !seq_running(s));

        const auto & prompt = seq_prompt(s);
        const size_t nt = prompt.tokens.size();

        if (nt > 0) {
            const char * where = s.slot != nullptr ? "slot" : "sequence";
            const int    which = s.slot != nullptr ? s.slot->id : s.seq_id;

            // [evict] the inputs to the choice, in one line: what the victim holds, what it costs to bring back at
            // the measured prefill rate, how long it has been idle, and the regime the pool is in
            {
                const double tps     = metrics.prompt.time > 0 ? (double) metrics.prompt.count / metrics.prompt.time * 1e6 : 0.0;
                const double rebuild = tps > 0 ? nt / tps : -1.0;
                size_t n_ids = 0;
                for (const auto & o : seqs) {
                    n_ids += o.seq_id >= 0;
                }
                SRV_INF("evicting sequence %d [%s]: %zu tokens, idle %.1f s, rebuild ~%.1f s at %.0f t/s; pool: %zu ids of %u at ~%.0f MiB fixed each, %zu cells held of %d (%s)\n",
                        s.seq_id, seq_need_str(need), nt, (ggml_time_us() - s.t_last_used) / 1e6, rebuild, tps,
                        n_ids, seq_ceiling(), seq_fixed_cost_mib(), pool_cells_held(), llama_n_ctx(ctx_tgt), why);
            }

            if (prompt_cache) {
                const bool saved = prompt_state_save(*prompt_cache, ctx_tgt, ctx_dft, s.seq_id, prompt);
                if (saved) {
                    prompt_cache->update();
                }
                if (saved || prompt_cache->contains(prompt)) {
                    SRV_WRN("purging %s %d with %zu tokens (saved to the prompt cache, %s)\n", where, which, nt, why);
                } else if (!force) {
                    SRV_WRN("%s %d: state exceeds the prompt cache limit - keeping its context instead of purging it\n", where, which);
                    return false;
                } else {
                    SRV_WRN("purging %s %d with %zu tokens (the prompt cache refused it, conversation lost, %s)\n", where, which, nt, why);
                }
            } else {
                SRV_WRN("purging %s %d with %zu tokens (no prompt cache, conversation lost, %s)\n", where, which, nt, why);
            }
        }

        seq_erase(s);

        return true;
    }

    // [seq] copy a yielded generation's state out and give up its id: the last rung, taken only when no id
    // can be had any other way. This is the 1.4-13 GB path on GLM; T2.7 routes it through the prompt cache.
    void seq_offload(server_sequence & s) {
        GGML_ASSERT(!s.seated() && !s.offloaded() && s.mid_flight());

        const size_t sz_tgt =           llama_state_seq_get_size_ext(ctx_tgt, s.seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t sz_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, s.seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        s.data_tgt.resize(sz_tgt);
        llama_state_seq_get_data_ext(ctx_tgt, s.data_tgt.data(), sz_tgt, s.seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (sz_dft > 0) {
            s.data_dft.resize(sz_dft);
            llama_state_seq_get_data_ext(ctx_dft, s.data_dft.data(), sz_dft, s.seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }

        seq_mem.seq_rm(s.seq_id, -1, -1);

        SRV_INF("offloaded sequence %d mid-flight (task %d, %d tokens in, %.1f MiB target + %.1f MiB draft state)\n",
                s.seq_id, s.task->id, s.n_decoded_at_suspend, sz_tgt / 1048576.0, sz_dft / 1048576.0);

        s.seq_id = -1;
    }

    // [seq] the inverse: an offloaded generation gets its state back under `id`. state_read_meta() does
    // seq_rm(dest) before it reads, so on failure the id's cells are gone whatever else says: the record
    // stays offloaded with its bytes intact and the id is left free. (T1.4)
    bool seq_restore(server_sequence & s, llama_seq_id id) {
        GGML_ASSERT(s.offloaded() && s.mid_flight());

        if (llama_state_seq_set_data_ext(ctx_tgt, s.data_tgt.data(), s.data_tgt.size(), id, LLAMA_STATE_SEQ_FLAGS_NONE) == 0) {
            SRV_ERR("restore: failed to restore target KV state (%zu bytes) for task %d\n", s.data_tgt.size(), s.task->id);
            seq_mem.seq_rm(id, -1, -1);
            return false;
        }
        if (ctx_dft && !s.data_dft.empty()) {
            if (llama_state_seq_set_data_ext(ctx_dft, s.data_dft.data(), s.data_dft.size(), id, LLAMA_STATE_SEQ_FLAGS_NONE) == 0) {
                // the target half is in; half a state is worse than none
                SRV_ERR("restore: failed to restore draft KV state (%zu bytes) for task %d\n", s.data_dft.size(), s.task->id);
                seq_mem.seq_rm(id, -1, -1);
                return false;
            }
        }

        SRV_INF("restored task %d under sequence %d (%.1f MiB target + %.1f MiB draft state)\n",
                s.task->id, id, s.data_tgt.size() / 1048576.0, s.data_dft.size() / 1048576.0);

        s.seq_id = id;
        s.data_tgt.clear();
        s.data_tgt.shrink_to_fit();
        s.data_dft.clear();
        s.data_dft.shrink_to_fit();

        return true;
    }

    // [seq-max] one more id, paid for now. Asks the loaded model what one more sequence costs (its state rows,
    // exact; its compute-buffer share, measured) and the devices what they have free, then raises the ceiling
    // on every context that shares the ids and on the speculative state - all or nothing.
    bool seq_ceiling_raise(bool urgent = false) {
        const uint32_t n_cur = seq_ceiling();
        const uint32_t n_new = n_cur + 1;

        if (!params_base.kv_unified) {
            return false;
        }
        if (n_new > seq_ceiling_cap()) {
            SRV_DBG("sequence ceiling stays at %u: the cap is %u\n", n_cur, seq_ceiling_cap());
            return false;
        }

        std::map<ggml_backend_buffer_type_t, size_t> need;
        for (llama_context * ctx : { ctx_tgt, ctx_dft }) {
            if (ctx == nullptr) {
                continue;
            }
            ggml_backend_buffer_type_t bufts[16];
            size_t sizes[16];
            const int32_t n = llama_seq_max_cost(ctx, n_new, bufts, sizes, 16);
            if (n < 0) {
                SRV_INF("sequence ceiling stays at %u: the %s context cannot change it\n", n_cur, ctx == ctx_tgt ? "target" : "draft");
                return false;
            }
            for (int32_t i = 0; i < std::min<int32_t>(n, 16); ++i) {
                need[bufts[i]] += sizes[i];
            }
        }
        pool_selftest_id_cost(need);

        // [pool] the raise must leave the floor: one more id after this one (this one takes the free id) and
        // --pool-min-ctx cells. When the device says no, cells nobody holds are given back and it is asked again;
        // past a deadline (urgent) the reserve itself may be spent
        const pool_cost_t reserve = urgent ? pool_cost_t{} : pool_reserve(/*ids_free_after*/ 0, pool_cells_free(), pool_size());

        std::string cost_str;
        if (!pool_fits(need, reserve, cost_str)) {
            pool_cost_t need_all = need;
            for (const auto & [buft, bytes] : reserve) {
                need_all[buft] += bytes;
            }
            if (!pool_shrink_for(need_all, "one more sequence id") || !pool_fits(need, reserve, cost_str)) {
                SRV_INF("sequence ceiling stays at %u: one more sequence needs %s\n", n_cur, cost_str.c_str());
                return false;
            }
        }

        if (!llama_set_n_seq_max(ctx_tgt, n_new)) {
            return false;
        }
        if (ctx_dft != nullptr && !llama_set_n_seq_max(ctx_dft, n_new)) {
            llama_set_n_seq_max(ctx_tgt, n_cur);
            return false;
        }
        if (spec && !common_speculative_set_n_seq(spec.get(), n_new)) {
            if (ctx_dft != nullptr) {
                llama_set_n_seq_max(ctx_dft, n_cur);
            }
            llama_set_n_seq_max(ctx_tgt, n_cur);
            return false;
        }

        SRV_INF("raised the sequence ceiling to %u (%s)\n", n_new, cost_str.empty() ? "no per-sequence cost reported" : cost_str.c_str());

        return true;
    }

    // [seq-max] give ids back when nothing needs them: down to the highest live id, never below the seat
    // count. Each step is a reallocation and a re-reserve, so this runs in the quiet moment when every seat
    // is idle and nothing waits, not on every release.
    bool seq_ceiling_shrink() {
        if (!params_base.kv_unified) {
            return false;
        }

        const uint32_t n_cur = seq_ceiling();
        const uint32_t n_new = std::max<uint32_t>((uint32_t) params_base.n_parallel, (uint32_t) (seq_id_highest() + 1));
        if (n_new >= n_cur) {
            return false;
        }

        if (!llama_set_n_seq_max(ctx_tgt, n_new)) {
            return false;
        }
        if (ctx_dft != nullptr && !llama_set_n_seq_max(ctx_dft, n_new)) {
            llama_set_n_seq_max(ctx_tgt, n_cur);
            return false;
        }
        if (spec) {
            common_speculative_set_n_seq(spec.get(), n_new);
        }

        SRV_INF("lowered the sequence ceiling to %u\n", n_new);

        return true;
    }

    //
    // [pool] ONE budget. The operator's design: "there is VRAM and model weight and the rest is well - a pool";
    // "i want a pool and whatever fits in here gets batched"; "i dont want to care about presizing".
    //
    // Two things draw from it and only one scales with depth: a sequence id (its state rows and compute share,
    // llama_seq_max_cost: 437 MiB on GLM, the same at 1 token or 250k) and its cells (llama_n_ctx_cost: 19.25 KiB
    // each on GLM). Before this the cells were carved out ONCE by -c and the ids competed for what was left - on
    // GLM the ceiling stopped at 5 while 1 GiB of the pool held 54,613 cells nobody used, which is four more ids.
    // Now the cell count moves too, and the trade is made at the margin in whichever direction is short:
    //
    //   cells short   grow the pool (a conversation's prompt, a batch that found no room, a state coming back);
    //                 if the device has nothing, idle ids are given back first (seq_ceiling_shrink) and asked again
    //   ids short     shrink the pool: cells nobody holds go back to the device, and the raise is asked again
    //
    // Ids and cells are not interchangeable at the margin - an id cannot be subdivided and a pool grown by less
    // than one id's worth frees no id - and they do not have to be, because of THE FLOOR, the one constant left:
    //
    //   reserve   one conversation's worth, PER DEVICE: one more id's fixed cost (unless an id is free) plus
    //             --pool-min-ctx cells (unless that many are free in the pool). --pool-min-ctx derives from
    //             n_batch: the largest step the batch builder places at once, so the smallest pool in which any
    //             request makes progress. It is the only knob; --seq-max-headroom stays what it was, the
    //             allocator's own slack (measured: the CUDA pool grows once on the first big prefill).
    //   rule      an ordinary grow or raise must leave the reserve on every device. Admission that needs nothing
    //             new never consults it. The deadline path (RULE 1, a sequence past its sleep bound) may spend it:
    //             that is what it is for - forward progress is structural, not something eviction has to be
    //             clever enough to preserve.
    //
    // Nothing here caps a conversation to protect concurrency: one conversation taking the whole pool is correct,
    // and concurrency is what yields (the ladder: evict, offload, suspend). The system never refuses; it degrades
    // to swapping. -c is where the pool STARTS, nothing more; --pool-static pins it for anyone who wants that.
    //
    // A resize moves the live cells (on the device when both tensor sets fit, through host memory when they do
    // not) and re-reserves the compute buffers, so it is not free: growth takes a quarter of the pool at a time
    // and a shrink gives back at least half the idle cells, so the next request does not resize again.
    //

    using pool_cost_t = std::map<ggml_backend_buffer_type_t, size_t>;

    bool pool_elastic() const {
        if (!params_base.kv_unified || params_base.pool_static) {
            return false;
        }
        // a draft that VIEWS the target's KV tensors (gemma4-assistant) would be left pointing at freed memory
        if (ctx_dft != nullptr) {
            char arch[64] = {0};
            llama_model_meta_val_str(llama_get_model(ctx_dft), "general.architecture", arch, sizeof(arch));
            if (std::strcmp(arch, "gemma4-assistant") == 0) {
                return false;
            }
        }
        return true;
    }

    uint32_t pool_size() const {
        return llama_n_ctx(ctx_tgt);
    }

    // [pool] SELF-TEST. The id trade - a raise refused by the device, cells given back, the raise asked
    // again - cannot be driven in the CPU suite: the CPU buffer type reports no device (ggml-backend.cpp,
    // ".device = NULL // FIXME"), so pool_fits() has nobody to ask and never says no, and a plain-KV model
    // prices an id at nothing (a unified attention cache adds no per-sequence rows). Until 2026-09-07 that
    // path had run on CUDA only, and the production log is where its first defect was found.
    //
    // LLAMA_SERVER_POOL_SELFTEST=<budget_mib>:<id_mib> prices a buffer type WITHOUT a device as one with
    // <budget_mib> in total, of which the pool's cells (at the pool's own per-cell cost, the same number
    // pool_shrink_for uses) and <id_mib> per id of the ceiling are spent; the id charge is added to what the
    // model reports. Buffer types with a real device are untouched, so this is inert on CUDA. Test scaffolding
    // only (tools/server/tests/unit/test_pool_elastic.py), never set in production.
    struct pool_selftest_t {
        bool   on       = false;
        size_t budget   = 0;
        size_t id_bytes = 0;
    };

    static const pool_selftest_t & pool_selftest() {
        static const pool_selftest_t st = []() {
            pool_selftest_t res;
            const char * e = std::getenv("LLAMA_SERVER_POOL_SELFTEST");
            if (e == nullptr) {
                return res;
            }
            double budget = 0.0;
            double id     = 0.0;
            if (sscanf(e, "%lf:%lf", &budget, &id) == 2 && budget >= 0.0 && id >= 0.0) {
                res.on       = true;
                res.budget   = (size_t) (budget * 1048576.0);
                res.id_bytes = (size_t) (id     * 1048576.0);
            }
            return res;
        }();
        return st;
    }

    // bytes one cell costs on `buft`, from what one padding step would add; 0 when the cells live elsewhere
    size_t pool_cell_bytes(ggml_backend_buffer_type_t buft) const {
        pool_cost_t step;
        if (!pool_grow_cost(pool_pad(pool_size() + 256), step)) {
            return 0;
        }
        return step.count(buft) ? step.at(buft) / 256 : 0;
    }

    // what `buft` has free: its device's answer, or under the self-test a priced answer for a buffer type
    // that has no device. false: nobody to ask (the resize will attempt and see)
    bool pool_buft_free(ggml_backend_buffer_type_t buft, size_t & free) const {
        auto * dev = ggml_backend_buft_get_device(buft);
        if (dev != nullptr) {
            size_t total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            return true;
        }
        const auto & st = pool_selftest();
        if (!st.on) {
            return false;
        }
        const size_t spent = pool_cell_bytes(buft) * pool_size() + st.id_bytes * seq_ceiling();
        free = spent >= st.budget ? 0 : st.budget - spent;
        return true;
    }

    // the self-test's charge for one more id, on every device-less buffer type that holds cells
    void pool_selftest_id_cost(pool_cost_t & need) const {
        const auto & st = pool_selftest();
        if (!st.on || st.id_bytes == 0) {
            return;
        }
        pool_cost_t step;
        if (!pool_grow_cost(pool_pad(pool_size() + 256), step)) {
            return;
        }
        for (const auto & [buft, bytes] : step) {
            if (ggml_backend_buft_get_device(buft) == nullptr) {
                need[buft] += st.id_bytes;
            }
            GGML_UNUSED(bytes);
        }
    }


    size_t pool_cells_free() const {
        const size_t held = pool_cells_held();
        const size_t n    = pool_size();
        return held >= n ? 0 : n - held;
    }

    uint32_t pool_min_ctx() const {
        return params_base.pool_min_ctx > 0 ? (uint32_t) params_base.pool_min_ctx : (uint32_t) llama_n_batch(ctx_tgt);
    }

    // [pool-restore] the cells a shrink must leave FREE beyond what is held. --pool-min-ctx, or what the largest
    // conversation the prompt cache could hand back needs to come back - its tokens plus the batch margin -
    // whichever is more. `n_largest` says which entry set it (0: --pool-min-ctx did).
    //
    // Why an entry at all: unheld cells are not idle. They are the capacity a cached conversation needs to
    // return, and a restore is all-or-nothing - state_read_meta finds every one of its cells or fails, and the
    // conversation is then prefilled from scratch (GLM, 2026-09-07: 141k tokens, 8.4 minutes, past the client's
    // timeout, looping). The shrink that caused it looked at 157 cache entries' worth of capacity and saw idle.
    //
    // Why the LARGEST single entry and not the sum: a shrink leaves room for any ONE of them to come back, which
    // is what a restore is; the sum would pin the pool at the size of everything ever cached and undo the trade.
    // Why the RAM tier only: --cache-ram is the operator's statement of what is hot, LRU-bounded, so the floor
    // moves with the workload; the disk tier is bounded only by --cache-disk (1 TiB in production) and adopts
    // files from previous runs, so "largest on disk" is the deepest conversation this model has ever seen on
    // this machine - a permanent pin. An entry that comes back from disk still gets its cells: the restore path
    // grows the pool for it (pool_room_for_restore), so the cost of being outside the floor is a resize plus
    // idle ids given back, not a reprefill. The floor is per device by construction: a cell count is one span
    // across every device that holds cells, so keeping N cells keeps their bytes on each of them.
    uint32_t pool_restore_floor(size_t & n_largest) const {
        n_largest = prompt_cache ? prompt_cache->n_tokens_largest_resident() : 0;

        const size_t min_ctx = pool_min_ctx();
        if (n_largest == 0) {
            return (uint32_t) min_ctx;
        }

        return (uint32_t) std::max<size_t>(min_ctx, n_largest + 1 + slots.size());
    }

    static uint32_t pool_pad(size_t n) {
        return (uint32_t) GGML_PAD(std::max<size_t>(n, 256), 256);
    }

    size_t seq_ids_free() const {
        size_t in_use = 0;
        for (const auto & s : seqs) {
            in_use += s.seq_id >= 0;
        }
        const uint32_t n = seq_ceiling();
        return in_use >= n ? 0 : n - in_use;
    }

    // what growing the pool to n_new cells costs, on every context that holds cells. false: a context cannot
    bool pool_grow_cost(uint32_t n_new, pool_cost_t & need) const {
        for (llama_context * ctx : { ctx_tgt, ctx_dft }) {
            if (ctx == nullptr) {
                continue;
            }
            if (ctx == ctx_dft && llama_n_ctx(ctx_dft) != llama_n_ctx(ctx_tgt)) {
                continue; // not in lockstep with the target: it is not resized either
            }
            ggml_backend_buffer_type_t bufts[16];
            size_t sizes[16];
            const int32_t n = llama_n_ctx_cost(ctx, n_new, bufts, sizes, 16);
            if (n < 0) {
                return false;
            }
            for (int32_t i = 0; i < std::min<int32_t>(n, 16); ++i) {
                need[bufts[i]] += sizes[i];
            }
        }
        return true;
    }

    // what one more id costs, on every context that shares the ids. false: a context cannot
    bool pool_id_cost(pool_cost_t & need) const {
        for (llama_context * ctx : { ctx_tgt, ctx_dft }) {
            if (ctx == nullptr) {
                continue;
            }
            ggml_backend_buffer_type_t bufts[16];
            size_t sizes[16];
            const int32_t n = llama_seq_max_cost(ctx, seq_ceiling() + 1, bufts, sizes, 16);
            if (n < 0) {
                return false;
            }
            for (int32_t i = 0; i < std::min<int32_t>(n, 16); ++i) {
                need[bufts[i]] += sizes[i];
            }
        }
        pool_selftest_id_cost(need);
        return true;
    }

    // the floor: one conversation's worth held back per device, given how many ids and cells would be free
    pool_cost_t pool_reserve(size_t ids_free_after, size_t cells_free_after, uint32_t n_pool_after) const {
        pool_cost_t res;
        if (ids_free_after == 0) {
            pool_id_cost(res);
        }
        const uint32_t min_ctx = pool_min_ctx();
        if (cells_free_after < min_ctx) {
            pool_grow_cost(pool_pad(n_pool_after + (min_ctx - cells_free_after)), res);
        }
        return res;
    }

    // does `need` fit on every device next to `reserve` and the headroom? `str` says what was asked, per device
    bool pool_fits(const pool_cost_t & need, const pool_cost_t & reserve, std::string & str) const {
        const size_t headroom = (size_t) params_base.seq_max_headroom_mib * 1024 * 1024;

        pool_cost_t all = need;
        for (const auto & [buft, bytes] : reserve) {
            all[buft] += 0; // make sure the device is listed even when the need there is zero
        }

        bool fits = true;
        str.clear();
        for (const auto & [buft, bytes] : all) {
            const size_t res = reserve.count(buft) ? reserve.at(buft) : 0;
            size_t free = 0;
            if (!pool_buft_free(buft, free)) {
                // no device to ask (the CPU buffer type reports none): attempt and see, as the resize does
                str += string_format("%s%s %.1f MiB + %.1f MiB reserve (no device to ask)", str.empty() ? "" : ", ",
                        ggml_backend_buft_name(buft), bytes / 1048576.0, res / 1048576.0);
                continue;
            }
            str += string_format("%s%s %.1f MiB + %.1f MiB reserve (%.1f MiB free)", str.empty() ? "" : ", ",
                    ggml_backend_buft_name(buft), bytes / 1048576.0, res / 1048576.0, free / 1048576.0);
            if (bytes + res + headroom > free) {
                fits = false;
            }
        }
        return fits;
    }

    // the cell count of every context that holds cells moves together, all or nothing
    bool pool_resize(uint32_t n_new, const char * why) {
        const uint32_t n_cur = pool_size();
        const bool dft_too = ctx_dft != nullptr && llama_n_ctx(ctx_dft) == n_cur;

        if (!llama_set_n_ctx(ctx_tgt, n_new)) {
            return false;
        }
        if (dft_too && !llama_set_n_ctx(ctx_dft, n_new)) {
            llama_set_n_ctx(ctx_tgt, n_cur);
            return false;
        }

        SRV_INF("[pool] %u -> %u cells (%zu held, %zu ids of %u): %s\n", n_cur, n_new, pool_cells_held(), seq_ceiling() - seq_ids_free(), seq_ceiling(), why);
        SRV_DBG("%s", n_new > n_cur ? "__TEST_TAG_POOL_GROW__\n" : "__TEST_TAG_POOL_SHRINK__\n");

        return true;
    }

    // cells are short by n_more: grow the pool if the device (after idle ids are given back) can pay for it and
    // the reserve stays. `urgent` may spend the reserve (RULE 1). false: not grown, nothing changed
    bool pool_grow(size_t n_more, bool urgent, const char * why) {
        if (!pool_elastic() || n_more == 0) {
            return false;
        }

        const uint32_t n_cur = pool_size();
        const size_t   held  = pool_cells_held();

        // a quarter of the pool at a time, unless only the exact need fits
        std::vector<uint32_t> targets;
        targets.push_back(pool_pad(n_cur + std::max<size_t>(n_more, n_cur / 4)));
        if (pool_pad(n_cur + n_more) != targets.back()) {
            targets.push_back(pool_pad(n_cur + n_more));
        }

        for (int pass = 0; pass < 2; ++pass) {
            for (const uint32_t n_new : targets) {
                pool_cost_t need;
                if (!pool_grow_cost(n_new, need)) {
                    SRV_INF("[pool] stays at %u cells: a context cannot change its cell count\n", n_cur);
                    return false;
                }
                const pool_cost_t reserve = urgent ? pool_cost_t{} : pool_reserve(seq_ids_free(), n_new > held ? n_new - held : 0, n_new);

                std::string str;
                if (pool_fits(need, reserve, str)) {
                    if (pool_resize(n_new, why)) {
                        SRV_INF("[pool] grew for %zu more cells: %s\n", n_more, str.c_str());
                        return true;
                    }
                    return false;
                }
                SRV_DBG("[pool] %u -> %u cells does not fit: %s\n", n_cur, n_new, str.c_str());
            }
            // cells short, ids idle: give the idle ids back and ask once more
            if (pass == 0 && !seq_ceiling_shrink()) {
                break;
            }
        }

        SRV_INF("[pool] stays at %u cells: %zu more do not fit (%s)\n", n_cur, n_more, why);
        return false;
    }

    // ids are short by `need` bytes per device: shrink the pool so the cells nobody holds pay for it, keeping
    // --pool-min-ctx free. false: not shrunk, nothing changed
    bool pool_shrink_for(const pool_cost_t & need, const char * why) {
        if (!pool_elastic()) {
            return false;
        }

        const uint32_t n_cur    = pool_size();
        const size_t   held     = pool_cells_held();
        const size_t   headroom = (size_t) params_base.seq_max_headroom_mib * 1024 * 1024;

        // [pool-restore] what stays free: --pool-min-ctx, or the largest cached conversation's way back
        size_t n_largest = 0;
        const uint32_t floor = pool_restore_floor(n_largest);
        const std::string floor_str = n_largest > 0
            ? string_format("the floor of %u: a %zu-token cached conversation plus the batch margin", floor, n_largest)
            : string_format("the floor of %u: --pool-min-ctx", floor);

        if (n_cur <= held + floor) {
            SRV_INF("[pool] stays at %u cells: %zu held, nothing idle beyond %s (%s)\n", n_cur, held, floor_str.c_str(), why);
            return false;
        }
        const size_t idle = n_cur - held - floor; // cells that could go

        // per-cell cost per device, from what one padding step would add
        pool_cost_t step;
        if (!pool_grow_cost(pool_pad(n_cur + 256), step)) {
            return false;
        }

        // cells to give back so every device gains what it lacks
        size_t n_give = 0;
        for (const auto & [buft, bytes] : need) {
            size_t free = 0;
            if (!pool_buft_free(buft, free)) {
                continue;
            }
            if (bytes + headroom <= free) {
                continue;
            }
            const size_t lack = bytes + headroom - free;
            const size_t per_cell = step.count(buft) ? step.at(buft) / 256 : 0;
            if (per_cell == 0) {
                SRV_INF("[pool] stays at %u cells: %s lacks %.1f MiB and holds no cells\n", n_cur, ggml_backend_buft_name(buft), lack / 1048576.0);
                return false; // the cells live elsewhere; giving them back helps nothing here
            }
            n_give = std::max(n_give, (lack + per_cell - 1) / per_cell);
        }
        if (n_give == 0) {
            return false;
        }
        if (n_give > idle) {
            SRV_INF("[pool] stays at %u cells: %zu would have to go and only %zu are idle beyond %s (%s)\n", n_cur, n_give, idle, floor_str.c_str(), why);
            return false;
        }

        // at least half the idle cells, so the next id does not resize again
        n_give = std::max(n_give, idle / 2);

        const uint32_t n_new = pool_pad(n_cur - n_give);
        if (n_new >= n_cur) {
            return false;
        }

        if (!pool_resize(n_new, why)) {
            return false;
        }
        {
            std::string bytes_str;
            for (const auto & [buft, per_cell] : step) {
                bytes_str += string_format("%s%s %.1f MiB", bytes_str.empty() ? "" : ", ", ggml_backend_buft_name(buft), (n_cur - n_new) * (per_cell / 256) / 1048576.0);
            }
            SRV_INF("[pool] gave back %u cells (%s) for %s, keeping %zu held and %s\n", n_cur - n_new, bytes_str.c_str(), why, held, floor_str.c_str());
        }
        return true;
    }

    // [pool] admission asks whether the WHOLE conversation fits - its id (if none is free) and its cells (if the
    // pool has too few), together, next to the reserve - and grows the pool for the cells when it does. When it
    // does not, nothing is refused here: the id path evicts an idle conversation for its id and the batch ladder
    // evicts for cells; the pool is grown then, at need, if it can be.
    void pool_admit(size_t n_tokens, const char * why) {
        if (!pool_elastic()) {
            return;
        }

        const size_t free_cells = pool_cells_free();
        const size_t margin     = 1 + slots.size(); // the next token of every seat, and this one's
        if (n_tokens + margin <= free_cells) {
            return; // cells are there; the id path asks its own question
        }

        const size_t n_more = n_tokens + margin - free_cells;
        const uint32_t n_new = pool_pad(pool_size() + n_more);

        pool_cost_t need;
        if (seq_ids_free() == 0 && !pool_id_cost(need)) {
            return;
        }
        if (!pool_grow_cost(n_new, need)) {
            return;
        }
        const pool_cost_t reserve = pool_reserve(seq_ids_free() > 0 ? seq_ids_free() - 1 : 0, 0, n_new);

        std::string str;
        if (!pool_fits(need, reserve, str)) {
            SRV_INF("[pool] %s (%zu tokens) does not fit whole: %s; it will be packed by eviction\n", why, n_tokens, str.c_str());
            return;
        }

        pool_grow(n_more, /*urgent*/ false, why);
    }

    // [seq] an id for a new sequence, in rising order of cost: one nobody holds; one more from the model if the
    // device has room (the ceiling grows); the LRU finished conversation's, which goes to the prompt cache; and
    // last a yielded generation's, whose state is copied out and back later - the price of having no room for
    // another id. -1 when even that is not allowed or possible.
    llama_seq_id seq_id_acquire(bool allow_offload, const char * why, bool urgent = false) {
        llama_seq_id id = seq_id_free();
        if (id >= 0) {
            return id;
        }

        // an id holding nothing (a cleared or erased conversation) costs nothing to take back, and less than a raise
        for (auto * s : seq_evictable()) {
            if (seq_prompt(*s).tokens.empty()) {
                id = s->seq_id;
                seq_evict(*s, /*force*/ true, why);
                return id;
            }
        }

        if (seq_ceiling_raise(urgent)) {
            id = seq_id_free();
            GGML_ASSERT(id >= 0);
            return id;
        }

        // a state the cache refuses is skipped on the first pass; if every candidate is refused the LRU one goes
        // anyway, as the launch path always did, because refusing the request is worse
        for (int pass = 0; pass < 2; ++pass) {
            for (auto * s : seq_evictable()) {
                id = s->seq_id;
                if (seq_evict(*s, /*force*/ pass == 1, why)) {
                    return id;
                }
            }
        }

        if (allow_offload) {
            auto cands = seq_mid_flight(/*resident_only*/ true);
            if (!cands.empty()) {
                id = cands.front()->seq_id;
                seq_offload(*cands.front());
                return id;
            }
        }

        return -1;
    }

    //
    // [seats] batch positions follow residency. The operator's spec: "i want a pool and whatever fits in here gets
    // batched." A seat is a host-side struct (a sampler, a prompt, pointers); the SEQUENCE it drives is what costs
    // memory (its state rows and compute-buffer share, which llama_seq_max_cost prices per device). So the seat
    // count is a consequence of what is resident and runnable, never a setting:
    //
    //   floor    --parallel: seats that always exist (the value nobody has to set; 1 is fine)
    //   cap      the ceiling cap (--seq-max, else the library maximum), and what one n_batch view can hold
    //            when every seat adds its sampled token and its draft
    //   growth   a seat is added FOR a sequence: a resident one with a new turn or a resume (it has an id), or a
    //            new conversation whose id the pool can hand out right now (seq_id_acquire, which asks the cost
    //            query before evicting anything and never displaces a mid-flight generation for a newcomer)
    //   shrink   the quiet moment gives seats above the floor back; their idle sequences stay resident
    //
    // Seats are cheap; sequences are not. Growing seats never grows the ceiling by itself: the ceiling only
    // moves through seq_ceiling_raise(), and only when the device says one more sequence fits.
    //

    uint32_t seat_floor() const {
        return (uint32_t) std::max(1, params_base.n_parallel);
    }

    // tokens one generating seat puts in a batch: its sampled token and at most n_draft_max drafted ones
    int32_t seat_tokens_per_batch() const {
        return 1 + std::max(0, common_speculative_n_max(&params_base.speculative));
    }

    // the most seats there can be: no more than there can be ids, no more than fit one n_batch view when all of
    // them generate at once (a seat's tokens straddling a view boundary is not a case the decode loop handles),
    // and no more than --parallel-max when someone wants the batch held narrower than the pool. The floor always
    // exists, as it always did, whatever the batch says.
    uint32_t seat_cap() const {
        const int32_t  n_batch  = llama_n_batch(ctx_tgt);
        const uint32_t by_batch = (uint32_t) std::max<int32_t>(1, n_batch / seat_tokens_per_batch());
        uint32_t cap = std::min(seq_ceiling_cap(), by_batch);
        if (params_base.n_parallel_max > 0) {
            cap = std::min(cap, (uint32_t) params_base.n_parallel_max);
        }
        return std::max(seat_floor(), cap);
    }

    // an idle seat: one holding nothing first, else the least recently used. nullptr when all are processing
    server_slot * seat_idle_pick() {
        server_slot * res = nullptr;
        for (auto & cand : slots) {
            if (cand.is_processing()) {
                continue;
            }
            if (res == nullptr ||
                (!cand.bound() && res->bound()) ||
                (cand.bound() == res->bound() && cand.t_last_used < res->t_last_used)) {
                res = &cand;
            }
        }
        return res;
    }

    void seat_init(server_slot & slot, int id) {
        slot.id      = id;
        slot.seq_id  = -1; // a sequence is bound at the first launch
        slot.ctx_tgt = ctx_tgt;
        slot.ctx_dft = ctx_dft;
        slot.mem.init(ctx_tgt, ctx_dft);
        slot.spec    = spec.get();
        slot.n_ctx   = n_ctx_slot();

        slot.mctx                   = mctx;
        slot.prompt.tokens.has_mtmd = mctx != nullptr;

        SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

        slot.callback_on_release = [this](int id_slot) {
            queue_tasks.pop_deferred_task(id_slot);
        };

        slot.callback_on_reset = [this](const server_slot & slot) {
            // flush the generated token stats before reset()
            if (slot.stats.n_gen > 0) {
                metrics_on_prediction(slot);
            }
        };

        slot.reset();
    }

    // one more seat. nullptr at the cap - the caller then waits for one to free up, as every request used to
    server_slot * seat_add(const char * why) {
        if (slots.size() >= seat_cap()) {
            SRV_DBG("seats stay at %zu: the cap is %u (%s)\n", slots.size(), seat_cap(), why);
            return nullptr;
        }

        slots.emplace_back();
        server_slot & slot = slots.back();
        seat_init(slot, (int) slots.size() - 1);

        if (slots.size() > seat_floor()) {
            SRV_INF("seats: %zu (grew for %s; floor %u, cap %u, ceiling %u)\n",
                    slots.size(), why, seat_floor(), seat_cap(), seq_ceiling());
        }

        return &slot;
    }

    // the quiet moment: seats above the floor go, from the back. An idle seat's sequence stays resident without
    // one (seat_release, zero copy) and is re-seated by its next turn. Only the ids and the pool cost anything,
    // and neither moves here.
    void seat_shrink() {
        const size_t n_before = slots.size();

        while (slots.size() > seat_floor()) {
            server_slot & slot = slots.back();
            if (slot.is_processing()) {
                break;
            }
            if (slot.bound() && !slot.seat_release()) {
                break; // an idle seat always releases; if it ever does not, keep it rather than lose its sequence
            }
            slots.pop_back();
        }

        if (slots.size() != n_before) {
            SRV_INF("seats: %zu (shrunk from %zu to the floor)\n", slots.size(), n_before);
        }
    }

    // [deadline] RULE 1 of the scheduler: a sequence past --slot-resume-after gets compute regardless. This takes
    // it from the running generation that has had its seat the longest (fair, and the one most likely to be
    // past its own turn): the seat is released, and with `offload` the sequence's state is copied out too, which
    // frees its id and its cells. A prefill cannot be taken (half its prompt is in the KV; seat_release refuses)
    // nor a parent or child (shared cells). nullptr when nothing running can be taken this pass.
    server_slot * pool_preempt_running(bool offload, const server_sequence & waiting, const char * why) {
        server_slot * victim = nullptr;
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.task) {
                continue;
            }
            if (slot.task->is_parent() || slot.task->is_child()) {
                continue;
            }
            if (victim == nullptr || slot.t_seated_us < victim->t_seated_us) {
                victim = &slot;
            }
        }
        if (victim == nullptr) {
            return nullptr;
        }

        server_sequence * v = victim->seq;
        const int64_t t_now = ggml_time_us();

        SLT_WRN(*victim, "preempted after %d generated tokens and %" PRId64 " ms on the seat: task %d %s (suspended %" PRId64 " ms)\n",
                (int) victim->stats.n_gen, (t_now - victim->t_seated_us) / 1000,
                waiting.task ? waiting.task->id : -1, why, ggml_time_ms() - waiting.t_suspended_ms);

        if (!victim->seat_release()) {
            // a generating seat with a task always releases; if it ever does not, nothing was touched
            return nullptr;
        }

        if (offload) {
            seq_offload(*v);
        }

        SRV_DBG("%s", "__TEST_TAG_DEADLINE_PREEMPT__\n");

        return victim;
    }

    // [seq] a fresh sequence on this seat; whatever idle sequence the seat held stays resident
    server_sequence * seq_new_for(server_slot & slot, const char * why, bool allow_offload = true) {
        const llama_seq_id id = seq_id_acquire(allow_offload, why);
        if (id < 0) {
            return nullptr;
        }

        if (slot.bound()) {
            slot.seat_release();
        }

        auto & s = seq_create(id);
        slot.seat_acquire(s);

        return &s;
    }

    bool seat_ensure_bound(server_slot & slot) {
        return slot.bound() || seq_new_for(slot, "slot action") != nullptr;
    }

    //
    // [seq] T2.5 the pressure ladder. llama_decode() could not place a batch and the pool has to give something
    // up. It runs per BATCH, not per launch: sequences grow after admission, and the production case (two
    // seats, both deep) exhausts the pool with nothing resident to evict. Rungs, in rising order of cost:
    //   0  defer     this batch already placed some tokens: the rest waits for the next one (nothing moves)
    //   1  evict     the LRU finished conversation, to the prompt cache (a copy, nothing was running)
    //   1b offload   a yielded generation waiting for a seat, to RAM (a copy, nothing was running)
    //   2  suspend   the largest RUNNING generation, with a copy - the 13 GB path on GLM. It WILL fire on that
    //                box and that is the design working, not failing: the alternative is rung 3
    //   3  fail      ONE sequence, the largest pending, spilled not destroyed (T0.3) - only when nothing above
    //                can move
    // One move per call; the caller retries the batch. Only under --kv-unified: without it every id has its own
    // stream, and one sequence's wall is nobody else's.
    //

    // [pool] rung -1: the batch found no room for its tokens from `off` on, and the pool can grow. Nobody is
    // evicted for cells the device would have given. A deadline-forced batch may spend the reserve.
    bool pool_grow_pending(int32_t off) {
        if (!pool_elastic()) {
            return false;
        }
        size_t n_pending = 0;
        for (const auto & [slot, n] : batch_pending_slots(off)) {
            n_pending += n;
            GGML_UNUSED(slot);
        }
        if (n_pending == 0) {
            return false;
        }
        const size_t free_cells = pool_cells_free();
        const size_t margin     = 1 + slots.size();
        const size_t n_more     = n_pending + margin > free_cells ? n_pending + margin - free_cells : n_pending;
        return pool_grow(n_more, /*urgent*/ false, "a batch found no room");
    }

    // rung 1: a finished conversation with cells leaves the pool - the shallowest first, cells being what is
    // short here (SEQ_NEED_CELLS, see seq_evictable). A refused save keeps it unless `force`. `keep` is a seat
    // being prepared for a task, whose own sequence is not a candidate (the restore drops its cells anyway)
    bool pool_evict_resident(bool force, const char * why, const server_slot * keep = nullptr) {
        if (!params_base.kv_unified) {
            return false;
        }

        for (auto * s : seq_evictable(SEQ_NEED_CELLS)) {
            if (seq_prompt(*s).tokens.empty()) {
                continue;
            }
            if (keep != nullptr && s->slot == keep) {
                continue;
            }
            if (seq_evict(*s, force, why, SEQ_NEED_CELLS)) {
                return true;
            }
        }

        return false;
    }

    // rung 1b: a yielded generation that holds cells while waiting for a seat gives them up
    bool pool_offload_waiting() {
        if (!params_base.kv_unified) {
            return false;
        }

        auto cands = seq_mid_flight(/*resident_only*/ true);
        if (cands.empty()) {
            return false;
        }

        seq_offload(*cands.front());

        return true;
    }

    // the cells the sequences with an id occupy: one per token under --kv-unified, where the pool is one span
    // and the server knows who holds what. Cells a parent shares with its children are counted once per
    // holder, so this over-counts them, which errs on the side of waiting. T2.6 makes it exact.
    size_t pool_cells_held() const {
        size_t n = 0;
        for (const auto & s : seqs) {
            if (s.seq_id >= 0) {
                n += seq_prompt(s).tokens.size();
            }
        }
        return n;
    }

    // room for a state, plus the next token of every running sequence and of itself: a state restored into
    // exactly its own cells is suspended again on the next batch, a copy each way for nothing. Without
    // --kv-unified only the restore itself can tell.
    bool pool_has_room_for(const server_sequence & s) const {
        if (!params_base.kv_unified) {
            return true;
        }

        size_t margin = 1;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                margin++;
            }
        }

        return pool_cells_held() + s.prompt.tokens.size() + margin <= (size_t) llama_n_ctx(ctx_tgt);
    }

    // the processing slots with tokens pending in the batch from `off` on, with how many each
    std::vector<std::pair<server_slot *, int32_t>> batch_pending_slots(int32_t off) {
        std::vector<std::pair<server_slot *, int32_t>> res;
        for (auto & slot : slots) {
            if (!slot.is_processing()) {
                continue;
            }
            int32_t n = 0;
            for (int32_t i = off; i < batch.size(); i++) {
                if (batch.tokens[i].seq_id == slot.seq_id) {
                    n++;
                }
            }
            if (n > 0) {
                res.emplace_back(&slot, n);
            }
        }
        return res;
    }

    // a slot's last n_pending batch tokens were never decoded: cut the prompt back so it says what the KV holds
    // (checkpoints past the cut go with it). false when the cut lands inside a media chunk - keep_first()
    // refuses before touching anything, so the prompt is as it was.
    bool slot_drop_pending(server_slot & slot, int32_t n_pending) {
        const int n_kv = slot.prompt.n_tokens() - n_pending;

        try {
            slot.prompt.tokens.keep_first(n_kv);
        } catch (const std::exception & e) {
            SLT_WRN(slot, "cannot cut the prompt back to %d tokens: %s\n", n_kv, e.what());
            return false;
        }

        slot.prompt.checkpoints.remove_if([&](const common_prompt_checkpoint & ckpt) {
            return ckpt.n_tokens > n_kv;
        });

        return true;
    }

    // take a slot's pending tokens out of the batch and remap the batch indices of everyone else
    void batch_drop_pending(server_slot & slot, int32_t off) {
        slot.i_batch = -1;

        const auto map = batch.remove_from(slot.seq_id, off);

        for (auto & s : slots) {
            if (s.i_batch >= 0) {
                s.i_batch = map[s.i_batch];
            }
            for (auto & i : s.spec_i_batch) {
                i = map[i];
            }
        }
    }

    // rung 0: the batch got some of its tokens in before the wall (off > 0), so the ones after it simply wait
    // for the next batch. Nothing is copied or failed. It matters for WHO the ladder picks next: n_batch halves
    // down to 1 on the way here, the last free cells go to whichever slot was batched first, and what is left
    // pending is then only the slot batched after it - the small innocent one, as a rule. The batch that follows
    // fails with nothing placed, every sequence in it pending, and rung 2 can choose the largest.
    bool pool_defer_pending(int32_t off) {
        if (off == 0) {
            return false;
        }

        int n_deferred = 0;

        for (const auto & [slot, n_pending] : batch_pending_slots(off)) {
            if (!slot_drop_pending(*slot, n_pending)) {
                continue; // stays pending; the rungs below deal with it
            }

            if (slot->state == SLOT_STATE_DONE_PROMPT) {
                // its last prompt chunk was in the pending part: it is a prompt in progress again
                slot->state = SLOT_STATE_PROCESSING_PROMPT;
            }
            if (slot->state == SLOT_STATE_GENERATING) {
                // the sampled token stays in `sampled` and is batched again; the draft is drawn again
                slot->spec_draft.clear();
                slot->spec_i_batch.clear();
            }

            batch_drop_pending(*slot, off);

            SLT_DBG(*slot, "KV pool full after %d of this batch's tokens: %d pending tokens wait for the next batch\n", off, n_pending);

            n_deferred++;
        }

        return n_deferred > 0;
    }

    // rung 2: the largest RUNNING generation with tokens pending gives up its seat and its cells (seat_release,
    // then the offload copy); the resume pass brings it back when there is room. Only with company: alone in the
    // batch there is nobody to make room for - it would be restored into the same full pool and suspended again,
    // forever - and that is rung 3's case. A prefill cannot be suspended (half its prompt is in the KV;
    // seat_release() refuses), nor a parent or child (shared cells, not explored): they fall through.
    bool pool_suspend_running(int32_t off) {
        // test hook: LLAMA_SERVER_POOL_NO_SUSPEND skips this rung so the terminal rung stays reachable by a
        // test. With two or more live sequences every exhaustion otherwise has a rung 0-2 move.
        static const bool disabled = std::getenv("LLAMA_SERVER_POOL_NO_SUSPEND") != nullptr;

        if (disabled || !params_base.kv_unified) {
            return false;
        }

        const auto pending = batch_pending_slots(off);
        if (pending.size() < 2) {
            return false;
        }

        server_slot * victim    = nullptr;
        int32_t       n_pending = 0;

        for (const auto & [slot, n] : pending) {
            if (slot->state != SLOT_STATE_GENERATING || !slot->task) {
                continue;
            }
            if (slot->task->is_parent() || slot->task->is_child()) {
                continue;
            }
            if (!victim || slot->prompt.n_tokens() > victim->prompt.n_tokens()) {
                victim    = slot;
                n_pending = n;
            }
        }

        if (!victim) {
            return false;
        }

        server_slot & slot = *victim;

        if (!slot_drop_pending(slot, n_pending)) {
            return false;
        }

        slot.spec_draft.clear();
        slot.spec_i_batch.clear();

        batch_drop_pending(slot, off);

        SLT_WRN(slot, "KV pool full: suspending the largest running generation with a copy (%d tokens in the KV, %d pending, %zu sequences in the batch)\n",
                slot.prompt.n_tokens(), n_pending, pending.size());

        server_sequence * s = slot.seq;

        const bool released = slot.seat_release();
        GGML_ASSERT(released);

        seq_offload(*s);

        SRV_DBG("%s", "__TEST_TAG_POOL_SUSPEND__\n");

        return true;
    }

    // [seq] the restore side of the ladder: an offloaded generation's state comes back under `id` once the pool
    // has room for it. Finished residents are evicted to make that room (rung 1; a refused save is forced on the
    // second pass, as everywhere). false: not restored, the record untouched, the id left free. `may_wait` says
    // whether the room can still appear on its own - something is running, or a resident yielded generation is
    // waiting for a seat and will finish - as opposed to a state that does not fit an empty pool.
    bool seq_restore_with_room(server_sequence & s, llama_seq_id id, bool & may_wait) {
        GGML_ASSERT(s.offloaded() && s.mid_flight());

        for (int pass = 0; pass < 2; ++pass) {
            for (;;) {
                if (!pool_has_room_for(s)) {
                    // [pool] the cells it needs, from the device first
                    const size_t need = s.prompt.tokens.size() + 1 + slots.size();
                    const size_t have = pool_cells_free();
                    if (need > have) {
                        pool_grow(need - have, /*urgent*/ false, "resuming a suspended generation");
                    }
                }
                if (pool_has_room_for(s) && seq_restore(s, id)) {
                    return true;
                }
                if (!pool_evict_resident(/*force*/ pass == 1, "resuming a suspended generation")) {
                    break;
                }
            }
        }

        may_wait = false;
        for (const auto & slot : slots) {
            may_wait |= slot.is_processing();
        }
        for (const auto & o : seqs) {
            may_wait |= &o != &s && o.mid_flight() && !o.seated() && !o.offloaded();
        }

        return false;
    }

    // [pool-restore] the prompt-cache restore side of the ladder, defect (b) of the 2026-09-07 incident: a cached
    // conversation about to be loaded into `slot` (prompt_load -> state_seq_set_data -> state_read_meta) needs
    // its cells BEFORE state_read_meta asks for them - it finds them all or fails, and the fallback is a full
    // prefill (141k tokens, 8.4 minutes on GLM, past the client's timeout, looping). pool_admit, pool_grow_pending
    // and seq_restore_with_room were pool-grow hooks; this path was not. Room is made the way the ladder makes
    // it: the pool grown from the device (idle ids given back first when the device says no - pool_grow's two
    // passes), then finished residents evicted for their cells, the shallowest first; a refused save is forced
    // on the second pass, as for a suspended generation coming back. The seat's own cells count as room: the
    // restore replaces them (state_read_meta seq_rm's the destination first). The entry is looked up again
    // after every step - an eviction's save can reshape the cache. false: the conversation will be prefilled,
    // and the log says why.
    bool pool_room_for_restore(server_slot & slot, const server_tokens & tokens_new) {
        if (!pool_elastic() || !prompt_cache) {
            return true; // static pool: the restore is attempted as before
        }

        const char * why = "a cached conversation coming back";

        auto need_more = [&](size_t & n_tokens) -> size_t {
            const server_prompt_cache_state * cand = prompt_cache->find(slot.prompt, tokens_new);
            if (cand == nullptr) {
                n_tokens = 0;
                return 0;
            }
            n_tokens = cand->prompt.tokens.size();
            const size_t need = n_tokens + 1 + slots.size();
            const size_t have = pool_cells_free() + (slot.bound() ? (size_t) slot.prompt.n_tokens() : 0);
            return need > have ? need - have : 0;
        };

        size_t n_tokens = 0;
        size_t n_more   = need_more(n_tokens);
        if (n_more == 0) {
            return true;
        }

        SRV_INF("[pool] %s: %zu tokens, %zu more cells than the pool has free\n", why, n_tokens, n_more);

        for (int pass = 0; pass < 2; ++pass) {
            for (;;) {
                if (pool_grow(n_more, /*urgent*/ false, why)) {
                    n_more = need_more(n_tokens);
                    if (n_more == 0) {
                        return true;
                    }
                    continue;
                }
                if (!pool_evict_resident(/*force*/ pass == 1, why, &slot)) {
                    break;
                }
                n_more = need_more(n_tokens);
                if (n_more == 0) {
                    return true;
                }
            }
        }

        SRV_WRN("[pool] %s: %zu tokens do not fit and the pool cannot grow (%u cells, %zu held, %zu free); it will be prefilled\n",
                why, n_tokens, pool_size(), pool_cells_held(), pool_cells_free());
        return false;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    // [seq] a seat for the task, and a sequence on it: the resident one sharing the longest prefix (re-seated
    // without a copy if it has no seat), else a fresh one. The seat's own idle sequence stays resident either
    // way; it is only evicted when its id is needed (seq_id_acquire).
    server_slot * get_available_slot(const server_task & task) {
        server_slot     * ret   = nullptr;
        server_sequence * match = nullptr;

        bool update_cache = false;

        // carried out of the LCP block purely so the decision can be logged below
        float f_keep_sel = -1.0f;
        float f_sim_sel  = -1.0f;

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
                if (ret->is_processing()) {
                    return ret; // the caller defers the task; nothing below may touch a running seat
                }
            }
        }

        // find the sequence that has at least n% prompt similarity: any finished one nobody is decoding, seated
        // or not - never a yielded generation, whose prompt is a task in progress. A requested slot pins the
        // conversation to that seat (slot save/restore relies on it), so another seat's sequence does not
        // qualify then; one that has no seat does, it was pushed off some seat and is nobody's.
        if (slot_prompt_similarity != 0.0f) {
            float f_sim_best = 0;

            for (auto & s : seqs) {
                if (s.offloaded() || s.mid_flight() || seq_running(s)) {
                    continue;
                }
                if (task.id_slot != -1 && s.slot != nullptr && s.slot != ret) {
                    continue;
                }

                const auto & tokens = seq_prompt(s).tokens;

                if (tokens.empty()) {
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const size_t lcp_len = tokens.get_common_prefix(task.tokens);
                const float f_sim_cur = float(lcp_len) / task.tokens.size();

                SRV_TRC(" - sequence %d: checking sim = %.3f (%zu/%zu) > %.3f\n", s.seq_id, f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

                if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                    f_sim_best = f_sim_cur;

                    match = &s;
                }
            }

            if (match != nullptr) {
                const float f_keep = (f_sim_best*task.tokens.size()) / seq_prompt(*match).tokens.size();

                SRV_INF("selected sequence %d by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f, %s\n",
                        match->seq_id, f_sim_best, slot_prompt_similarity, f_keep,
                        match->slot != nullptr ? "on its seat" : "resident without a seat");

                f_keep_sel = f_keep;
                f_sim_sel  = f_sim_best;

                // save whenever ANYTHING is lost. everything in the sequence past the
                // common prefix is destroyed further down by keep_first()/seq_rm(),
                // so f_keep is the fraction PRESERVED - not evidence that this is
                // the same conversation continuing.
                //
                // with a large shared system prompt that distinction matters: two
                // unrelated conversations sharing a 6.5k preamble score f_keep > 0.9
                // while the entire distinguishing tail is thrown away. at 0.5f every
                // conversation shorter than ~13k tokens looked like a continuation
                // of every other one.
                //
                // f_keep == 1.0 is the genuine continuation: the sequence is a strict
                // prefix of the incoming prompt, nothing is lost, nothing to save.
                if (f_keep < 1.0f) {
                    update_cache = true;
                }

                // a seated match is used from its own seat
                if (match->slot != nullptr) {
                    ret = match->slot;
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 "\n", t_last);
            }
        }

        // [seats] every seat is busy. A pinned request waits for its seat. Anyone else gets a new one if the pool
        // has room for their sequence: a resident match needs no id (it holds one already - "resident without a
        // seat" is exactly what a seat is for); a new conversation asks for an id without displacing anything
        // mid-flight - a yielded generation is only offloaded for a newcomer by a seat that has freed up.
        bool bound_here = false;
        if (ret == nullptr && task.id_slot == -1 && slots.size() < seat_cap()) {
            if (match != nullptr) {
                ret = seat_add("a resident conversation with a new turn");
            } else {
                pool_admit(task.n_tokens(), "a new conversation");
                const llama_seq_id id = seq_id_acquire(/*allow_offload*/ false, "a new conversation");
                if (id >= 0) {
                    ret = seat_add("a new conversation");
                    if (ret != nullptr) {
                        ret->seat_acquire(seq_create(id));
                        bound_here = true;
                    }
                    // else: the id stays free, nothing holds it; the task is deferred like any other
                }
            }
        }

        if (ret == nullptr) {
            return nullptr;
        }

        if (match != nullptr) {
            if (match->slot != ret) {
                if (ret->bound()) {
                    ret->seat_release();
                }
                ret->seat_acquire(*match);
                SLT_INF(*ret, "re-seated resident sequence %d (%d tokens, zero copy)\n", ret->seq_id, ret->prompt.n_tokens());
            }
        } else if (!bound_here && seq_new_for(*ret, "new conversation") == nullptr) {
            SLT_ERR(*ret, "%s", "no sequence id could be acquired for the task\n");
            return nullptr;
        }

        if (ret) {
            // cache prompts only for completion tasks
            const bool cache_usable = prompt_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            // THESE ARE TWO DIFFERENT QUESTIONS and they had one answer.
            //
            //   do_save: is this slot about to lose context worth keeping?
            //   do_load: is there something in the cache better than what it holds?
            //
            // sharing a flag meant a high f_keep silently skipped the LOOKUP as
            // well as the save, so a conversation could sit in the cache, be a
            // perfect match for the incoming prompt, and never be consulted.
            //
            // consulting is unconditionally safe: load() seeds its comparison with
            // the slot's OWN f_keep/f_sim and only replaces the prompt if some
            // entry strictly beats it on both.
            const bool do_save = cache_usable && update_cache;
            const bool do_load = cache_usable;

            // the save/load decision is otherwise invisible below INFO: every step of
            // it logs at SRV_TRC (verbosity 4), and when update_cache is false
            // NOTHING is emitted at all - the skip has to be inferred from the
            // absence of a line. That made a regression costing 72% of all prefill
            // indistinguishable from the cache simply not being there.
            if (cache_usable) {
                SLT_INF(*ret, "prompt cache: save = %d, load = %d, f_keep = %.3f, f_sim = %.3f, %zu entries / %.1f MiB\n",
                        do_save, do_load, f_keep_sel, f_sim_sel,
                        prompt_cache->states.size(),
                        prompt_cache->size() / 1048576.0);
            }

            if (do_save || do_load) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                if (do_save) {
                    ret->prompt_save(*prompt_cache);
                }

                // [pool-restore] the cells first, then the restore; a restore that cannot have them is
                // still attempted, so that its failure is the KV cache's own line in the log
                if (do_load) {
                    pool_room_for_restore(*ret, task.tokens);
                }

                if (do_load && !ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear();
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // the pool has no room for even one token (n_batch == 1, ret == 1). fail ONE slot and let the others retry.
    // returns false if no processing slot has tokens pending in the batch
    bool fail_slot_under_pressure(int32_t off, const std::string & err) {
        // every pending token is unplaceable, so "the one that did not fit" is not defined by position:
        // batch.tokens[off] is just whichever slot was batched first. free the most room instead.
        server_slot * victim = nullptr;
        for (int32_t i = off; i < batch.size(); i++) {
            for (auto & slot : slots) {
                if (slot.seq_id == batch.tokens[i].seq_id && slot.is_processing()) {
                    if (!victim || slot.prompt.n_tokens() > victim->prompt.n_tokens()) {
                        victim = &slot;
                    }
                }
            }
        }

        if (!victim) {
            return false;
        }

        server_slot & slot = *victim;

        int32_t n_pending = 0;
        for (int32_t i = off; i < batch.size(); i++) {
            if (batch.tokens[i].seq_id == slot.seq_id) {
                n_pending++;
            }
        }

        SLT_ERR(slot, "%s n_tokens = %d, n_pending = %d\n", err.c_str(), slot.prompt.n_tokens(), n_pending);

        send_error(slot, err);

        // the KV holds everything up to the pending tokens. spill that, do not destroy it.
        if (prompt_cache && slot.task->type == SERVER_TASK_TYPE_COMPLETION) {
            const int n_kv = slot.prompt.n_tokens() - n_pending;
            try {
                slot.prompt.tokens.keep_first(n_kv);
                slot.prompt.checkpoints.remove_if([&](const common_prompt_checkpoint & ckpt) {
                    return ckpt.n_tokens > n_kv;
                });

                if (slot.prompt_save(*prompt_cache)) {
                    prompt_cache->update();
                    SLT_WRN(slot, "spilled %d tokens to the prompt cache\n", n_kv);
                    SLT_DBG(slot, "%s", "__TEST_TAG_POOL_EXHAUSTED_SPILL__\n");
                }
            } catch (const std::exception & e) {
                // keep_first() refuses to cut inside a media chunk
                SLT_WRN(slot, "cannot spill: %s\n", e.what());
            }
        }

        slot.release();
        slot.prompt_clear();
        slot.i_batch = -1;

        const auto map = batch.remove_from(slot.seq_id, off);

        for (auto & s : slots) {
            if (s.i_batch >= 0) {
                s.i_batch = map[s.i_batch];
            }
            for (auto & i : s.spec_i_batch) {
                i = map[i];
            }
        }

        return true;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // [seq] a seat launched outside get_available_slot() (child tasks) may hold no sequence yet
        if (!slot.bound() && seq_new_for(slot, "launch") == nullptr) {
            send_error(task, "no sequence id is available for this request", ERROR_TYPE_SERVER);
            return false;
        }

        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            // TODO: tmp until backend sampling is fully implemented
            slot.sampler_bind(task);

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());

            if (spec && !common_speculative_get_synth_probs(spec.get()).empty()) {
                const uint32_t seed = task.params.sampling.seed == LLAMA_DEFAULT_SEED
                    ? std::random_device{}()
                    : task.params.sampling.seed;
                slot.spec_synth_rng.seed(seed);
            }
        } else {
            slot.smpl.reset();
        }

        // the per-request limit takes priority over the global one
        slot.n_predict_max = task.params.n_predict != -1 ? task.params.n_predict : params_base.n_predict;

        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;
        slot.t_seated_us = ggml_time_us();

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }
        } else {
            // the token ends the text with a partial UTF-8 character, so no text can be sent yet.
            // still emit the token: stats.n_gen was already advanced for it, and in stream mode the
            // final response carries no tokens, so a skipped token here is never sent at all
            result.text_to_send = "";
        }

        slot.add_token(result);
        if (slot.task->params.stream) {
            send_partial_response(slot, result, false);
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_gen = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), (int) slot.stats.n_gen, slot.n_ctx);
        }

        // check the limits
        if (slot.stats.n_gen > 0 && slot.has_next_token && !slot.has_budget()) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_gen = %d, n_predict = %d\n", (int) slot.stats.n_gen, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_gen = %d, n_indent = %d\n", (int) slot.stats.n_gen, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && slot.stats.t_gen_ms() > slot.task->params.t_max_predict_ms) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_gen = %d, t_max_predict_ms = %d ms\n", (int) slot.stats.n_gen, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_gen = %d, n_remaining = %d, next token: %5d '%s'\n", (int) slot.stats.n_gen, slot.n_remaining(), result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.stats.n_prompt_cached;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = slot.stats.t_elapsed_us() / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->stats = slot.stats;
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->stats           = slot.stats;
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.stats.n_gen;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.stats.n_prompt_cached;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.seq_id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.seq_id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files, init_opt);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true, init_opt)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        // [seats] child completions get seats like anyone else; their ids are acquired at launch
        while (free_slots.size() < n_slots_needed) {
            server_slot * slot = seat_add("child completions");
            if (slot == nullptr) {
                break;
            }
            free_slots.push_back(slot);
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // [checkpoint-exp] each retained checkpoint sits at least this many times further
    // from the tip than the one after it. 2 gives 8k/16k/32k/64k/128k on min-step 8192.
    static constexpr int64_t CHECKPOINT_EXP_FACTOR = 2;

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        const int id_task = slot.task->id;

        // [checkpoint-exp] Thin checkpoints exponentially by distance from the tip,
        // rather than spacing them uniformly.
        //
        // Uniform spacing costs O(n) checkpoints on a long conversation - ~24 on a
        // 200k one at min-step 8192 - and each is 145.563 MiB of state carried inside
        // every prompt-cache entry. Measured on 2026-09-08: entries averaged 40.6
        // KiB/token against GLM's raw KV of 19.25, so checkpoints were roughly HALF of
        // the prompt cache, and the cache is what OOM-killed the server four times.
        //
        // A rollback lands near the tip in practice, so keep a checkpoint only if it
        // sits at least CHECKPOINT_EXP_FACTOR times further from the tip than the last
        // one kept: 8k, 16k, 32k, 64k, 128k back. That is O(log n) checkpoints spanning
        // the WHOLE history and densest where a divergence actually happens, instead of
        // O(n) evenly spread. ~5 instead of ~24 on 200k, and unlike simply lowering
        // --ctx-checkpoints it does not leave the older history uncovered.
        //
        // Checkpoints belonging to the current task are never thinned, as before.
        {
            const int64_t tip  = slot.prompt.n_tokens();
            const int64_t step = std::max<int64_t>(1, params_base.checkpoint_min_step);

            std::list<common_prompt_checkpoint> kept;

            int64_t keep_dist = 0; // distance from the tip of the last checkpoint kept

            for (auto it = slot.prompt.checkpoints.rbegin(); it != slot.prompt.checkpoints.rend(); ++it) {
                const int64_t dist = tip - it->n_tokens;

                // keep_dist == 0 means nothing has been kept yet: the checkpoint nearest
                // the tip is always retained, however close it is. It is the one a
                // rollback is most likely to land on, and it is created within
                // --checkpoint-min-step of the tip, so a distance test would drop it.
                if (it->id_task == id_task || keep_dist == 0 || dist >= std::max(step, keep_dist*CHECKPOINT_EXP_FACTOR)) {
                    if (it->id_task != id_task) {
                        keep_dist = dist;
                    }
                    kept.push_front(std::move(*it));
                    continue;
                }

                SLT_TRC(slot, "erasing context checkpoint too close to a later one (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", dist = %" PRId64 ", size = %.3f MiB)\n",
                        it->pos_min, it->pos_max, it->n_tokens, dist, (float) it->size() / 1024 / 1024);
            }

            slot.prompt.checkpoints = std::move(kept);
        }

        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        // replace an existing checkpoint at the same n_tokens instead of appending a duplicate
        {
            const int64_t n_tokens_new = slot.prompt.n_tokens() - n_tokens_cur;
            for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end(); ) {
                if (it->n_tokens == n_tokens_new) {
                    SLT_TRC(slot, "superseding context checkpoint at n_tokens = %" PRId64 "\n", it->n_tokens);
                    it = slot.prompt.checkpoints.erase(it);
                } else {
                    ++it;
                }
            }
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();

        cur.id_task = id_task;

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        cur.update_tgt(ctx_tgt, slot.seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        cur.update_dft(ctx_dft, slot.seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        // stash the draft's speculative state with the checkpoint
        common_speculative_get_state(spec.get(), slot.seq_id, cur.data_spec);

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    // returns false to decline the task, it is offered again after the decode is done
    bool process_single_task(server_task && task, bool is_yielding) {
        // while yielding, an encode / decode is running and only reading the server state is safe
        if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET) {
            SRV_DBG("decoding, decline task, id_task = %d\n", task.id);
            return false;
        }

        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else if (!launch_slot_with_task(*slot, std::move(task))) {
                        SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                        break; // drop the task
                    }

                    if (params_base.cache_idle_slots) {
                        // [seq] every finished sequence nobody is decoding, seated or not. With a unified cache this
                        // is the eager pressure valve: nothing stays resident past a launch (T2.5 makes it lazy);
                        // --no-cache-idle-slots keeps residents until an id or the pool is actually needed
                        for (auto * s : seq_evictable()) {
                            const auto & prompt = seq_prompt(*s);

                            SRV_TRC("saving idle sequence %d to prompt cache\n", s->seq_id);

                            const bool saved = prompt_state_save(*prompt_cache, ctx_tgt, ctx_dft, s->seq_id, prompt);
                            if (saved) {
                                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                prompt_cache->update();
                            }

                            if (params_base.kv_unified) {
                                // [TAG_IDLE_SLOT_CLEAR]
                                // Only discard the context once it is safely stored.
                                // prompt_state_save() returns false when the state exceeds the cache
                                // size limit; clearing anyway threw away a conversation that was
                                // never cached, and the skip is only logged inside the cache.
                                if (saved) {
                                    seq_erase(*s);
                                } else if (prompt.tokens.size() > 0) {
                                    SRV_WRN("sequence %d: state exceeds the prompt cache limit - keeping its context instead of clearing it\n", s->seq_id);
                                }
                                // an EMPTY sequence also returns false here, from the
                                // `prompt.tokens.size() == 0` early-out. It has nothing to keep and
                                // nothing to clear, and reporting it as a size-limit refusal sent an
                                // investigation looking for a limit that was two orders of magnitude
                                // away (2,580 MiB state against an 81,920 MiB --cache-ram).
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                    // [seq] or the task is a yielded generation waiting for a seat. Left there it resumes into a
                    // dead connection and holds a seat, and its KV, for the rest of a generation nobody will
                    // read. What it generated so far stays as a finished conversation, like a cancelled seat
                    // keeps its prompt; an offloaded one has nothing to keep.
                    for (auto & s : seqs) {
                        if (s.mid_flight() && s.task->id == task.id_target) {
                            SRV_INF("cancel: dropping suspended task %d (%d tokens in)\n",
                                    task.id_target, s.n_decoded_at_suspend);
                            if (s.offloaded()) {
                                seq_erase(s);
                            } else {
                                s.drop_mid_flight();
                            }
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %s on unknown completion id=%s, no live slot\n",
                                task.params.control_action.c_str(), task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        if (slot.is_processing()) {
                            n_processing_slots++;
                        }
                    }
                    SRV_DBG("n_processing_slots = %d\n", n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->metrics             = metrics;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_GET:
                {
                    json slots_data = json::array();

                    int n_idle_slots = 0;

                    for (server_slot & slot : slots) {
                        if (!slot.is_processing()) {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot.to_json(slots_debug == 0));
                    }
                    SRV_DBG("n_idle_slots = %d\n", n_idle_slots);

                    auto res = std::make_unique<server_task_result_slots>();
                    res->id           = task.id;
                    res->slots_data   = std::move(slots_data);
                    res->n_idle_slots = n_idle_slots;

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // [seq] the seat's own sequence is what is saved or restored; a seat without one gets a fresh one
                    if (!seat_ensure_bound(*slot)) {
                        send_error(task, "No sequence id is available for this slot", ERROR_TYPE_SERVER);
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    std::vector<char> packed;
                    try {
                        packed = slot->prompt.tokens.serialize();
                    } catch (const std::exception & err) {
                        send_error(task, err.what(), ERROR_TYPE_NOT_SUPPORTED);
                        break;
                    }

                    GGML_ASSERT(packed.size() % sizeof(llama_token) == 0);
                    const size_t nwrite = llama_state_seq_save_file(
                        ctx_tgt, filepath.c_str(), slot->seq_id,
                        reinterpret_cast<const llama_token *>(packed.data()), packed.size() / sizeof(llama_token));
                    if (nwrite == 0) {
                        send_error(task, "Unable to save slot", ERROR_TYPE_SERVER);
                        break;
                    }

                    // [slot-aux] checkpoints + draft KV are not covered by the state file
                    if (!slot_aux_save(filepath, ctx_dft, slot->seq_id, slot->prompt.checkpoints)) {
                        SLT_WRN(*slot, "%s", "failed to save slot aux data (checkpoints/draft); "
                                             "restore will fall back to full prompt reprocessing\n");
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // [seq] the seat's own sequence is what is saved or restored; a seat without one gets a fresh one
                    if (!seat_ensure_bound(*slot)) {
                        send_error(task, "No sequence id is available for this slot", ERROR_TYPE_SERVER);
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    size_t nread = 0;
                    try {
                        size_t n_packed = 0;
                        llama_tokens packed;
                        nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->seq_id, nullptr, 0, &n_packed);
                        if (nread != 0) {
                            packed.resize(std::max<size_t>(1, n_packed));
                            nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->seq_id, packed.data(), packed.size(), &n_packed);
                        }
                        if (nread == 0) {
                            throw std::runtime_error("No available space in KV cache or invalid slot save file");
                        }
                        packed.resize(n_packed);

                        server_tokens restored = server_tokens::deserialize(packed, mctx != nullptr);

                        if (restored.size() > (size_t) slot->n_ctx) {
                            throw std::runtime_error("Restored prompt does not fit in the slot context");
                        }

                        if (!restored.validate(ctx_tgt)) {
                            throw std::runtime_error("Invalid tokens in slot save file");
                        }

                        slot->prompt.clear();
                        slot->prompt.tokens = std::move(restored);

                        // [slot-aux] restore checkpoints + draft KV. Without the
                        // checkpoints the next prompt-processing pass takes the
                        // do_reset branch and reprocesses everything, which makes
                        // the restore pointless on hybrid/recurrent models.
                        if (!slot_aux_load(filepath, ctx_dft, slot->seq_id, slot->prompt.checkpoints)) {
                            SLT_WRN(*slot, "%s", "no usable slot aux data (checkpoints/draft); "
                                                 "prompt may be fully reprocessed\n");
                        }
                    } catch (const std::exception & err) {
                        slot->prompt_clear();
                        send_error(task, std::string("Unable to restore slot: ") + err.what(), ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = slot->prompt.tokens.size();
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    if (slot->bound()) {
                        slot->prompt_clear();
                    }

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }

        return true;
    }

    void iterate(std::deque<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
        // [seq] yielded generations were waiting on these same seats; the failure that
        // emptied them is theirs too, and a dropped entry would be a client waiting forever
        for (auto * s : seq_mid_flight(/*resident_only*/ false)) {
            send_error(*s->task, reason, ERROR_TYPE_SERVER);
            if (s->offloaded()) {
                seq_erase(*s);
            } else {
                s->drop_mid_flight();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    void update_slots() {
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // [preempt] P2 / [seats] / [deadline]: put yielded generations back on a seat.
        //
        // One scheduler, two rules (the operator's: "a hardcapped priority queue, second dimension - can we pack it").
        //
        // RULE 2, pack: a resident sequence with pending work is batched. It takes an idle seat, or a NEW one,
        // which is nobody's loss. NEW work still wins a contested idle seat: a suspended task has had a turn, the
        // deferred ones have not, and with room to grow the two never contend anyway.
        //
        // RULE 1, the hard ceiling on time asleep: once a suspended sequence has waited --slot-resume-after it
        // gets compute REGARDLESS. If no seat, id or room can be had any other way, the running generation that
        // has had its seat the longest is suspended in its place (pool_preempt_running) - with a copy when the
        // id or the cells are what is short, zero-copy when only the seat is. Good packing makes this rare: a
        // sequence that fits never waits. The deadline is the guarantee for a pool that genuinely cannot hold
        // everything, where the alternative is a long generation starving behind other long generations with
        // nothing queued to make anyone yield.
        for (;;) {
            bool progressed = false;

            for (auto * sp : seq_mid_flight(/*resident_only*/ false)) {
                server_sequence & s = *sp;

                const bool     nobody_new = queue_tasks.queue_tasks_deferred_size() == 0;
                const int64_t  waited_ms  = ggml_time_ms() - s.t_suspended_ms;
                const bool     aged       = params_base.slot_resume_after_ms >= 0 &&
                                            waited_ms >= params_base.slot_resume_after_ms;
                // the teeth: --slot-deadline-preempt (default off). Without it an aged sequence wins a seat that frees
                // up on its own and nothing is taken from anyone, as before
                const bool     force      = aged && params_base.slot_deadline_preempt;

                // can this one be seated at all? an idle seat (when nobody new is waiting for it, or this one has aged
                // past waiting), a new seat, or - past the deadline, with the teeth on - a running generation's
                const bool can_seat = ((nobody_new || aged) && seat_idle_pick() != nullptr) ||
                                      slots.size() < seat_cap() ||
                                      force;
                if (!can_seat) {
                    // oldest suspension first: if this one cannot be seated, the younger ones cannot either
                    break;
                }

                const int n_gen_at = s.n_decoded_at_suspend;

                // the seat a preemption freed for this sequence, if one was needed on the way
                server_slot * taken = nullptr;

                if (s.offloaded()) {
                    // it needs an id back: a free one, one more from the model, or a finished conversation's (saved
                    // first) - never another yielded generation's, that would only move the problem along. Past the
                    // deadline, a running generation's (RULE 1).
                    llama_seq_id id = seq_id_acquire(/*allow_offload*/ false, "resuming a suspended generation", /*urgent*/ force);
                    if (id < 0 && force) {
                        taken = pool_preempt_running(/*offload*/ true, s, "is past its suspension deadline and needs an id");
                        if (taken != nullptr) {
                            id = seq_id_free();
                        }
                    }
                    if (id < 0) {
                        continue;
                    }

                    // [seq] T2.5, the restore side of the ladder: room first. Finished residents are evicted for it
                    // (rung 1); a pool held by RUNNING sequences means waiting, not failing - they finish or are
                    // suspended themselves, and the room appears. Past the deadline the room is MADE (RULE 1). Otherwise
                    // this one is skipped, not the list: a resident generation behind it may be holding exactly the cells
                    // it waits for, and has to run to free them.
                    bool may_wait = false;
                    bool restored = seq_restore_with_room(s, id, may_wait);
                    while (!restored && force && may_wait) {
                        server_slot * t = pool_preempt_running(/*offload*/ true, s, "is past its suspension deadline and needs room");
                        if (t == nullptr) {
                            break;
                        }
                        if (taken == nullptr) {
                            taken = t;
                        }
                        restored = seq_restore_with_room(s, id, may_wait);
                    }

                    if (!restored) {
                        if (may_wait) {
                            if (s.n_resume_failures++ == 0) {
                                SRV_WRN("failed to resume a suspended generation (%d tokens in): no room in the KV pool yet, waiting\n", n_gen_at);
                            }
                            continue;
                        }

                        // nothing runs and nothing can be evicted, so the state does not fit an empty pool. One retry
                        // on a later pass; after that the task is answered, because a dropped entry is a client waiting
                        // forever for a generation nobody is running. (T1.4)
                        if (s.n_resume_failures++ == 0) {
                            SRV_WRN("failed to resume a suspended generation (%d tokens in), will retry once\n", n_gen_at);
                            continue;
                        }
                        SRV_ERR("failed to resume a suspended generation (%d tokens in), giving up\n", n_gen_at);
                        send_error(*s.task, "failed to resume the suspended generation: no room in the KV cache", ERROR_TYPE_SERVER);
                        seq_erase(s);
                        continue;
                    }

                    s.n_resume_failures = 0;
                }

                // the seat, in rising order of cost: the one a preemption just freed; an idle one; a new one; and past
                // the deadline, a running generation's - zero-copy, its sequence stays resident (RULE 1 for the seat)
                server_slot * slot = taken;
                if (slot == nullptr && (nobody_new || aged)) {
                    slot = seat_idle_pick();
                }
                if (slot == nullptr) {
                    slot = seat_add("resuming a suspended generation");
                }
                if (slot == nullptr && force) {
                    slot = pool_preempt_running(/*offload*/ false, s, "is past its suspension deadline and needs a seat");
                }
                if (slot == nullptr) {
                    // it has its id and its cells; a seat frees up or the deadline takes one on a later pass
                    continue;
                }

                if (slot->bound()) {
                    slot->seat_release(); // its idle sequence stays resident
                }
                slot->seat_acquire(s);

                if (aged) {
                    SLT_INF(*slot, "resumed on the deadline after %" PRId64 " ms (bound %d ms), %zu queued\n",
                            waited_ms, params_base.slot_resume_after_ms, queue_tasks.queue_tasks_deferred_size());
                }

                progressed = true;
                break; // the list changed: start over
            }

            if (!progressed) {
                break;
            }
        }

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");

                // [seq-max] [seats] the quiet moment: no batch in flight, nothing waiting
                if (queue_tasks.queue_tasks_deferred_size() == 0) {
                    seat_shrink();
                    seq_ceiling_shrink();
                }

                metrics_flush_idle();

                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
            batch.render();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));

            // the batch is half-built and not rendered, skip now to avoid UB
            return;
        }

        GGML_ASSERT(batch.slot_batched || batch.size() == 0);

        if (batch.slot_batched) {
            auto & slot_batched      = batch.slot_batched;
            auto & alora_scale       = batch.alora_scale;
            auto & alora_disabled_id = batch.alora_disabled_id;

            // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        llama_batch batch_view;
        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch_view = batch.get_view(off, n_tokens);
                bool ok = decode(n_batch, off, batch_view);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off, batch_view);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }
        }
    }

    void pre_decode() {
        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                slot.mem.seq_rm (slot.seq_id, n_keep            , n_keep + n_discard);
                slot.mem.seq_add(slot.seq_id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        });

        // start populating the batch for this iteration
        batch.clear();

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        std::vector<server_slot *> generating;
        std::vector<server_slot *> drafting;

        // determine which slots are generating and drafting
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            generating.push_back(&slot);

            if (spec) {
                common_speculative_get_draft_params(spec.get(), slot.seq_id).drafting = false;

                const bool use_ckpt_tgt = ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
                const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                const int n_draft_max = slot.get_n_draft_max();

                if (n_draft_max > 0) {
                    GGML_ASSERT(slot.can_speculate());

                    if (!slot.spec_draft.empty()) {
                        // we have a previous (partial) draft to reuse
                        if (use_ckpt_tgt) {
                            GGML_ASSERT(!slot.spec_ckpt.empty());
                        }
                    } else {
                        GGML_ASSERT(slot.spec_i_batch.empty());

                        slot.spec_ckpt.update_pos(
                                slot.prompt.n_tokens(),
                                llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.seq_id),
                                llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.seq_id));

                        if (use_ckpt_dft) {
                            slot.spec_ckpt.update_dft(ctx_dft, slot.seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();

                        common_speculative_get_draft_params(spec.get(), slot.seq_id) = {
                            /* .drafting = */ true,
                            /* .n_max    = */ n_draft_max,
                            /* .n_past   = */ slot.prompt.n_tokens(),
                            /* .id_last  = */ slot.sampled,
                            /* .prompt   = */ &slot.spec_prompt,
                            /* .result   = */ &slot.spec_draft,
                        };

                        drafting.push_back(&slot);
                    }
                }
            }
        });

        // generate the actual drafts (if any)
        if (!drafting.empty()) {
            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
        }

        // make checkpoints if needed
        iterate(drafting, [&](server_slot & slot) {
            auto & draft = slot.spec_draft;
            auto & ckpt  = slot.spec_ckpt;

            slot.stats.n_draft_tokens += draft.size();

            // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
            const bool use_ckpt_dft = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

            if (ctx_dft) {
                if (use_ckpt_dft) {
                    ckpt.load_dft(ctx_dft, slot.seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.seq_id, ckpt.pos_max + 1, -1)) {
                    GGML_ABORT("failed to remove sequence %d\n", slot.seq_id);
                }
            }

            if (!draft.empty()) {
                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                   (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_tgt));

                const bool use_ckpt_dft =
                   (ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && draft.size() > llama_n_rs_seq(ctx_dft));

                if (use_ckpt_tgt) {
                    //const int64_t t_start = ggml_time_us();

                    ckpt.update_tgt(ctx_tgt, slot.seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                    //const int64_t t_total = ggml_time_us() - t_start;
                    //printf("checkpoint total: %f ms\n", t_total / 1000.0);

                    SLT_DBG(slot, "created speculative checkpoint (pos_min = %d, pos_max = %d, n_tokens = %d, size = %.3f MiB, draft = %.3f MiB)\n",
                            ckpt.pos_min, ckpt.pos_max, slot.prompt.n_tokens(),
                            (float) ckpt.size() / 1024 / 1024,
                            (float) ckpt.data_dft.size() / 1024 / 1024);
                }

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft, slot.seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }
        });

        // update the batch with the sampled/drafted tokens
        iterate(generating, [&](server_slot & slot) {
            slot.handle_last_sampled_token(batch);
        });

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // [ratio] the third resource: positions in the batch loop. A batch that carries a prefill chunk AND the
        // generating slots' tokens gives each generating slot one token per batch, and a prefill batch takes as
        // long as its chunk (3.1 s per 512 tokens on GLM): the generating slot's rate becomes the prefill's batch
        // rate. It is seated, it is resident, and it is starving - measured 0.34 t/s against 43. Shrinking n_ubatch
        // only collapses the prefill. The lever that works is interleaving whole batches: N decode-only batches
        // between two prefill batches, prefill keeping its full chunk. --decode-per-prefill N; 0 keeps the greedy
        // merge. A prefill with nothing generating beside it is never held back.
        bool admit_prefill = true;
        if (params_base.decode_per_prefill > 0 && !generating.empty()) {
            bool prefill_pending = false;
            for (const auto & slot : slots) {
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    prefill_pending = true;
                    break;
                }
            }
            if (prefill_pending) {
                if (n_decode_only < params_base.decode_per_prefill) {
                    admit_prefill = false;
                    n_decode_only++;
                    SRV_DBG("prefill held back: decode-only batch %d of %d (%zu generating)\n",
                            n_decode_only, params_base.decode_per_prefill, generating.size());
                } else {
                    n_decode_only = 0;
                }
            } else {
                n_decode_only = 0;
            }
        } else {
            n_decode_only = 0;
        }

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            iterate(slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }

                if (!slot.is_processing()) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    return;
                }

                // [ratio] this batch is decode-only
                if (!admit_prefill && (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED)) {
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.stats.update_prompt_start();

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_media();

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // the divergence: the old and new prompts agree on content up to here and nowhere
                                // past it. everything the reuse loop below moves came from past this point.
                                const int n_past_div = n_past;

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_media());

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            if (ctx_tgt_has_recurrent_state) {
                                                // the KV for these tokens could be moved, but the recurrent state cannot: it is
                                                // one accumulator over the whole history, so it still contains everything
                                                // between the divergence and here. keeping it (or a checkpoint taken past the
                                                // divergence) had the model answering from tokens the client had deleted
                                                // (measured on glm5next: "4 document parts" for a truth of "three", GLM-TODO
                                                // T3.5). the state has to be rebuilt from the divergence, and rebuilding it
                                                // runs the full stack over the same tokens, which regenerates their KV anyway -
                                                // so a shift here is memory churn with nothing to show for it. leave n_past at
                                                // the divergence; the checkpoint search below rolls back from there.
                                                SLT_INF(slot, "chunk of %zu tokens matches at [%zu, %zu) -> [%zu, %zu) but the recurrent state past the divergence at %d cannot be shifted with it; rebuilding from %d\n",
                                                        n_match, head_c, head_c + n_match, head_p, head_p + n_match, n_past_div, n_past_div);
                                                break;
                                            }

                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            slot.mem.seq_rm (slot.seq_id, head_p, head_c);
                                            slot.mem.seq_add(slot.seq_id, head_c, head_c + n_match, kv_shift);

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);

                                    // content invalidation. a checkpoint is a snapshot of the memory after the first
                                    // n_tokens of the OLD prompt. the shift above moved tokens from past the divergence
                                    // to new positions, so a checkpoint taken past the divergence describes content the
                                    // new prompt does not have - while its positions may now sit inside [0, pos_next),
                                    // which is all the position-based search and prune further down can see. the
                                    // context-shift path drops every checkpoint after it shifts (slot.prompt.clear());
                                    // this is the same rule, keeping the ones that stay valid. the prune has to run
                                    // before the search, not after it, or the stale snapshot is what gets restored.
                                    if (n_past > n_past_div) {
                                        for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                            if (it->n_tokens > n_past_div) {
                                                SLT_INF(slot, "erased context checkpoint taken past the divergence (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", divergence = %d, size = %.3f MiB)\n",
                                                        it->pos_min, it->pos_max, it->n_tokens, n_past_div, (float) it->size() / 1024 / 1024);
                                                it = slot.prompt.checkpoints.erase(it);
                                            } else {
                                                ++it;
                                            }
                                        }
                                    }
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.seq_id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.seq_id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - slots_n_diff, 0);
                                    const int np1 = std::min<int>(n_past + slots_n_diff + 2, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                            if (cur.pos_max > pos_next) {
                                                return false;
                                            }
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        it->load_tgt(ctx_tgt, slot.seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        it->load_dft(ctx_dft, slot.seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                                        // restore the draft's speculative state
                                        common_speculative_set_state(spec.get(), slot.seq_id, it->data_spec);

                                        pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                        n_past   = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);
                                        SLT_TRC(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) it->size() / 1024 / 1024);
                                    }

                                    if (do_reset) {
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.stats.n_prompt_cached    = n_past;
                        slot.stats.n_prompt_processed = 0;

                        metrics.add_prompt_cached(n_past);

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    // note: the prompt timing is advanced in post_decode(), so it does not cover
                    //       the tokens added to the batch below
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    slot.mem.seq_rm(slot.seq_id, p0, -1);

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    // - the model supports partial sequence removal but only up to a fixed bound
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0);

                    bool has_mtmd = false;

                    // check if we should process the mtmd chunk
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the mtmd chunk
                        // note: it submits its own decode, potentially be async
                        //       so the timing is queued and flushed on the next sync
                        metrics_pre_decode();

                        // encode on the worker thread, so we can still handle metrics tasks
                        size_t n_tokens_out = 0;
                        int32_t res = 0;
                        queue_tasks.yield_to_queue([&]() {
                            res = process_mtmd_chunk(slot, slot.mbatch, cur_token_idx, n_tokens_out);
                        });

                        if (res != 0) {
                            SLT_ERR(slot, "failed to process mtmd chunk, res = %d\n", res);
                            send_error(slot, "failed to process mtmd chunk", ERROR_TYPE_SERVER);
                            slot.release();
                            return; // the slot is done, skip it entirely
                        }

                        metrics_queue_prompt(n_tokens_out);
                        slot.stats.n_prompt_processed += n_tokens_out;
                        slot.stats.update_prompt_last();

                        // add the mtmd chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            // the chunk is already in the KV cache at this point, so we don't need to keep its data around
                            slot.prompt.tokens.push_back_placeholder(chunk.get());
                        }

                        has_mtmd = true;
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();

                    // schedule fallback: true when no checkpoint has been created for checkpoint_min_step
                    // tokens, counted from the last checkpoint or from the start of the prompt. user-message
                    // boundaries stay the preferred placement; this only fires when none has shown up for
                    // that long. without it a prompt whose user boundaries all sit near its end gets
                    // checkpoints only there, and a rollback to anywhere earlier finds nothing usable and
                    // reprocesses the whole prefix (measured: 33k prompt, rollback to 18k, 18k tokens
                    // thrown away, GLM-TODO T3.5c). a min step of 0 means "no minimum" and never schedules.
                    const auto is_checkpoint_due = [&](int n_tokens) {
                        if (params_base.checkpoint_min_step <= 0) {
                            return false;
                        }
                        const auto & checkpoints = slot.prompt.checkpoints;
                        const int64_t last = checkpoints.empty() ? 0 : checkpoints.back().n_tokens;
                        return n_tokens > last + params_base.checkpoint_min_step;
                    };

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        add_ok &= batch.add(slot.seq_id,
                            cur_tok,
                            /* pos       = */ slot.prompt.tokens.pos_next(),
                            /* output    = */ slot.need_embd(),
                            /* is_prompt = */ true);
                        slot.prompt.tokens.push_back(cur_tok);

                        // break at the last user message, or at user messages at least min step past the last checkpoint
                        if (do_checkpoint && spans.is_user_start(slot.prompt.n_tokens())) {
                            const auto pos = slot.prompt.n_tokens();
                            const auto & checkpoints = slot.prompt.checkpoints;

                            if (pos == last_user_pos || checkpoints.empty() || pos > checkpoints.back().n_tokens + params_base.checkpoint_min_step) {
                                break;
                            }
                        }

                        // break on the schedule fallback so that the next batch starts on a checkpoint
                        if (do_checkpoint && is_checkpoint_due(slot.prompt.n_tokens())) {
                            break;
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    const bool is_user_start = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;
                    const bool is_sched_start = is_checkpoint_due(n_tokens_start);

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.stats.n_gen = 0;
                        slot.next_yield_at = 0;
                        slot.i_batch     = batch.size() - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message, is due on the checkpoint_min_step schedule, or we are near the
                        // end of the prompt
                        if (!is_user_start && !near_prompt_end && !is_sched_start) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.seq_id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.seq_id);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        create_checkpoint(slot, n_tokens_cur, pos_min, pos_max);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(int32_t & n_batch, int32_t off, llama_batch & batch_view) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        metrics_pre_decode();

        if (batch.size() == 0) {
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // TODO @ngxson : dft model may have different n_embd than the tgt model, so we check & reject if that's the case
        // this case is not currently used by any models, but may need to be supported in the future
        //
        // model_dft is null for draft types that build the draft from the target
        // itself rather than from a separate draft model - draft-mtp assigns only
        // the context. the has_draft path above checks for exactly this.
        if (spec && batch.has_embd && model_dft) {
            if (llama_model_n_embd_inp(model_dft) != llama_model_n_embd_inp(model_tgt)) {
                SRV_ERR("%s", "unsupported batch.has_embd + spec case\n");
                throw std::runtime_error("unsupported batch.has_embd + spec case");
            }
        }

        bool has_output = false;
        for (int i = off; i < off + batch_view.n_tokens; ++i) {
            has_output |= batch.tokens[i].output;
        }

        // yield to the queue, so we can still handle metrics tasks while decoding
        // note: the sync is done here too, so that the wait is also covered by the yield
        int ret = 0;
        queue_tasks.yield_to_queue([&]() {
            ret = llama_decode(ctx_tgt, batch_view);
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });

        if (ret != 0) {
            if (n_batch == 1 && ret == 1) {
                SRV_ERR("Context size has been exceeded. off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

                // [seq] T2.5: the pool has no cell for a single token. The ladder, one move per retry; rung 1
                // ran on the way down (below), so here it only catches what finished since. This used to
                // error, release and clear EVERY processing slot; then one; now one only when nothing else can go.
                if (pool_grow_pending(off)                     ||
                    pool_defer_pending(off)                    ||
                    pool_evict_resident(false, "KV pool full") ||
                    pool_offload_waiting()                     ||
                    pool_suspend_running(off)                  ||
                    pool_evict_resident(true,  "KV pool full") ||
                    fail_slot_under_pressure(off, "Context size has been exceeded.")) {
                    return false; // retry the rest of the batch
                }
            }

            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear();
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache. A finished conversation
            // goes first (rung 1, cheap and due anyway); the copies wait until halving has proved them necessary
            if (!pool_grow_pending(off) && !pool_evict_resident(false, "KV pool full")) {
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        } else {
            // success, apply batch metrics
            metrics_post_decode(off, batch_view.n_tokens, has_output);
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (spec) {
            bool ok = true;
            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec.get(), batch_view);
            });

            if (!ok) {
                // the TARGET decode has already succeeded - this is the success
                // branch, metrics_post_decode() ran just above. everything the
                // client asked for has been computed correctly. all that failed
                // is maintaining the draft's shadow of that state.
                //
                // speculation is an optimization: the target verifies every
                // drafted token before it is emitted, so a stale or empty draft
                // costs acceptance rate and nothing else.
                SRV_WRN("%s", "failed to process speculative batch - continuing without speculation for this batch\n");
            }
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    void post_decode(int32_t n_batch_tokens, int32_t off, llama_batch & batch_view) {
        // for checking if a given batch index is inside batch_view
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        iterate(slots, [&](server_slot & slot) {
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                if (slot.can_speculate()) {
                    common_speculative_begin(spec.get(), slot.seq_id, slot.prompt.tokens.get_text_tokens());
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
            }

            slot.i_batch = -1;

            common_sampler_accept(slot.smpl.get(), id, true);

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.stats.n_gen += 1;

            if (slot.stats.n_gen == 1) {
                slot.stats.update_prompt_last();
                slot.t_print_last = t_now;
                slot.n_gen_last = 0;
            }

            slot.stats.update_gen_last();

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, slot)) {
                // release slot because of stop condition
                slot.print_timings();
                send_final_response(slot);
                slot.release();

                return;
            }

            slot.print_timings_tg();

            // [preempt] P1 correctness harness, kept: suspends and IMMEDIATELY
            // resumes the same slot, exercising state capture with no scheduling
            // involved. A run with LLAMA_SERVER_PREEMPT_SELFTEST=N must produce
            // output identical to one without it (temp 0, fixed seed).
            {
                static const int selftest_every = []() {
                    const char * e = std::getenv("LLAMA_SERVER_PREEMPT_SELFTEST");
                    return e ? std::atoi(e) : 0;
                }();
                if (selftest_every > 0 &&
                    slot.state == SLOT_STATE_GENERATING &&
                    slot.stats.n_gen > 0 &&
                    slot.stats.n_gen % (uint64_t) selftest_every == 0) {
                    server_sequence * s = slot.seq;
                    if (slot.seat_release()) {
                        slot.seat_acquire(*s);
                    }
                }
            }

            slot_maybe_yield(slot);
        });

        // speculative decoding - main model sample and accept
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() ||
                    slot.spec_draft.empty() || slot.spec_i_batch.empty()) {
                return;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            GGML_ASSERT(n_draft > 0);

            // verify and try to accept the draft
            {
                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));

                GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1);
                const auto & synth_probs = common_speculative_get_synth_probs(spec.get());
                auto accepted = synth_probs.empty()
                    ? common_sampler_sample_and_accept_n(slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft)
                    : server_sample_and_accept_synth(
                            slot.smpl.get(), slot.ctx_tgt, slot.spec_i_batch, slot.spec_draft,
                            synth_probs, slot.spec_synth_rng, slot.spec_is_replay);
                slot.spec_i_batch.clear();

                GGML_ASSERT(accepted.size() >= 1);

                const uint32_t n_rollback = slot.spec_draft.size() + 1 - accepted.size();

                const bool use_ckpt_tgt =
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_rollback > llama_n_rs_seq(ctx_tgt));

                // check for partial draft acceptance
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
                        if (trace > 0) {
                            SLT_INF(slot, "accepted %2zu/%2zu draft tokens (restore checkpoint)\n", accepted.size() - 1, slot.spec_draft.size());
                        }

                        // partial acceptance is not supported by the context -> truncate the draft and restore the state
                        slot.spec_is_replay = true;
                        slot.spec_draft = std::move(accepted);

                        const auto & ckpt = slot.spec_ckpt;

                        SLT_DBG(slot, "restoring speculative checkpoint (pos_min = %d, pos_max = %d, size = %zu)\n", ckpt.pos_min, ckpt.pos_max, ckpt.size());

                        ckpt.load_tgt(slot.ctx_tgt, slot.seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                        if (slot.ctx_dft) {
                            ckpt.load_dft(slot.ctx_dft, slot.seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                        }

                        slot.mem.seq_rm(slot.seq_id, ckpt.pos_max + 1, -1);

                        slot.prompt.tokens.keep_first(ckpt.n_tokens);
                        common_sampler_copy(smpl_save.get(), slot.smpl.get());

                        return;
                    }
                }

                if (trace > 0) {
                    SLT_INF(slot, "accepted %2zu/%2zu draft tokens\n", accepted.size() - 1, n_draft);
                }

                common_speculative_accept(spec.get(), slot.seq_id, accepted.size() - 1);

                slot.spec_draft = std::move(accepted);
            }

            const auto ids = std::move(slot.spec_draft);

            size_t n_accepted = ids.size() - 1;
            if (slot.spec_is_replay && n_accepted > 0) {
                n_accepted--;
            }
            slot.spec_is_replay = false;

            slot.stats.update_gen_last();

            // update how many tokens out of those tested were accepted
            slot.stats.n_draft_accepted += n_accepted;
            slot.stats.n_draft_verif_steps += 1;

            auto & n_accepted_per_pos = slot.n_accepted_per_pos;
            if (n_accepted_per_pos.empty()) {
                n_accepted_per_pos.resize(common_speculative_n_max(spec.get()), 0);
            }
            for (size_t i = 0; i < n_accepted && i < n_accepted_per_pos.size(); ++i) {
                n_accepted_per_pos[i]++;
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert({ids.begin(), ids.end() - 1});

            slot.sampled = ids.back(); // last accepted token
            SLT_DBG(slot, "add accepted tokens: sampled=%d, ids.size=%zu, n_draft=%zu\n", slot.sampled, ids.size(), n_draft);

            slot.mem.seq_rm(slot.seq_id, slot.prompt.tokens.pos_next(), -1);

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // TODO: set result.probs

                slot.stats.n_gen += 1;

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    slot.release();

                    return;
                }
            }

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) n_accepted, (int) n_draft, slot.prompt.n_tokens());

            // [preempt] same tail as the non-speculative path above. This slot
            // returned early from that lambda (it had a draft), so the trigger
            // there never saw it - measured: 0 yields over a 1500-token
            // generation with draft-simple while a request waited the whole time.
            slot_maybe_yield(slot);
        });
    }

    // [preempt] P3: the real trigger, reached once per decode step by BOTH the
    // plain and the speculative accept paths. Yield ONLY while other work is
    // waiting - with an empty queue this costs nothing, which is the whole
    // point ("when agents count and slots number match we better not yield").
    // Never during prefill: suspend() enforces that.
    //
    // Threshold rather than `n_gen % quantum == 0`: a speculative step advances
    // n_gen by 1 + n_accepted (measured 9 per step with an 8-token draft), so a
    // modulo test only lands when the jump happens to hit a multiple. The
    // threshold is crossed by any jump and re-armed one quantum past wherever
    // n_gen actually is.
    void slot_maybe_yield(server_slot & slot) {
        const uint64_t quantum = (uint64_t) params_base.slot_quantum;

        if (quantum == 0 || slot.state != SLOT_STATE_GENERATING || slot.stats.n_gen == 0) {
            return;
        }

        if (slot.next_yield_at == 0) {
            slot.next_yield_at = quantum;
        }

        if (slot.stats.n_gen < slot.next_yield_at) {
            return;
        }

        // re-arm whether or not we yield: the quantum is a check cadence, and
        // an empty queue at this check means the next look is a quantum away
        slot.next_yield_at = slot.stats.n_gen + quantum;

        // read BEFORE suspend(): it releases the slot, which pops a deferred
        // task into the main queue, so the count reads 0 afterwards.
        // [D15] only work that could take THIS seat counts: an inference task not pinned to another seat. A slot
        // action pinned here waits for the seat to go idle on its own, a parent needs more than one seat, and a
        // request pinned elsewhere gains nothing from this yield - each of those used to make every seat yield.
        const size_t n_waiting = queue_tasks.count_deferred_if([&](const server_task & t) {
            switch (t.type) {
                case SERVER_TASK_TYPE_COMPLETION:
                case SERVER_TASK_TYPE_INFILL:
                case SERVER_TASK_TYPE_EMBEDDING:
                case SERVER_TASK_TYPE_RERANK:
                    break;
                default:
                    return false;
            }
            if (t.is_parent()) {
                return false;
            }
            return t.id_slot == -1 || t.id_slot == slot.id;
        });
        if (n_waiting == 0) {
            return;
        }

        // the sequence stays where it is (resident, mid-flight); only the seat is given up
        server_sequence * s = slot.seq;
        if (slot.seat_release()) {
            SLT_INF(slot, "yielded the slot after %d tokens, %zu waiting\n",
                    s->n_decoded_at_suspend, n_waiting);
        }
    }

    // context size of a single slot, capped by --kv-unified-per-slot and by the training context of the model.
    // [pool] with an elastic pool the pool of the moment is no cap: it grows for the conversation
    int n_ctx_slot() const {
        int res = pool_elastic() ? llama_model_n_ctx_train(model_tgt) : llama_n_ctx_seq(ctx_tgt);

        if (params_base.kv_unified_per_slot > 0) {
            res = std::min(res, params_base.kv_unified_per_slot);
        }

        return std::min(res, llama_model_n_ctx_train(model_tgt));
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }

    //
    // metrics helpers
    //

    // call before submitting a decode, so that the queued prompt stats can be timed
    void metrics_pre_decode() {
        t_decode_start = ggml_time_us();
    }

    // the batch is submitted, but its compute may not be done yet
    void metrics_queue_prompt(uint64_t n_tokens) {
        if (n_tokens == 0) {
            return;
        }
        if (n_prompt_queued == 0) {
            t_prompt_start = t_decode_start;
        }
        n_prompt_queued += n_tokens;
    }

    // call only after the context is synchronized, otherwise the time is meaningless
    void metrics_flush_prompt() {
        if (n_prompt_queued == 0) {
            return;
        }
        metrics.add_prompt(n_prompt_queued, ggml_time_us() - t_prompt_start);
        n_prompt_queued = 0;
    }

    // has_output is computed by the caller, which also already synchronized the context if it is set
    void metrics_post_decode(int32_t off, int32_t n_tokens, bool has_output) {
        metrics.n_decode++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                metrics.n_busy_slots++;
            }
        }
        for (const auto & s : seqs) {
            metrics.n_tokens_max = std::max(metrics.n_tokens_max, (uint64_t) seq_prompt(s).n_tokens());
        }

        // apply enqueued prompt tokens stats
        // note: a slot can be released before we get here, which clears its stats
        //       the tokens were still computed, counted in the global metrics, not in slot
        uint64_t n_prompt_tokens = 0;

        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];

            if (!t.is_prompt) {
                continue; // generated tokens are handled after sampling
            }

            n_prompt_tokens++;

            auto * slot = get_slot_by_seq_id(t.seq_id);
            if (slot != nullptr && slot->stats.is_set()) {
                slot->stats.n_prompt_processed++;
            }
        }

        metrics_queue_prompt(n_prompt_tokens);

        if (has_output) {
            // the context is already synchronized, so the timings are correct
            metrics_flush_prompt();
        }

        // advance the prompt timing of the slots that had tokens in this batch
        // note: a second pass, it must run after the sync to reflect the compute
        const int64_t t_now = ggml_time_us();
        for (int i = off; i < off + n_tokens; ++i) {
            const auto & t = batch.tokens[i];
            auto * slot = get_slot_by_seq_id(t.seq_id);
            if (t.is_prompt && slot != nullptr && slot->stats.is_set()) {
                slot->stats.set_prompt_last(t_now);
            }
        }
    }

    // flush any queued prompt metrics if all slots are now idle
    void metrics_flush_idle() {
        if (n_prompt_queued == 0) {
            return;
        }

        llama_synchronize(ctx_tgt);
        metrics_flush_prompt();
    }

    void metrics_on_prediction(const server_slot & slot) {
        const uint64_t t_us    = slot.stats.t_gen_us();
        const uint64_t n       = slot.stats.n_gen;
        const uint64_t n_steps = slot.stats.n_gen_steps();

        metrics.predict       .add(n, n_steps, t_us);
        metrics.predict_bucket.add(n, n_steps, t_us);

        metrics.n_draft_tokens      += slot.stats.n_draft_tokens;
        metrics.n_draft_accepted    += slot.stats.n_draft_accepted;
        metrics.n_draft_verif_steps += slot.stats.n_draft_verif_steps;

        auto & dst = metrics.n_accepted_per_pos;
        const auto & src = slot.n_accepted_per_pos;

        if (dst.size() < src.size()) {
            dst.resize(src.size(), 0);
        }
        for (size_t i = 0; i < src.size(); i++) {
            dst[i] += src[i];
        }
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

void server_context::flush_prompt_cache() {
    if (!impl) {
        return;
    }

    impl->flush_prompt_cache();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* has_inp_video          */ impl->chat_params.allow_video,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* slot_n_ctx             */ impl->n_ctx_slot(),
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
        /* model_ftype            */ ftype_name,
    };
}

// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_res_spipe {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    res->set_req(&req); // will also set spipe if needed

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files, ctx_server.init_opt));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        // message delimiters for checkpointing
        json delims = json_value(data, "message_delimiters", json::array());
        auto delimiters = common_chat_msg_delimiters_parse(delims);
        delimiters.tokenize(ctx_server.vocab);

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->logit_bias_eog,
                    data);

            task.params.message_spans = task.tokens.find_message_spans(delimiters);

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->set_next([res_this = res.get(), res_type, sse_ping_interval](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = [&res_this]() {
                return res_this->should_stop();
            };

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        });
    }

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();

    // note: this must be registered before load_model()
    //       so that on sleep phase, the callback is called before ctx is destroyed
    queue_tasks.on_sleeping_state([this](bool is_sleeping) {
        update_cached_responses(is_sleeping);
    });
}

static json get_res_model_info(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return {
        {"id",       meta.model_name},
        {"aliases",  meta.model_aliases},
        {"tags",     meta.model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta.model_vocab_type},
            {"n_vocab",     meta.model_vocab_n_tokens},
            {"n_ctx",       meta.slot_n_ctx},
            {"n_ctx_train", meta.model_n_ctx_train},
            {"n_embd",      meta.model_n_embd_inp},
            {"n_params",    meta.model_n_params},
            {"size",        meta.model_size},
            {"ftype",       meta.model_ftype},
        }},
    };
}

static json get_res_models(const server_context_meta & meta) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    return json{
        {"models", json::array({
            {
                {"name",  meta.model_name},
                {"model", meta.model_name},
                {"modified_at", ""},
                {"size", ""},
                {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                {"type", "model"},
                {"description", ""},
                {"tags", json::array({""})},
                {"capabilities", meta.has_mtmd ? json::array({"completion","multimodal"}) : json::array({"completion"})},
                {"parameters", ""},
                {"details", {
                    {"parent_model", ""},
                    {"format", "gguf"},
                    {"family", ""},
                    {"families", json::array({""})},
                    {"parameter_size", ""},
                    {"quantization_level", ""}
                }}
            }
        })},
        {"object", "list"},
        {"data", json::array({
            get_res_model_info(meta),
        })}
    };
}

static json get_res_props(const server_context_meta & meta, const common_params & params, bool is_sleeping) {
    // note: do NOT use ctx_server here, otherwise it's not possible to use this during sleep

    task_params tparams;
    tparams.sampling = params.sampling;
    json default_generation_settings_for_props = json {
        { "params", tparams.to_json(true) },
        { "n_ctx",  meta.slot_n_ctx },
    };

    std::string tmpl_default = common_chat_templates_source(meta.chat_params.tmpls.get(), "");
    std::string tmpl_tools   = common_chat_templates_source(meta.chat_params.tmpls.get(), "tool_use");

    json props = {
        { "default_generation_settings", default_generation_settings_for_props },
        { "total_slots",                 params.n_parallel },
        { "model_alias",                 meta.model_name },
        { "model_ftype",                 meta.model_ftype },
        { "model_path",                  meta.model_path },
        { "modalities",                  json {
            {"vision", meta.has_inp_image},
            {"video",  meta.has_inp_video},
            {"audio",  meta.has_inp_audio},
        } },
        { "media_marker",                get_media_marker() },
        { "endpoint_slots",              params.endpoint_slots },
        { "endpoint_props",              params.endpoint_props },
        { "endpoint_metrics",            params.endpoint_metrics },
        { "ui",                          params.ui },
        { "ui_settings",                 meta.json_ui_settings },
        { "chat_template",               tmpl_default },
        { "chat_template_caps",          meta.chat_template_caps },
        { "bos_token",                   meta.bos_token_str },
        { "eos_token",                   meta.eos_token_str },
        { "build_info",                  meta.build_info },
        { "is_sleeping",                 is_sleeping },
        { "cors_proxy_enabled",          params.ui_mcp_proxy },
    };
    if (params.use_jinja) {
        if (!tmpl_tools.empty()) {
            props["chat_template_tool_use"] = tmpl_tools;
        }
    }

    return props;
}

json server_routes::get_model_info() const {
    return get_res_model_info(*meta);
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response(true);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // render response using cached_metrics
        auto use_cached_metrics = [&]() {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->headers["Process-Start-Time-Unix"] = std::to_string(cached_metrics.t_start);
            server_task_result_metrics tmp;
            tmp.metrics = cached_metrics;
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = tmp.to_metrics();
            // the gauges are averaged over the window between two scrapes
            cached_metrics.reset_bucket();
            should_reset_buckets = true;
        };

        if (queue_tasks.is_sleeping()) {
            use_cached_metrics();

        } else {
            // request slots data using task queue
            {
                server_task task(SERVER_TASK_TYPE_METRICS);
                task.id = res->rd.get_new_id();
                // the gauges are averaged over the window between two scrapes
                task.metrics_reset_bucket = true;
                res->rd.post_task(std::move(task), true); // high-priority task
            }

            // a task posted right before sleeping is never processed, do not wait for it
            auto result = res->rd.next([&]{
                return req.should_stop() || queue_tasks.is_sleeping();
            });
            if (!result) {
                if (!req.should_stop()) {
                    use_cached_metrics();
                }
                return res;
            }

            if (result->is_error()) {
                res->error(result->to_json());
                return res;
            }

            auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
            GGML_ASSERT(res_task != nullptr);

            res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->metrics.t_start);
            res->content_type = "text/plain; version=0.0.4";
            res->status = 200;
            res->data = res_task->to_metrics();
        }

        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_SLOT_GET);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        auto * res_task = dynamic_cast<server_task_result_slots*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->to_json());
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_props);
        } else {
            res->ok(get_res_props(*meta, params, false));
        }
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true, ctx_server.init_opt);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, ctx_server.init_opt, req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);
        // note: do NOT use ctx_server here, this endpoint must be accessible during sleep
        if (queue_tasks.is_sleeping()) {
            std::unique_lock<std::mutex> lock(mutex_cache);
            res->ok(cached_models);
        } else {
            res->ok(get_res_models(*meta));
        }
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens").get<llama_tokens>();
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i], ctx_server.init_opt);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true, ctx_server.init_opt);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize").get<int>();
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const mtmd_helper_init_opt & init_opt, const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(mctx, prompt.get<std::string>(), files, init_opt, true).size();
    } else {
        n_tokens = tokenize_mixed(vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}

void server_routes::update_cached_responses(bool is_sleeping) {
    // caller is task_queue, so ctx_server can be accessed without holding locks
    std::unique_lock<std::mutex> lock(mutex_cache);

    if (is_sleeping) {
        cached_models  = get_res_models(*meta);
        cached_props   = get_res_props(*meta, params, true);
        cached_metrics = ctx_server.get_metrics();

        should_reset_buckets = false;

        SRV_DBG("%s\n", "cached responses updated");

    } else if (should_reset_buckets) {
        // a scrape during sleep already reported these buckets
        ctx_server.reset_metrics_bucket();

        should_reset_buckets = false;
    }
}

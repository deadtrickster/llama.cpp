// A KV pool resize under --split-mode tensor. The meta device refuses any tensor read that is not a whole
// split chunk, and n_ctx_resize's ordering fence read ONE byte of the new KV tensor (2026-09-18, lab2x1:
// `ggml-backend-meta.cpp:1617: GGML_ASSERT(size % chunk_size_full == 0)` at server init, every start).
// Needs a GPU device (ctest skips with 77 otherwise).

#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } } while (0)

static void decode_n(llama_context * ctx, const llama_vocab * vocab, int n, llama_pos pos0) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        batch.token   [i] = 3 + (pos0 + i) % (n_vocab - 3);
        batch.pos     [i] = pos0 + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id  [i][0] = 0;
        batch.logits  [i] = i == n - 1;
    }
    batch.n_tokens = n;
    const int32_t ret = llama_decode(ctx, batch);
    CHECK(ret == 0, "decode of %d tokens at pos %d failed: ret = %d", n, (int) pos0, ret);
    llama_batch_free(batch);
}

int main(int argc, char ** argv) {
    CHECK(argc == 2, "usage: %s <model.gguf>", argv[0]);
    llama_backend_init();

    // the meta device over every GPU (the meta device over the CPU device alone segfaults at context
    // creation - a separate defect; this test is about the resize fence, so it runs where tensor mode runs)
    std::vector<ggml_backend_dev_t> devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            devs.push_back(dev);
        }
    }
    if (devs.empty()) {
        printf("test-kv-resize-tensor: SKIP (no GPU device; tensor mode needs one)\n");
        return 77;
    }
    devs.push_back(nullptr);

    llama_model_params mp = llama_model_default_params();
    mp.split_mode   = LLAMA_SPLIT_MODE_TENSOR;
    mp.devices      = devs.data();
    mp.n_gpu_layers = 999; // every layer on the meta device, so the KV lives in split buffers
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    CHECK(model, "failed to load %s with split_mode tensor", argv[1]);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = 512;
    cp.n_batch         = 64;
    cp.n_ubatch        = 64;
    cp.n_seq_max       = 1;
    cp.kv_unified      = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; // required by tensor mode
    cp.no_perf         = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    CHECK(ctx, "failed to create the context");

    decode_n(ctx, vocab, 16, 0);

    // grow: the new KV tensors are allocated, the old rows copied over, and the copy is fenced with a read
    CHECK(llama_set_n_ctx(ctx, 1024), "grow 512 -> 1024 refused");
    CHECK(llama_n_ctx(ctx) == 1024, "n_ctx is %u after the grow", llama_n_ctx(ctx));
    decode_n(ctx, vocab, 16, 16);

    // shrink back
    CHECK(llama_set_n_ctx(ctx, 512), "shrink 1024 -> 512 refused");
    decode_n(ctx, vocab, 16, 32);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    printf("test-kv-resize-tensor: OK\n");
    return 0;
}

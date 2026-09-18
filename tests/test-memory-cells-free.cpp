// [pool] llama_memory_n_cells_free() is the truth the server's pool accounting derives from:
// size - used, the minimum over a memory's cell-backed parts, -1 for a memory with no token cells.
// Exercised on a hybrid model (attention + recurrent: the composite path) and a pure recurrent one.
//
//   test-memory-cells-free <model.gguf>

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } } while (0)

static llama_context * make_ctx(llama_model * model, uint32_t n_ctx) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = n_ctx;
    cp.n_batch    = 64;
    cp.n_ubatch   = 64;
    cp.n_seq_max  = 1;
    cp.kv_unified = true;
    cp.no_perf    = true;
    return llama_init_from_model(model, cp);
}

static void decode_n(llama_context * ctx, const llama_vocab * vocab, int n, llama_pos pos0) {
    // in-vocab ids, not BOS: the tiny test vocabs have no BOS and llama_vocab_bos() is LLAMA_TOKEN_NULL
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

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    CHECK(model, "failed to load %s", argv[1]);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const uint32_t n_ctx = 512;
    llama_context * ctx = make_ctx(model, n_ctx);
    CHECK(ctx, "failed to create a context");
    llama_memory_t mem = llama_get_memory(ctx);

    const int64_t free0 = llama_memory_n_cells_free(mem);
    printf("empty: n_cells_free = %lld (n_ctx %u)\n", (long long) free0, llama_n_ctx(ctx));

    if (free0 < 0) {
        // no token cells (pure recurrent): the answer is "unknown", consistently, and nothing else
        decode_n(ctx, vocab, 16, 0);
        CHECK(llama_memory_n_cells_free(mem) == -1, "a memory without cells reported %lld after a decode",
              (long long) llama_memory_n_cells_free(mem));
        printf("OK: no token cells, reports -1 before and after decoding (%s)\n", argv[1]);
    } else {
        // the pool is the padded n_ctx; used starts at 0
        CHECK(free0 == (int64_t) llama_n_ctx(ctx), "empty memory reports %lld free, n_ctx is %u", (long long) free0, llama_n_ctx(ctx));

        decode_n(ctx, vocab, 40, 0);
        const int64_t free1 = llama_memory_n_cells_free(mem);
        CHECK(free1 == free0 - 40, "after 40 tokens: %lld free, expected %lld", (long long) free1, (long long) (free0 - 40));

        decode_n(ctx, vocab, 24, 40);
        const int64_t free2 = llama_memory_n_cells_free(mem);
        CHECK(free2 == free0 - 64, "after 64 tokens: %lld free, expected %lld", (long long) free2, (long long) (free0 - 64));

        // seq_rm of the whole sequence gives every cell back. (Not a partial range: a hybrid memory's
        // recurrent part refuses to drop a suffix of a sequence - its state is not a function of
        // position - and that refusal is correct, it is what the checkpoints are for.)
        CHECK(llama_memory_seq_rm(mem, 0, -1, -1), "seq_rm of the whole sequence failed");
        const int64_t free3 = llama_memory_n_cells_free(mem);
        CHECK(free3 == free0, "after seq_rm of the sequence: %lld free, expected %lld", (long long) free3, (long long) free0);

        decode_n(ctx, vocab, 8, 0);
        CHECK(llama_memory_n_cells_free(mem) == free0 - 8, "after re-decoding 8: %lld free, expected %lld",
              (long long) llama_memory_n_cells_free(mem), (long long) (free0 - 8));

        llama_memory_clear(mem, true);
        CHECK(llama_memory_n_cells_free(mem) == free0, "after clear: %lld free, expected %lld",
              (long long) llama_memory_n_cells_free(mem), (long long) free0);

        printf("OK: %lld -> %lld -> %lld -> %lld -> %lld (%s)\n",
               (long long) free0, (long long) free1, (long long) free2, (long long) free3, (long long) free0, argv[1]);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

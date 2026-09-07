#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>

// A recurrent cache cannot have its positions shifted. seq_add moves cell.pos and leaves
// r_l/s_l untouched, so after a shift the cache holds state computed over one history while
// claiming to sit at the positions of another. Nothing downstream can tell: the server's
// --cache-reuse path then restores checkpoints and serves answers that still depend on
// tokens the client deleted (measured on glm5next, GLM-TODO T3.5).
//
// llama_memory_can_shift() is the one switch every shifting path consults (server load-time
// refusal of --ctx-shift / --cache-reuse, per-request n_cache_reuse, common_init ctx_shift),
// so it must be false for any cache with a resident recurrent layer - pure recurrent and
// hybrid alike - and must stay true for an attention-only cache, which shifts correctly.
//
// Run against the dummy models from test-llama-archs -o: mamba (recurrent), qwen35 and
// nemotron_h (hybrid, through llama_memory_hybrid), llama (attention-only control).
int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init model/context\n", __func__);
        return 1;
    }

    const bool holds_recurrent_state = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
    const bool can_shift             = llama_memory_can_shift(llama_get_memory(ctx));

    fprintf(stderr, "%s : recurrent-or-hybrid = %d, can_shift = %d\n", __func__, holds_recurrent_state, can_shift);

    if (holds_recurrent_state && can_shift) {
        fprintf(stderr, "%s : FAIL - a cache holding recurrent state reports that it can shift positions\n", __func__);
        return 1;
    }

    if (!holds_recurrent_state && !can_shift) {
        fprintf(stderr, "%s : FAIL - an attention-only cache lost the ability to shift positions\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s : OK\n", __func__);
    return 0;
}

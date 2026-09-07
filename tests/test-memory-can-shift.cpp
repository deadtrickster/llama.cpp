#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>

// A recurrent cache CAN have its positions shifted. seq_add moves cell.pos and leaves r_l/s_l
// untouched, and that is right: the state is not a function of position. It is a function of
// CONTENT, so a caller that shifts because tokens were removed from the history (the server's
// --cache-reuse) holds state computed over a history the client no longer has, and must roll
// that state back to the last point the histories share - which is the server's job, done in
// tools/server/server-context.cpp. Refusing the shift at the memory level instead (the T3.5a
// stopgap) also switched off --ctx-shift and the per-request n_cache_reuse for every recurrent
// and hybrid model, for a defect that lives in one caller.
//
// llama_memory_can_shift() is the one switch every shifting path consults, so this pins what it
// says: true for attention-only, recurrent and hybrid caches alike; false only where the KV
// cache itself cannot apply a K-shift - M-RoPE positions (n_pos_per_embd > 1).
//
// Run against the dummy models from test-llama-archs -o: mamba (recurrent), nemotron_h (hybrid,
// through llama_memory_hybrid), llama (attention-only), qwen35 (hybrid with M-RoPE: the one
// that must stay non-shiftable, for a reason that has nothing to do with its recurrent half).
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

    const auto rope_type = llama_model_rope_type(model);
    const bool mrope     = rope_type == LLAMA_ROPE_TYPE_MROPE || rope_type == LLAMA_ROPE_TYPE_IMROPE;

    const bool expected  = !mrope;
    const bool can_shift = llama_memory_can_shift(llama_get_memory(ctx));

    fprintf(stderr, "%s : recurrent-or-hybrid = %d, mrope = %d, can_shift = %d, expected = %d\n", __func__,
            holds_recurrent_state, mrope, can_shift, expected);

    if (can_shift != expected) {
        if (mrope) {
            fprintf(stderr, "%s : FAIL - an M-RoPE cache reports that it can shift positions; K-shift cannot be applied there\n", __func__);
        } else if (holds_recurrent_state) {
            fprintf(stderr, "%s : FAIL - a cache holding recurrent state refuses position shifts; the state does not depend on position, "
                            "and the refusal takes --ctx-shift and --cache-reuse down with it\n", __func__);
        } else {
            fprintf(stderr, "%s : FAIL - an attention-only cache lost the ability to shift positions\n", __func__);
        }
        return 1;
    }

    fprintf(stderr, "%s : OK\n", __func__);
    return 0;
}

#include "common.cuh"

#define CUDA_SLICE_MEAN_BLOCK_SIZE 256

// fused [GGML_UNARY_OP_SIGMOID ->] GGML_OP_MUL -> reshape -> (view, GGML_OP_ADD)... -> GGML_OP_SCALE:
//   dst[e, t] = scale * sum_c a[e + c*n_inner, t] * f(b[e + c*n_inner, t]) + bias
// the mean over the n_slice streams of a hyper-connection residual, f = sigmoid or identity.
void ggml_cuda_op_slice_mean(ggml_backend_cuda_context & ctx,
        const ggml_tensor * a, const ggml_tensor * b, bool sigmoid_b,
        int64_t n_inner, int64_t n_slice, int64_t n_tokens, ggml_tensor * dst);

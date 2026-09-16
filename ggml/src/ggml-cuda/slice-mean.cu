#include "slice-mean.cuh"

// A hyper-connection block reads its input as the mean over the hc residual streams of
// norm(x) * sigmoid(gate): as ggml ops that is the gated product over the wide tensor, a cont
// of the first stream slice, hc-1 adds of the other slices and a scale, six launches that
// write and re-read the wide intermediate. One kernel: each thread owns one (e, t) and walks
// the slices in the same order the adds did.
template <bool sigmoid_b>
static __global__ void slice_mean_f32(
        const float * a, const float * b, float * dst,
        const float scale, const float bias,
        const int64_t n_inner, const int64_t n_slice, const int64_t n_tokens) {
    ggml_cuda_pdl_lc();
    const int64_t n   = n_inner*n_tokens;
    const int64_t idx = blockIdx.x*(int64_t) blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    const int64_t e = idx % n_inner;
    const int64_t t = idx / n_inner;

    ggml_cuda_pdl_sync();

    const int64_t base = t*n_slice*n_inner + e;

    // bit-exact with the separate ops: the sigmoid is unary.cu's, the product rounds before the
    // adds (no fma contraction), the adds run in slice order, and the final `scale * x + bias`
    // is scale.cu's expression (which nvcc contracts, as it does there)
    float sum = 0.0f;
    for (int64_t c = 0; c < n_slice; ++c) {
        const float av = a[base + c*n_inner];
        float       bv = b[base + c*n_inner];
        if (sigmoid_b) {
            bv = 1.0f / (1.0f + expf(-bv));
        }
        const float p = __fmul_rn(av, bv);
        sum = c == 0 ? p : __fadd_rn(sum, p);
    }

    dst[idx] = scale * sum + bias;
}

void ggml_cuda_op_slice_mean(ggml_backend_cuda_context & ctx,
        const ggml_tensor * a, const ggml_tensor * b, bool sigmoid_b,
        int64_t n_inner, int64_t n_slice, int64_t n_tokens, ggml_tensor * dst) {
    GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_nelements(a) == n_inner*n_slice*n_tokens && ggml_nelements(dst) == n_inner*n_tokens);

    float scale;
    float bias;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (const float *) dst->op_params + 1, sizeof(float));

    const int64_t n = n_inner*n_tokens;
    const int64_t num_blocks = (n + CUDA_SLICE_MEAN_BLOCK_SIZE - 1) / CUDA_SLICE_MEAN_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_SLICE_MEAN_BLOCK_SIZE, 0, ctx.stream());
    if (sigmoid_b) {
        ggml_cuda_kernel_launch(slice_mean_f32<true>,  launch_params, (const float *) a->data, (const float *) b->data, (float *) dst->data, scale, bias, n_inner, n_slice, n_tokens);
    } else {
        ggml_cuda_kernel_launch(slice_mean_f32<false>, launch_params, (const float *) a->data, (const float *) b->data, (float *) dst->data, scale, bias, n_inner, n_slice, n_tokens);
    }
}

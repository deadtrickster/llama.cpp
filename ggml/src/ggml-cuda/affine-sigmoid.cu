#include "affine-sigmoid.cuh"

// The per-stream mixing coefficients of a hyper-connection block (DeepSeek-V4, GLM-5.3) are
// sigmoid(mixes*scale + base)*k + c on a handful of values per token; as four ops that is four
// launches of a few microseconds each, twice per layer. One kernel, general broadcast semantics.
static __global__ void affine_sigmoid_f32(
        const char * GGML_CUDA_RESTRICT x, const char * GGML_CUDA_RESTRICT s, const char * GGML_CUDA_RESTRICT b, float * GGML_CUDA_RESTRICT dst,
        const float scale, const float bias,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const int64_t nbx0, const int64_t nbx1, const int64_t nbx2, const int64_t nbx3,
        const int64_t nes0, const int64_t nes1, const int64_t nes2, const int64_t nes3,
        const int64_t nbs0, const int64_t nbs1, const int64_t nbs2, const int64_t nbs3,
        const int64_t neb0, const int64_t neb1, const int64_t neb2, const int64_t neb3,
        const int64_t nbb0, const int64_t nbb1, const int64_t nbb2, const int64_t nbb3) {
    ggml_cuda_pdl_lc();
    const int64_t n   = ne0*ne1*ne2*ne3;
    const int64_t idx = blockIdx.x*(int64_t) blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    const int64_t i0 = idx % ne0;
    int64_t       t  = idx / ne0;
    const int64_t i1 = t % ne1;
    t /= ne1;
    const int64_t i2 = t % ne2;
    const int64_t i3 = t / ne2;

    ggml_cuda_pdl_sync();

    const float xv = *(const float *) (x + i0*nbx0 + i1*nbx1 + i2*nbx2 + i3*nbx3);
    const float sv = *(const float *) (s + (i0 % nes0)*nbs0 + (i1 % nes1)*nbs1 + (i2 % nes2)*nbs2 + (i3 % nes3)*nbs3);
    const float bv = *(const float *) (b + (i0 % neb0)*nbb0 + (i1 % neb1)*nbb1 + (i2 % neb2)*nbb2 + (i3 % neb3)*nbb3);

    const float y = xv*sv + bv;

    dst[idx] = scale/(1.0f + expf(-y)) + bias;
}

void ggml_cuda_op_affine_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * mul, ggml_tensor * add, ggml_tensor * dst) {
    const ggml_tensor * x = mul->src[0];
    const ggml_tensor * s = mul->src[1];
    const ggml_tensor * b = add->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32 && s->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(x, dst) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_can_repeat(s, x) && ggml_can_repeat(b, x));

    float scale;
    float bias;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (const float *) dst->op_params + 1, sizeof(float));

    const int64_t n = ggml_nelements(dst);
    const int64_t num_blocks = (n + CUDA_AFFINE_SIGMOID_BLOCK_SIZE - 1) / CUDA_AFFINE_SIGMOID_BLOCK_SIZE;

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_AFFINE_SIGMOID_BLOCK_SIZE, 0, ctx.stream());
    ggml_cuda_kernel_launch(affine_sigmoid_f32, launch_params,
        (const char *) x->data, (const char *) s->data, (const char *) b->data, (float *) dst->data, scale, bias,
        x->ne[0], x->ne[1], x->ne[2], x->ne[3],
        (int64_t) x->nb[0], (int64_t) x->nb[1], (int64_t) x->nb[2], (int64_t) x->nb[3],
        s->ne[0], s->ne[1], s->ne[2], s->ne[3],
        (int64_t) s->nb[0], (int64_t) s->nb[1], (int64_t) s->nb[2], (int64_t) s->nb[3],
        b->ne[0], b->ne[1], b->ne[2], b->ne[3],
        (int64_t) b->nb[0], (int64_t) b->nb[1], (int64_t) b->nb[2], (int64_t) b->nb[3]);
}

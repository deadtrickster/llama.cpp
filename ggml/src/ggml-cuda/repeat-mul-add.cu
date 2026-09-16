#include "repeat-mul-add.cuh"

// A hyper-connection block scatters its output back into the hc residual streams as
// res + b[e, t] * w[c, t]: as ggml ops that is a REPEAT of b to the stream width, a MUL by the
// per-stream weight and the residual ADD, three launches over the widest tensor of the layer.
// One kernel, general broadcast semantics on both operands.
template <bool has_res>
static __global__ void repeat_mul_add_f32(
        const char * res, const char * b, const char * c, float * dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const int64_t nbr0, const int64_t nbr1, const int64_t nbr2, const int64_t nbr3,
        const int64_t neb0, const int64_t neb1, const int64_t neb2, const int64_t neb3,
        const int64_t nbb0, const int64_t nbb1, const int64_t nbb2, const int64_t nbb3,
        const int64_t nec0, const int64_t nec1, const int64_t nec2, const int64_t nec3,
        const int64_t nbc0, const int64_t nbc1, const int64_t nbc2, const int64_t nbc3) {
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

    const float bv = *(const float *) (b + (i0 % neb0)*nbb0 + (i1 % neb1)*nbb1 + (i2 % neb2)*nbb2 + (i3 % neb3)*nbb3);
    const float cv = *(const float *) (c + (i0 % nec0)*nbc0 + (i1 % nec1)*nbc1 + (i2 % nec2)*nbc2 + (i3 % nec3)*nbc3);

    // bit-exact with the separate ops: the mul rounds to f32 before the add (no fma contraction)
    const float p = __fmul_rn(bv, cv);
    if (has_res) {
        const float rv = *(const float *) (res + i0*nbr0 + i1*nbr1 + i2*nbr2 + i3*nbr3);
        dst[idx] = __fadd_rn(rv, p);
    } else {
        dst[idx] = p;
    }
}

void ggml_cuda_op_repeat_mul_add(ggml_backend_cuda_context & ctx, ggml_tensor * repeat, ggml_tensor * mul, ggml_tensor * add) {
    const ggml_tensor * b   = repeat->src[0];
    const ggml_tensor * c   = mul->src[0] == repeat ? mul->src[1] : mul->src[0];
    const ggml_tensor * res = add ? (add->src[0] == mul ? add->src[1] : add->src[0]) : nullptr;
    ggml_tensor * dst = add ? add : mul;

    GGML_ASSERT(b->type == GGML_TYPE_F32 && c->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_can_repeat(b, dst) && ggml_can_repeat(c, dst));
    GGML_ASSERT(!res || (res->type == GGML_TYPE_F32 && ggml_are_same_shape(res, dst)));

    const int64_t n = ggml_nelements(dst);
    const int64_t num_blocks = (n + CUDA_REPEAT_MUL_ADD_BLOCK_SIZE - 1) / CUDA_REPEAT_MUL_ADD_BLOCK_SIZE;

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_REPEAT_MUL_ADD_BLOCK_SIZE, 0, ctx.stream());
    const char * res_d = res ? (const char *) res->data : nullptr;
    auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, launch_params,
            res_d, (const char *) b->data, (const char *) c->data, (float *) dst->data,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
            res ? (int64_t) res->nb[0] : 0, res ? (int64_t) res->nb[1] : 0, res ? (int64_t) res->nb[2] : 0, res ? (int64_t) res->nb[3] : 0,
            b->ne[0], b->ne[1], b->ne[2], b->ne[3],
            (int64_t) b->nb[0], (int64_t) b->nb[1], (int64_t) b->nb[2], (int64_t) b->nb[3],
            c->ne[0], c->ne[1], c->ne[2], c->ne[3],
            (int64_t) c->nb[0], (int64_t) c->nb[1], (int64_t) c->nb[2], (int64_t) c->nb[3]);
    };
    if (res) {
        launch(repeat_mul_add_f32<true>);
    } else {
        launch(repeat_mul_add_f32<false>);
    }
}

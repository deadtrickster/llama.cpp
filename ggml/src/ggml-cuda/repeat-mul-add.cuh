#include "common.cuh"

#define CUDA_REPEAT_MUL_ADD_BLOCK_SIZE 256

// fused GGML_OP_REPEAT + GGML_OP_MUL [+ GGML_OP_ADD]:
//   dst = [res +] repeat(b) * c
// with b and c broadcast into dst the way ggml_repeat / ggml_mul broadcast (ggml_can_repeat).
void ggml_cuda_op_repeat_mul_add(ggml_backend_cuda_context & ctx, ggml_tensor * repeat, ggml_tensor * mul, ggml_tensor * add);

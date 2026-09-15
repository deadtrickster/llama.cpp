#include "common.cuh"

#define CUDA_AFFINE_SIGMOID_BLOCK_SIZE 256

// fused GGML_OP_MUL + GGML_OP_ADD + GGML_UNARY_OP_SIGMOID + GGML_OP_SCALE:
//   dst = sigmoid(x * s + b) * scale + bias
// with s and b broadcast over x the way ggml_mul / ggml_add broadcast (ggml_can_repeat).
void ggml_cuda_op_affine_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * mul, ggml_tensor * add, ggml_tensor * dst);

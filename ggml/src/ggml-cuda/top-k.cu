#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#if !defined(GGML_CUDA_USE_CUB) || defined(GGML_USE_HIP) || defined(CUB_TOP_K_AVAILABLE)

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

// The gather is deterministic: every (row, block) owns a contiguous column range, counts its
// greater/equal elements first, and then places them by column rank. Slots handed out by atomics
// in arrival order made both the tie set at the k-th value and the output order vary from run to
// run, which showed up as different greedy continuations of the same 33k prompt.

// per (row, block): how many keys above the threshold and how many equal to it
template<int BLOCK_SIZE>
static __global__ void top_k_radix_count(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ counts,
        int ncols,
        int blocks_per_row,
        int cols_per_block) {
    const int row       = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid       = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    const uint32_t prefix = states[row].prefix;

    const int col_beg = row_block * cols_per_block;
    const int col_end = min(ncols, col_beg + cols_per_block);

    int n_greater = 0;
    int n_equal   = 0;
    for (int col = col_beg + tid; col < col_end; col += BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        n_greater += key >  prefix;
        n_equal   += key == prefix;
    }
    n_greater = warp_reduce_sum(n_greater);
    n_equal   = warp_reduce_sum(n_equal);

    __shared__ int s_greater[BLOCK_SIZE / WARP_SIZE];
    __shared__ int s_equal  [BLOCK_SIZE / WARP_SIZE];
    if (tid % WARP_SIZE == 0) {
        s_greater[tid / WARP_SIZE] = n_greater;
        s_equal  [tid / WARP_SIZE] = n_equal;
    }
    __syncthreads();
    if (tid == 0) {
        int g = 0, e = 0;
        for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; ++w) {
            g += s_greater[w];
            e += s_equal[w];
        }
        counts[2*((size_t) row * blocks_per_row + row_block) + 0] = g;
        counts[2*((size_t) row * blocks_per_row + row_block) + 1] = e;
    }
}

// exclusive scan of a flag over the block, Hillis-Steele in shared memory; returns the thread's offset
template<int BLOCK_SIZE>
static __device__ __forceinline__ int top_k_block_exclusive_scan(int v, int * s_scan, int & total) {
    const int tid = threadIdx.x;
    s_scan[tid] = v;
    __syncthreads();
    for (int off = 1; off < BLOCK_SIZE; off *= 2) {
        const int t = tid >= off ? s_scan[tid - off] : 0;
        __syncthreads();
        s_scan[tid] += t;
        __syncthreads();
    }
    total = s_scan[BLOCK_SIZE - 1];
    const int excl = s_scan[tid] - v;
    __syncthreads();
    return excl;
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const top_k_radix_state * __restrict__ states,
        const int * __restrict__ counts,
        int ncols,
        int k,
        int blocks_per_row,
        int cols_per_block) {
    const int row       = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid       = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    const top_k_radix_state state = states[row];

    // output layout: the keys above the threshold in column order, then the first `rank` keys equal
    // to it in column order
    const int n_greater_total = k - state.rank;

    // this block's starting ranks: the counts of the blocks before it in the row
    int base_greater = 0;
    int base_equal   = 0;
    for (int b = 0; b < row_block; ++b) {
        base_greater += counts[2*((size_t) row * blocks_per_row + b) + 0];
        base_equal   += counts[2*((size_t) row * blocks_per_row + b) + 1];
    }

    __shared__ int s_scan[BLOCK_SIZE];

    const int col_beg = row_block * cols_per_block;
    const int col_end = min(ncols, col_beg + cols_per_block);
    for (int tile = col_beg; tile < col_end; tile += BLOCK_SIZE) {
        const int col = tile + tid;
        uint32_t key = 0;
        bool in_range = col < col_end;
        if (in_range) {
            key = top_k_float_to_ordered(row_src[col]);
        }
        const int is_greater = in_range && key >  state.prefix;
        const int is_equal   = in_range && key == state.prefix;

        int tot_greater, tot_equal;
        const int off_greater = top_k_block_exclusive_scan<BLOCK_SIZE>(is_greater, s_scan, tot_greater);
        const int off_equal   = top_k_block_exclusive_scan<BLOCK_SIZE>(is_equal,   s_scan, tot_equal);

        if (is_greater) {
            row_dst[base_greater + off_greater] = col;
        } else if (is_equal) {
            const int r = base_equal + off_equal;
            if (r < state.rank) {
                row_dst[n_greater_total + r] = col;
            }
        }
        base_greater += tot_greater;
        base_equal   += tot_equal;
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);
    // contiguous, block-aligned column ranges so that ranks within a row follow column order
    const int cols_per_block = ((ncols + blocks_per_row - 1) / blocks_per_row + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    ggml_cuda_pool_alloc<int> counts_alloc(pool, (size_t) nrows * blocks_per_row * 2);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();
    int * counts     = counts_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_count<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(src, states, counts, ncols, blocks_per_row, cols_per_block);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(src, dst, states, counts, ncols, k, blocks_per_row, cols_per_block);
}

#endif // !defined(GGML_CUDA_USE_CUB) || defined(GGML_USE_HIP) || defined(CUB_TOP_K_AVAILABLE)

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // one CUB top-k per row is ~3 launches per row: a sparse-attention indexer selecting 2048 of n_kv
    // keys for every token of a 512-token prefill microbatch, on every layer, was 120k launches and
    // 10% of a GLM-5.3 prefill. past a few rows the multi-row radix select (11 launches for all rows)
    // wins; GGML_CUDA_TOPK_RADIX_MIN_ROWS moves the threshold (0 keeps CUB always)
    static const int radix_min_rows = []() {
        const char * env = getenv("GGML_CUDA_TOPK_RADIX_MIN_ROWS");
        return env ? atoi(env) : 16;
    }();
    if (radix_min_rows > 0 && nrows >= radix_min_rows && ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
#if defined(GGML_USE_HIP)
    if (ncols > 1024) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    } else {
#endif // defined(GGML_USE_HIP)
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
#if defined(GGML_USE_HIP)
    }
#endif // defined(GGML_USE_HIP)
#endif
}

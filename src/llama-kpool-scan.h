#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

// The pooled-indexer input (llama_kv_cache_set_input_kpool) derives, per sequence, a map from
// KV cells to fixed-size pools by scanning every cell of the pool extent: four O(n_kv) passes with
// a division and a bitset test per cell. Measured on a live GLM-5.3-Flash server with several
// resident conversations: 7% of the decode thread, ~5 ms per verify step, for a result that only
// changes when the cells change. This is that result, kept on the cache and reused while the
// cells' generation, the extent and the pool geometry are the ones it was computed for.
struct llama_kpool_scan {
    // key
    uint64_t     gen   = UINT64_MAX;
    int64_t      n_kv  = -1;
    uint32_t     r     = 0;
    llama_seq_id seq   = -1;

    // pass 1: pool index range of the sequence's cells (b = pos / r)
    bool    found = false;
    int64_t b_min = 0;
    int64_t b_max = 0;

    // pass 2, valid for one (n_run) - the run length after the per-stream rebalancing
    int64_t n_run  = -1;
    int64_t b_base = 0;

    std::vector<llama_pos> pos_at;   // [n_kv] position of the cell, -1 if not this sequence's
    std::vector<int32_t>   pool_of;  // [n_kv] complete pool the cell belongs to, -1 if none
    std::vector<int32_t>   filled;   // [n_run] members found per pool
    std::vector<int32_t>   cells;    // [n_run*r] cell index per pool slot (0 where unfilled)
};

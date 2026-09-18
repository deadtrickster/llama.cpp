// common_state_buf's slab pool must be bounded in BYTES and must not hand a huge slab to a small request.
// Measured 2026-09-18 on lab2x1: the pool kept 12 freed ~24 GB slabs (a count cap), the RAM prompt cache
// reported 45 GB while the process held 134 GB anonymous, and the kernel OOM-killed the server at 174 GB.

#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int failures = 0;

#define CHECK(cond, ...) do { if (!(cond)) { failures++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

static const size_t MIB = 1ull << 20;

int main() {
    // 1. freed slabs beyond the byte cap are unmapped, not pooled
    common_state_buf_pool_set_cap(256 * MIB);
    {
        std::vector<common_state_buf> bufs(8);
        for (auto & b : bufs) {
            b.resize(64 * MIB); // each becomes its own 64 MiB slab (fresh() rounds to 64 MiB)
            b.data()[0] = 1;
        }
    }
    {
        size_t n_slabs = 0, bytes = 0;
        common_state_buf_pool_stats(n_slabs, bytes);
        CHECK(bytes <= 256 * MIB, "pool holds %zu MiB in %zu slabs after freeing 8 x 64 MiB with a 256 MiB cap", bytes / MIB, n_slabs);
        CHECK(n_slabs >= 1, "the pool kept nothing although the cap allows 256 MiB");
    }

    // 2. a small request does not take a huge slab (first-fit made a 300 MB checkpoint hold a 24 GB slab)
    common_state_buf_pool_set_cap(0);        // drain what 1. left
    common_state_buf_pool_set_cap(4096 * MIB);
    {
        common_state_buf big;
        big.resize(1024 * MIB);
        big.data()[0] = 1;
    } // 1 GiB slab is now pooled (cap 4 GiB)
    {
        common_state_buf small;
        small.resize(1 * MIB);
        CHECK(small.capacity() < 1024 * MIB, "a 1 MiB request took a %zu MiB slab", small.capacity() / MIB);
        // a 600 MiB request must not take it either: 1.7x over-allocation is 40 GB of RSS behind a 55 GB
        // cache on the lab box (measured 2026-09-18: RSS 146 GB, cache 55.5 GB, pool capped at 32 GiB)
        common_state_buf mid;
        mid.resize(600 * MIB);
        CHECK(mid.capacity() < 1024 * MIB, "a 600 MiB request took a %zu MiB slab", mid.capacity() / MIB);
    }
    common_state_buf_pool_set_cap(0);
    common_state_buf_pool_set_cap(4096 * MIB);
    {
        common_state_buf big;
        big.resize(1024 * MIB);
        big.data()[0] = 1;
    }
    {
        // ... while one within an eighth of the slab does reuse it (a conversation re-saved a turn later)
        common_state_buf near;
        near.resize(960 * MIB);
        CHECK(near.capacity() == 1024 * MIB, "a 960 MiB request did not reuse the pooled 1 GiB slab (got %zu MiB)", near.capacity() / MIB);
    }

    // 3. a cap of zero keeps nothing
    common_state_buf_pool_set_cap(0);
    {
        common_state_buf b;
        b.resize(64 * MIB);
    }
    {
        size_t n_slabs = 0, bytes = 0;
        common_state_buf_pool_stats(n_slabs, bytes);
        CHECK(bytes == 0 && n_slabs == 0, "cap 0 still pooled %zu MiB in %zu slabs", bytes / MIB, n_slabs);
    }

    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("test-state-buf: OK\n");
    return 0;
}

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

    // 3. live accounting: what the slabs in use map, what RSS sees for them
    common_state_buf_pool_set_cap(0);
    {
        size_t n_live = 0, live = 0;
        common_state_buf_live_stats(n_live, live);
        const size_t live0 = live;
        {
            common_state_buf a, b;
            a.resize(64 * MIB); a.data()[0] = 1;
            b.resize(64 * MIB); b.data()[0] = 1;
            common_state_buf_live_stats(n_live, live);
            CHECK(live >= live0 + 128 * MIB, "two 64 MiB buffers in use, live grew by %zu MiB", (live - live0) / MIB);
            CHECK(n_live >= 2, "n_live is %zu with two buffers in use", n_live);
        }
        common_state_buf_live_stats(n_live, live);
        CHECK(live == live0, "buffers freed, live is %zu MiB over the baseline", (live - live0) / MIB);
    }

    // 4. a small buffer maps a small slab: a 1 MiB checkpoint blob used to map 64 MiB (fresh() rounded every
    // request up to 64 MiB); with 32 checkpoints x 3 buffers per conversation that was ~4 GB of rounding each
    {
        common_state_buf small;
        small.resize(1 * MIB);
        CHECK(small.capacity() <= 4 * MIB, "a 1 MiB request mapped %zu MiB", small.capacity() / MIB);
        common_state_buf big;
        big.resize(300 * MIB);
        CHECK(big.capacity() % (64 * MIB) == 0 && big.capacity() >= 300 * MIB, "a 300 MiB request mapped %zu MiB (not a 64 MiB multiple)", big.capacity() / MIB);
    }

    // 5. clear() keeps the mapping for reuse; clear() + shrink_to_fit() gives it back. The cache's release paths
    // (spill with release_ram, an entry loaded into a slot, an obsolete stub) all say clear(); shrink_to_fit()
    // and meant "free it" - with shrink_to_fit a no-op every released entry kept its full slab, unattributed
    // (2026-09-19: "other 15.5 GiB" in the [mem] line 28 minutes after a restart)
    {
        size_t n_live = 0, live0 = 0, live = 0;
        common_state_buf_live_stats(n_live, live0);
        common_state_buf b;
        b.resize(64 * MIB); b.data()[0] = 1;
        b.clear();
        CHECK(b.capacity() >= 64 * MIB, "clear() alone released the mapping (capacity %zu MiB)", b.capacity() / MIB);
        b.shrink_to_fit();
        CHECK(b.capacity() == 0, "clear() + shrink_to_fit() kept %zu MiB mapped", b.capacity() / MIB);
        common_state_buf_live_stats(n_live, live);
        CHECK(live == live0, "the released slab still counts as live (%zu MiB over the baseline)", (live - live0) / MIB);
        b.resize(1 * MIB);
        CHECK(b.size() == 1 * MIB && b.data() != nullptr, "the buffer is unusable after shrink_to_fit");
    }

    // 6. a cap of zero keeps nothing
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

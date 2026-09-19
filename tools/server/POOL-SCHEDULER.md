# The KV pool scheduler

Status: stages 1 and 2 implemented 2026-09-18 (`llama_memory_n_cells_free`, `pool_admit` / `pool_make_room`);
stages 3 and 4 open. Replaces the reactive pressure ladder in `server-context.cpp`.

## Why

Every pool failure of 2026-09-17/18 was the same failure: two bookkeepings disagreeing.

- a resume reserved a sequence id; the ceiling shrink did not know and took it back (abort)
- the restore ladder said "room", `state_read_meta` said no (115k prefilled)
- `pool_cells_reclaimable` excluded a *yielded* generation whose cells were the whole point,
  although rung 1b exists to offload exactly that (207k re-prefilled for 14 min)
- the sub-batch cut keyed on `i_batch`, which a speculative group never sets (cut never moved)
- the halving retry stranded a slot's logits outside the decoded view (`get_logits_ith` assert,
  the server hung on its port)

None is wrong locally. The structure is: the pressure path is **reactive** - the ladder runs only
after `llama_decode` returns `ret = 1`, inside a batch whose slot indices, speculative groups and
logits flags are already committed. Any repair at that point can invalidate what is in the batch.
And "who holds what" lives in five places (slot, sequence record, prompt-cache entry, offloaded
state, mid-flight flag) with five accountings that drift. Patching a rung exposes the next hole.

## The model

A shared pool `P` (cells; the KV cache is the truth, not a server-side count). Jobs take slices.

    job j:  C_j   constant part - its KV so far. 0 when offloaded. A prefill's C_j GROWS: every
                  step converts D_j into C_j.
            D_j   dynamic part - next-step need. 1 + n_draft generating; one chunk prefilling.
            where resident | RAM | disk
            q_j   quantum used this turn (tokens generated, or chunks prefilled)

A seat is NOT a resource. It is "in the batch this tick." Conflating the two is the yielded-resident
bug: the quantum handed over the seat while the job kept its C_j in the pool.

Feasibility of a tick, checked BEFORE the batch is built:

    sum over resident j of C_j  +  sum over j in batch of D_j  <=  P

`ret = 1` from `llama_decode` becomes an assertion. The halving loop and the stranded-index class of
bugs go with it: indices are assigned after the resident set is fixed.

## The tick

1. Fill the batch greedily with resident jobs whose D_j fits ("batch as many as possible").
2. A job wants in (a restore, or a resident whose D_j no longer fits): pick victims by ONE cost,
   cost = C_j (the copy out, and the copy back later). In order:
     a. yielded residents - free, they are not computing (rung 1b, offload)
     b. idle finished residents, smallest C_j first (rung 1, evict to the prompt cache)
     c. a prefill mid-way, largest C_j (rung 2b, park)
     d. a running generation, largest C_j (rung 2, suspend)
   The rungs stay as ACTIONS the scheduler emits. They stop deciding for themselves, and "what can
   I move" is the same function that picks the victim - today feasibility and victim choice are two
   functions, and they disagree.
3. Everything that does not fit waits offloaded; it comes back when it fits, into room for C_j + a
   turn (below), never into exactly its own cells.

## The degeneracy, and its one rule

When constant parts dominate - `C_a + C_b > P` - a and b can never be co-resident. The schedule MUST
alternate, and every alternation pays ~2*C of copies (out and back). The only way that is not
thrash is a turn that amortizes the copy:

    quantum_j  proportional to  C_j

A job that cost 200k cells to bring in gets a turn long enough that its copy is a few percent of it
(~3 s per 200k swap at ~30 t/s: >= 500 tokens, not 16). This is the resume hysteresis, derived
instead of guessed, and it is the ONLY tunable that matters. Prefills obey it too: park at chunk
boundaries, never one that just cost a swap to bring back. The reciprocal: prefer keeping a large
C_j resident when choosing victims - evicting a small C_j is cheap.

Fairness is a latency bound, not a throughput one: in the degenerate regime the pool's throughput
is fixed by the copies; the quantum decides how long each conversation waits.

## What this replaces

- `pool_grow_pending / pool_defer_pending / pool_evict_resident / pool_offload_waiting /
  pool_suspend_running / fail_slot_under_pressure` as a ladder run on `ret = 1`: become actions.
- the `n_batch /= 2` retry loop in `decode()`: deleted. `ret = 1` fails the ONE slot whose need the
  scheduler mis-sized, with a log line, never a halving.
- `pool_cells_free / pool_cells_held / pool_cells_reclaimable / pool_has_room_for /
  pool_room_for*`: one `pool_fits(resident set, batch)` asked of the KV cache.
- `--slot-quantum`, `--slot-resume-after`, `--decode-per-prefill`, `--pool-min-grow`: the first two
  become the quantum rule; the third a scheduler weight; the fourth goes away (a grow that is a
  crumb is the KV's business, the scheduler just sees P).

## Tests

Feasibility becomes a pure function of (P, {C_j, D_j}). The scheduler is unit-tested with a fake
pool and fake jobs - no model, no device, no fake-device budget, no emergent wall. The tiny-model
server tests then check only that the actions do what the scheduler asked (offload/park/evict/
restore), which they already do.

The emergent-wall tests of 2026-09-18 (`LLAMA_SERVER_POOL_SELFTEST` calibrated to a budget) were
the wrong strategy for this subsystem: they raced the device and went red/green by timing.

## What stage 2 taught (measured, 2026-09-18)

- The victim is chosen by cost alone, the asking slot included. Protecting the asker (`keep`) made the
  shallow generation the victim beside a deep one asking for its next cell; the tests that name the
  largest as the victim, and the restore that expects the deep one to wait, both need the asker eligible.
- A refused slot with an EMPTY batch behind it can never be admitted: nothing runs, so nothing finishes
  and frees cells, and `pool_make_room` already found nothing to move. It fails and spills. Letting it
  "wait" spun the update loop at full speed - 49 GB of log from one test on a tmpfs.
- The wait is logged once per episode (`slot.pool_waiting`), not per tick.
- Anything collected before admission can point at a slot admission then moved out: the `generating`
  list and `batch.slot_batched` both did (`GGML_ASSERT(task)`, `seq_id = -1` in the batch). Collect,
  admit, then trust nothing collected without re-checking `is_processing()`.
- The emergent-wall tests survive stage 2 with their budgets recalibrated (the KV's own free count is
  larger than the old held count, so a wall calibrated against the latter lands ~300 cells later).

## The RAM side of an id (measured 2026-09-19)

An id's cost is not only its KV cells. A conversation keeps up to `--ctx-checkpoints` (32) context
checkpoints, ~180-330 MiB each on Qwen3.8-27B, wherever it sits: on its seat, in the sequence registry
when it yields or finishes, and in the prompt cache until the ladder thins them. Only the cache has a
ladder. The pool raises its id ceiling into free VRAM (`--seq-max-headroom`) and never asks what those
ids cost in RAM: "5 ids of 5" at 22:xx was ~40 GiB of checkpoints beside a 56 GiB cache and a 32 GiB
slab pool, on a 185 GB box - the term behind the second OOM kill after the slab fixes.

Two consequences for the scheduler:
- `C_j` has a RAM component (checkpoints + a pending offload copy), and feasibility of one more resident id
  must be checked against RAM as well as cells - today the `[mem]` line only reports it.
- The ladder's rungs (drop middle checkpoints, keep newest + pinned, drop the rest) belong to seats and
  the registry as much as to the cache: a `--checkpoint-ram` budget across all three, LRU idle sequence
  first. Until then `--ctx-checkpoints 8` in the launcher bounds the per-id cost.

## Order of work

1. The table and `pool_fits()` asked of the KV cache - the truth. Small, mechanical, makes the rest
   testable.
2. Admission before decode: build the batch from the feasible set; delete the halving loop. This is
   where the abort lives; it goes with the repro harness in place.
3. The quantum rule for both generations and prefills; `--decode-per-prefill` as a weight.
4. Delete the ladder's decision code; keep the actions.

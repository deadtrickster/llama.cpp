# Prompt-cache eviction: granularity over eviction, deep-reuse aware

This documents the lab2x1 prompt-cache eviction policy (`glm-all`). It replaces
the historical "LRU whole-entry spill" with a ladder that sheds *granularity*
first, distinguishes active from dead conversations, and treats the disk tier as
a long-lived store rather than a scratch space.

## Why

The RAM tier (`--cache-ram`) was a resting level, not a ceiling: while the
server was awake the cache climbed to its limit and sat there, so an
OOM/SIGKILL threw away everything resident, and a hot-but-shallow conversation
(fast tool calls) grew its checkpoint sprawl until it starved everyone else.
Neither RAM nor disk is infinite, so eviction must be a *score*, not a binary.

## The signal: `deep_reuse`

A checkpoint's reuse count is essentially 0 or 1 — append-only growth never
touches deep checkpoints, and a rollback lands on one checkpoint then diverges
so it is dead. A per-checkpoint counter is therefore noise. The right
granularity is one bit per conversation:

- `deep_reuse` flips true when a restore lands on a **non-tip** context
  checkpoint (the conversation actually revisited its own deep history).
- It is set on the slot during a checkpoint restore, travels with the
  `server_prompt`, and is persisted in the L2 spill header (v3; v2 files read
  as `false`).

A never-rolled-back conversation is append-only, so its deep checkpoints are
dead weight and shed first. A rolled-back conversation has proven it uses deep
history, so it keeps granularity longer.

## The ladder

Per entry, under RAM pressure, in increasing cost:

| rung | what is shed | restore cost |
|---|---|---|
| 0→1 | middle checkpoints dropped, newest + oldest kept | a deep rollback reprocesses from the oldest anchor |
| 1→2 | oldest anchor dropped, newest kept | a rollback past the tip reprocesses from 0 |
| 2→3 | MTP draft state dropped | decode is unaccelerated until MTP warms up |
| exhausted | entry spilled to disk | a disk read |

The victim is the resident entry with the lowest `(deep_reuse, t_last_used,
degrade_level)` — never-rolled-back before rolled-back, older before newer, so
the busy conversation keeps its granularity and the loss spreads within a class.

## Active vs dead

- `--cache-active-seconds` (default 300): an entry hit within this window is
  *active*.
- Under pressure, the least-recently-used **inactive** resident entry spills
  first; active entries spill only when the pool still cannot fit (10+
  conversations genuinely do not).
- The periodic spill (`--cache-spill-seconds`, default 60) moves only *inactive*
  entries to disk, so an active conversation is not evicted from RAM every
  interval, and a hard kill loses at most one interval of the RAM tier.

## Disk tier

- `--cache-reap-seconds` (default 172800 = 48h): a spilled entry idle longer
  than this is reaped from disk regardless of the size budget. This is the
  "keep as much as possible, at the cost of granularity" rule — 48h of spills
  live on NVMe and are only dropped by age, not by a too-active neighbour's
  sprawl.
- Size-based LRU trimming still applies once `--cache-disk` is exceeded.

## Compaction

A compaction is a prompt that is *not* a prefix-extension of any cached entry
(history replaced by a summary). On it, the surviving entries are the dead
pre-compaction snapshots: they are spilled once (restorable) and then dropped
from the index, so one conversation stops holding every historical depth. This
is correct for the single-active-conversation shape the server is sized for; a
multi-conversation server would key entries by session and drop only the
compacted one's own history.

## Parameters

| flag | default | meaning |
|---|---|---|
| `--cache-spill-seconds` | 60 | periodic background spill of inactive entries (0 = off) |
| `--cache-active-seconds` | 300 | hit within N seconds = active, stays resident (0 = LRU decides) |
| `--cache-reap-seconds` | 172800 | reap spilled entries idle longer than N seconds (0 = size-only) |

# Prompt-cache eviction: granularity over eviction, deep-reuse aware

This documents the lab2x1 prompt-cache policy (`glm-all`). It replaces the
historical "LRU whole-entry spill" with a ladder that sheds *granularity* first,
distinguishes active from dead conversations, mirrors the RAM tier to disk for
crash survivability, and treats the disk tier as a long-lived store.

## Why

The RAM tier (`--cache-ram`) is a budget GLM is meant to *use*, not a ceiling it
should hover under. The historical failure modes were: the cache sat at its
limit and an OOM/SIGKILL threw away everything resident (there was effectively
no swap), and a hot-but-shallow conversation (fast tool calls) grew its
checkpoint sprawl until it starved everyone else. The fix is three levers:
fill RAM and use it, mirror it to disk so a crash is cheap, and shed
*checkpoints* — not conversations — under pressure. Swap absorbs OS-level
overflow so the kernel never has to kill GLM.

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
| exhausted | entry evicted (RAM released; it was already mirrored to disk) | a disk read |

The victim is the resident entry with the lowest `(deep_reuse, t_last_used,
degrade_level)` — never-rolled-back before rolled-back, older before newer, so
the busy conversation keeps its granularity and the loss spreads within a class.

## Mirror: crash survivability without draining RAM

An entry has a `resident` bit (bulk buffers in RAM) *and* a spill file (bulk
buffers on disk); they are no longer mutually exclusive.

- `--cache-spill-seconds` (default 60): every N seconds `mirror_resident()`
  writes each resident entry **through to disk while keeping it resident**. A
  power cut / SIGKILL then loses at most one interval of the RAM tier, but the
  RAM tier stays full — the mirror is a safety net, not an eviction.
- Eviction is the only thing that releases RAM: when the tier is full, the
  ladder degrades, then the LRU entry is *spilled* (its RAM copy dropped; the
  disk copy is already there or is written now).

So the RAM tier is fully occupied, the disk tier is a write-through mirror plus
the evicted overflow, and swap is the final OS backstop for anything beyond the
mirror's interval.

## Active vs dead

- `--cache-active-seconds` (default 300): an entry hit within this window is
  *active*.
- This only orders eviction: under pressure the least-recently-used **inactive**
  resident entry is spilled first; active entries spill only when the pool still
  cannot fit. The mirror is unconditional — it does not care about activity.

## Disk tier

- `--cache-reap-seconds` (default 172800 = 48h): a fully-spilled entry idle
  longer than this is reaped from disk regardless of the size budget — "keep as
  much as possible, at the cost of granularity". Mirrored entries (still in RAM)
  are never reaped by age.
- Size-based LRU trimming still applies once `--cache-disk` is exceeded, and
  also only touches fully-spilled entries.

## Compaction

A compaction is a prompt that is *not* a prefix-extension of any cached entry
(history replaced by a summary). On it, the surviving entries are the dead
pre-compaction snapshots: they are mirrored once (restorable) and then dropped
from the index, so one conversation stops holding every historical depth. This
is correct for the single-active-conversation shape the server is sized for; a
multi-conversation server would key entries by session and drop only the
compacted one's own history.

## Parameters

| flag | default | meaning |
|---|---|---|
| `--cache-spill-seconds` | 60 | periodic write-through mirror of resident entries (0 = off) |
| `--cache-active-seconds` | 300 | hit within N seconds = active, last to spill under pressure (0 = LRU decides) |
| `--cache-reap-seconds` | 172800 | reap fully-spilled entries idle longer than N seconds (0 = size-only) |

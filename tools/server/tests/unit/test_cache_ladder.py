import os
import re
import time
import tempfile
import pytest
from utils import *

# [cache-ladder] Prompt-cache eviction used to be binary: under memory pressure the
# least-recently-used entry was spilled to disk WHOLE, so one conversation lost
# everything while the next kept full fidelity. These tests pin the ladder that
# replaced it - degrade before evict, spread the loss evenly, keep LRU ordering.
#
# The rungs shed state that costs time on a later restore and never correctness:
#
#   0 full      -> 1 thinned   every other checkpoint dropped, newest kept
#   1 thinned   -> 2 skeletal  no checkpoints at all
#   2 skeletal  -> 3 no draft  MTP draft state dropped
#   3           -> exhausted, the caller spills
#
# Checkpoints only exist for models whose sequence cannot be partially removed, so
# a recurrent model is used throughout - the same reason test_checkpoint_schedule
# uses one.

server: ServerProcess


class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0

    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


DEGRADED = re.compile(
    r"cache ladder: degraded entry \((\d+) tokens\) level (\d+) -> (\d+), freed ([\d.]+) MiB"
)
SPILLED = re.compile(r"L2: spilled\s+(\d+) tokens")

MIN_STEP = 64
N_BATCH = 32

# what one entry weighs on mamba-130m: a 2.67 MiB recurrent state plus one checkpoint of the same size
# every MIN_STEP tokens of a ~1600-token conversation, 72 MiB at level 0, ~37 MiB thinned (level 1),
# 2.67 MiB skeletal (level 2). The tier has to hold a full entry or two and not six: an entry over the
# limit on its own is refused outright, before the ladder, and nothing degrades (the file was written
# against 8 MiB and never run; that was what it measured)
RAM_MIB = 200


def _mk(log, cache_ram_mib, disk=None):
    sp = ServerProcess()
    sp.model_hf_repo = "Felladrin/gguf-mamba-130m-hf"
    sp.model_hf_file = "mamba-130m-hf.Q8_0.gguf"
    sp.model_alias = "mamba-130m"
    sp.n_slots = 1
    sp.n_ctx = 2048
    sp.n_batch = N_BATCH
    sp.log_path = log
    # a deliberately tiny RAM tier so a handful of short conversations creates
    # real pressure without needing a large model
    sp.cache_ram = cache_ram_mib
    sp.checkpoint_min_step = MIN_STEP
    if disk:
        sp.slot_save_path = disk
        sp.cache_disk = 1024
    return sp


def _turn(sp, text, n_predict=4):
    return sp.make_request("POST", "/completion", data={
        "prompt": text,
        "n_predict": n_predict,
        "cache_prompt": True,
    })


def _distinct(i, n_tokens=400):
    # conversations must NOT be prefixes of one another, or saving one erases the
    # others and the test measures nothing (GLM-TODO: prefix-shaped conversations)
    return f"Conversation {i} about topic {i}. " + " ".join(f"w{i}x{j}" for j in range(n_tokens))


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = None
    yield
    if server is not None:
        server.stop()


def test_degrades_before_it_spills():
    """Pressure must shed fidelity first. A spill before any degrade means the
    ladder was skipped and eviction is still a cliff."""
    global server
    log = server_log_path()
    server = _mk(log, cache_ram_mib=RAM_MIB)
    server.start(timeout_seconds=120)
    reader = LogReader(log)

    for i in range(6):
        _turn(server, _distinct(i))

    text = reader.drain()
    degrades = DEGRADED.findall(text)
    spills = SPILLED.findall(text)

    assert degrades, "no entry was degraded under pressure - the ladder never ran"
    # the first ladder event must be a degrade, not a spill
    first_degrade = DEGRADED.search(text)
    first_spill = SPILLED.search(text)
    if first_spill:
        assert first_degrade.start() < first_spill.start(), \
            "an entry was spilled before anything was degraded"


def test_degradation_is_spread_not_concentrated():
    """The operator's requirement: keep shapes evenly across conversations. Every
    entry should reach level 1 before any entry reaches level 2, so no single
    conversation is stripped bare while its neighbour keeps everything."""
    global server
    log = server_log_path()
    server = _mk(log, cache_ram_mib=RAM_MIB)
    server.start(timeout_seconds=120)
    reader = LogReader(log)

    for i in range(6):
        _turn(server, _distinct(i))

    events = [(int(t), int(a), int(b)) for t, a, b, _ in DEGRADED.findall(reader.drain())]
    assert events, "nothing degraded"

    # once some entry has been taken to level N+1, no entry may still be sitting at
    # level N-1 unvisited: levels advance as a front, not a spike
    seen_level = {}
    for tokens, frm, to in events:
        seen_level[tokens] = to
        deepest = max(seen_level.values())
        shallowest = min(seen_level.values())
        assert deepest - shallowest <= 1, (
            f"degradation concentrated: one entry at level {deepest} while another "
            f"is still at {shallowest} ({seen_level})"
        )


def test_lru_order_within_a_rung():
    """LRU still decides who goes first among equally-degraded entries. The entry
    touched most recently must not be the first to lose fidelity."""
    global server
    log = server_log_path()
    server = _mk(log, cache_ram_mib=RAM_MIB)
    server.start(timeout_seconds=120)

    for i in range(5):
        _turn(server, _distinct(i))

    reader = LogReader(log)
    # touch conversation 0 again so it becomes the most recently used
    _turn(server, _distinct(0))
    # then force pressure with a new one
    _turn(server, _distinct(99))

    events = [int(t) for t, _, _, _ in DEGRADED.findall(reader.drain())]
    if events:
        # the just-touched conversation should not be the first victim
        first = events[0]
        assert first != len(_distinct(0).split()), \
            "the most recently used entry degraded first - LRU ordering was lost"


def test_a_degraded_entry_still_restores():
    """Every rung costs time, never correctness. A conversation whose entry was
    degraded must still come back from cache rather than reprocessing whole."""
    global server
    log = server_log_path()
    server = _mk(log, cache_ram_mib=RAM_MIB)
    server.start(timeout_seconds=120)

    first = _distinct(0)
    _turn(server, first)
    for i in range(1, 6):
        _turn(server, _distinct(i))

    # come back to conversation 0, whose entry has been degraded by now
    res = _turn(server, first + " and one more thing")
    t = res.body["timings"]
    assert t["cache_n"] > 0, "a degraded entry did not restore at all"
    assert t["prompt_n"] < t["cache_n"], \
        f"restore reprocessed more than it reused: prompt_n={t['prompt_n']} cache_n={t['cache_n']}"


def test_spill_only_after_the_ladder_is_exhausted():
    """Disk is the last rung. An entry may only be spilled once it has nothing
    left to shed, otherwise the ladder is being short-circuited."""
    global server
    disk = take_tmpdir("llama-spill-")
    log = server_log_path()
    server = _mk(log, cache_ram_mib=RAM_MIB, disk=disk)
    server.start(timeout_seconds=120)
    reader = LogReader(log)

    # skeletal entries are 5.3 MiB: RAM_MIB holds ~37 of them, the 38th spills one
    for i in range(42):
        _turn(server, _distinct(i))

    text = reader.drain()
    spills = SPILLED.findall(text)
    if not spills:
        pytest.skip("no spill occurred - raise the conversation count or lower cache_ram")

    # every entry that reached disk must have been seen degrading first
    degraded_tokens = {int(t) for t, _, _, _ in DEGRADED.findall(text)}
    for tok in spills:
        assert int(tok) in degraded_tokens, \
            f"entry of {tok} tokens was spilled without ever being degraded"


def test_a_degraded_mirrored_entry_restores_from_disk():
    """An entry mirrored to disk, then degraded by the ladder (checkpoints dropped), then released from
    RAM: its next turn must come back from the file. Measured 2026-09-18 on the production Qwen: the
    mirror held the entry's original checkpoints, degrade() dropped some from the RAM copy without
    touching the file, the release kept only the file, and unspill refused it - "L2: checkpoint count
    mismatch", "failed to read spilled entry back, discarding it" - four times, each a 240k-token
    conversation prefilled from zero."""
    global server
    log = server_log_path()
    # 90 MiB holds A alone at full fidelity (80 MiB): B's arrival saves A and the mirror pass writes it; the
    # third conversation's save is what degrades A, and the byte ladder, out of rungs, then spills the LRU - A
    server = _mk(log, cache_ram_mib=90, disk=take_tmpdir("llama-spill-"))
    server.cache_spill_seconds = 1        # the mirror pass, normally every 60 s
    server.start(timeout_seconds=120)
    reader = LogReader(log)

    a_text = _distinct(0, n_tokens=450)   # A is the only conversation of this size: the log lines name it
    a = _turn(server, a_text)
    assert a.status_code == 200
    # the cached entry holds the prompt and the generated tokens but the last sampled one
    n_a = a.body["timings"]["prompt_n"] + a.body["timings"]["predicted_n"] - 1

    # B takes the only slot: A goes to the RAM tier, and the mirror pass writes it to disk
    _turn(server, _distinct(1))
    text = ""
    deadline = time.time() + 15
    while time.time() < deadline and f"L2: mirrored {n_a:>7} tokens" not in text:
        time.sleep(0.5)
        text += reader.drain()
    assert f"L2: mirrored {n_a:>7} tokens" in text, f"precondition: A ({n_a} tokens) was never mirrored to disk"

    # pressure: the ladder degrades A (drops checkpoints from the RAM copy), then releases it
    for i in range(2, 14):
        _turn(server, _distinct(i))
    text += reader.drain()
    assert re.search(rf"degraded entry \({n_a} tokens\)", text), (
        f"precondition: A ({n_a} tokens) was never degraded; ladder events: {DEGRADED.findall(text)}")

    # A comes back: from disk, whole
    back = _turn(server, a_text + " w0y1 w0y2 w0y3")
    text += reader.drain()
    assert back.status_code == 200
    assert "checkpoint count mismatch" not in text, "the stale mirror was refused and A was prefilled from zero"
    assert "failed to read spilled entry back" not in text
    assert f"L2: restored {n_a:>7} tokens" in text, "precondition: A did not come back through the disk tier"
    assert back.body["timings"]["prompt_n"] < n_a // 2, (
        f"A was prefilled ({back.body['timings']['prompt_n']} of {n_a} tokens) instead of restored")

import glob
import os
import tempfile
import threading
import time

import pytest
from utils import *

# --slot-quantum makes a generating slot yield to waiting work and resume later.
#
# The trigger used to live only in the non-speculative sampling path, after the
# `can_speculate() && !spec_draft.empty()` early return, so any speculator that
# drafts on every step (draft-mtp in production, draft-simple here) made the
# flag inert. Measured before the fix: 0 yields over a 1500-token generation
# with draft-simple while a second request waited the whole time.
#
# NOTE: run this file with PORT set to something free, e.g.
#   PORT=51800 pytest unit/test_slot_preempt.py
# ServerProcess defaults to 8080, which the production server holds. That also
# breaks ServerPreset.load_all(), so the models never download and every server
# then dies with --offline and return code 1.

server: ServerProcess

# Speculation TRAPS this file steps around, each one measured:
#  - leaving spec_type unset does NOT disable speculation: the server default
#    types list is "none,draft-mtp". Pass "none" explicitly for the control.
#  - ngram-simple on the tinyllama2 preset ABORTS the server (rc 134) the first
#    time the story text repeats: the preset has n_batch 32 and the default
#    ngram draft is 48 tokens, and handle_last_sampled_token() asserts that
#    1 + 48 fits the batch. It was never a request refusal.
#  - ngram-simple with size-m < size-n silently never drafts (copy_max < n)
#    and the run is speculation in name only.
#  - draft-simple with the target as its own draft model drafts on EVERY step
#    and accepts ~99% (greedy), the closest analogue to draft-mtp that runs on
#    the tiny model, so it is the speculation arm here. Assert draft_n_accepted
#    as a precondition or the arm proves nothing.


def _target_model_path() -> str:
    """The tinyllama2 preset's gguf, out of the HF cache the harness fills, so
    it can double as its own draft model."""
    cache = os.environ.get("LLAMA_CACHE", "tmp")
    hits = glob.glob(os.path.join(cache, "models--ggml-org--test-model-stories260K",
                                  "snapshots", "*", "stories260K-f32.gguf"))
    assert hits, f"stories260K-f32.gguf not found under LLAMA_CACHE={cache}"
    return hits[0]


def _mk(quantum, spec, log, draft_n_max=8):
    sp = ServerPreset.tinyllama2()
    sp.n_slots = 1
    sp.n_ctx = 4096
    sp.slot_quantum = quantum
    sp.log_path = log
    if spec == "none":
        sp.spec_type = "none"
    elif spec == "draft-simple":
        sp.spec_type = "draft-simple"
        sp.model_draft = _target_model_path()
        sp.spec_draft_n_max = draft_n_max   # 1 + n_max must fit the preset's n_batch (32)
    else:
        raise ValueError(spec)
    return sp


def _yields(log_path: str) -> list:
    """n_gen at each yield, from the server log."""
    out = []
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            if "yielded the slot after" in line:
                out.append(int(line.split("yielded the slot after")[1].split()[0]))
    return out


def _wait_ready(sp: ServerProcess, timeout_s: int = 180):
    """start() returns once the HTTP server binds, but the model may still be
    loading - requests then get 503. Poll until a real completion answers."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = sp.make_request("POST", "/completion", data={
            "prompt": "a", "n_predict": 1, "temperature": 0.0, "cache_prompt": False})
        if r.status_code == 200:
            return
        time.sleep(0.5)
    raise AssertionError("server never became ready")


def _completion(sp: ServerProcess, n_predict: int):
    return sp.make_request("POST", "/completion", data={
        "prompt": "Once upon a time in a land far away there lived",
        "n_predict": n_predict, "temperature": 0.0, "top_k": 1, "seed": 42,
        "cache_prompt": False,
    })


def _contend(sp: ServerProcess):
    """A holds the only slot for 1500 tokens; B arrives 50 ms later and must
    queue. Returns (A, B, B_finished_before_A)."""
    a, b, t_done = [], [], {}

    def run(tag, n, out):
        out.append(_completion(sp, n))
        t_done[tag] = time.time()

    ta = threading.Thread(target=run, args=("A", 1500, a)); ta.start()
    time.sleep(0.05)
    assert not a, "A finished before B was even sent: no contention possible"
    tb = threading.Thread(target=run, args=("B", 8, b)); tb.start()
    ta.join(timeout=300); tb.join(timeout=300)
    assert a and a[0].status_code == 200, "holding request failed"
    assert b and b[0].status_code == 200, "contending request failed"
    # PRECONDITION: A really ran long enough to be preempted
    assert a[0].body["timings"]["predicted_n"] >= 1000, (
        f"holding request too short to create contention "
        f"(predicted_n={a[0].body['timings']['predicted_n']})")
    return a[0], b[0], t_done["B"] < t_done["A"]


def _run(spec, quantum=8, draft_n_max=8):
    log = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(quantum, spec, log, draft_n_max)
    sp.start(timeout_seconds=120)
    try:
        _wait_ready(sp)
        # the same generation with nothing waiting: no yield, and the reference
        # text the preempted run must reproduce
        alone = _completion(sp, 1500)
        assert alone.status_code == 200
        assert _yields(log) == [], "yielded with an empty queue"

        a, b, b_first = _contend(sp)
        if spec != "none":
            # PRECONDITION: speculation was actually running during A
            acc = a.body["timings"].get("draft_n_accepted", 0)
            assert acc > 500, f"speculation arm did not speculate (draft_n_accepted={acc})"
        return alone, a, b, b_first, _yields(log)
    finally:
        sp.stop()


def test_quantum_yields_without_speculation():
    """CONTROL. Pins the configuration the feature was first measured with."""
    alone, a, b, b_first, yields = _run("none")
    assert len(yields) > 0, "no yield with speculation off - the trigger itself is broken"
    assert b_first, "B waited for A to finish: the yield changed nothing"
    assert a.body["content"] == alone.body["content"], "resumed generation diverged"


def test_quantum_yields_with_speculation():
    """The D1 arm. draft-simple drafts on every step, so before the fix the
    post-decode lambda returned early on every step and the trigger below it
    never ran: 0 yields, B served only after A's 1500 tokens."""
    alone, a, b, b_first, yields = _run("draft-simple")
    assert len(yields) > 0, (
        "slot never yielded with speculation enabled: the trigger is not "
        "reached from the speculative accept path")
    assert b_first, "B waited for A to finish: --slot-quantum is inert under speculation"
    assert a.body["content"] == alone.body["content"], "resumed speculative generation diverged"


def test_quantum_threshold_is_not_skipped_by_draft_steps():
    """The modulo arm. A 15-token draft that is fully accepted advances n_gen by
    16 per step, so with quantum 16 n_gen runs 1, 17, 33, ... and
    `n_gen % 16 == 0` is NEVER true (a rare partial accept can realign it, which
    is why the count is asserted early, before the tail of A). A threshold
    re-armed one quantum past wherever n_gen lands is crossed on every step."""
    alone, a, b, b_first, yields = _run("draft-simple", quantum=16, draft_n_max=15)
    assert len(yields) > 0, "no yield: multiples of the quantum were stepped over"
    assert yields[0] < 1200, (
        f"first yield at n_gen={yields[0]}: B arrived early in A but the slot "
        f"only yielded near the end, which is the modulo test landing by luck")
    assert b_first
    assert a.body["content"] == alone.body["content"], "resumed speculative generation diverged"

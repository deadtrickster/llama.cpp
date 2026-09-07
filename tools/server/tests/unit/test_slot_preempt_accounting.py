import os
import tempfile
import threading
import time

import pytest
from utils import *

# Accounting around --slot-quantum suspend/resume: what the server REPORTS about
# a yielded generation, as opposed to whether the yield itself happens (that is
# test_slot_preempt.py's job).
#
# Two facts pinned here:
#   T1.7  a yielded generation is counted once in /metrics, not once per yield
#         plus once at the end
#   T1.8  the shutdown flush does not warn about a slot whose prompt is
#         already in the cache - that is the normal case, not a refusal
#
# NOTE: run with PORT set to something free, e.g.
#   PORT=51830 pytest unit/test_slot_preempt_accounting.py
# ServerProcess defaults to 8080, which the production server holds.

server: ServerProcess


def _mk(log, quantum=None, spec_type="none", n_slots=1, metrics=False, backend_sampling=False):
    sp = ServerPreset.tinyllama2()
    sp.n_slots = n_slots
    sp.backend_sampling = backend_sampling
    sp.n_ctx = 4096
    sp.slot_quantum = quantum
    # "none" explicitly: the server default types list is "none,draft-mtp"
    # and drafts still fire when --spec-type is left unset, which stops the
    # quantum trigger from ever firing
    sp.spec_type = spec_type
    sp.server_metrics = metrics
    sp.log_path = log
    return sp


def _log_count(log_path: str, needle: str) -> int:
    with open(log_path, "r", errors="replace") as f:
        return f.read().count(needle)


def _wait_ready(sp: ServerProcess, timeout_s: int = 180):
    """start() returns once the HTTP server binds; the model may still be
    loading. Poll until a real completion answers 200."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = sp.make_request("POST", "/completion", data={
            "prompt": "a", "n_predict": 1, "temperature": 0.0, "cache_prompt": False})
        if r.status_code == 200:
            return
        time.sleep(0.5)
    raise AssertionError("server never became ready")


def _busy(sp: ServerProcess, n_predict: int, out: list):
    out.append(sp.make_request("POST", "/completion", data={
        "prompt": "Once upon a time in a land far away there lived",
        "n_predict": n_predict, "temperature": 0.0, "top_k": 1, "seed": 42,
        "cache_prompt": False,
    }))


def _metric(sp: ServerProcess, name: str) -> float:
    res = sp.make_request("GET", "/metrics")
    assert res.status_code == 200
    assert isinstance(res.body, str)
    for line in res.body.splitlines():
        if line.startswith(name + " "):
            return float(line.split(" ", 1)[1])
    raise AssertionError(f"{name} not in /metrics")


@pytest.mark.parametrize("backend_sampling", [False, True])
def test_yielded_generation_is_counted_once_in_metrics(backend_sampling):
    """T1.7. suspend() ran callback_on_reset -> metrics_on_prediction with the
    slot's partial stats, resume() restored those same stats, and the final
    release counted them again. tokens_predicted_total therefore exceeded what
    the clients were actually sent by n_decoded_at_suspend per yield.

    The backend_sampling=True arm is the same measurement with the sampler
    bound in the backend. It also drives the resume()-side rebind (T1.9)
    through a real yield; whether the resumed slot samples on the backend or
    fell back to the CPU is NOT observable from outside - the output is the
    same either way - so that arm proves the path runs, not which sampler it
    used."""
    log = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log, quantum=8, n_slots=1, metrics=True, backend_sampling=backend_sampling)
    sp.start(timeout_seconds=120)
    try:
        _wait_ready(sp)
        before = _metric(sp, "llamacpp:tokens_predicted_total")

        a, b = [], []
        ta = threading.Thread(target=_busy, args=(sp, 1500, a)); ta.start()
        time.sleep(0.05)                      # A holds the only slot; B must queue
        tb = threading.Thread(target=_busy, args=(sp, 8, b)); tb.start()
        ta.join(timeout=300); tb.join(timeout=300)

        assert a and a[0].status_code == 200, "holding request failed"
        assert b and b[0].status_code == 200, "contending request failed"
        assert a[0].body["timings"]["predicted_n"] >= 1000, (
            "the holding request finished too early to create contention "
            f"(predicted_n={a[0].body['timings']['predicted_n']})")
        # PRECONDITION: a yield actually happened, otherwise there is nothing
        # to double-count and equality below would prove nothing
        n_yields = _log_count(log, "yielded the slot")
        assert n_yields > 0, "precondition: no yield happened"

        after = _metric(sp, "llamacpp:tokens_predicted_total")
        sent = a[0].body["timings"]["predicted_n"] + b[0].body["timings"]["predicted_n"]
        counted = after - before
        assert counted == sent, (
            f"/metrics counted {counted:.0f} predicted tokens but the clients received "
            f"{sent} ({n_yields} yield(s)): the partial stats at suspend were "
            "published and then counted again at the final release"
        )
    finally:
        sp.stop()


def test_flush_does_not_warn_about_an_already_cached_slot():
    """T1.8. prompt_save() returns false both when the cache refuses the state
    for size AND when the prompt is already in the cache. flush_prompt_cache()
    warned on any false, so the normal case - an idle slot whose conversation
    the idle-slot purge already copied into the cache - was reported as a
    refusal on every shutdown."""
    log = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log, n_slots=1)
    sp.start(timeout_seconds=120)
    try:
        _wait_ready(sp)
        prompt = "A quiet harbour town where every morning the fishing boats returned"
        req = {"prompt": prompt, "n_predict": 8, "temperature": 0.0, "top_k": 1,
               "seed": 7, "cache_prompt": True}
        r1 = sp.make_request("POST", "/completion", data=req)
        assert r1.status_code == 200
        # the second identical request makes the idle-slot purge copy the slot's
        # state into the cache before reuse; greedy sampling regenerates the same
        # tokens, so afterwards the slot holds exactly what the cache holds
        r2 = sp.make_request("POST", "/completion", data=req)
        assert r2.status_code == 200
        assert r2.body["timings"]["cache_n"] > 0, "precondition: prompt was not cached"
        assert r2.body["content"] == r1.body["content"], "precondition: generations differ"
    finally:
        sp.stop()   # SIGTERM -> clean_up -> flush_prompt_cache()

    assert _log_count(log, "flush:") > 0, "precondition: the shutdown flush never ran"
    assert _log_count(log, "prompt_save() refused it") == 0, (
        "flush warned about a slot whose prompt was already in the cache; "
        "that is the normal case, not a size refusal"
    )

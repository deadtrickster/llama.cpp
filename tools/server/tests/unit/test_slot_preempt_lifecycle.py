import glob
import http.client
import json
import os
import re
import socket
import tempfile
import threading
import time

import pytest
from utils import *

# Lifecycle of queue_suspended - the deque a yielded generation lives in while
# another task holds its slot. Three seams where the entry used to be invisible
# because the code walked `slots` only:
#
#   T1.6  a CANCEL for a suspended task found nothing, so a client that hung up
#         while suspended still got its full generation into a dead socket
#   T1.5  the shutdown flush saved the slots into the prompt cache and let the
#         suspended entries die with the process
#   T1.4  a failed resume() dropped the entry: no error to the client, and the
#         slot's KV cells were already gone (state_read_meta does seq_rm before
#         it reads) while the slot's prompt still claimed them
#
# stories260K generates ~4000 tok/s on this box (~800 with -t 1), so nothing
# here is timed by sleeping: contention is built from token counts (n_predict,
# prompt length, --batch-size) and the yield is asserted from the log before
# anything else. The slot context is capped at the model's 2048-token training
# context whatever --ctx-size says.
#
# NOTE: run with PORT set to something free, e.g.
#   PORT=51860 pytest unit/test_slot_preempt_lifecycle.py
# ServerProcess defaults to 8080, which the production server holds.

PROMPT_A = "Once upon a time in a land far away there lived a brave knight who"
PROMPT_B = "In a small village by the sea an old fisherman mended his nets every morning"

# the server checks a waiting client's socket once per second (HTTP_POLLING_SECONDS)
DISCONNECT_POLL_S = 1.0


def _mk(log, quantum, n_slots=1, n_ctx=2048, n_threads=None):
    sp = ServerPreset.tinyllama2()
    sp.n_slots = n_slots
    sp.n_ctx = n_ctx
    sp.n_threads = n_threads
    sp.n_predict = None           # the preset's --n-predict 64 is not a cap we want
    sp.server_slots = True
    sp.slot_quantum = quantum
    # "none" explicitly: the server default types list is "none,draft-mtp"
    # and drafts still fire when --spec-type is left unset, which stops the
    # quantum trigger from ever firing
    sp.spec_type = "none"
    sp.log_path = log
    return sp


def _log_count(log_path: str, needle: str) -> int:
    with open(log_path, "r", errors="replace") as f:
        return f.read().count(needle)


def _wait_log(log_path: str, needle: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _log_count(log_path, needle) > 0:
            return True
        time.sleep(0.02)
    return False


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


def _completion(prompt, n_predict: int, **extra) -> dict:
    data = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
            "seed": 42, "cache_prompt": False}
    data.update(extra)
    return data


def _busy(sp: ServerProcess, prompt, n_predict: int, out: list, **extra):
    try:
        out.append(sp.make_request("POST", "/completion", data=_completion(prompt, n_predict, **extra),
                                   timeout=120))
    except Exception as e:
        out.append(e)


def _raw_post(sp: ServerProcess, data: dict, timeout: float) -> http.client.HTTPConnection:
    """A completion over a bare socket, so the test can hang up on it or bound
    how long it waits for an answer."""
    conn = http.client.HTTPConnection(sp.server_host, sp.server_port, timeout=timeout)
    conn.request("POST", "/completion", body=json.dumps(data),
                 headers={"Content-Type": "application/json"})
    return conn


def _tokens(sp: ServerProcess, text: str) -> list:
    r = sp.make_request("POST", "/tokenize", data={"content": text})
    assert r.status_code == 200
    return r.body["tokens"]


def _slots_idle(sp: ServerProcess) -> bool:
    r = sp.make_request("GET", "/slots")
    assert r.status_code == 200, r.body
    return not any(s["is_processing"] for s in r.body)


def test_cancel_reaches_a_suspended_task():
    """T1.6. A holds the only slot, B queues, A yields. A's client hangs up
    while A is suspended. The CANCEL handler searched `slots`, so A stayed in
    queue_suspended, resumed when B finished, and generated ~1500 tokens into
    a closed socket while holding the slot (and its KV) the whole time.

    -t 1 so that B (a full 2048-token context, ~2.4 s) outlives the server's
    1 s socket poll: the CANCEL has to land while A is still suspended."""
    log = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log, quantum=8, n_slots=1, n_threads=1)
    sp.start(timeout_seconds=120)
    try:
        _wait_ready(sp)

        # A: over a bare socket so it can be closed mid-flight
        conn = _raw_post(sp, _completion(PROMPT_A, 1500), timeout=120)
        time.sleep(0.05)                       # A is generating; B must queue behind it
        b = []
        tb = threading.Thread(target=_busy, args=(sp, PROMPT_B, 3000, b)); tb.start()

        # PRECONDITION: the yield happened, so A is in queue_suspended
        assert _wait_log(log, "yielded the slot", 10), "precondition: no yield happened"

        # A's client goes away. The server sees the closed socket on its next
        # result poll and posts a CANCEL for A's task id.
        conn.close()
        assert _wait_log(log, "cancel task, id_task", 3 * DISCONNECT_POLL_S), \
            "precondition: the disconnect never turned into a CANCEL"
        assert tb.is_alive(), \
            "precondition: B finished before the CANCEL arrived, so A was already resumed"

        tb.join(timeout=120)
        assert b and not isinstance(b[0], Exception) and b[0].status_code == 200, f"B failed: {b}"

        # a wrongly-surviving A resumes the moment B releases the slot; give it
        # a moment to show up so an absent log line means absent, not early
        time.sleep(0.5)
        assert _log_count(log, "resumed at") == 0, \
            "the cancelled task resumed into a dead connection after B finished"
        assert _slots_idle(sp), "a slot is still processing after every live client is done"
    finally:
        sp.stop()

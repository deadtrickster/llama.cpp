import os
import tempfile
import time

import pytest
from utils import *

# --seq-max N decouples the context's sequence capacity (llama n_seq_max) from
# the number of server slots (--parallel). Slots still take ids [0, n_parallel);
# the ids above are a residency ceiling that nothing hands out yet (T2.1 of the
# adaptive-parallel work). These tests pin down what the flag does TODAY:
#   - unset: n_seq_max == n_parallel, unchanged behaviour
#   - N > n_parallel: the server starts, serves, and the context reports N
#   - N < n_parallel: refused at startup
#   - without --kv-unified the KV cache is split into N streams, so the
#     per-slot context shrinks to n_ctx / N. That is a real cost of the flag.
#
# NOTE: run this file with PORT set to something free, e.g.
#   PORT=51885 pytest unit/test_seq_max.py
# ServerProcess defaults to 8080, which the production server holds.


def _mk(log_path: str) -> ServerProcess:
    sp = ServerPreset.tinyllama2()
    sp.n_slots = 2
    sp.n_ctx = 1024  # a multiple of 4 * the KV pad (256), so n_ctx / n_seq_max is exact below
    sp.n_predict = 8
    sp.temperature = 0.0
    sp.log_path = log_path
    return sp


def _wait_ready(sp: ServerProcess, timeout_s: int = 120):
    """start() returns once the HTTP server binds, but the model may still be
    loading. Poll until the server actually answers a completion."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = sp.make_request("POST", "/completion", data={
            "prompt": "a", "n_predict": 1, "temperature": 0.0, "cache_prompt": False})
        if r.status_code == 200:
            return
        time.sleep(0.5)
    raise AssertionError("server never became ready")


def _log(path: str) -> str:
    with open(path, errors="replace") as f:
        return f.read()


def test_seq_max_unset_equals_parallel():
    log_path = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log_path)
    sp.start(timeout_seconds=120)
    try:
        _wait_ready(sp)
    finally:
        sp.stop()

    # the server prints llama_n_seq_max(ctx_tgt): what the context was really allocated with
    assert "initializing, n_slots = 2, n_seq_max = 2," in _log(log_path)


def test_seq_max_above_parallel_starts_and_serves():
    log_path = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log_path)
    sp.seq_max = 4
    sp.kv_unified = True  # one KV stream, so the extra ids cost no per-slot context
    sp.start(timeout_seconds=120)
    try:
        _wait_ready(sp)

        res = sp.make_request("POST", "/completion", data={
            "prompt": "Once upon a time",
            "n_predict": 8,
            "temperature": 0.0,
        })
        assert res.status_code == 200, res.body
        assert len(res.body["content"]) > 0

        # both slots still work, and there are still exactly two of them
        for id_slot in (0, 1):
            res = sp.make_request("POST", "/completion", data={
                "prompt": "Once upon a time", "n_predict": 4, "temperature": 0.0, "id_slot": id_slot})
            assert res.status_code == 200, res.body
            assert res.body["id_slot"] == id_slot

        props = sp.make_request("GET", "/props")
        assert props.status_code == 200
        assert props.body["total_slots"] == 2
        # unified: the whole pool per slot - and [pool] the pool grows for a conversation, so a slot's context is
        # the model's (tinyllama2: 2048), not the 1024 cells -c starts the pool at
        assert props.body["default_generation_settings"]["n_ctx"] == 2048
    finally:
        sp.stop()

    # the server prints llama_n_seq_max(ctx_tgt): what the context was really allocated with
    assert "initializing, n_slots = 2, n_seq_max = 4," in _log(log_path)


def test_seq_max_below_parallel_is_refused():
    log_path = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log_path)
    sp.seq_max = 1
    with pytest.raises(RuntimeError, match="Server process died"):
        sp.start(timeout_seconds=60)

    log = _log(log_path)
    assert "--seq-max (1) must be >= --parallel (2)" in log


def test_seq_max_without_kv_unified_splits_the_context():
    """Without --kv-unified the attention cache is one stream per sequence id,
    so n_ctx is divided by n_seq_max, not by n_parallel: raising --seq-max
    shrinks every slot's context. Recorded here so the cost is visible."""
    log_path = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log_path)
    sp.seq_max = 4
    sp.start(timeout_seconds=120)
    try:
        _wait_ready(sp)
        props = sp.make_request("GET", "/props")
        assert props.status_code == 200
        assert props.body["total_slots"] == 2
        assert props.body["default_generation_settings"]["n_ctx"] == 1024 // 4
    finally:
        sp.stop()

import os
import tempfile
import threading
import time

import pytest
from utils import *

# --kv-unified puts every slot on ONE fixed pool and entitles each of them to
# all of it (n_ctx_seq == n_ctx). N slots can therefore hold more than the pool
# between them, and the pool has no eviction of its own: when llama_decode()
# cannot place a batch it returns 1 and the server has to make room itself.
#
# Production runs -c 327680 --parallel 2 --kv-unified-per-slot 262144, i.e. 1.6x
# oversubscribed. These tests reproduce the same shape at a fraction of the
# scale and are about what happens to the CONVERSATIONS when the pool is full.
#
# NOTE: run this file with PORT set to something free, e.g.
#   PORT=51810 pytest unit/test_kv_pool_oversubscribe.py
# ServerProcess defaults to 8080, which the production server holds.

POOL = 1024

SENTENCE = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
)


class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0

    def drain(self):
        with open(self.path, errors="replace") as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


def _mk(log_path: str, n_ctx: int = POOL, n_predict: int = 4) -> ServerProcess:
    sp = ServerPreset.tinyllama2()
    sp.n_slots = 2
    sp.n_ctx = n_ctx
    sp.n_batch = 256
    sp.kv_unified = True
    sp.n_predict = n_predict
    sp.temperature = 0.0
    sp.cache_ram = 100
    sp.log_path = log_path
    return sp


def _wait_ready(sp: ServerProcess, timeout_s: int = 180):
    """start() returns once the HTTP server binds, but the model may still be
    loading - requests then get 503 {"message": "Loading model"}. Poll until the
    server actually answers a completion."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = sp.make_request("POST", "/completion", data={
            "prompt": "a", "n_predict": 1, "temperature": 0.0, "cache_prompt": False})
        if r.status_code == 200:
            return
        time.sleep(0.5)
    raise AssertionError("server never became ready")


def _prompt_of(sp: ServerProcess, n_tokens: int) -> str:
    """A prompt of roughly n_tokens tokens, built from SENTENCE."""
    r = sp.make_request("POST", "/tokenize", data={"content": SENTENCE})
    assert r.status_code == 200
    per = len(r.body["tokens"])
    assert per > 0
    return SENTENCE * (n_tokens // per + 1)


def _complete(sp: ServerProcess, prompt: str, id_slot: int, n_predict: int = 4) -> dict:
    res = sp.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": id_slot,
        "n_predict": n_predict,
        "temperature": 0.0,
        "cache_prompt": True,
    })
    assert res.status_code == 200, res.body
    return res.body["timings"]


def test_pressure_purge_spills_the_idle_conversation():
    """T0.1. Slot 0 holds a finished conversation A. A prompt B on slot 1 that
    does not fit next to it makes llama_decode() fail, and the server purges the
    idle slot to make room (try_clear_idle_slots). The purged conversation must
    land in the prompt cache, not vanish: sending A again afterwards has to be
    served from the cache.

    --no-cache-idle-slots is set so the launch-time save/clear in
    get_available_slot() cannot save A first and mask the purge path."""
    log_path = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log_path)
    sp.no_cache_idle_slots = True
    sp.start(timeout_seconds=120)
    try:
        _wait_ready(sp)
        log = LogReader(log_path)
        log.drain()

        prompt_a = _prompt_of(sp, 300)
        prompt_b = _prompt_of(sp, 800)

        t_a = _complete(sp, prompt_a, id_slot=0)
        n_a = t_a["prompt_n"] + t_a["cache_n"]
        assert t_a["cache_n"] == 0

        t_b = _complete(sp, prompt_b, id_slot=1)
        n_b = t_b["prompt_n"] + t_b["cache_n"]

        # PRECONDITIONS: the two conversations do not fit together, and the
        # server actually purged slot 0 to make room for B. Without the purge
        # the test proves nothing.
        assert n_a + n_b > POOL, f"no pressure: {n_a} + {n_b} <= {POOL}"
        purge = log.drain()
        assert "purging slot 0 with" in purge, "slot 0 was never purged under pressure"

        # a tiny request on slot 1 pushes B out of the pool, so A has room to
        # come back. It does not touch slot 0.
        _complete(sp, "Hello", id_slot=1)

        t_a2 = _complete(sp, prompt_a, id_slot=0)
    finally:
        sp.stop()

    # A was idle, complete, and purged only for lack of room. That is a spill
    # to the prompt cache, not a deletion: the resend must hit the cache. The
    # final token is always reprocessed.
    assert t_a2["cache_n"] == n_a - 1, (
        f"the purged conversation is gone: cache_n={t_a2['cache_n']}, expected {n_a - 1}")
    assert t_a2["prompt_n"] == 1


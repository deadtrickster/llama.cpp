import os
import re
import tempfile
import threading

import pytest
from utils import *

# [pool] The KV cell count is no longer fixed by -c. Under --kv-unified the pool
# STARTS at -c and moves at runtime: it grows when a conversation needs cells
# the pool does not have (its prompt at admission, a batch that found no room,
# a suspended state coming back) and shrinks when a sequence id is needed and
# the device has no memory for it (the id trade cannot be driven on CPU, where
# the device reports no memory; that path is covered by the CUDA run in the
# report, not here). One constant remains: --pool-min-ctx cells plus one id's
# fixed cost are held back on every device, so one conversation can always
# make progress. --pool-static pins the pool for anyone who wants that.
#
# What this file proves on CPU, against a 512-cell start:
#   - a prompt larger than the pool is admitted and completes: the pool grew for it
#     (before: "input (N tokens) is larger than the max context size (512 tokens)")
#   - two conversations that together exceed the pool are both served
#   - --pool-static keeps the old refusal: -c is exactly what it was
#
# NOTE: run with PORT set to something free below 32768, e.g.
#   PORT=51920 pytest unit/test_pool_elastic.py

SENTENCE = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
)

START = 512


def _mk(log_path: str, n_slots: int = 1, pool_static: bool = False) -> ServerProcess:
    sp = ServerPreset.tinyllama2()
    sp.n_slots = n_slots
    sp.n_ctx = START            # where the pool starts; n_ctx_train of tinyllama2 is 2048
    sp.n_batch = 256
    sp.kv_unified = True
    sp.n_predict = 8
    sp.temperature = 0.0
    sp.cache_ram = 0
    sp.spec_type = "none"
    sp.pool_static = pool_static
    sp.log_path = log_path
    return sp


def _log(path: str) -> str:
    with open(path, errors="replace") as f:
        return f.read()


def _prompt_of(sp: ServerProcess, n_tokens: int) -> str:
    r = sp.make_request("POST", "/tokenize", data={"content": SENTENCE})
    assert r.status_code == 200
    per = len(r.body["tokens"])
    assert per > 0
    return SENTENCE * (n_tokens // per + 1)


def _complete(sp: ServerProcess, prompt: str):
    return sp.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": 8, "temperature": 0.0, "cache_prompt": True})


def _grow_lines(log: str):
    return re.findall(r"\[pool\] (\d+) -> (\d+) cells", log)


def test_prompt_larger_than_the_pool_grows_it():
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "server.log")
        sp = _mk(log)
        sp.start(timeout_seconds=180)
        try:
            assert "[pool] elastic: starts at 512 cells" in _log(log)

            prompt = _prompt_of(sp, 700)
            r = _complete(sp, prompt)
            assert r.status_code == 200, r.body
            assert r.body["tokens_evaluated"] >= 700

            grows = _grow_lines(_log(log))
            assert grows, "no [pool] resize line: the pool did not grow for a 700-token prompt"
            n_from, n_to = int(grows[0][0]), int(grows[0][1])
            assert n_from == START
            assert n_to >= 768, f"grew to {n_to}, which does not hold 700 tokens"
            assert n_to % 256 == 0
        finally:
            sp.stop()


def test_two_conversations_beyond_the_pool_are_both_served():
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "server.log")
        sp = _mk(log, n_slots=2)
        sp.start(timeout_seconds=180)
        try:
            prompt = _prompt_of(sp, 400)
            results = [None, None]

            def go(i):
                results[i] = _complete(sp, prompt + ("A" if i == 0 else "B"))

            ts = [threading.Thread(target=go, args=(i,)) for i in range(2)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()

            for i in range(2):
                assert results[i].status_code == 200, results[i].body
                assert results[i].body["tokens_evaluated"] >= 400

            text = _log(log)
            grows = _grow_lines(text)
            assert grows, "two 400-token conversations on a 512-cell pool: nothing grew"
            assert int(grows[-1][1]) >= 1024, f"pool ended at {grows[-1][1]} cells, which does not hold both"
            assert "is larger than the max context size" not in text
        finally:
            sp.stop()


def test_pool_static_keeps_the_old_refusal():
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "server.log")
        sp = _mk(log, pool_static=True)
        sp.start(timeout_seconds=180)
        try:
            assert "[pool] static: 512 cells" in _log(log)

            prompt = _prompt_of(sp, 700)
            r = _complete(sp, prompt)
            assert r.status_code == 400, r.body
            assert "context size" in str(r.body)
            assert not _grow_lines(_log(log))
        finally:
            sp.stop()

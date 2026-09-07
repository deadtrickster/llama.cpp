import os
import re
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


def _gen(sp: ServerProcess, id_slot: int, n_predict: int, out: list):
    try:
        out.append(sp.make_request("POST", "/completion", data={
            "prompt": "Once upon a time in a land far away there lived a brave knight who",
            "id_slot": id_slot,
            "n_predict": n_predict,
            "ignore_eos": True,
            "temperature": 0.0, "top_k": 1, "seed": 42,
            "cache_prompt": True,
        }))
    except Exception as e:                      # a died server shows up here
        out.append(e)


def test_pool_exhaustion_fails_one_sequence_and_spills_it():
    """T0.3, the ladder's terminal rung. Two generating slots on a 2048 pool,
    each asked for 1400 tokens. No slot is idle, so nothing can be purged; the
    batch-halving ladder bottoms out at n_batch == 1 and llama_decode() still
    cannot place a token. The server used to answer that by erroring,
    releasing and CLEARING every processing slot - both conversations gone,
    the innocent one included.

    Required: exactly one request fails, the other completes in full, and the
    failed one is spilled to the prompt cache rather than destroyed.

    Since T2.5 this shape is handled one rung earlier: the larger running
    sequence is suspended with a copy and both complete (the tests below). With
    two or more live sequences every exhaustion has such a move, so the
    terminal rung is only reachable by switching that one off -
    LLAMA_SERVER_POOL_NO_SUSPEND, a test hook - which is what keeps this test
    testing what it says."""
    log_path = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log_path, n_ctx=2048, n_predict=4096)
    sp.debug = True                              # the spill tag is a DBG line
    saved = dict(os.environ)
    os.environ["LLAMA_SERVER_POOL_NO_SUSPEND"] = "1"
    try:
        sp.start(timeout_seconds=120)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    try:
        _wait_ready(sp)
        log = LogReader(log_path)
        log.drain()

        a, b = [], []
        ta = threading.Thread(target=_gen, args=(sp, 0, 1400, a))
        tb = threading.Thread(target=_gen, args=(sp, 1, 1400, b))
        ta.start(); tb.start()
        ta.join(timeout=300); tb.join(timeout=300)

        assert sp.process is None or sp.process.poll() is None, "server died under pool exhaustion"
        for name, out in (("A", a), ("B", b)):
            assert out, f"{name}: no response"
            assert not isinstance(out[0], Exception), f"{name}: {out[0]!r}"

        text = log.drain()
    finally:
        sp.stop()

    # PRECONDITION: the pool really ran out. Without this line the two requests
    # simply fit and nothing below is being tested.
    assert "Context size has been exceeded" in text, "the pool was never exhausted"

    results = {"A": a[0], "B": b[0]}
    survivors = [n for n, r in results.items() if r.status_code == 200]
    failed    = [n for n, r in results.items() if r.status_code != 200]
    report = ", ".join(f"{n}: HTTP {r.status_code}" for n, r in results.items())

    # exactly one sequence could not be placed; the other one is innocent
    assert len(survivors) == 1, f"expected exactly one survivor, got {report}"
    assert len(failed) == 1

    survivor = results[survivors[0]]
    assert survivor.body["timings"]["predicted_n"] == 1400, (
        f"the surviving request was cut short: {survivor.body['timings']}")

    # the failed one was spilled, not destroyed
    assert "__TEST_TAG_POOL_EXHAUSTED_SPILL__" in text, (
        "the failed sequence was cleared without being saved to the prompt cache")


# T2.5 - the pressure ladder, checked per BATCH. Sequences grow after admission,
# so the pool can exhaust with two RUNNING sequences and nothing resident to
# evict (production: two seats, both deep). Rungs, in rising order of cost:
#   1  evict the LRU finished conversation      (to the prompt cache)
#   2  suspend the largest RUNNING generation   (state copied to RAM, restored
#                                                when there is room again)
#   3  fail ONE sequence and spill it (T0.3)    (only when 1 and 2 cannot help)
#
# The scenario below is the production shape at 1/160 scale: two generations
# that together need ~3100 cells on a 2048 pool.

TWO_DEEP_N_PREDICT = 1400
SUSPEND_LINE = re.compile(r"id\s+0 \| task \d+ \| KV pool full: suspending")


def _tokens(sp: ServerProcess, text: str) -> list:
    r = sp.make_request("POST", "/tokenize", data={"content": text})
    assert r.status_code == 200
    return r.body["tokens"]


def _gen_tokens(sp: ServerProcess, prompt, id_slot: int, n_predict: int, out: list, done: dict, name: str):
    try:
        out.append(sp.make_request("POST", "/completion", data={
            "prompt": prompt,
            "id_slot": id_slot,
            "n_predict": n_predict,
            "ignore_eos": True,
            "temperature": 0.0, "top_k": 1, "seed": 42,
            "cache_prompt": True,
        }, timeout=300))
    except Exception as e:                      # a died server shows up here
        out.append(e)
    done[name] = time.time()


def _two_deep(log_path: str, extra_b: int = 0, env: dict | None = None):
    """A on slot 0 with a ~300-token prompt, B on slot 1 with a ~20-token one,
    both generating TWO_DEEP_N_PREDICT tokens with EOS ignored on a 2048 pool.
    A is the larger sequence throughout.

    extra_b adds tokens to B's prompt. Whether the batch that hits the wall has
    placed A's token before it fails (n_batch halves down to 1, so the last free
    cells go to whichever slot was batched first - A) depends on the parity of
    the free count at that moment, and that is what decides which sequence is
    the only one still pending. Both parities must give the same answer.

    Returns (a, b, log text, b finished before a)."""
    sp = _mk(log_path, n_ctx=2048, n_predict=4096)
    sp.debug = True                              # the ladder's tags are DBG lines
    saved = dict(os.environ)
    if env:
        os.environ.update(env)
    try:
        sp.start(timeout_seconds=120)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    try:
        _wait_ready(sp)
        log = LogReader(log_path)
        log.drain()

        prompt_a = _prompt_of(sp, 300)
        toks_b = _tokens(sp, SENTENCE * 3)[:20 + extra_b]
        assert len(toks_b) == 20 + extra_b

        a, b, done = [], [], {}
        ta = threading.Thread(target=_gen_tokens, args=(sp, prompt_a, 0, TWO_DEEP_N_PREDICT, a, done, "A"))
        tb = threading.Thread(target=_gen_tokens, args=(sp, toks_b, 1, TWO_DEEP_N_PREDICT, b, done, "B"))
        ta.start(); tb.start()
        ta.join(timeout=300); tb.join(timeout=300)

        assert sp.process is None or sp.process.poll() is None, "server died under pool exhaustion"
        for name, out in (("A", a), ("B", b)):
            assert out, f"{name}: no response"
            assert not isinstance(out[0], Exception), f"{name}: {out[0]!r}"

        text = log.drain()
    finally:
        sp.stop()

    # PRECONDITION: the pool really ran out. Without this line the two requests
    # simply fit and nothing below is being tested.
    assert "Context size has been exceeded" in text, "the pool was never exhausted"

    return a[0], b[0], text, done["B"] < done["A"]


@pytest.mark.parametrize("extra_b", [0, 1])
def test_pool_exhaustion_suspends_the_largest_running_sequence(extra_b):
    """Rung 2. Two running sequences, nothing resident, the pool is full. The
    ladder must suspend the LARGEST running one (A, slot 0) with a copy and let
    the other keep decoding - not fail either of them. B, never interrupted,
    completes in full."""
    log_path = os.path.join(tempfile.mkdtemp(), "srv.log")
    a, b, text, _ = _two_deep(log_path, extra_b=extra_b)

    report = f"A: HTTP {a.status_code}, B: HTTP {b.status_code}"
    assert "__TEST_TAG_POOL_EXHAUSTED_SPILL__" not in text, (
        f"rung 3 fired: a sequence was failed although a running one could have been suspended ({report})")
    assert SUSPEND_LINE.search(text), (
        f"the largest running sequence (A, slot 0) was not the one suspended ({report})")
    assert "offloaded sequence" in text, "the suspended sequence was not copied out of the pool"

    assert b.status_code == 200, f"the sequence that was kept running failed: {report}"
    assert b.body["timings"]["predicted_n"] == TWO_DEEP_N_PREDICT, (
        f"the surviving request was cut short: {b.body['timings']}")


def test_pool_exhaustion_restores_the_suspended_sequence():
    """The restore side of the ladder. A was suspended for B; when B finishes,
    its finished conversation is all that stands between A's state and the
    room it needs. The resume must evict it (rung 1, to the prompt cache) and
    bring A back, so A completes in full too, after B: nobody errors."""
    log_path = os.path.join(tempfile.mkdtemp(), "srv.log")
    a, b, text, b_first = _two_deep(log_path)

    report = f"A: HTTP {a.status_code}, B: HTTP {b.status_code}"
    assert "offloaded sequence" in text, f"precondition: nothing was suspended ({report})"

    assert a.status_code == 200 and b.status_code == 200, f"a request errored under pool pressure: {report}"
    assert "restored task" in text, "the suspended state never came back"
    assert "resumed at" in text, "the suspended generation never resumed"
    for name, r in (("A", a), ("B", b)):
        assert r.body["timings"]["predicted_n"] == TWO_DEEP_N_PREDICT, (
            f"{name} was cut short: {r.body['timings']}")
    assert b_first, "A came back before B was done with the pool"

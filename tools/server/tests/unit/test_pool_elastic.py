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

START = 512      # -c: the ceiling the pool may grow to
FLOOR = 256      # where it starts: --pool-min-ctx, derived from -b 256 here


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
            assert f"[pool] elastic: starts at {FLOOR} cells" in _log(log)

            prompt = _prompt_of(sp, 700)
            r = _complete(sp, prompt)
            assert r.status_code == 200, r.body
            assert r.body["tokens_evaluated"] >= 700

            # the first resize is the start (-c down to the floor); the grows follow, a quarter at a time
            grows = [(int(a), int(b)) for a, b in _grow_lines(_log(log)) if int(b) > int(a)]
            assert grows, "no [pool] resize line: the pool did not grow for a 700-token prompt"
            assert grows[0][0] == FLOOR
            assert grows[-1][1] >= 768, f"grew to {grows[-1][1]}, which does not hold 700 tokens"
            assert all(n_to % 256 == 0 for _, n_to in grows)
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


# ---------------------------------------------------------------------------------------------------------
# [pool-restore] The other direction of the trade, and the conversation that has to come BACK.
#
# From the production log of 2026-09-07 night (GLM-5.3-Flash, glm-all @ 9de4a257f), verbatim:
#
#   pool_resize: [pool] 327680 -> 204544 cells (79333 held, 4 ids of 4): one more sequence id
#   prompt cache: save = 0, load = 1, ... 157 entries / 7584.7 MiB
#   E state_read_meta: failed to find 140667 available cells in kv cache
#   E state_seq_set_data: error loading state: failed to restore kv cache
#   W slot prompt_load: id 0 | task -1 | failed to load prompt from cache
#   launch_slot_: id 0 | task 14997        <- full 141k prefill, 8.4 minutes
#
# The pool shrank to buy a sequence id. One second later a cached conversation needing 140,667 cells tried
# to restore into ~125,000 free ones, failed - a restore is all-or-nothing - and was prefilled from scratch:
# 506 s at 278 t/s, past the client's 300 s timeout, so the client retried and the whole thing looped.
#
# Two defects, two tests:
#   (a) the shrink counted unheld cells as idle. They are the capacity a cached conversation needs to come
#       back. The floor a shrink leaves is now the larger of --pool-min-ctx and the largest entry the prompt
#       cache could hand back (its RAM tier: what --cache-ram says is hot), plus the batch margin.
#   (b) the restore was not a pool-grow hook. A cached conversation that finds too few cells now asks the
#       pool for them - grown from the device, idle ids given back, finished residents evicted - and is
#       prefilled only when none of that can be had.
#
# The id trade cannot be driven on CPU (the CPU buffer type has no device to say no, and a plain-KV model
# prices an id at nothing), so these run under LLAMA_SERVER_POOL_SELFTEST=<budget_mib>:<id_mib>, which
# prices the device-less buffer type as a device of that size with that much per id - inert on CUDA. The
# numbers below are worked from stories260K's 640 bytes per cell, padding of 256 and --pool-min-ctx 256
# (derived from -b 256). llama_n_ctx_cost prices a cell at 1,152 bytes here (640 of KV plus the compute
# buffer's share). The pool starts at the floor and grows with A to 2304 cells, 2.6 MiB; one id 1.25 MiB;
# a raise needs the id plus one more in reserve, 2.4 MiB, and the fake device of 5.75 MiB has 1.9 MiB free,
# so B's raise is refused, the shrink gives back the 364 idle cells (to 2048) and A is evicted for the id.
# C's raise is refused the same way, and the shrink then has A in the cache to leave room for: 2048 cells
# hold 1943 plus the margin and B, nothing is idle beyond that floor, and the seat goes to C when B is done.
#
# What is asserted is the LOG LINE (the state_read_meta error must not appear) and the TOKEN COUNTS
# (prompt_n 1, cache_n the conversation): a silent full prefill answers correctly, and a test that only
# checks the answer passes on the defect.
# ---------------------------------------------------------------------------------------------------------

import time

PROMPT_B = "In a small village by the sea an old fisherman mended his nets every morning"
PROMPT_C = "The mountain pass was closed by snow and the travellers waited in the inn"

POOL_R    = 3072
SELFTEST  = "5.75:1.25"
N_A       = 1900       # tokens in the cached conversation; n_ctx_train of tinyllama2 is 2048


def _mk_r(log_path: str, cache_dir: str, selftest: bool) -> ServerProcess:
    # LLAMA_TESTS_KEEP_LOGS=<dir>: keep the server logs there instead of the temp dir, to read after a run
    keep = os.environ.get("LLAMA_TESTS_KEEP_LOGS")
    if keep:
        log_path = os.path.join(keep, f"{os.getpid()}-{os.path.basename(os.path.dirname(log_path))}-{os.path.basename(log_path)}")
    sp = _mk(log_path)
    sp.n_ctx = POOL_R
    sp.n_threads = 1               # slow enough that B is still generating when C arrives
    sp.cache_ram = 100
    sp.slot_save_path = cache_dir  # the disk tier: test (b) hands the entry across a restart
    sp.debug = True
    # the fake device prices buffer types WITHOUT a device: on a CUDA build the cells would sit on a real
    # card and nothing here would bite, so these run with no card in sight, at the CPU buffer's price
    sp.n_gpu_layer = 0
    sp.env = {"CUDA_VISIBLE_DEVICES": ""}
    if selftest:
        sp.env["LLAMA_SERVER_POOL_SELFTEST"] = SELFTEST
    return sp


def _gen(sp: ServerProcess, prompt: str, n_predict: int):
    return sp.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "cache_prompt": True,
        "ignore_eos": True}, timeout=120)


def _shrink_lines(log: str):
    """(from, to, held) of every shrink the pool made for an id"""
    res = []
    for m in re.finditer(r"\[pool\] (\d+) -> (\d+) cells \((\d+) held, .*\): one more sequence id", log):
        n_from, n_to, held = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if n_to < n_from:
            res.append((n_from, n_to, held))
    return res


def _pressure(sp: ServerProcess):
    """B holds the only seat with a long generation and C arrives while it runs: C needs an id, the fake
    device cannot pay for it without cells, and the pool is asked to shrink - the trade the production
    log shows. Whoever is in the prompt cache at that moment is what the shrink has to leave room for."""
    res = {}

    def go(k, prompt, n):
        res[k] = _gen(sp, prompt, n)

    tb = threading.Thread(target=go, args=("B", PROMPT_B, 1000))
    tb.start()
    time.sleep(0.05)
    tc = threading.Thread(target=go, args=("C", PROMPT_C, 600))
    tc.start()
    for t in (tb, tc):
        t.join()
    for k in "BC":
        assert res[k].status_code == 200, (k, res[k].body)


def _cache_a_then_pressure(sp: ServerProcess, prompt_a: str):
    """A is served in full on the one id there is and stays resident. B arrives and needs that id: the raise
    is asked first - the pool gives back what is idle beyond A's HELD cells, the id still does not fit -
    and A is evicted into the prompt cache for it. Then the pressure phase, with A in the cache. No request
    may precede A here: a sequence left by one would hold the id and make A's arrival the raise.
    Returns A's token count."""
    r = _gen(sp, prompt_a, 4)
    assert r.status_code == 200, r.body
    n_a = r.body["tokens_evaluated"]
    assert n_a >= N_A

    _pressure(sp)

    return n_a


def _after_a_is_cached(log: str, n_a: int) -> str:
    """the log from the moment A entered the prompt cache. Before it A is resident and its cells are HELD,
    so a shrink there is judged against --pool-min-ctx, correctly; the floor is about entries in the cache"""
    for m in re.finditer(r"purging (?:slot|sequence) \d+ with (\d+) tokens \(saved to the prompt cache", log):
        if int(m.group(1)) >= n_a:
            return log[m.start():]
    raise AssertionError(f"the {n_a}-token conversation never entered the prompt cache: nothing is proven")


def _return_of_a(sp: ServerProcess, prompt_a: str):
    r = _gen(sp, prompt_a, 4)
    assert r.status_code == 200, r.body
    return r.body["timings"]


def test_shrink_keeps_room_for_the_largest_cached_conversation():
    """(a) A shrink for an id may not take the cells the largest entry in the prompt cache's RAM tier
    needs to come back. Every shrink line must leave held + that entry + the batch margin, and A must
    return from its cells."""
    with tempfile.TemporaryDirectory() as d:
        log = os.path.join(d, "server.log")
        cache_dir = os.path.join(d, "cache")
        os.mkdir(cache_dir)
        sp = _mk_r(log, cache_dir, selftest=True)
        sp.start(timeout_seconds=180)
        try:
            prompt_a = _prompt_of(sp, N_A)
            n_a = _cache_a_then_pressure(sp, prompt_a)

            text = _after_a_is_cached(_log(sp.log_path), n_a)
            shrinks = _shrink_lines(text)
            refused = "idle beyond the floor" in text
            assert shrinks or refused, \
                "the id trade never ran (no shrink, no refusal): the self-test hook did not bite, nothing is proven"
            for n_from, n_to, held in shrinks:
                assert n_to >= held + n_a + 2, \
                    f"[pool] {n_from} -> {n_to} with {held} held leaves no room for the {n_a}-token cached conversation"

            t = _return_of_a(sp, prompt_a)
            text = _log(sp.log_path)
            assert "available cells in kv cache" not in text, "state_read_meta could not place the cached conversation"
            assert "failed to load prompt from cache" not in text
            assert t["prompt_n"] == 1 and t["cache_n"] == n_a - 1, \
                f"A was prefilled again instead of restored: prompt_n={t['prompt_n']}, cache_n={t['cache_n']}"
        finally:
            sp.stop()


def test_cached_conversation_larger_than_the_pool_grows_it_back():
    """(b) The entry is on DISK - handed across a restart, which is how production's index looks - so the
    floor (RAM tier) does not cover it and the shrink goes below it: that is the incident. The restore must
    then grow the pool back rather than prefill 1,900 tokens."""
    with tempfile.TemporaryDirectory() as d:
        cache_dir = os.path.join(d, "cache")
        os.mkdir(cache_dir)

        log1 = os.path.join(d, "server1.log")
        sp = _mk_r(log1, cache_dir, selftest=False)
        sp.start(timeout_seconds=180)
        try:
            prompt_a = _prompt_of(sp, N_A)
            r = _gen(sp, prompt_a, 4)
            assert r.status_code == 200, r.body
            n_a = r.body["tokens_evaluated"]
            r = _gen(sp, PROMPT_B, 4)          # pushes A into the prompt cache
            assert r.status_code == 200, r.body
        finally:
            sp.stop()                          # a clean shutdown spills the cache to disk

        log2 = os.path.join(d, "server2.log")
        sp = _mk_r(log2, cache_dir, selftest=True)
        sp.start(timeout_seconds=180)
        try:
            _pressure(sp)              # A stays on disk: nothing here touches it before the shrink

            # the pool of the new server started at the floor and grew only for B and C: it is below the
            # entry on disk, which the floor (RAM tier) never covered. (Before the pool started minimal
            # the situation came from a shrink for an id; the fact is the same: too few cells for A.)
            text = _log(sp.log_path)
            sizes = [int(b) for _, b in _grow_lines(text)]
            assert sizes and sizes[-1] < n_a + 2, \
                f"the pool holds the {n_a}-token entry already ({sizes}): the situation was not created, nothing is proven"

            t = _return_of_a(sp, prompt_a)
            text = _log(sp.log_path)
            assert "available cells in kv cache" not in text, "state_read_meta could not place the cached conversation"
            assert "failed to load prompt from cache" not in text
            assert t["prompt_n"] == 1 and t["cache_n"] == n_a - 1, \
                f"A was prefilled again instead of restored: prompt_n={t['prompt_n']}, cache_n={t['cache_n']}"
            assert re.search(r"\[pool\] \d+ -> \d+ cells .*: a cached conversation coming back", text), \
                "the pool did not grow for the restore"
        finally:
            sp.stop()


# ---------------------------------------------------------------------------------------------------------
# [seq] the id a resume is handed lives nowhere until the restore lands. Measured 2026-09-17 on GLM: the
# resume of a suspended 200k-token generation acquired id 3 by raising the ceiling to 4, asked the pool to
# grow for its cells, and the grow's own shrink rung - "cells short, ids idle: give the idle ids back" -
# saw no record holding 3 and lowered the ceiling to 3 again. The restore into id 3 failed, its clean-up
# seq_rm(3) hit common_memory's abort, and the server dumped core with three conversations on it.
#
# Driven here with the fake device: ids cost nothing (every raise succeeds) and the pool has a budget it
# reaches with A and B on it. A is suspended for B; D takes A's id while A waits; A's resume then has to
# raise for an id and cannot grow for its cells.
# ---------------------------------------------------------------------------------------------------------

def _wait_for(path: str, pattern: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if re.search(pattern, _log(path)):
            return True
        time.sleep(0.05)
    return False


def test_resume_keeps_the_id_it_was_handed_when_the_pool_cannot_grow():
    log = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log)
    sp.n_ctx = 4096           # the budget is the wall, not -c; a seat still holds at most n_ctx_train (2048)
    sp.n_threads = 1
    sp.n_gpu_layer = 0
    sp.debug = True
    # no CUDA device at all (a CUDA build otherwise puts the host copies on CUDA_Host, at another price):
    # 1.2 KiB per cell on the CPU buffer, 2048 cells (2.4 MiB) fit the 2.5 MiB and 2304 do not; an id is free
    sp.env = {"LLAMA_SERVER_POOL_SELFTEST": "2.5:0", "CUDA_VISIBLE_DEVICES": ""}
    sp.start(timeout_seconds=120)
    try:
        prompt_a = "Alpha. " + _prompt_of(sp, 1500)
        prompt_b = "Bravo. " + _prompt_of(sp, 20)
        prompt_d = "Delta. " + _prompt_of(sp, 1000)

        res = {}

        def go(k, prompt, n):
            try:
                res[k] = _gen(sp, prompt, n)
            except Exception as e:      # a died server shows up here
                res[k] = e

        # A ends under a seat's 2048; A and B together do not fit the budget, so A is suspended for B
        ta = threading.Thread(target=go, args=("A", prompt_a, 500))
        ta.start()
        assert _wait_for(log, r"id\s+0 \| task \d+ \| n_gen = \d+, n_remaining", 60), "A never started generating"
        tb = threading.Thread(target=go, args=("B", prompt_b, 600))
        tb.start()

        assert _wait_for(log, r"offloaded sequence \d+ mid-flight", 120), "the pool never suspended anyone"

        # D takes the id A gave up and stays resident on it: A's resume has to raise for an id
        r = _gen(sp, prompt_d, 4)
        assert r.status_code == 200, r.body

        ta.join(timeout=300)
        tb.join(timeout=300)
        text = _log(log)
    finally:
        sp.stop()

    for k in "AB":
        assert k in res, f"{k}: no response"
        assert not isinstance(res[k], Exception), f"{k}: the server died: {res[k]!r}"

    # PRECONDITIONS: A's resume raised the ceiling for its id, and the pool could not grow for its cells
    i_raise = text.find("raised the sequence ceiling to 3")
    assert i_raise >= 0, "the resume never needed a raise: nothing is proven"
    assert re.search(r"more do not fit \(resuming a suspended generation\)", text), "the pool grew for the resume: nothing is proven"

    # the id the resume holds is not "idle" for the grow's shrink rung: the ceiling stays where the raise
    # put it until the restore has landed under that id
    i_restored = text.find("restored task", i_raise)
    assert i_restored > 0, "the suspended generation never came back"
    between = text[i_raise:i_restored]
    assert "lowered the sequence ceiling" not in between, (
        "the grow's shrink rung took back the id the resume was handed")
    assert "larger than n_seq_max" not in text
    assert "failed to restore target KV state" not in text

    assert res["A"].status_code == 200, f"A: {res['A'].body}"
    assert res["A"].body["timings"]["predicted_n"] == 500, f"A was cut short: {res['A'].body['timings']}"
    assert "resumed at" in text, "the suspended generation never resumed"

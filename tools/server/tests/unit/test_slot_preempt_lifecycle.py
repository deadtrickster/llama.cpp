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


def _mk(log, quantum, n_slots=1, n_ctx=2048, n_threads=None, kv_unified=False):
    sp = ServerPreset.tinyllama2()
    sp.n_slots = n_slots
    # [seats] these tests are about what happens to a SUSPENDED task, and a task is only suspended when the
    # seats are contended: hold the seat count where the test was written (seats follow the pool otherwise)
    sp.parallel_max = n_slots
    sp.n_ctx = n_ctx
    sp.n_threads = n_threads
    # [seq] with a unified cache the server grows the sequence ceiling on demand, so a yielded
    # generation stays RESIDENT (cells kept, zero copy); without it the ceiling cannot move and
    # the yielded generation is OFFLOADED (state copied out). Both paths must satisfy these tests.
    sp.kv_unified = kv_unified
    # [pool] a suspended state that must FAIL to come back needs a pool that cannot grow for it; on CPU the
    # elastic pool would (the device never says no): pin it, which is what --pool-static is for
    sp.pool_static = True
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


@pytest.mark.parametrize("kv_unified", [False, True])
def test_cancel_reaches_a_suspended_task(kv_unified):
    """T1.6. A holds the only slot, B queues, A yields. A's client hangs up
    while A is suspended. The CANCEL handler searched `slots`, so A stayed in
    queue_suspended, resumed when B finished, and generated ~1500 tokens into
    a closed socket while holding the slot (and its KV) the whole time.

    -t 1 so that B (a full 2048-token context, ~2.4 s) outlives the server's
    1 s socket poll: the CANCEL has to land while A is still suspended."""
    log = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log, quantum=8, n_slots=1, n_threads=1, kv_unified=kv_unified)
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


def _read_stream_until_quiet(resp: http.client.HTTPResponse) -> str:
    """Collect the content of an SSE completion until the server goes quiet
    (socket timeout) - for a suspended task that is everything sent so far."""
    content = ""
    try:
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.decode("utf-8", errors="replace").strip()
            if line.startswith("data: ") and "[DONE]" not in line:
                content += json.loads(line[6:]).get("content", "")
    except (socket.timeout, TimeoutError):
        pass
    return content


@pytest.mark.parametrize("kv_unified", [False, True])
def test_suspended_task_survives_shutdown(tmp_path, kv_unified):
    """T1.5. A yields to B; SIGTERM lands while A is suspended and B holds the
    slot. flush_prompt_cache() walked `slots` only, so B was saved and A - whose
    KV state and prompt were sitting in queue_suspended in exactly the format
    the cache stores - died with the process. With the registry A is either
    OFFLOADED (bytes, no context needed) or RESIDENT (its cells are read out of
    the context, like a seated sequence's); the flush must cover both.

    Two facts after the shutdown: the spill directory holds an entry of exactly
    A's prompt + generated-so-far length, and a restart on the same
    --slot-save-path serves the conversation A's client actually has (prompt +
    the streamed partial output) from the cache. The bare prompt alone would
    not hit: the cache refuses entries it would keep less than 25% of
    (f_keep < 0.25, "don't trash large prompts"), which is its general policy
    for any long conversation and not this flush's business.

    The clients hang up after the flush: server_context::terminate() stops the
    task queue only, so a handler whose client is still connected polls forever
    and ctx_http.stop() waits on it (pre-existing, not the subject here)."""
    cache_dir = str(tmp_path)
    log = os.path.join(cache_dir, "srv1.log")
    sp = _mk(log, quantum=8, n_slots=1, n_threads=1, kv_unified=kv_unified)
    sp.cache_ram = 100
    sp.slot_save_path = cache_dir
    sp.start(timeout_seconds=120)
    rc = None
    conns = []
    try:
        _wait_ready(sp)
        n_prompt_a = len(_tokens(sp, PROMPT_A)) + 1     # + BOS
        conns.append(_raw_post(sp, _completion(PROMPT_A, 1500, stream=True), timeout=1.0))
        time.sleep(0.05)
        conns.append(_raw_post(sp, _completion(PROMPT_B, 3000), timeout=60))

        assert _wait_log(log, "yielded the slot", 10), "precondition: no yield happened"
        # what A's client has in hand at this point
        partial = _read_stream_until_quiet(conns[0].getresponse())
        assert len(partial) > 0, "precondition: A streamed nothing before it yielded"

        # SIGTERM, the same thing stop() sends, but we want the exit status
        sp.process.terminate()
        assert _wait_log(log, "remaining cache entries", 30), "precondition: the shutdown spill never ran"
        for c in conns:
            c.close()
        rc = sp.process.wait(timeout=30)
    finally:
        sp.stop()
    assert rc == 0, f"server exited with {rc}"
    assert _log_count(log, "flush:") > 0, "precondition: the shutdown flush never ran"

    # fact 1: the suspended state is on disk, as an entry of A's exact length.
    # spill files are named l2p-<model>-<fingerprint>-<n_tokens>-<hash>.spill
    with open(log, errors="replace") as f:
        m = re.search(r"suspended after (\d+) generated tokens", f.read())
    assert m, "precondition: no suspend line in the log"
    n_suspended = n_prompt_a + int(m.group(1))
    spilled = sorted(int(os.path.basename(p).split("-")[3]) for p in glob.glob(os.path.join(cache_dir, "l2p-*.spill")))
    assert any(abs(n - n_suspended) <= 1 for n in spilled), \
        f"no spill entry of ~{n_suspended} tokens (A's prompt {n_prompt_a} + {m.group(1)} generated); on disk: {spilled}"

    # fact 2: a new process serves A's conversation from that entry. Same KV layout as the first
    # process: the spill files are keyed by a fingerprint that includes it, so a server started
    # without --kv-unified indexes none of what a unified one spilled
    sp2 = _mk(os.path.join(cache_dir, "srv2.log"), quantum=8, n_slots=1, kv_unified=kv_unified)
    sp2.cache_ram = 100
    sp2.slot_save_path = cache_dir
    sp2.start(timeout_seconds=120)
    try:
        _wait_ready(sp2)
        r = sp2.make_request("POST", "/completion", data={
            "prompt": PROMPT_A + partial, "n_predict": 4, "temperature": 0.0, "cache_prompt": True})
        assert r.status_code == 200, r.body
        t = r.body["timings"]
    finally:
        sp2.stop()

    # detokenize/retokenize can move a boundary or two, so "the whole prompt and
    # most of the partial output" rather than an exact count. Without the flush
    # this is 0 or 1 (BOS).
    assert t["cache_n"] >= n_prompt_a + len(partial.split()) // 2, \
        f"suspended conversation lost at shutdown: cache_n={t['cache_n']} of {t['prompt_n'] + t['cache_n']}"


def test_failed_resume_answers_the_client_and_clears_the_slot():
    """T1.4. Forces a REAL restore failure. state_read_meta() first does
    seq_rm(dest) and then find_slot(cont=false), so it fails exactly when the
    pool has fewer free cells than the saved state - and the cells it just
    freed belong to the resume slot's previous occupant, so the bulk has to be
    on the OTHER slot.

    Layout, --kv-unified pool of 1024 cells, --batch-size 2:
      A (slot 0): 100-token prompt, then generation
      B (slot 1): 600-token prompt sent right after, n_predict 300. Slot 0 fills
                  the batch first, so B prefills 1 token per A step.
      C:          deferred behind both - that arms A's quantum trigger. B is
                  still in prefill at A's 320th token (600 > 320), and a prefill
                  never yields, so A is the one that goes: 100 + 320 = 420 cells.
                  C then takes slot 0 for 330 tokens; B keeps growing 1/step and
                  is at ~650 cells when C finishes and A tries to come back:
                  1024 - 650 < 420.

    Before the fix that failure was logged and the entry destroyed: A's client
    waited forever, and slot 0 kept C's prompt with none of its cells in KV."""
    log = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log, quantum=320, n_slots=2, n_ctx=1024)
    sp.kv_unified = True
    sp.n_batch = 2
    sp.start(timeout_seconds=120)
    try:
        _wait_ready(sp)
        toks = _tokens(sp, " ".join(["the cat sat on the mat and looked at the dog"] * 120))
        assert len(toks) >= 700, f"precondition: only {len(toks)} tokens to build prompts from"
        prompt_a = toks[:100]
        prompt_b = toks[:600]
        prompt_c = toks[600:620]
        prompt_d = toks[600:640]          # C's prompt is a prefix of D's

        conn_a = _raw_post(sp, _completion(prompt_a, 1500), timeout=20)
        time.sleep(0.005)
        b, c = [], []
        tb = threading.Thread(target=_busy, args=(sp, prompt_b, 300, b)); tb.start()
        time.sleep(0.005)
        tc = threading.Thread(target=_busy, args=(sp, prompt_c, 330, c)); tc.start()

        # PRECONDITIONS: A yielded, and its resume failed for real
        assert _wait_log(log, "yielded the slot", 15), "precondition: no yield happened"
        assert _wait_log(log, "failed to resume a suspended generation", 15), \
            "precondition: the resume did not fail, so the layout did not force a restore failure"

        # the stated defect: nobody tells A's client
        try:
            resp = conn_a.getresponse()
            body = resp.read()
        except socket.timeout:
            pytest.fail("A's client hung after the failed resume: no error was sent")
        assert resp.status in (200, 500), f"unexpected status {resp.status}: {body!r}"

        tb.join(timeout=60); tc.join(timeout=60)
        assert b and b[0].status_code == 200, f"B failed: {b}"
        assert c and c[0].status_code == 200, f"C failed: {c}"

        # the corruption: the failed restore did seq_rm(0) before it read, so slot 0
        # has no cells for the prompt it still claims (C's). A request on slot 0
        # whose prompt extends C's must NOT be told any of it is cached.
        r = sp.make_request("POST", "/completion", data={
            "prompt": prompt_d, "id_slot": 0, "n_predict": 1, "temperature": 0.0, "cache_prompt": True})
        assert r.status_code == 200, r.body
        assert r.body["timings"]["cache_n"] == 0, \
            f"slot 0 reused {r.body['timings']['cache_n']} cached tokens that are not in the KV"
    finally:
        sp.stop()

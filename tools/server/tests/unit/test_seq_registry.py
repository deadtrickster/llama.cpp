import os
import re
import tempfile
import threading
import time

import pytest
from utils import *

# T2.3 / T2.4 of the adaptive-parallel work (GLM-TODO section 14/15): sequences
# outlive seats. A seat is a server_slot (--parallel); a sequence is a
# llama_seq_id and the KV cells it holds (--seq-max, T2.1). Every sequence is a
# record in one registry with three states:
#
#   RUNNING    seated and decoding
#   RESIDENT   holds cells, nobody decoding it - seated-idle or unseated
#   OFFLOADED  no cells, no id; mid-flight state in RAM until an id frees up
#
# The user-visible payoff, and the property tested hardest here: a sequence
# that is RESIDENT but not RUNNING resumes WITHOUT reprocessing its prompt and
# WITHOUT copying its state anywhere. Before the registry a seat needed for a
# different conversation destroyed the one it held (or copied it to the prompt
# cache and back - the 1.4-13 GB path on GLM).
#
# Everything runs with --cache-ram 0 unless stated: with no prompt cache the
# only way a second turn can report cache_n > 0 is that its cells were still in
# the KV under their own sequence id. --no-cache-idle-slots for the same reason
# (with it on, every launch offloads every idle sequence; that eager valve is
# T2.5's business, not this file's).
#
# NOTE: run with PORT set to something free, e.g.
#   PORT=51910 pytest unit/test_seq_registry.py
# ServerProcess defaults to 8080, which the production server holds.

PROMPT_A = "Once upon a time in a land far away there lived a brave knight who"
PROMPT_B = "In a small village by the sea an old fisherman mended his nets every morning"
PROMPT_C = "The mountain pass was closed by snow and the travellers waited in the inn"


def _mk(log, n_slots=1, seq_max=None, cache_ram=0, quantum=None, n_ctx=2048, kv_unified=True, parallel_max=None):
    sp = ServerPreset.tinyllama2()
    sp.n_slots = n_slots
    # [seats] seats follow the pool now, so "one seat, two ids" needs the cap said out loud (--parallel-max);
    # the tests that leave it unset are the ones asserting that seats follow
    sp.parallel_max = parallel_max
    sp.n_ctx = n_ctx
    sp.n_predict = None           # the preset's --n-predict 64 is not a cap we want
    sp.temperature = 0.0
    sp.kv_unified = kv_unified    # one pool; ids above --parallel cost no per-slot context
    sp.seq_max = seq_max
    sp.cache_ram = cache_ram
    sp.no_cache_idle_slots = True
    sp.spec_type = "none"         # the server default drafts, which changes token counts
    sp.slot_quantum = quantum
    sp.n_threads = 4
    sp.log_path = log
    return sp


def _log(path: str) -> str:
    with open(path, "r", errors="replace") as f:
        return f.read()


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


def _tokens(sp: ServerProcess, text: str) -> list:
    r = sp.make_request("POST", "/tokenize", data={"content": text})
    assert r.status_code == 200
    return r.body["tokens"]


def _gen(sp: ServerProcess, prompt, n_predict: int, **extra) -> dict:
    """Greedy, EOS ignored, tokens returned: the continuation is a function of
    the prompt alone, so two turns can be checked against one long generation."""
    data = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
            "seed": 42, "cache_prompt": True, "return_tokens": True, "ignore_eos": True}
    data.update(extra)
    r = sp.make_request("POST", "/completion", data=data, timeout=120)
    assert r.status_code == 200, r.body
    return r.body


def test_resident_sequence_resumes_without_reprocessing():
    """One seat, two ids. A's conversation is on the seat when B arrives. B
    needs the seat, not A's cells: A must stay RESIDENT under its own id and
    its next turn must be served from those cells - prompt_n 1, cache_n the
    whole conversation - and continue exactly where one uninterrupted
    generation would have. Before the registry the seat was the sequence: B's
    launch cleared A (no cache to save it to), and A's next turn paid a full
    re-prefill."""
    log = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log, n_slots=1, seq_max=2)
    sp.start(timeout_seconds=120)
    try:
        _wait_ready(sp)
        toks_a = _tokens(sp, PROMPT_A)
        toks_b = _tokens(sp, PROMPT_B)
        n1, n2 = 32, 32

        # the reference: A generated in one go. Its first n1 tokens are what a
        # first turn of A produces; the rest is what the second turn must.
        ref = _gen(sp, toks_a, n1 + n2)
        assert len(ref["tokens"]) == n1 + n2, ref["timings"]

        # B takes the seat
        b = _gen(sp, toks_b, 8)
        assert b["timings"]["cache_n"] == 0

        # A's second turn: the conversation so far, plus nothing new
        turn2 = _gen(sp, toks_a + ref["tokens"][:n1], n2)
        t = turn2["timings"]
    finally:
        sp.stop()

    # the payoff, by token counts: nothing but the final token is reprocessed
    # (the server always re-evaluates at least one token to get logits)
    assert t["cache_n"] == len(toks_a) + n1 - 1, \
        f"A's cells were not resident: cache_n={t['cache_n']}, prompt_n={t['prompt_n']}"
    assert t["prompt_n"] == 1, t
    # and by content: the resident cells are A's, not something else's
    assert turn2["tokens"] == ref["tokens"][n1:], "the second turn diverged from the uninterrupted generation"


def test_id_pressure_evicts_lru_and_keeps_the_rest_resident():
    """One seat, two ids (--seq-max 2 caps the ceiling), three conversations.
    C cannot have an id without taking one: the least recently used resident
    (A) goes, B stays. B's next turn is served from its cells; A's next turn
    is a full re-prefill (no cache to bring it back from) and, needing an id
    of its own, evicts the LRU resident of that moment (C) - which is why B is
    asked first. Before the registry both paid the full re-prefill."""
    log = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log, n_slots=1, seq_max=2)
    sp.start(timeout_seconds=120)
    try:
        _wait_ready(sp)
        toks_a = _tokens(sp, PROMPT_A)
        toks_b = _tokens(sp, PROMPT_B)
        toks_c = _tokens(sp, PROMPT_C)

        a = _gen(sp, toks_a, 16)
        b = _gen(sp, toks_b, 16)
        c = _gen(sp, toks_c, 16)
        assert c["timings"]["cache_n"] == 0

        b2 = _gen(sp, toks_b + b["tokens"], 4)["timings"]
        a2 = _gen(sp, toks_a + a["tokens"], 4)["timings"]
    finally:
        sp.stop()

    assert b2["cache_n"] == len(toks_b) + 16 - 1, f"B was not kept resident: {b2}"
    assert b2["prompt_n"] == 1
    assert a2["cache_n"] == 0, f"A should have been evicted for C: {a2}"
    assert a2["prompt_n"] == len(toks_a) + 16
    assert "purging sequence" in _log(log) or "purging slot" in _log(log), "no eviction was logged"


MAMBA_GGUF = os.path.join(os.environ.get("LLAMA_CACHE", "tmp"), "models--Felladrin--gguf-mamba-130m-hf",
                          "snapshots", "83ef2222e1a0437d26bda213537afc195a53b3ad", "mamba-130m-hf.Q8_0.gguf")


def _mk_mamba(log):
    """A recurrent model, straight from the cached file: the one whose per-id
    state rows are actually reallocated when the ceiling moves."""
    sp = ServerProcess()
    sp.model_file = MAMBA_GGUF
    sp.model_hf_repo = None       # the harness defaults these to stories260K, which would win over --model
    sp.model_hf_file = None
    sp.model_alias = "mamba-130m"
    sp.n_slots = 1
    sp.n_ctx = 1024
    sp.n_batch = 64
    sp.n_ubatch = 64
    sp.n_predict = None
    sp.temperature = 0.0
    sp.kv_unified = True
    sp.cache_ram = 0
    sp.no_cache_idle_slots = True
    sp.spec_type = "none"
    sp.n_threads = 4
    sp.debug = True               # --verbose: the reallocation is logged by llama_memory_recurrent, below the server's default level
    sp.log_path = log
    return sp


def _resident_second_turn(sp: ServerProcess, log: str, n1: int = 8):
    """A, then B on the single seat, then A's second turn. Returns (turn2 timings, log)."""
    toks_a = _tokens(sp, PROMPT_A)
    toks_b = _tokens(sp, PROMPT_B)
    a = _gen(sp, toks_a, n1)
    b = _gen(sp, toks_b, 4)
    assert b["timings"]["cache_n"] == 0
    t = _gen(sp, toks_a + a["tokens"], 4)["timings"]
    return t, len(toks_a), _log(log)


def test_ceiling_grows_on_demand_without_seq_max():
    """No --seq-max at all: the context starts with one id per seat. When B
    needs an id and A holds the only one, the server asks the model what one
    more costs, finds the device has it, and raises the ceiling by one instead
    of evicting A. A's second turn is then served from its cells."""
    log = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log, n_slots=1, seq_max=None)
    sp.start(timeout_seconds=120)
    try:
        _wait_ready(sp)
        t, n_a, text = _resident_second_turn(sp, log)
    finally:
        sp.stop()

    assert "raised the sequence ceiling to 2" in text, "the ceiling never moved"
    assert t["cache_n"] == n_a + 8 - 1 and t["prompt_n"] == 1, f"A was not resident after the raise: {t}"


@pytest.mark.skipif(not os.path.exists(MAMBA_GGUF), reason="mamba-130m is not in LLAMA_CACHE")
def test_ceiling_grows_on_a_recurrent_model():
    """The same on Mamba, where the raise reallocates the recurrent state rows
    under a live sequence (the log line comes from llama_memory_recurrent).
    A's cells - its state row - must survive the move: second turn resident."""
    log = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk_mamba(log)
    sp.start(timeout_seconds=300)
    try:
        _wait_ready(sp, timeout_s=300)
        t, n_a, text = _resident_second_turn(sp, log)
    finally:
        sp.stop()

    assert "seq_max_resize: 1 -> 2 cells" in text, "the recurrent rows were not reallocated"
    # the server asked the model what one more sequence costs; a recurrent model answers with its rows
    m = re.search(r"raised the sequence ceiling to 2 \(CPU ([0-9.]+) MiB", text)
    assert m and float(m.group(1)) > 0, "the raise did not report a per-sequence cost from the model"
    assert t["cache_n"] == n_a + 8 - 1 and t["prompt_n"] == 1, f"A was not resident after the raise: {t}"


def test_ceiling_shrinks_when_idle_and_grows_back():
    """--seq-max 3 with one seat: three ids are paid for at startup. Once the
    server is idle with one conversation live the ceiling comes down to what is
    used (1, never below the seat count); the next conversation raises it again,
    up to the cap. Both moves are logged by the server."""
    log = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log, n_slots=1, seq_max=3)
    sp.start(timeout_seconds=120)
    try:
        _wait_ready(sp)
        toks_a = _tokens(sp, PROMPT_A)
        toks_b = _tokens(sp, PROMPT_B)
        _gen(sp, toks_a, 4)
        time.sleep(0.3)   # the shrink runs in the idle pass after the release
        assert "lowered the sequence ceiling to 1" in _log(log), "the ceiling did not come down while idle"
        _gen(sp, toks_b, 4)
        text = _log(log)
    finally:
        sp.stop()

    assert "raised the sequence ceiling to 2" in text, "the ceiling did not grow back for B"


def _contend(sp: ServerProcess, prompt_a: str, prompt_b: str):
    """A holds the only seat for 1500 tokens; B arrives 50 ms later and must
    queue. Returns (A, B, B_finished_before_A)."""
    a, b, t_done = [], [], {}

    def run(tag, prompt, n, out):
        out.append(sp.make_request("POST", "/completion", data={
            "prompt": prompt, "n_predict": n, "temperature": 0.0, "top_k": 1,
            "seed": 42, "cache_prompt": False}, timeout=300))
        t_done[tag] = time.time()

    ta = threading.Thread(target=run, args=("A", prompt_a, 1500, a)); ta.start()
    time.sleep(0.05)
    assert not a, "A finished before B was even sent: no contention possible"
    tb = threading.Thread(target=run, args=("B", prompt_b, 8, b)); tb.start()
    ta.join(timeout=300); tb.join(timeout=300)
    assert a and a[0].status_code == 200, "holding request failed"
    assert b and b[0].status_code == 200, "contending request failed"
    assert a[0].body["timings"]["predicted_n"] >= 1000, "holding request too short to create contention"
    return a[0], b[0], t_done["B"] < t_done["A"]


def _yield_run(seq_max, prompt_b=PROMPT_B, kv_unified=True, n_ctx=2048):
    log = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log, n_slots=1, seq_max=seq_max, quantum=8, kv_unified=kv_unified, n_ctx=n_ctx, parallel_max=1)
    sp.start(timeout_seconds=120)
    try:
        _wait_ready(sp)
        alone = sp.make_request("POST", "/completion", data={
            "prompt": PROMPT_A, "n_predict": 1500, "temperature": 0.0, "top_k": 1,
            "seed": 42, "cache_prompt": False}, timeout=300)
        assert alone.status_code == 200
        a, b, b_first = _contend(sp, PROMPT_A, prompt_b)
    finally:
        sp.stop()
    text = _log(log)
    assert text.count("yielded the slot") > 0, "precondition: A never yielded"
    assert b_first, "precondition: B waited for A to finish, the yield changed nothing"
    return alone, a, text


def _generate_1500(sp: ServerProcess, env=None):
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
        r = sp.make_request("POST", "/completion", data={
            "prompt": PROMPT_A, "n_predict": 1500, "temperature": 0.0, "top_k": 1,
            "seed": 42, "cache_prompt": False}, timeout=300)
        assert r.status_code == 200
        return r.body["content"]
    finally:
        sp.stop()


def test_seat_move_is_exact():
    """T2.4's correctness argument, isolated: LLAMA_SERVER_PREEMPT_SELFTEST=N
    releases and re-acquires the seat every N generated tokens with nothing
    else in the pool. 1500 tokens through ~30 moves must equal a server that
    never moved - the field list moved by seat_release()/seat_acquire()
    mirrors reset() exactly, or this shows the difference."""
    plain = _generate_1500(_mk(os.path.join(tempfile.mkdtemp(), "srv.log"), n_slots=1, seq_max=2))
    log = os.path.join(tempfile.mkdtemp(), "srv.log")
    moved = _generate_1500(_mk(log, n_slots=1, seq_max=2), env={"LLAMA_SERVER_PREEMPT_SELFTEST": "50"})
    n_moves = _log(log).count("resumed at")
    assert n_moves >= 25, f"precondition: only {n_moves} seat moves happened"
    assert moved == plain, "the seat move changed the generation"


def test_yield_with_id_headroom_is_zero_copy():
    """T2.4. One seat, two ids, --slot-quantum. A yields to B: A's cells stay
    where they are under A's id, B decodes under the other id, A comes back by
    being re-seated. No state is copied in either direction, B is answered
    before A finishes, and A's continuation is identical to the uncontended
    one. Before the registry a yield copied the whole sequence out and the
    resume copied it back.

    Without --kv-unified on purpose: each id has its own KV stream, so B's
    resident conversation cannot touch A's cell layout. In the unified pool it
    can - B's cells sit inside A's span - and that is enough to move the CPU
    matmul's rounding and flip a greedy near-tie hundreds of tokens later
    (measured: 58 seat moves with an empty pool are bit-identical over 1500
    tokens, a contended run with a resident contender diverges at token 400 to
    900 depending on where the yield landed). The layout effect is the pool's,
    not the seat move's; the unified arm below asserts the mechanism only.
    n_ctx 4096 because the split leaves each stream 2048."""
    alone, a, text = _yield_run(seq_max=2, kv_unified=False, n_ctx=4096)
    assert a.body["content"] == alone.body["content"], "resumed generation diverged"
    assert "resumed at" in text
    assert "stays resident, zero copy" in text, "the yield did not leave the sequence resident"
    assert "offloaded sequence" not in text, "a sequence was offloaded although an id was free"
    assert "restored task" not in text, "state was copied back although it never left"


def test_yield_with_id_headroom_long_contender_is_still_zero_copy():
    """The same with a long contender: B's whole conversation stays resident
    next to A's, nothing is copied, and A finishes its 1500 tokens with no
    second prefill."""
    alone, a, text = _yield_run(seq_max=2, prompt_b=PROMPT_B)
    assert a.body["timings"]["predicted_n"] == 1500
    assert a.body["timings"]["prompt_n"] == alone.body["timings"]["prompt_n"], "the resume re-prefilled the prompt"
    assert "stays resident, zero copy" in text
    assert "offloaded sequence" not in text and "restored task" not in text


def test_yield_without_id_headroom_offloads_and_still_resumes():
    """The contrast: one seat, ONE id, and --seq-max 1 so the ceiling may not
    grow. B cannot decode without A's id, so A is offloaded (the copy) and
    restored when the id frees up. Same result, just the price of the copy -
    which is what production paid before the ceiling could move."""
    alone, a, text = _yield_run(seq_max=1)
    assert a.body["content"] == alone.body["content"], "resumed generation diverged"
    assert "offloaded sequence" in text, "no id was free, so A must have been offloaded"
    assert "restored task" in text, "the offloaded state was never restored"


def test_resume_saves_the_finished_conversation_it_displaces():
    """The 'real D3' T1 left open for the registry: A yields to B, B finishes
    and its conversation sits on the seat, A resumes INTO that seat. The old
    resume pass restored A over B without saving B - the one launch-time path
    that skipped the prompt cache. With one id and the cache on, A's resume
    must evict B through the cache, so B's next turn is a cache hit. --seq-max 1
    holds the ceiling at one id; with room to grow B would simply get a second
    id and nothing would be displaced."""
    log = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log, n_slots=1, seq_max=1, cache_ram=100, quantum=8)
    sp.start(timeout_seconds=120)
    try:
        _wait_ready(sp)
        toks_b = _tokens(sp, PROMPT_B)

        a, b_out = [], []

        def run_a():
            a.append(sp.make_request("POST", "/completion", data={
                "prompt": PROMPT_A, "n_predict": 1500, "temperature": 0.0, "top_k": 1,
                "seed": 42, "cache_prompt": False}, timeout=300))

        def run_b():
            b_out.append(_gen(sp, toks_b, 8))

        ta = threading.Thread(target=run_a); ta.start()
        time.sleep(0.05)
        assert not a, "A finished before B was sent"
        tb = threading.Thread(target=run_b); tb.start()
        tb.join(timeout=300); ta.join(timeout=300)
        assert a and a[0].status_code == 200
        assert b_out
        b = b_out[0]
        assert "yielded the slot" in _log(log), "precondition: A never yielded"
        assert "resumed at" in _log(log), "precondition: A never resumed"

        # B's next turn, after A is done with the seat
        b2 = _gen(sp, toks_b + b["tokens"], 4)["timings"]
    finally:
        sp.stop()

    assert b2["cache_n"] == len(toks_b) + 8 - 1, \
        f"B's conversation was destroyed by A's resume instead of spilled: {b2}"
    assert b2["prompt_n"] == 1


#
# Pool-driven seats. The operator's spec, verbatim: "i want a pool and whatever fits in here gets batched."
# A seat (server_slot) is a batch position; before this, --parallel fixed their number and the sequence
# ceiling moved underneath it, so residency adapted and batching did not. Now seats follow residency: a
# resident sequence with pending work is batched, and admission asks the one question the cost query
# answers - does one more sequence fit right now, on this model, on each device. --parallel is a floor.
#
# The property, as an assertion: with a pool that can hold N conversations, N conversations make progress
# CONCURRENTLY - without anyone having set N. Asserted by observing generation, not by reading a config.
#

import json
import requests

PROMPTS = [
    "Once upon a time in a land far away there lived a brave knight who",
    "In a small village by the sea an old fisherman mended his nets every morning",
    "The mountain pass was closed by snow and the travellers waited in the inn",
    "Deep in the forest a fox and a rabbit argued about who owned the river",
    "The baker woke before dawn to light the ovens and knead the bread",
    "A little girl found a key in the garden and wondered which door it opened",
]


def _stream(sp: ServerProcess, prompt: str, n_predict: int, out: dict, tag: str):
    """One streamed completion. Records when its first and last tokens ARRIVED, so
    overlapping [first, last] windows are direct evidence of concurrent decoding."""
    url = f"http://{sp.server_host}:{sp.server_port}/completion"
    r = requests.post(url, json={
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1, "seed": 42,
        "cache_prompt": False, "stream": True, "ignore_eos": True}, stream=True, timeout=600)
    t_first = None
    t_last = None
    gaps = []
    text = ""
    n = 0
    for raw in r.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace")
        if not line.startswith("data: "):
            continue
        d = json.loads(line[6:])
        now = time.time()
        if t_first is None:
            t_first = now
        else:
            gaps.append(now - t_last)
        t_last = now
        n += 1
        text += d.get("content", "")
        if d.get("stop"):
            break
    out[tag] = {"status": r.status_code, "t_first": t_first, "t_last": t_last, "n": n,
                "text": text, "max_gap": max(gaps) if gaps else 0.0}


def _run_concurrently(sp: ServerProcess, prompts, n_predict: int, stagger_s: float = 0.0) -> dict:
    out = {}
    threads = []
    for i, p in enumerate(prompts):
        t = threading.Thread(target=_stream, args=(sp, p, n_predict, out, f"r{i}"))
        t.start()
        threads.append(t)
        if stagger_s:
            time.sleep(stagger_s)
    for t in threads:
        t.join(timeout=600)
    assert len(out) == len(prompts), f"{len(prompts) - len(out)} requests never returned"
    for tag, r in out.items():
        assert r["status"] == 200, (tag, r)
        assert r["n"] >= n_predict, f"{tag} produced {r['n']} < {n_predict} tokens"
    return out


def _max_overlap(out: dict) -> int:
    """The largest number of generations that were between their first and last token at one instant."""
    events = []
    for r in out.values():
        events.append((r["t_first"], 1))
        events.append((r["t_last"], -1))
    events.sort(key=lambda e: (e[0], e[1]))  # an end at the same instant as a start counts as not overlapping
    cur = best = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    return best


def _seat_counts(log_text: str) -> list:
    """Every seat count the server logged when it grew or shrank."""
    return [int(m) for m in re.findall(r"seats: (\d+) \(", log_text)]


def test_pool_holds_n_so_n_run_concurrently():
    """Four conversations that the pool can hold, --parallel 1, no --seq-max: all four must be
    generating at the same instant. Before pool-driven seats --parallel was the batch width and the
    three others waited their turn on the one seat, however much room the pool had."""
    log = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log, n_slots=1, seq_max=None, n_ctx=8192)
    sp.start(timeout_seconds=120)
    try:
        _wait_ready(sp)
        out = _run_concurrently(sp, PROMPTS[:4], n_predict=400)
    finally:
        sp.stop()

    overlap = _max_overlap(out)
    first_done = min(r["t_last"] for r in out.values())
    last_start = max(r["t_first"] for r in out.values())
    assert overlap == 4, (
        f"only {overlap} of 4 generations ran at once: the last first-token arrived {last_start - first_done:+.3f} s "
        f"after the first generation finished. Nobody set 4; the pool holds 4; 4 must run.")
    counts = _seat_counts(_log(log))
    assert counts and max(counts) == 4, f"the seat count never followed the pool: {counts}"


def test_pool_bound_admits_exactly_what_fits():
    """The other half of 'whatever fits gets batched': what does NOT fit waits. --seq-max 2 stands in
    for the cost query refusing a third id (that is the GLM situation at ceiling 5). Four requests,
    --parallel 1: exactly two generate at once, never three, the seat count never exceeds the ceiling,
    and all four complete."""
    log = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log, n_slots=1, seq_max=2, n_ctx=8192)
    sp.start(timeout_seconds=120)
    try:
        _wait_ready(sp)
        out = _run_concurrently(sp, PROMPTS[:4], n_predict=400)
    finally:
        sp.stop()

    assert _max_overlap(out) == 2, f"expected exactly 2 concurrent generations under a 2-id ceiling, saw {_max_overlap(out)}"
    counts = _seat_counts(_log(log))
    assert counts and max(counts) == 2, f"seats went past the ceiling or never grew: {counts}"


def test_suspended_generation_gets_compute_within_the_deadline():
    """RULE 1 of the scheduler: a hard ceiling on time asleep. The adversarial shape: every seat held by a
    long generation with NOTHING queued behind it, plus one suspended sequence. One id (--seq-max 1), so
    A's yield to B offloads A; B is as long as A and nobody else arrives, so no quantum yield ever fires
    for A's benefit. --slot-resume-after is the bound. Before this A waited for B to finish - the
    resume pass could win a free seat but nothing ever freed one."""
    bound_ms = 300
    log = os.path.join(tempfile.mkdtemp(), "srv.log")
    sp = _mk(log, n_slots=1, seq_max=1, quantum=8, n_ctx=4096)
    sp.slot_resume_after = bound_ms
    sp.n_threads = 1              # the model's training context caps a generation at 2048 tokens; slow it instead
    sp.start(timeout_seconds=120)
    try:
        _wait_ready(sp)
        alone = sp.make_request("POST", "/completion", data={
            "prompt": PROMPTS[0], "n_predict": 2000, "temperature": 0.0, "top_k": 1,
            "seed": 42, "cache_prompt": False, "ignore_eos": True}, timeout=600)
        assert alone.status_code == 200
        t_alone = alone.body["timings"]["predicted_ms"]
        # PRECONDITION: B alone runs several bounds long, or waiting for it to finish would meet the deadline by accident
        assert t_alone > 4 * bound_ms, f"the long generation only takes {t_alone:.0f} ms; slow it down or lower the bound"

        out = _run_concurrently(sp, PROMPTS[:2], n_predict=2000, stagger_s=0.05)
    finally:
        sp.stop()

    text = _log(log)
    assert "offloaded sequence" in text, "precondition: A was never offloaded, the id was not contended"
    waits = [int(m) for m in re.findall(r"resumed at \d+ generated tokens after (\d+) ms suspended", text)]
    assert waits, "no generation was ever resumed"
    # the guarantee: nothing slept past the bound (plus one batch to notice)
    slack_ms = 300
    assert max(waits) <= bound_ms + slack_ms, (
        f"a suspended generation waited {max(waits)} ms for compute against a {bound_ms} ms bound "
        f"(all waits: {waits}); B alone takes {t_alone:.0f} ms, so it waited for B to finish")
    # and the time-slicing changed nothing about what was generated
    assert out["r0"]["text"] == alone.body["content"], "A's continuation diverged across the swaps"

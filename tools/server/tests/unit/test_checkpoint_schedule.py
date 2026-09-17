import os
import re
import tempfile
import pytest
from utils import *

# Context checkpoints are placed at user-message boundaries, with --checkpoint-min-step as a
# floor between them. A prompt whose boundaries all sit near its end - or that has none, as
# with a raw /completion - therefore gets checkpoints only in its last n_ubatch tokens, and a
# later rollback to anywhere earlier finds nothing usable and reprocesses the whole prefix
# (measured in production: 33k prompt, rollback to 18k, 18k tokens thrown away, GLM-TODO T3.5c).
#
# This test asserts the fallback: with no user boundary for checkpoint_min_step tokens, a
# checkpoint is created on that schedule anyway. It reads placements from the server log
# because the checkpoint list is not exposed over HTTP. Checkpoints exist only for models
# whose sequence cannot be partially removed, so a recurrent model is used.

server: ServerProcess


class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0
    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


MIN_STEP = 64
N_BATCH  = 32

CREATED = re.compile(r"created context checkpoint \d+ of \d+ \(pos_min = -?\d+, pos_max = -?\d+, n_tokens = (\d+)")


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerProcess()
    server.model_hf_repo = "Felladrin/gguf-mamba-130m-hf"
    server.model_hf_file = "mamba-130m-hf.Q8_0.gguf"
    server.model_alias = "mamba-130m"
    server.n_ctx = 1024
    server.n_batch = N_BATCH
    server.n_ubatch = N_BATCH
    server.n_slots = 1
    server.n_predict = 4
    server.seed = 42
    server.temperature = 0.0
    server.n_ctx_checkpoints = 16
    server.checkpoint_min_step = MIN_STEP
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)


def test_checkpoints_follow_min_step_without_user_boundaries():
    global server
    server.start(timeout_seconds=300)
    log = LogReader(server.log_path)
    log.drain()

    # a raw completion prompt carries no message spans, so there is no user boundary anywhere
    prompt = ("The quick brown fox jumps over the lazy dog. " * 40).strip()
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": 2,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n = res.body["timings"]["prompt_n"]
    assert prompt_n > 4 * MIN_STEP, f"prompt too short to exercise the schedule: {prompt_n}"

    placements = [int(m) for m in CREATED.findall(log.drain())]
    print(f"prompt_n = {prompt_n}, checkpoints at n_tokens = {placements}")

    # the fallback must cover the whole prefix: from 0 and between consecutive checkpoints the
    # uncovered span may not exceed one schedule step plus one batch of slack
    edges = [0] + sorted(placements)
    gaps = [b - a for a, b in zip(edges, edges[1:])]
    assert placements, "no checkpoints were created at all"
    assert max(gaps) <= MIN_STEP + N_BATCH, \
        f"a rollback into a {max(gaps)}-token hole would reprocess it all; placements = {placements}, prompt_n = {prompt_n}"
    assert min(placements) <= MIN_STEP + N_BATCH, \
        f"first checkpoint only at {min(placements)} of {prompt_n}; nothing below covers an early rollback"


CACHE_RAM_MIB = 56   # calibrated: our 4-turn entry is ~37 MiB and the newcomer ~29; both fit only once ours sheds its middle checkpoints
RESTORED = re.compile(r"restored context checkpoint \(pos_min = -?\d+, pos_max = -?\d+, n_tokens = (\d+)")
DELIMS   = [{"role": "user", "delimiter": "USER:"}, {"role": "assistant", "delimiter": "ASSISTANT:"}]


def _turn(prompt: str, n_predict: int):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": n_predict, "cache_prompt": True, "temperature": 0.0,
        "message_delimiters": DELIMS,
    })
    assert res.status_code == 200, res.body
    return res.body["content"], res.body["timings"]


def test_turn_start_checkpoints_survive_thinning():
    """A client whose request failed recorded nothing and re-sends the conversation
    without its last exchange, so its prompt diverges at the start of the PREVIOUS
    reply. The checkpoint created at the start of that user message belonged to
    an earlier task, and the later turns' thinning passes were free to drop it
    (measured on the production server: divergence at 103,866, nearest checkpoint
    at 101,544, 12.3k tokens re-prefilled). The last two turn-start checkpoints
    must survive thinning, so the retry re-prefills that user message and its
    reply, nothing older."""
    global server
    server.n_ctx = 2048
    server.checkpoint_min_step = 16   # frequent schedule checkpoints, so the thinning pass has work to do
    server.n_ctx_checkpoints = 32
    server.cache_ram = CACHE_RAM_MIB
    server.start(timeout_seconds=300)
    log = LogReader(server.log_path)
    log.drain()

    filler = "The quick brown fox jumps over the lazy dog. " * 6
    t1 = f"USER: {filler}\nASSISTANT:"
    r1, _ = _turn(t1, 48)
    t2 = f"{t1}{r1}\nUSER: Tell me more. {filler}\nASSISTANT:"
    r2, _ = _turn(t2, 96)                 # the reply the client will lose
    t3 = f"{t2}{r2}\nUSER: And then? {filler}\nASSISTANT:"
    r3, _ = _turn(t3, 96)                  # the reply the client will lose
    t4 = f"{t3}{r3}\nUSER: Go on. {filler}\nASSISTANT:"
    _, _ = _turn(t4, 48)                   # the turn after it, so turn 3's checkpoints sit mid-list
    log.drain()

    # another conversation takes the slot: ours goes to the prompt cache, whose room is tight
    # enough (cache_ram) that saving the newcomer makes the cache shed our middle checkpoints
    _, _ = _turn(f"USER: Something else entirely. {filler * 2}\nASSISTANT:", 8)
    mid = log.drain()

    # the retry: the conversation without the last exchange, i.e. turn 3 sent again. Saving the
    # newcomer to make room for ours is what runs the cache ladder on our entry.
    _, timings = _turn(t3, 8)
    text = log.drain()
    assert "cache ladder: degraded" in mid + text, "precondition: the cache never came under pressure"

    n_t2 = server.make_request("POST", "/tokenize", data={"content": t2}).body["tokens"]
    n_t3 = server.make_request("POST", "/tokenize", data={"content": t3}).body["tokens"]
    turn3_user_len = len(n_t3) - len(n_t2) - len(server.make_request("POST", "/tokenize", data={"content": r2}).body["tokens"])

    restored = [int(m) for m in RESTORED.findall(text)]
    assert restored, f"the retry did not restore any checkpoint: {timings}"
    # the retry must land on turn 3's user-message start - the checkpoint that also serves an edit or a
    # regeneration of that turn - not on whatever else the ladder happened to keep
    turn3_start = len(n_t2) + len(server.make_request("POST", "/tokenize", data={"content": r2}).body["tokens"])
    assert abs(restored[0] - turn3_start) <= 2, (
        f"the retry restored from {restored[0]}, not from turn 3's start at {turn3_start}: "
        f"the turn-start checkpoint did not survive the cache ladder (prompt_n = {timings['prompt_n']})")
    assert timings["prompt_n"] <= turn3_user_len + N_BATCH + 8, (
        f"the retry re-prefilled {timings['prompt_n']} tokens; turn 3's user message is {turn3_user_len}")

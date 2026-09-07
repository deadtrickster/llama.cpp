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

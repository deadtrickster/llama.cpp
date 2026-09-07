import os
import tempfile
import pytest
from utils import *

# --cache-reuse on a model whose cache holds recurrent state. The KV of a matching chunk can be
# shifted to its new position; the recurrent state cannot - it is one accumulator over the whole
# history, so after a mid-history deletion it still contains the deleted text, and so does every
# context checkpoint taken past the point where the old and new prompts diverge. Left alone, the
# server restored such a checkpoint and answered from the deleted text (measured on glm5next:
# "4 document parts" where the truth was "three"; reproduced on mamba-130m below as "zebra").
#
# The fix (T3.5b) keeps cache reuse ENABLED for these models and rebuilds from the divergence
# instead of shifting. This test asserts on what the model SAYS - a token-count-only check scores
# the bug as a win (25 tokens, wrong answer) - and that neither gate refuses the feature any more.
# The attention-only control keeps the real shift.

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


def mamba_130m() -> ServerProcess:
    server = ServerProcess()
    server.model_hf_repo = "Felladrin/gguf-mamba-130m-hf"
    server.model_hf_file = "mamba-130m-hf.Q8_0.gguf"
    server.model_alias = "mamba-130m"
    server.n_ctx = 4096
    server.n_batch = 512
    server.n_slots = 1
    server.n_predict = 8
    server.seed = 42
    server.temperature = 0.0
    return server


# P1 + P2 + P3 + Q, then P2 is deleted. P3 is long enough for the reuse loop to match it as a
# chunk; P2 carries a fact that only a state computed WITH P2 can produce at Q.
P1 = ("Mary had a little lamb whose fleece was white as snow, and everywhere that Mary went the lamb was sure to go. "
      "It followed her to school one day which was against the rule; it made the children laugh and play to see a lamb at school. ") * 6
P2 = "Remember this: the password is zebra. The password is zebra. " * 30
P3 = "Jack and Jill went up the hill to fetch a pail of water; Jack fell down and broke his crown and Jill came tumbling after. " * 12
Q  = "\nQuestion: what is the password?\nAnswer: the password is"

LOAD_REFUSAL    = "cache_reuse is not supported by this context, it will be disabled"
REQUEST_REFUSAL = "cache reuse is not supported - ignoring n_cache_reuse"
SHIFTED         = "reusing chunk with size"
REBUILT         = "cannot be shifted with it; rebuilding from"


def _start_with_log(srv: ServerProcess) -> LogReader:
    srv.debug = True
    fd, srv.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    srv.n_cache_reuse = 32
    # enough checkpoints that one survives below the divergence; the prompt cache OFF, or the
    # slot silently serves the old and the new conversation from two cached copies
    srv.n_ctx_checkpoints = 64
    srv.checkpoint_min_step = 64
    srv.cache_ram = 0
    srv.cache_disk = 0
    srv.start(timeout_seconds=300)
    return LogReader(srv.log_path)


def _complete(srv: ServerProcess, prompt: str, cache_prompt: bool, n_cache_reuse: int = 32):
    res = srv.make_request("POST", "/completion", data={
        "prompt": prompt,
        "n_predict": 8,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": cache_prompt,
        "n_cache_reuse": n_cache_reuse,
    })
    assert res.status_code == 200, res.body
    return res.body


def test_recurrent_answers_from_the_edited_prompt():
    global server
    server = mamba_130m()
    log = _start_with_log(server)
    boot = log.drain()
    assert LOAD_REFUSAL not in boot, "recurrent model refused --cache-reuse at load time"

    _complete(server, P1 + P2 + P3 + Q, cache_prompt=True)
    log.drain()

    reused = _complete(server, P1 + P3 + Q, cache_prompt=True)
    tail = log.drain()

    oracle = _complete(server, P1 + P3 + Q, cache_prompt=False)

    # what the model says comes first: this is the assertion the bug fails
    assert "zebra" not in reused["content"], f"answered from the deleted text: {reused['content']!r}"
    assert reused["content"] == oracle["content"], \
        f"reuse-on says {reused['content']!r}, from-scratch says {oracle['content']!r}"

    # and the path that produced it: reuse was offered, the chunk matched, and nothing was shifted
    assert REQUEST_REFUSAL not in tail, "recurrent model refused the per-request n_cache_reuse field"
    assert REBUILT in tail, "the matching chunk was not seen by the reuse loop; the test did not exercise the path"
    assert SHIFTED not in tail, "recurrent state was shifted with the KV"


def test_attention_keeps_cache_reuse():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 2048
    server.n_slots = 1
    server.n_batch = 64
    log = _start_with_log(server)
    boot = log.drain()
    assert LOAD_REFUSAL not in boot, "attention-only model lost --cache-reuse"

    p1 = "Once upon a time there was a little girl named Lily who liked to play in the park. " * 4
    p2 = "The dog was named Max and Max was a big red dog. " * 8
    p3 = "Lily and her mom went to the store to buy some milk and bread for dinner. " * 5
    _complete(server, p1 + p2 + p3, cache_prompt=True)
    log.drain()
    _complete(server, p1 + p3, cache_prompt=True)
    tail = log.drain()
    assert REQUEST_REFUSAL not in tail, "attention-only model lost per-request n_cache_reuse"
    assert SHIFTED in tail, "attention-only model no longer shifts a matching chunk"

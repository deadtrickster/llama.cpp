import os
import re
import tempfile
import pytest
from utils import *

# A recurrent cache cannot shift positions: seq_add moves cell.pos and leaves the recurrent
# state untouched, so --cache-reuse on such a model restores state computed over a history
# the client no longer has (GLM-TODO T3.5). The server must refuse cache reuse for it at
# load time AND ignore the per-request n_cache_reuse field, which bypasses the launcher flag.
# Both gates key on llama_memory_can_shift(), which is what this test observes through the
# server log - a pure attention model is the control and must keep both paths open.

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
    server.n_ctx = 2048
    server.n_batch = 64
    server.n_slots = 1
    server.n_predict = 4
    server.seed = 42
    server.temperature = 0.0
    return server


PROMPT = ("The quick brown fox jumps over the lazy dog. " * 12).strip()

LOAD_REFUSAL     = "cache_reuse is not supported by this context, it will be disabled"
REQUEST_REFUSAL  = "cache reuse is not supported - ignoring n_cache_reuse"


def _start_with_log(srv: ServerProcess) -> LogReader:
    srv.debug = True
    fd, srv.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    srv.n_cache_reuse = 256
    srv.start(timeout_seconds=300)
    return LogReader(srv.log_path)


def _two_requests(srv: ServerProcess) -> None:
    # first fills the cache, second - with the launcher flag possibly disabled - asks for
    # reuse per request, which is the door --cache-reuse 0 alone would leave open
    for _ in range(2):
        res = srv.make_request("POST", "/completion", data={
            "prompt": PROMPT,
            "n_predict": 2,
            "cache_prompt": True,
            "n_cache_reuse": 256,
        })
        assert res.status_code == 200, res.body


def test_recurrent_refuses_cache_reuse():
    global server
    server = mamba_130m()
    log = _start_with_log(server)
    boot = log.drain()
    assert LOAD_REFUSAL in boot, "recurrent model accepted --cache-reuse at load time"

    _two_requests(server)
    assert REQUEST_REFUSAL in log.drain(), "recurrent model honoured the per-request n_cache_reuse field"


def test_attention_keeps_cache_reuse():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 2048
    server.n_slots = 1
    server.n_batch = 64
    log = _start_with_log(server)
    boot = log.drain()
    assert LOAD_REFUSAL not in boot, "attention-only model lost --cache-reuse"

    _two_requests(server)
    assert REQUEST_REFUSAL not in log.drain(), "attention-only model lost per-request n_cache_reuse"

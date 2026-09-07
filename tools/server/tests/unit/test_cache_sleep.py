import time

import pytest
from utils import *

# --sleep-idle-seconds tears the llama_context down and rebuilds it on the next
# request. These tests are about what a single conversation - one that lives
# only in a slot, never pushed into the prompt cache by a second one - goes
# through on that path. test_cache_persist.py covers the two-conversation shape
# where the first is already in the cache before the sleep.

PROMPT_A = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers."
)


def make_server(cache_dir: str) -> ServerProcess:
    server = ServerPreset.tinyllama2()
    server.n_slots = 1
    server.n_ctx = 512
    server.n_predict = 4
    server.temperature = 0.0
    server.cache_ram = 100
    server.slot_save_path = cache_dir
    server.sleep_idle_seconds = 1
    return server


def complete(server: ServerProcess, prompt: str) -> dict:
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200, f"{res.status_code}: {res.body}"
    return res.body["timings"]


def complete_when_ready(server: ServerProcess, prompt: str, timeout: float = 60.0) -> dict:
    # start() returns once /health answers, which is before the model is loaded;
    # completions answer 503 "Loading model" until then
    start = time.time()
    while True:
        res = server.make_request("POST", "/completion", data={
            "prompt": prompt,
            "id_slot": 0,
            "cache_prompt": True,
        })
        if res.status_code == 200:
            return res.body["timings"]
        assert res.status_code == 503, f"{res.status_code}: {res.body}"
        assert time.time() - start < timeout, "model never finished loading"
        time.sleep(0.1)


def wait_for_sleep(server: ServerProcess, timeout: float = 30.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        res = server.make_request("GET", "/props")
        assert res.status_code == 200
        if res.body["is_sleeping"]:
            return
        time.sleep(0.1)
    pytest.fail("server did not go to sleep")


def test_single_conversation_survives_sleep_wake(tmp_path):
    # T1.2: a conversation reaches the prompt cache lazily, when a later task
    # pushes it out of its slot. With one conversation nothing ever pushes it, so
    # it is still only in the slot when the server sleeps. Sleep must save the
    # slots before it frees the context, the way exit already does.
    server = make_server(str(tmp_path))
    server.start()
    try:
        t_a = complete_when_ready(server, PROMPT_A)
        n_prompt_a = t_a["prompt_n"] + t_a["cache_n"]
        assert t_a["cache_n"] == 0

        wait_for_sleep(server)

        # this request wakes the model back up; the slot was rebuilt empty on
        # reload, so a cache hit here can only have come from the sleep flush
        t_a2 = complete(server, PROMPT_A)
    finally:
        server.stop()

    # the final token is always reprocessed
    assert t_a2["cache_n"] == n_prompt_a - 1, f"conversation lost on sleep: {t_a2}"
    assert t_a2["prompt_n"] == 1

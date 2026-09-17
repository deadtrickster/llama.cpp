import glob
import os
import time

import pytest
from utils import *

# The prompt cache spills to disk when the server shuts down and indexes what it
# finds there when it starts. These tests are about the seam between two server
# processes, so each one starts a server, stops it, and starts another on the
# same --slot-save-path.

PROMPT_A = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers."
)

PROMPT_B = (
    "In a small village by the sea, an old fisherman mended his nets "
    "every morning and told the children stories about the storms he "
    "had survived and the islands he had never quite managed to reach."
)


def make_server(cache_dir: str, n_ctx: int = 512) -> ServerProcess:
    server = ServerPreset.tinyllama2()
    server.n_slots = 1
    server.n_ctx = n_ctx
    server.n_predict = 4
    server.temperature = 0.0
    server.cache_ram = 100
    server.slot_save_path = cache_dir
    return server


def spill_files(cache_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(cache_dir, "l2p-*.spill")))


def complete(server: ServerProcess, prompt: str) -> dict:
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    return res.body["timings"]


def test_prompt_cache_hit_after_restart(tmp_path):
    cache_dir = str(tmp_path)

    server = make_server(cache_dir)
    server.start()
    try:
        # PROMPT_A is processed in full, then pushed out of the slot by PROMPT_B,
        # which is what puts it into the prompt cache
        t_a = complete(server, PROMPT_A)
        n_prompt_a = t_a["prompt_n"] + t_a["cache_n"]
        assert t_a["cache_n"] == 0

        complete(server, PROMPT_B)
    finally:
        server.stop()

    # a clean shutdown must leave the cache on disk, not delete it
    files = spill_files(cache_dir)
    assert len(files) > 0, f"no spill files in {cache_dir}: {os.listdir(cache_dir)}"

    server2 = make_server(cache_dir)
    server2.start()
    try:
        t_a2 = complete(server2, PROMPT_A)
    finally:
        server2.stop()

    # the whole point: the new process reused the old process's KV instead of
    # reprocessing the prompt. The final token is always reprocessed.
    assert t_a2["cache_n"] == n_prompt_a - 1
    assert t_a2["prompt_n"] == 1


def test_prompt_cache_not_shared_between_model_configs(tmp_path):
    cache_dir = str(tmp_path)

    server = make_server(cache_dir)
    server.start()
    try:
        t_a = complete(server, PROMPT_A)
        n_prompt_a = t_a["prompt_n"] + t_a["cache_n"]
        complete(server, PROMPT_B)
    finally:
        server.stop()

    assert len(spill_files(cache_dir)) > 0

    # a different context size means a differently shaped sequence state. The
    # bytes on disk would still load; they would just be wrong. The model key in
    # the file name is what stops that, so the entries must not even be indexed.
    server2 = make_server(cache_dir, n_ctx=256)
    server2.start()
    try:
        t_a2 = complete(server2, PROMPT_A)
    finally:
        server2.stop()

    assert t_a2["cache_n"] == 0
    assert t_a2["prompt_n"] == n_prompt_a

    # and the other config's files are still there, untouched
    assert len(spill_files(cache_dir)) > 0


def test_prompt_cache_survives_sleep_wake(tmp_path):
    # the motivating case: --sleep-idle-seconds unloads the model between two
    # models sharing one box, and that goes through the same destroy() path as
    # shutting down. Before this, every conversation lost its KV on every swap.
    cache_dir = str(tmp_path)

    server = make_server(cache_dir)
    server.sleep_idle_seconds = 1
    server.start()
    try:
        t_a = complete(server, PROMPT_A)
        n_prompt_a = t_a["prompt_n"] + t_a["cache_n"]
        complete(server, PROMPT_B)

        start = time.time()
        while time.time() - start < 30.0:
            res = server.make_request("GET", "/props")
            assert res.status_code == 200
            if res.body["is_sleeping"]:
                break
            time.sleep(0.1)
        else:
            pytest.fail("server did not go to sleep")

        assert len(spill_files(cache_dir)) > 0

        # this request wakes the model back up
        t_a2 = complete(server, PROMPT_A)
    finally:
        server.stop()

    assert t_a2["cache_n"] == n_prompt_a - 1


def test_restored_conversation_survives_a_crash(tmp_path):
    """A conversation restored from disk lived only in its slot: the restore
    was a move and the file was unlinked. Measured 2026-09-17 on GLM: a
    144,700-token conversation came back from disk at 22:42, the server
    crashed at 22:46, and its next turn found an 88k-token snapshot - the
    60-second mirror writes cache entries, never live slots. The file stays
    now: a restore is a copy, and the disk-only index entry that names the
    file is released only once a longer snapshot of the conversation is on
    disk (a later mirror, an eviction, a clean stop)."""
    cache_dir = str(tmp_path)

    server = make_server(cache_dir)
    server.start()
    try:
        t_a = complete(server, PROMPT_A)
        n_prompt_a = t_a["prompt_n"] + t_a["cache_n"]
        complete(server, PROMPT_B)          # pushes A into the prompt cache
    finally:
        server.stop()                       # a clean stop spills A to disk
    assert spill_files(cache_dir), "precondition: nothing on disk"

    # A comes back from disk and keeps generating in its slot ...
    server2 = make_server(cache_dir)
    server2.start()
    t_a2 = complete(server2, PROMPT_A)
    assert t_a2["cache_n"] == n_prompt_a - 1, "precondition: A did not come back from disk"
    # ... and the server dies with A live in the slot: no flush, no spill, no mirror
    server2.process.kill()
    server2.process.wait(timeout=10)
    server2.process = None

    assert spill_files(cache_dir), "the restore consumed the only copy of the conversation on disk"

    server3 = make_server(cache_dir)
    server3.start()
    try:
        t_a3 = complete(server3, PROMPT_A)
    finally:
        server3.stop()

    assert t_a3["cache_n"] == n_prompt_a - 1, \
        f"A was prefilled from scratch after the crash: prompt_n={t_a3['prompt_n']}, cache_n={t_a3['cache_n']}"

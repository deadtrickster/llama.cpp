import threading
import pytest
from utils import *

# We use a F16 MOE gguf as main model, and q4_0 as draft model

server = ServerPreset.stories15m_moe()

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"

def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    # set default values
    server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 4
    server.spec_draft_n_max = 8
    server.fa = "off"


@pytest.fixture(autouse=True)
def fixture_create_server():
    return create_server()


def test_with_and_without_draft():
    global server
    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.2,
        "top_k": 5,
        "seed": 4242,
        "n_predict": 16,
        "return_tokens": True,
    }

    server.model_draft = None  # disable draft model
    server.spec_type = None
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    tokens_no_draft = res.body["tokens"]
    server.stop()

    # create new server with draft model
    create_server()
    server.start()
    res = server.make_request("POST", "/completion", data=request)
    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    tokens_draft = res.body["tokens"]

    assert tokens_no_draft == tokens_draft

    server.stop()
    create_server()
    assert server.spec_draft_n_max is not None
    server.spec_synth_rates = [0.0] * server.spec_draft_n_max
    server.start()
    res = server.make_request("POST", "/completion", data=request)

    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["timings"]["draft_n_accepted"] == 0
    assert res.body["tokens"] == tokens_no_draft


def test_different_draft_min_draft_max():
    global server
    test_values = [
        (1, 2),
        (1, 4),
        (4, 8),
        (4, 12),
        (8, 16),
    ]
    last_content = None
    for draft_min, draft_max in test_values:
        server.stop()
        server.spec_draft_n_min = draft_min
        server.spec_draft_n_max = draft_max
        server.start()
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 16,
        })
        assert res.status_code == 200
        if last_content is not None:
            assert last_content == res.body["content"]
        last_content = res.body["content"]


def test_synth_is_deterministic():
    global server
    assert server.spec_draft_n_max is not None
    server.spec_synth_rates = [0.75 ** (i + 1) for i in range(server.spec_draft_n_max)]
    server.start()

    request = {
        "prompt": "I believe the meaning of life is",
        "temperature": 0.2,
        "top_k": 5,
        "seed": 4242,
        "n_predict": 32,
    }
    responses = [server.make_request("POST", "/completion", data=request) for _ in range(2)]

    for res in responses:
        assert res.status_code == 200
        assert res.body["timings"]["draft_n"] > 0
    assert responses[0].body["timings"]["draft_n"] == responses[1].body["timings"]["draft_n"]
    assert responses[0].body["timings"]["draft_n_accepted"] == responses[1].body["timings"]["draft_n_accepted"]


def test_synth_ignores_target_tokens():
    global server
    assert server.spec_draft_n_max is not None
    server.spec_synth_rates = [1.0] * server.spec_draft_n_max
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "seed": 4242,
        "n_predict": 32,
    })

    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["timings"]["draft_n_accepted"] == res.body["timings"]["draft_n"]

    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "temperature": 0.0,
        "seed": 4242,
        "n_predict": 6,
        "grammar": 'root ::= "a"{5,5}',
    })
    assert res.status_code == 200, res.body

    res = server.make_request("POST", "/completion", data={
        "prompt": "Respond with only: OK",
        "temperature": 0.0,
        "seed": 4242,
        "n_predict": 64,
        "ignore_eos": True,
    })
    assert res.status_code == 200, res.body
    assert res.body["tokens_predicted"] == 64
    assert res.body["stop_type"] == "limit"


def test_slot_ctx_not_exceeded():
    global server
    server.n_ctx = 256
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0


def test_with_ctx_shift():
    global server
    server.n_ctx = 256
    server.enable_ctx_shift = True
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Hello " * 248,
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 256,
        "speculative.p_min": 0.0,
    })
    assert res.status_code == 200
    assert len(res.body["content"]) > 0
    assert res.body["tokens_predicted"] == 256
    assert res.body["truncated"] == True


@pytest.mark.parametrize("n_slots,n_requests", [
    (1, 2),
    (2, 2),
])
def test_multi_requests_parallel(n_slots: int, n_requests: int):
    global server
    server.n_slots = n_slots
    server.start()
    tasks = []
    for _ in range(n_requests):
        tasks.append((server.make_request, ("POST", "/completion", {
            "prompt": "I believe the meaning of life is",
            "temperature": 0.0,
            "top_k": 1,
        })))
    results = parallel_function_calls(tasks)
    for res in results:
        assert res.status_code == 200
        assert match_regex("(wise|kind|owl|answer)+", res.body["content"])


@pytest.mark.xfail(strict=False, reason=(
    "the halving retry under pool pressure drops n_batch BELOW a speculative group's size (8 < 9 here; "
    "2 < 3 and 4->2 < 3 in the two production aborts of 2026-09-18), the cut cannot keep the group whole, "
    "and sampling reads a group index whose batch slot carries no logits: get_logits_ith 'invalid logits "
    "id N' -> GGML_ASSERT(logits != nullptr) in sampling.cpp:154, the server dies. Reproduces ~2 of 3 runs "
    "with pool_cells_free() from the memory, ~1 of 3 with the old held-count (a thread-timing race decides "
    "whether both groups share the batch at the wall). Removed by admission-before-decode, "
    "tools/server/POOL-SCHEDULER.md stage 2, which deletes the halving loop. Flips to strict pass there."))
def test_sub_batch_cut_keeps_a_later_draft_group_whole():
    """Two slots each verifying a draft group (1 sampled token + n_max drafts) on a
    pool that runs out: llama_decode refuses the 18-token batch, the retry halves
    n_batch, and the cut lands between the two groups - [0, 9) is decoded, slot
    1's group [9, 18) waits for the next sub-batch. post_decode used to refuse
    that ("speculative batch index 9 is not inside the current sub-batch
    [0, 9)"), a 500 for the slot whose group was simply LATER, not split.
    Measured 2026-09-17 on GLM: a 9-token batch, n_batch halved to 4, slot 1
    answered 500 mid-turn. A group entirely outside the view is somebody else's
    sub-batch; only a split group is an error."""
    global server
    create_server()
    server.n_slots = 2
    server.n_ctx = 256              # stories15M trains at 256: the pool is 256 cells, two seats of up to 256
    server.kv_unified = True
    server.kv_unified_per_slot = 256
    server.pool_static = True
    server.n_predict = 4096
    server.spec_synth_rates = [1.0] * server.spec_draft_n_max   # every draft accepted: full groups every step
    server.start()

    # ~19 tokens each plus 140 generated: 318 cells on a 256-cell pool
    prompt_a = "Alpha. " + " ".join(f"a{i}" for i in range(8))
    prompt_b = "Bravo. " + " ".join(f"b{i}" for i in range(8))
    out = {}

    def go(k, prompt, id_slot):
        try:
            out[k] = server.make_request("POST", "/completion", data={
                "prompt": prompt, "id_slot": id_slot, "n_predict": 140, "ignore_eos": True,
                "temperature": 0.0, "cache_prompt": True}, timeout=300)
        except Exception as e:
            out[k] = e

    ta = threading.Thread(target=go, args=("A", prompt_a, 0))
    tb = threading.Thread(target=go, args=("B", prompt_b, 1))
    ta.start(); tb.start()
    ta.join(timeout=300); tb.join(timeout=300)

    for k in "AB":
        assert k in out and not isinstance(out[k], Exception), f"{k}: {out.get(k)!r}"
        assert out[k].status_code == 200, f"{k}: {out[k].body}"
        assert "speculative batch index" not in str(out[k].body)
        assert out[k].body["timings"]["predicted_n"] == 140, f"{k} was cut short: {out[k].body['timings']}"

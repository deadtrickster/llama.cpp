import base64
import os

import pytest
import requests

from utils import *

#
# Speculative decoding (MTP / EAGLE3) on a prompt that contains an image.
#
# An image reaches the target as an embedding batch. The MTP and EAGLE3 drafts
# cannot consume one and skip it, so the target advances over the image's
# positions and the draft does not. The next text batch then starts past a hole
# in the draft's KV, and what happens next depends on the DRAFT context's rope
# type rather than on anything about the request:
#
#   n_pos_per_embd == 1 (no rope)  the batch is rejected outright, llama_decode
#                                  returns -1 and the server answers HTTP 500
#                                  "failed to process speculative batch"
#   n_pos_per_embd == 4 (M-RoPE)   the batch is accepted as a forward jump, the
#                                  hole stays, and draft acceptance around the
#                                  image quietly degrades
#
# Both are wrong; only the first is loud. These tests cover both halves.
#
# GATE: none of the models in the standard test set carries both an MTP/EAGLE3
# draft head and a vision projector, so these tests are opt-in. Point them at a
# model that has one:
#
#   LLAMA_TEST_SPEC_MTMD_HF_REPO   HF repo id, e.g. "ggml-org/Qwen3.6-27B-GGUF:Q8_0"
#                                  (the ":quant" form lets the registry supply the mmproj)
#   LLAMA_TEST_SPEC_MTMD_HF_FILE   optional, a specific file within the repo
#   LLAMA_TEST_SPEC_MTMD_MMPROJ    optional, a URL for the projector when the
#                                  registry does not provide one
#   LLAMA_TEST_SPEC_MTMD_DRAFT     optional, a draft model file (required for draft-eagle3)
#   LLAMA_TEST_SPEC_MTMD_TYPE      "draft-mtp" (default) or "draft-eagle3"
#   LLAMA_TEST_SPEC_MTMD_NGL       optional, layers to offload (default 99)
#

HF_REPO = os.environ.get("LLAMA_TEST_SPEC_MTMD_HF_REPO")
HF_FILE = os.environ.get("LLAMA_TEST_SPEC_MTMD_HF_FILE")
MMPROJ = os.environ.get("LLAMA_TEST_SPEC_MTMD_MMPROJ")
DRAFT = os.environ.get("LLAMA_TEST_SPEC_MTMD_DRAFT")
SPEC_TYPE = os.environ.get("LLAMA_TEST_SPEC_MTMD_TYPE", "draft-mtp")
NGL = int(os.environ.get("LLAMA_TEST_SPEC_MTMD_NGL", "99"))

pytestmark = pytest.mark.skipif(
    HF_REPO is None,
    reason="set LLAMA_TEST_SPEC_MTMD_HF_REPO to a model that has both an MTP/EAGLE3 head and an mmproj",
)

IMG_URL_CAT = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"
IMG_URL_TRUCK = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"

server: ServerProcess


def _img_base64(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    return base64.b64encode(response.content).decode("utf-8")


@pytest.fixture(autouse=True)
def create_server(tmp_path):
    global server
    os.environ["LLAMA_MEDIA_MARKER"] = "<__media__>"
    server = ServerProcess()
    server.offline = False
    server.model_hf_repo = HF_REPO
    server.model_hf_file = HF_FILE
    server.model_alias = "spec-mtmd"
    server.mmproj_url = MMPROJ
    server.model_draft = DRAFT
    server.spec_type = SPEC_TYPE
    server.spec_draft_n_max = 3
    server.n_gpu_layer = NGL
    server.n_ctx = 8192
    server.n_batch = 512
    server.n_slots = 1
    server.n_predict = 32
    server.temperature = 0.0
    server.seed = 42
    server.slot_save_path = str(tmp_path)
    server.server_slots = True
    # The development box runs an unrelated production server on the harness
    # default port; a stray request answered by it would silently pass.
    server.server_port = 8300


def _completion(prompt, **extra):
    data = {
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 32,
        "cache_prompt": True,
        "prompt": prompt,
        **extra,
    }
    return server.make_request("POST", "/completions", data=data)


def _image_prompt(url: str) -> dict:
    # an image in the MIDDLE of the prompt, so text follows it and the draft has
    # to decode a batch whose positions start past the image
    return {
        "prompt_string": "Look at this picture: <__media__>\nDescribe what you see in one sentence.",
        "multimodal_data": [_img_base64(url)],
    }


def test_spec_image_mid_conversation_answers():
    """(a) an image mid-prompt must answer, not 500.

    Before the resync this returns HTTP 500 "failed to process speculative
    batch" on any draft whose context has no rope, which is every MTP draft
    exported from a target without rope (e.g. glm5next)."""
    global server
    server.start()

    res = _completion(_image_prompt(IMG_URL_CAT))
    assert res.status_code == 200, res.body
    assert len(res.body["content"]) > 0

    # a second, different image on the same slot exercises the resync a second
    # time, now on top of a draft cache that was already cleared once
    res = _completion(_image_prompt(IMG_URL_TRUCK))
    assert res.status_code == 200, res.body
    assert len(res.body["content"]) > 0


def test_spec_image_matches_no_spec():
    """The target verifies every drafted token, so the answer must not depend on
    whether speculation is on."""
    global server
    server.spec_type = None
    server.model_draft = None
    server.start()
    res = _completion(_image_prompt(IMG_URL_CAT), return_tokens=True)
    assert res.status_code == 200, res.body
    tokens_no_spec = res.body["tokens"]
    server.stop()

    server.spec_type = SPEC_TYPE
    server.model_draft = DRAFT
    server.start()
    res = _completion(_image_prompt(IMG_URL_CAT), return_tokens=True)
    assert res.status_code == 200, res.body
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["tokens"] == tokens_no_spec


def test_spec_acceptance_near_image():
    """(b) the SILENT half of the bug.

    On a draft whose context uses M-RoPE the hole is tolerated rather than
    rejected, so the request answers either way and the only visible symptom is
    the acceptance rate. Compare generation right after an image against
    steady-state text on the same server: without the resync the draft attends
    across a hole and its accepted fraction collapses.

    Note what this asserts and what it does not. A single run can only show that
    acceptance near an image is in the same ballpark as acceptance on text; the
    claim that it IMPROVED is a before/after comparison and has to be made by
    running this test on both builds and comparing the printed rates."""
    global server
    server.start()

    res = _completion("Describe a cat in one sentence.")
    assert res.status_code == 200, res.body
    draft_n_text = res.body["timings"]["draft_n"]
    accepted_text = res.body["timings"]["draft_n_accepted"]
    assert draft_n_text > 0, "the draft never ran on a plain text prompt"
    rate_text = accepted_text / draft_n_text

    res = _completion(_image_prompt(IMG_URL_CAT))
    assert res.status_code == 200, res.body
    draft_n_img = res.body["timings"]["draft_n"]
    accepted_img = res.body["timings"]["draft_n_accepted"]
    assert draft_n_img > 0, "the draft never ran after an image"
    rate_img = accepted_img / draft_n_img

    print(f"acceptance: text={rate_text:.3f} after-image={rate_img:.3f}")

    assert rate_img > 0.0
    assert rate_img >= 0.5 * rate_text


def test_spec_slot_save_restore_with_image():
    """(c) the draft's KV is saved and restored separately from the target's.

    After the resync the draft's saved range is a SUFFIX of the target's: it
    starts just after the last media span instead of at position 0. A restore
    followed by a truncation to the common prefix must still leave the two
    contexts consistent, whether the prefix lands after the image (draft ends at
    p0-1) or inside it (draft is emptied)."""
    global server
    server.swa_full = True
    server.start()

    prompt = _image_prompt(IMG_URL_CAT)

    res = _completion(prompt, id_slot=0)
    assert res.status_code == 200, res.body
    content = res.body["content"]
    prompt_n_full = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "spec_mtmd.bin"})
    assert res.status_code == 200, res.body
    n_saved = res.body["n_saved"]

    res = server.make_request("POST", "/slots/0?action=erase")
    assert res.status_code == 200, res.body

    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "spec_mtmd.bin"})
    assert res.status_code == 200, res.body
    assert res.body["n_restored"] == n_saved

    # continue the restored conversation: the prefix lands AFTER the image
    res = _completion(prompt, id_slot=0)
    assert res.status_code == 200, res.body
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["content"] == content

    # and now a prompt whose common prefix ends INSIDE the image, which is the
    # case that could re-trip the draft's position check
    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "spec_mtmd.bin"})
    assert res.status_code == 200, res.body

    res = _completion(_image_prompt(IMG_URL_TRUCK), id_slot=0)
    assert res.status_code == 200, res.body
    assert len(res.body["content"]) > 0

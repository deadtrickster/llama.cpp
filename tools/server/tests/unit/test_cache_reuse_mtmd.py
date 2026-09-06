from utils import *

# cache reuse used to be disabled outright whenever a multimodal projector was
# loaded, even for prompts that contain no media at all. the gate is now on the
# CONTENT of the prompt, so text-only turns on a multimodal server get reuse and
# turns containing media fall back to plain prefix matching.

server: ServerProcess

IMG_URL_0 = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"

DEFAULT_PORT = 8200


def server_port() -> int:
    # PORT is set per xdist worker by the session fixture, so read it late, not at
    # import time. never run on 8080: a foreign server there answers /health with
    # 200 and start() cannot tell it apart from ours, which voids every result
    port = int(os.environ.get("PORT", DEFAULT_PORT))
    assert port != 8080, "refusing to run on the default port, set PORT"
    return port


N_CACHE_REUSE = 32


def make_server(n_cache_reuse: int | None) -> ServerProcess:
    os.environ['LLAMA_MEDIA_MARKER'] = '<__media__>'
    sp = ServerPreset.tinygemma3()
    sp.server_port  = server_port()
    sp.offline      = False  # the blobs may not be pre-downloaded
    sp.n_ctx        = 2048
    sp.n_slots      = 1      # both turns must land in the same slot
    # gemma3 is a SWA model: without --swa-full the KV cache below the sliding
    # window is dropped and the server re-processes the whole prompt regardless
    # of cache reuse, which would make this test measure SWA rather than reuse
    sp.swa_full     = True
    sp.n_predict    = 8
    sp.n_cache_reuse = n_cache_reuse
    return sp


# the two prompts share a long PREFIX and a long SUFFIX, and the second one drops
# a block from the middle. an exact-prefix match can only reuse the PREFIX; cache
# reuse can also shift the SUFFIX down and reuse it.
# note: the reuse loop only advances head_c, so it re-aligns on a deletion, not on
# an insertion or a same-length replacement
PREFIX = "".join(f"line {i} of the header\n"  for i in range(30))
MIDDLE = "".join(f"line {i} of the middle\n"  for i in range(30))
SUFFIX = "".join(f"line {i} of the trailer\n" for i in range(120))

PROMPT_LONG  = PREFIX + MIDDLE + SUFFIX
PROMPT_SHORT = PREFIX + SUFFIX


def complete(prompt: str, extra: dict | None = None) -> dict:
    data = {
        "prompt": prompt,
        "temperature": 0.0,
        "top_k": 1,
        "n_predict": 4,
        "cache_prompt": True,
    }
    if extra:
        data.update(extra)
    res = server.make_request("POST", "/completion", data=data)
    assert res.status_code == 200, res.body
    return res.body["timings"]


def test_cache_reuse_text_only_prompt_on_multimodal_server():
    """(a) text-only turns diverging MID-history must reuse the tail.

    Fails before the content-based gate: with an mmproj loaded, --cache-reuse
    was reset to 0 at load time, so only the common prefix was reused."""
    global server
    server = make_server(N_CACHE_REUSE)
    server.start()

    t1 = complete(PROMPT_LONG)
    print(f"first turn: cache_n={t1['cache_n']} prompt_n={t1['prompt_n']}")

    t2 = complete(PROMPT_SHORT)
    n_total = t2["cache_n"] + t2["prompt_n"]
    print(f"second turn: cache_n={t2['cache_n']} prompt_n={t2['prompt_n']} n_total={n_total}")

    # the suffix is ~4x the prefix, so without reuse prompt_n is most of the prompt
    assert t2["cache_n"] > 0.8 * n_total
    assert t2["prompt_n"] < 0.2 * n_total


def test_cache_reuse_falls_back_for_prompt_with_image():
    """(b) a prompt containing media must still answer and must not crash."""
    global server
    server = make_server(N_CACHE_REUSE)
    server.start()

    # warm the slot with a text-only turn first, so the reuse path is live
    complete(PROMPT_LONG)

    res = server.make_request("POST", "/chat/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "What is this:\n"},
                {"type": "image_url", "image_url": {"url": IMG_URL_0}},
            ]},
        ],
    })
    assert res.status_code == 200, res.body
    choice = res.body["choices"][0]
    assert "assistant" == choice["message"]["role"]
    assert match_regex("(cat)+", choice["message"]["content"])

    # the reuse path must still work after the image turn, and not abort
    complete(PROMPT_LONG)
    t = complete(PROMPT_SHORT)
    n_total = t["cache_n"] + t["prompt_n"]
    print(f"after image: cache_n={t['cache_n']} prompt_n={t['prompt_n']} n_total={n_total}")
    assert t["cache_n"] > 0.8 * n_total


def test_cache_reuse_from_request_body_on_multimodal_server():
    """(c) "n_cache_reuse" in the request BODY bypasses the load-time default
    (server-schema.cpp), so it reaches the reuse path even with no --cache-reuse
    on the command line. It must not abort, and it must actually reuse."""
    global server
    server = make_server(None)  # no --cache-reuse on the command line
    server.start()

    extra = {"n_cache_reuse": N_CACHE_REUSE}

    complete(PROMPT_LONG, extra)

    t2 = complete(PROMPT_SHORT, extra)
    n_total = t2["cache_n"] + t2["prompt_n"]
    print(f"body n_cache_reuse: cache_n={t2['cache_n']} prompt_n={t2['prompt_n']} n_total={n_total}")

    assert t2["cache_n"] > 0.8 * n_total
    assert t2["prompt_n"] < 0.2 * n_total

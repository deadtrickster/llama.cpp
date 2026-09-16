# glm-all measurement scripts

The scripts behind the numbers in the top-level README's glm-all section. None depends on this box;
each takes its inputs on the command line.

| script | what |
|---|---|
| `nsys-step-anatomy.py` | per-verify-step anatomy of a decode from an `nsys` recording: busy time per device, idle gaps by device handoff, kernel time by class, launches per step |
| `ppl-paired.py` | paired per-chunk ΔNLL between two `llama-perplexity` logs over the same text (the text variance cancels; the tool's ± does not) |
| `logprob-diff.py` | first-token top-k logprob comparison between two `/completion` responses (`n_probs`), the token-level view of a numerical change |
| `greedy-compare.sh` | temperature-0 completion against a running server, sha of the text vs a saved reference; the bit-exactness gate for changes that should not move numerics |
| `bw.cu` | host↔device bandwidth and small-transfer latency per CUDA device; what a slow link actually delivers |

Recording a decode for `nsys-step-anatomy.py` (the server must be started under `nsys launch`; it cannot attach):

    nsys launch --session-new=glm -t cuda,osrt --cuda-graph-trace=node llama-server ...
    nsys start --session=glm -o decode      # after a warm-up request
    ... one decode request ...
    nsys stop  --session=glm
    nsys stats --force-export=true -q --report cuda_gpu_kern_sum decode.nsys-rep   # produces decode.sqlite
    python3 nsys-step-anatomy.py decode.sqlite

Check the server is idle around the recorded request (`/metrics`, `requests_processing`); a recording that overlaps
another client's prefill is not a decode profile.

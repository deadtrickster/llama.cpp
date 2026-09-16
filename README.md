# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## glm-all: GLM-5.3-Flash serving changes

This branch carries the GLM-5.3-Flash (`glm5next`) port plus the serving changes below.
Hardware is described only as "2 cards" (the two main GPUs), "3 cards" (plus a small third GPU over a slow link) and "CPU" (expert weights in host RAM).
Commit hashes are listed so any row can be bisected; each reason is one sentence about the mechanism.

| area | commits | why | effect |
|---|---|---|---|
| prompt-cache restore actually restores | `f86781985` `f7105d5d1` `1a5cc9b73` | disk restore dropped checkpoints so hybrid models re-prefilled everything; a failed save wiped the slot; `-1` meant a tighter limit | a restore costs the tail after the last checkpoint, not the prompt |
| KV state buffers | `362322b03` `c8af5f217` | pinned, pooled state buffers let the driver DMA directly; checkpoints are immutable, so share them copy-on-write | slot swaps at DMA speed, no duplicate checkpoint memory |
| cache save/load decisions | `e969bbcc9` `78bde7e63` `185f33458` `3e14258c5` | one `f_keep` boolean gated save and load and picked by list order; save on loss, always consult, longest prefix wins | similar conversations no longer destroy each other's tails |
| disk tier: durable, self-describing spill files | `28b747fe8` `b2393a9b9` `7bed071c3` `2b998a669` | files named by model key and token hash carry a header, so a restart or another process can index them | the cache survives a restart |
| disk tier: nothing lost on the way down | `f6e221927` `1597db29c` `427180c74` `3b7236a7d` `3c8b5b37a` | destructors, sleeps and router kills each lost the cache; spill explicitly, save live slots first, size the stop timeout | eviction, sleep and shutdown keep every conversation |
| disk tier: write-through mirror and reaping | `5547557c3` `e1046ea07` `c7834ad8d` `332ecdfd8` `5c3f2b860` `ccdb386d8` `bf91ea247` `1804d5374` | a timed write-through mirror survives a hard kill; superseded snapshots reap first; sleep only under RAM pressure (`1804d5374` untested draft) | a crash loses seconds, the disk holds one snapshot per conversation |
| speculative decoding around images | `04eccf41e` `13e4408f4` `7504bd2ef` `8dab9d5fb` `eb2165791` `99ac96151` `4ae930d8b` | the draft cannot consume an embedding batch and fell behind the target: HTTP 500 or silent acceptance collapse; resync it | image turns answer; acceptance recovers after the image |
| cache reuse with a projector loaded | `083c04652` `e8169aeb1` `4bb229414` `83d0527c6` | reuse was gated on "projector present" instead of "prompt has media", so text turns on a vision server never reused | text turns reuse; media prompts keep exact-prefix matching |
| decode preemption | `1b0d23781` `1ac1b607e` `7571942dc` `7bc92d229` `20ad100d4` `4a9d9a1b9` `55e6b39bf` `23abf9f04` | run-to-completion let one deep reasoning turn hold a seat for minutes; slots suspend at a quantum and resume | a cheap turn no longer waits behind a long one |
| one elastic KV pool | `5a29b50f9` `052b09c8f` `76691521c` `f62ac65c6` `1f613518e` `e0dd42e00` `a6696fffd` `a23b7f524` `86894272d` `076a53244` `981be4145` `906f84951` `f52da8fc4` `c78a7f531` | cells, sequence ids and seats drew on one memory but cells were fixed; the pool now resizes, seats follow residency | no pre-sizing; whatever fits is resident and batched |
| pool under pressure | `fb106a81a` `fcef1d130` `1397b97d8` `978b4c46e` `a62668208` | a full pool purged conversations unsaved and failed every slot; now one spills, shrinks keep restore room, restores ask first | pressure degrades one conversation instead of losing several |
| memory fit | `257de77d2` `042ea80ea` `ad616d9e4` `0f81548c1` `d936f654c` `783128be7` | user placement made the fitter refuse, the recurrent cache allocated under `no_alloc`, test models could not exercise placement | the fitter sizes the context around whatever the user pinned |
| recurrent state is not a function of position | `ae86607ed` `0a78ba758` `666038904` `31737f195` `675068409` | shifting cells left SSM state stale; invalidate by content instead, checkpoint on a schedule, thin exponentially away from the tip | correct reuse on hybrid models; checkpoints stop dominating the cache |
| streaming and metrics | `ffec7c4fb` `bdb23b440` | a token ending in a partial UTF-8 character emitted no frame; throughput gauges read 0 until a generation ended | every token reaches the client; live rates while generating |
| experts of three MoE layers moved from CPU to the third card | (config) | only activations cross the slow link; the CPU was the bottleneck, not the two cards | decode 45.3 → 72.1 t/s |
| reuse the decode graph | `bf374b4b2` | the pooled-indexer input never answered `can_reuse`, so every step rebuilt, re-allocated and re-captured the graph | 72.1 → 76.7 t/s |
| kernels: split-K, concat, batched gate fusion, hc-coefficient fusion | `950de14d1` `fd7a9c0f1` `b90dc6e66` `bd8ec7021` `655ab8cc0` | a 16384→24 projection ran 12 blocks; a concat idled 250/256 threads; shared expert never fused; four launches per 12 floats | 76.7 → 78.0 → 83.8 t/s (split-K last) |
| memoize the pooled-indexer scan | `5b8f3d874` | the cell→pool scan rescanned the whole pool extent every step (7% of the thread); memoized on a cells generation counter | host work per step no longer grows with pool occupancy |
| checkpoints where conversations actually branch | `50dc7e8c1` `ce4938254` `69c5e707d` | thinning left holes wider than a re-prefill; a restore landing above the branch point re-prefilled the whole tail; MTP entries refused a draft-less server | a branch re-prefills its own tail, not the shared preamble |
| a second model: Qwen3.8-Flash-Next (`qwen4exp`) | `a213bfcc4` | the indexer cache did not follow the elastic pool, so the model asserted at load | loads and serves on the same server build |
| pin the lazy-read table | `be988ee34` | 22 rows of a 54 GB host table were disk reads mid-step; the uploaded GPU weights held the page cache | warm decode 103 → 114 t/s (2 cards, Qwen) |
| the draft context re-captured its CUDA graph every step | `8178a5594` `ac5348ebd` | catch-up and draft graphs shared one cache key; pipelined input copies alternated its first node's address | 2 captures per step gone; mean step 17.4 ms baseline |
| hyper-connection glue fusions | `295905d9f` `ab99e3bf6` `0d4d9e699` | scale→unary→scale, repeat→mul→add and the stream mean were 14 launches per layer on a few kilobytes | 4584 → 3654 launches, 17.40 → 16.52 ms/step |
| pipelined CUDA graph launch | `13bae0325` `eeecf9452` | `cudaGraphLaunch` costs the host 0.13 µs per node while the device waits; launch the split in norm-aligned chunks | 16.47 → 15.95 ms/step; GLM idle decode 83.8 → 85.9 t/s |
| quantize activations once per graph | `8dd776b5f` `a23f63d5d` | the same block input was quantized to q8_1 for each mat-vec that read it; one buffer per tensor per compute, in its own pool | Qwen −118 launches/step, device busy −0.26 ms; bit-identical |
| batched top-k for the sparse indexer | `e56af5620` | one CUB top-k per row was 120k launches per 5k-token prefill; the multi-row radix select with a deterministic gather (the atomic one gave different continuations per run) | GLM 33k prefill 705 → 848 t/s |
| a fusion that was not bit-exact | `1db055c23` | `-use_fast_math` rounds a fused `a * sigmoid(b)` differently from the two kernels on GLM; reverted for 0.04 ms | greedy output identical again |

Decode figures: 33k-token prompt, 200 generated tokens, n=6, idle server, 3 cards; they are this box's numbers, the mechanisms are not.
Qwen figures: 2 cards, MTP draft with 2 drafted tokens (the launcher default; 5 measured 92.8 t/s against 104 at 2 or 3), backend sampling, `--load-mode mmap+mlock`;
"mean step" is the nsys-measured wall time of one verify+draft cycle, which does not depend on what was sampled. Qwen end to end: warm decode 92.8 → 116–120 t/s, 4.65 → 3.9 s/turn.
Every kernel change is checked against the CPU reference in `test-backend-ops` (new MUL_MAT, CONCAT, AFFINE_SIGMOID and MUL_MAT_VEC_FUSION cases).
With split-K disabled (`GGML_CUDA_DISABLE_MMVQ_SPLIT_K=1`) temperature-0 output on the reference prompt is byte-identical through every commit;
split-K changes the summation order, so it is judged on the distribution:

| wikitext-2 test, 4096-token chunks, 8-token ubatches (the mat-vec path) | PPL |
|---|---|
| split-K on, 10 chunks | 2.9700 ± 0.0396 |
| split-K off, 10 chunks | 2.9657 ± 0.0394 |
| paired per-chunk ΔNLL, 10 chunks | +0.0014 ± 0.0031 nats/token (t = 1.49) |
| split-K on, 50 chunks | 3.2292 ± 0.0196 |
| split-K off, 50 chunks | 3.2291 ± 0.0196 |
| paired per-chunk ΔNLL, 50 chunks | +0.00003 ± 0.00061 nats/token (t = 0.05; 23 chunks up, 27 down) |

For scale, the same first-token measurement puts the CPU→third-card expert move at a larger perturbation (max Δlogprob 0.44 over the top-50) than split-K (0.33).

The batched top-k changes the order the selected keys are summed in, so it is judged the same way, prefill-shaped (`-ub 512`, the indexer's 512-row case):

| wikitext-2 test, 4096-token chunks, 512-token ubatches | PPL |
|---|---|
| radix top-k, 50 chunks | 3.2347 ± 0.0197 |
| per-row CUB top-k, 50 chunks | 3.2345 ± 0.0197 |
| paired per-chunk ΔNLL, 50 chunks | +0.00006 ± 0.00010 nats/token (t = 0.63; 34 chunks up, 10 down) |

### Before and after, all three models

Same benches throughout: "idle" is a 33k-token prompt on an idle server (prefill and decode), "warm" is four ~35k-token turns of a live conversation, "short" is a 300-token sampled chat answer. "Before" is the launcher default of the morning of the Qwen work (GLM: the row above it); "after" is the pushed build with the launcher defaults it left (n_max 2, CPU sampling, `--load-mode mmap+mlock` for the lazy table).

| model | prefill t/s | idle decode t/s | warm decode t/s / s per turn | short sampled t/s |
|---|---|---|---|---|
| GLM-5.3-Flash, 3 cards | 703 → 848 | 83.8 → 86.8 | (turn-to-turn noise ±15 on this model) | — |
| Qwen3.8-Flash-Next, 2 cards | 1411 → 2243 | ~105 → 129 | 92.8 / 4.65 → 114 / 3.78 | ~80 → 111–118 |
| Qwen3.8-27B dense, 2 cards | 3044 → 3167 | 81.3 → 86.0 | 77.2 / 5.16 → 99 / 4.32 | 57 → 76–88 |

Backend sampling (`--backend-sampling`) is off in the launchers: it buys ~5 t/s of sampled decode and costs 17–25% of prefill on both Qwen models (a sampler graph rides along with every prompt microbatch: Flash 2243 → 1863, dense 3167 → 2364).

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain

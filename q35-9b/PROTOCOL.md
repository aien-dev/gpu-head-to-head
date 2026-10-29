# Qwen3.5-9B on one GB10: our Omega engine, stock llama.cpp on CUDA, and stock llama.cpp on Omega

Version 3, 2026-10-27, frozen before any timed number exists. Changes:
- v1 (sha256 d57ff4dc…) put OURS FP8 against llama.cpp Q8_0. That breaks the first ask in ggml-org/llama.cpp discussion #31092, so v1 is withdrawn.
- v2 made the headline pair BF16 against BF16.
- v3 adds Drake's third configuration (stock llama.cpp on our GPU stack with no NVIDIA CUDA user-space libraries) and an independent correctness oracle (HF transformers).

No timed cell has run under any version. The GPU runs so far were untimed: the BF16 llama.cpp reference capture, the OURS FP8 and BF16 correctness checks, and the HF oracle.

Drake in chat, 2026-10-27:
- "run a test between Llama CPP and our system, Omega CUDA++ ... use the Qwen 3.5 9B model"
- "make sure we are doing it collaboratively in the repo with Llama CPP ... Make sure we're doing what she asked"
- "Omega versus Llama CPP (pure Omega); Pure Llama with CUDA; Omega and Llama together with no CUDA"

## The three configurations
| # | Configuration | Arms | Same-file pairing |
|---|---|---|---|
| 1 | Pure Omega: our own engine, no llama.cpp, no NVIDIA user-space libs | OURS16 | OURS16 vs LLAMA16: same BF16 values. Layout differs, see results/WEIGHTS.md |
| 2 | Pure llama.cpp on NVIDIA CUDA (stock release b11913) | LLAMA16, LLAMA8 | the baseline for 1 and 3 |
| 3 | Stock llama.cpp with our GPU backend (aien-dev/ggml-omega `OmegaGB10`), no libggml-cuda, no NVIDIA user-space libs | LLOMEGA8, LLOMEGA16 if BF16 lands | LLOMEGA8 vs LLAMA8 and LLOMEGA16 vs LLAMA16: the identical GGUF file, identical llama.cpp front end. Only the GPU backend differs |

In configuration 3, any op ggml-omega does not take runs on ggml's CPU backend. The op placement table (GPU or CPU, share of decode time) is published with the numbers. A slow result there is reported as it is.

## What the llama.cpp side asked for (SIDDARTHAREDDY8, discussion #31092, 2026-10-24) and how this run meets it
| Ask | Here |
|---|---|
| Same model file, same quantization, same prompt, same context size | Config 3 against config 2 uses the identical GGUF file (Q8_0, and BF16 if supported), so this ask is met exactly. Config 1 cannot read GGUF, so it uses the HF safetensors the GGUF was converted from. Every tensor was checked to hold the same values (130 byte-identical, the rest reordered or exactly transformed by the converter: results/WEIGHTS.md). Same prompts, same chat template, context 32768 for every arm. |
| Prompt processing and generation reported separately | pp tok/s and tg tok/s are separate columns. They are never blended. |
| State the offload | LLAMA*: `-ngl 999`, all layers on the GPU, verified from the load log. LLOMEGA*: `-ngl 999` onto the OmegaGB10 device, plus a per-op table of what ran on the GPU and what fell back to the CPU. OURS*: every layer on the GB10 GPU, with embedding lookup and sampling locations stated from the code. |
| Repetitions and spread | 3 rounds × 3 reps per cell (after warm-ups). Reported as mean ± sd and median with min..max. llama-bench `-r 5` is included as a cross-check. |
| Correctness before speed | The `agree` cell runs first: greedy tokens from both arms on the same 16 prompts, with the first divergence index reported. Garbage output voids that arm. |
| Freeze machine state, publish the recipe | Recorded per session: driver version, GPU clocks and power state (`nvidia-smi -q -d CLOCK,POWER,PERFORMANCE`), load and other processes. Exact commands, llama.cpp b11913 version string, both commits, and every file hash are published in the repo. |
| Disclose that the stacks differ | LLAMA* uses NVIDIA's user-space libraries (libcudart, libcublas, libcuda). OURS* and LLOMEGA* load none of them, which is checked from /proc/PID/maps. Every arm uses NVIDIA's open kernel module and GSP firmware. |

## Arms
| Arm | Engine | Weights | Role |
|---|---|---|---|
| OURS16 | aien-serve-http `--engine omega` (sc #790 + omega-mv #905 build) | BF16 safetensors, HF Qwen/Qwen3.5-9B rev c2022362 | headline, config 1 |
| LLAMA16 | llama-server b11913, CUDA 13.4 runtime, `-ngl 999 -fa on` | BF16 GGUF, sha256 5d6fe481… | headline, config 2 |
| LLAMA8 | same binary | Q8_0 GGUF, sha256 e1dc8c27… | headline, config 2 (pair for LLOMEGA8) |
| LLOMEGA8 | stock llama-server, backend dir = CPU + libggml-omega.so only | the same Q8_0 GGUF file | headline, config 3 |
| LLOMEGA16 | same as LLOMEGA8 | the same BF16 GGUF file | headline, config 3, only if ggml-omega gains BF16 MUL_MAT; otherwise NOT_RUN |
| OURS8 | aien-serve-http, FP8 block-128 checkpoint | FP8 | reference only, never a headline (no llama.cpp FP8 type exists) |

## Correctness oracle
HF transformers 5.16.1 (nvcr.io/nvidia/vllm:26.09-py3) runs greedy BF16 on the same 16 prompts with the same chat template and thinking off: results/ref/HF-ORACLE.jsonl. It is the model's reference implementation and shares no code with any arm. Every arm's `agree` output is compared with the oracle and with LLAMA16: first divergence in characters and in tokens, using the model's tokenizer.

## Proof of "no NVIDIA user-space libraries"
For OURS* and LLOMEGA*, the shared objects the running server mapped are captured from /proc/PID/maps (maps-so.txt). It must list none of libcuda, libcudart, libcublas, libcublasLt, libnvrtc, libnvidia-ptxjitcompiler. LLAMA* is the negative control and must list them. NVIDIA's open kernel module and GSP firmware are used by every arm and are disclosed.

## Cells (one HTTP client for all arms: HeadToHead repo bench/bench.py, temperature 0, thinking off)
1. `agree`: c=1, 128 tokens, 16 decode prompts, 1 rep each. Correctness gate, runs first.
2. `decode-c1`: c=1, max 256 tokens, decode.jsonl, 2 warm-ups + 3 reps. Gives tg tok/s.
3. `prefill-p512`, `prefill-p2k`, `prefill-p8k`: c=1, max 1 token, `--nonce`, 2 warm-ups + 3 reps. Gives pp tok/s and TTFT.
4. `decode-c8`: LLAMA arms with `-np 8` only, as an aggregate reference. NOT_RUN for OURS, whose dense door has no batching.
- Server cpus 5-9,15-19, client cpus 0-1. Quiet gate before every cell: bench/quiet.py `--window 60 --max-load 6`. A cell that runs without a quiet window is marked LOADED.

## Design
- 3 rounds. In each round every arm runs once with a fresh server start, and load time is recorded. Arm order per round comes from `random.Random(9)`, and the order is logged.
- No re-runs by choice. A round is repeated only for a recorded machine fault, and both attempts are reported.

## Metrics
- tg tok/s per request = (completion_tokens − 1) / (t_last − t_first).
- pp tok/s per request = prompt_tokens / (t_first − t_send). This includes HTTP and the first sampled token, and that is stated.
- Per arm and cell: mean ± sd, median, min..max over all non-warm-up requests, computed in Wolfram with the queries saved.
- Headline: OURS16 / LLAMA16 ratio of median tg (decode-c1) and of median pp per prompt size.

## Not claimed
- One machine, one model, single-stream generation. Not batch serving. Any cell not run is NOT_RUN.

## Resumption amendment, 2026-10-29 (before any new timed run)

The resumed campaign uses `bench/bench-node.mjs` on Node v24.21.0 as a
verification-only measurement instrument. Python is not executed. It preserves
the request fields, prompt rotation, warm-up selection, nonce prefix, streaming
content/reasoning handling and busy-response retry policy of `bench.py`.
Synthetic HTTP controls live in `bench/test-bench-node.mjs`; they exercise
fragmented UTF-8/SSE, request shape, retries, failed/truncated responses, missing
usage, concurrent requests, warm-ups and refusal to overwrite an existing output.

The instrument records monotonic nanoseconds as decimal strings (`t_send_ns`,
`t_first_ns`, `t_last_ns`). It calculates no rates or statistics. Wolfram converts
those raw timestamps and applies the metric definitions above; queries and
results accompany the report. The default request timeout is the same as the
original client: Wolfram query
`UnitConvert[Quantity[1800, "Seconds"], "Milliseconds"]` returned
`Quantity[1800000, "Milliseconds"]`.

Unlike the old client, malformed events, missing token usage and streams ending
without `[DONE]` fail the request and make the client exit unsuccessfully.
Failed rows are retained. Files from different client implementations are not
pooled into one timing summary. The recovered R1 remains historical evidence;
the resumed client needs its own complete repeated campaign before a headline
comparison. No new timed run has been accepted under this amendment.

Backend operator checks are a separate qualification level from model output.
The existing `ggml-omega/tests/e2e-gb10.sh` requires an exclusive model window
with the resident model stopped. That test remains NOT_RUN while the shared
service is retained. Operator-only tests performed beside that service cannot
replace this gate. This amendment grants no permission to stop the service.

The archived Q35 launcher named a CUDA-runtime directory that does not exist.
The resumed launcher uses the actual extracted companion release directory,
`h2h/dl/llama-bin/cudart-llama-b11913-bin-ubuntu-cuda-13.4-arm64`.
Its release archive SHA-256 is
`f6a5a0236082be3a10e9418a47ba389d23097c1fe8926581835f22e3cdeb599a`.
Library maps are retained for every arm. Historical maps show system-library
paths, so historical and resumed CUDA rows are not treated as identical runtime
configurations or pooled together.

## Authorized model-window follow-up, 2026-10-29

Drake explicitly authorized stopping the shared model unless it is doing a test.
The saved operating rule now requires checking live activity and the shared
board before stopping it, then restoring the original setup after testing.
Exclusive windows performed under that instruction have now exercised the
mixed backend's fixed-prompt model check. Qwen3.5-9B BF16 and Q8 each matched
the CPU reference over all 16 generated tokens. Nonzero MUL_MAT, RMS_NORM and
SWIGLU calls prove actual placement; post-inference process maps contained none
of the forbidden NVIDIA user-space libraries. The checkpoint files are the same
ones verified above. Full chat-corpus agreement and timing remain separate gates.

The backend test driver and comparator were hardened after controls reproduced
false success for empty output and a changed greedy token. The driver now
preserves comparator failure, requires all three executed operations and checks
an after-inference library snapshot. Twenty-one comparator controls and eight
host-only driver controls passed. Backend test-source commit:
`f4d4014` in `aien-dev/ggml-omega`, local experimental branch at this checkpoint.
Its committed `evidence/gb10-model-20261029.json` binds the model results to the
backend, CUDApp, llama.cpp and test-source hashes and records the numerical
differences. Greedy agreement is not full numerical equality.

The original shared model was restored after both completed windows, with the
same binary, model, configuration and healthy endpoint. A subsequent full-corpus
attempt refused to stop it while another worker had requests in progress. No
comparison cell is accepted from that refusal, and no active request was stopped.
The independent reference observations, repeated timing, quiet-window evidence,
larger-model arms and final comparison report remain incomplete.

## Full-server observations, 2026-10-29

The resumed full-server attempt exposed two defects that the bounded model
driver did not cover. Private CUDApp SHA helpers could resolve to incompatible
`libmtmd` symbols; an export map now keeps all backend definitions private except
`ggml_backend_init`. Empty graph nodes could also reach a chip routine requiring
nonempty input; the GPU executor now skips them, matching the CPU executor.
The original failures, debugger trace and failing empty-graph reproduction are
preserved. Export isolation and empty-graph regression controls pass on the
fixed backend, source `9b1d3d5` (local branch).

With those fixes, LLOMEGA16 completed every original chat-corpus request without
request errors. All three intended GPU operations executed, and the
after-inference library snapshot passed. Responses differ from the resumed
stock BF16 baseline on d04, d14 and d15. Against the frozen independent HF
reference, three sequences are identical and thirteen first-divergence tuples
have exact previously recorded observations. These observations do not establish
full numerical equivalence or excuse later differences. No final accuracy or
timing verdict follows from this run.

The first full LLOMEGA8 run completed its requests but failed the unchanged
placement gate: MUL_MAT was absent from its GPU profile. It remains a failed
placement experiment. Source inspection found llama.cpp's CPU-only repacked
weight buffers incompatible with this backend's host-buffer interface. The
historical Omega launcher used `-nr`; the resumed command had omitted it. The
corrected Q8 run uses stock `--no-repack`, the identical GGUF and the same staged
backend. Its placement and output remain unverified until the run completes.
This is a recorded configuration correction; results are not pooled across the
two configurations.

Local evidence: `handoffs/stack-comparison-20261029/llomega16-full-corpus-receipt.json`,
`llomega16-token-audit.json`, `llomega8-token-audit.json`,
`startup-collision-fix.json`, `q8-placement-diagnosis.json` and
`llomega-correctness-3/`. The independent-reference observations still missing
for other arms, repeated timing, quiet-window controls and larger-model arms
remain outstanding.

# gpu-head-to-head

Same-machine, end-to-end comparison of three inference stacks serving **Qwen3.6-35B-A3B** on one
NVIDIA DGX Spark (GB10, sm_121): our `aien-serve-http`, vLLM, and llama.cpp. This is bar **P0** of
the GPU-stack plan: prove the numbers before claiming them, and publish them even when we lose.

- Protocol (predeclared, committed before the first run): [PROTOCOL.md](PROTOCOL.md)
- Results table: planned at `results/RESULTS.md`, not present in this tree.
- Raw request rows, server logs, memory samples, versions: [results/](results/)
- Client and scripts: [bench/](bench/)

Every number here is end to end (HTTP client on the same host). Chip-only numbers from other repos
are never mixed into these tables.

## Connected repositories

This repo is a child of the GPU stack. It pins its parents by sha:

| Parent | Pinned sha | Role here |
|--------|-----------|-----------|
| [aien-dev/CUDApp](https://github.com/aien-dev/CUDApp) | `780a0af1f08c1786ea53a4e8702ca8bb1396fdbb` (main, 2026-10-27) | GPU stack source of truth (CUDA++) |
| [aien-dev/aien-sovereign-core-mv](https://github.com/aien-dev/aien-sovereign-core-mv) (was `aien-sovereign-core`) | `f0230a59eec985c3290ac78c0cc04322bacf9f0a` | source of the measured `aien-serve-http` binary `bin-19173f9e` |

The measured binary `aien-serve-http` sha256 `19173f9ec3b609fe79af644f84741d8808b92ea8eb3ecec4bc945af6f22b3aeb`
links `libomega_gpu.a` sha256 `110295ef23dc7561eea371636347c005835f37b82c22f92401aa8a887a3ca990`
built from omega-mv `ship/motor-l387` `6de999da` (see `results/logs/versions/BUILD.txt`). It predates
the CUDApp cutover; the CUDApp pin records the GPU-stack state at the time of the measurement.

## Host tools

`bench/*.py` are host measurement tools (Python 3 stdlib; `make_prompts.py` and `agree.py` also use
`tokenizers` from inside the pinned vLLM container). They are not product code and ship nothing.
Omega replacement: none yet; an Omega HTTP/SSE client is the replacement path once the Omega
standard library has sockets and threads.

## Measurement controls

Before a resumed 35B run, check the external instruments with:

```sh
python3 bench/test-run-arm.py
python3 bench/test-run-arm-lifecycle.py
python3 bench/test-unload-gates.py
python3 bench/test-shared-model-guard.py
python3 bench/test-bench-python.py
python3 bench/test-extract-admission.py
bash -n bench/run_arm.sh
```

Protocol amendments A6 and A7 describe owned-process cleanup, per-cell admission records, failed-stream detection and refusal to
overwrite previous evidence. Passing these CPU-only controls does not qualify a model or complete
the GPU benchmark. Historical rows without an admission ledger retain their original limitations.

The separate `shared-model-guard.py` checks the saved executable, arguments, working directory,
required environment, idle health, connections, GPU inventory and the coordinator-reviewed BOARD
checksum. Its stop mode signals a bound Linux process handle and refuses an exited handle. The
controller must record intent and undo before stopping, hold the GPU lock and restore the same
shared setup afterward. The complete controller/restoration integration remains unfinished.
Guard controls use fixtures and harmless child processes; they never access the shared model,
its port or GPU inventory.

Session cleanup now enforces the protocol's unload gate: no GPU compute processes remain and
available memory is no more than 2 GiB below its pre-launch baseline. Missing or malformed
memory readings and failed GPU queries also fail the session and remove its DONE marker.
`unload-status.txt` records both checks; more available memory is acceptable. The check observes
memory after process cleanup without flushing caches or killing unrelated processes. A failure
requires investigation before another arm starts. Fixture tests exercise the actual cleanup
function; real-model unload and shared-service restoration still require hardware evidence.

## License

Apache-2.0 WITH LLVM-exception, see [LICENSE](LICENSE).

## Human-Directed Agentic Engineering

Development follows Human-Directed Agentic Engineering (HDAE): humans define goals,
architecture, constraints, acceptance criteria, and stopping points; agents implement,
execute checks, diagnose failures, and revise within those boundaries. Human review
retains responsibility for claims, acceptance, and release decisions. This describes
the development method, not a change to runtime autonomy or historical evidence.

For the documented method and a systems-software case study, see Gregory M. Kurtzer
(2026), *Ginux: Building an Independent Kernel That Runs Linux Userspace through
Human-Directed Agentic Engineering*, technical report, v1, Zenodo,
<https://doi.org/10.5281/zenodo.23682548>.

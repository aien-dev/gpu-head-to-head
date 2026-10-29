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
python3 bench/test-shared-window.py
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
shared setup afterward.
Guard controls use fixtures and harmless child processes; they never access the shared model,
its port or GPU inventory.

Session cleanup now enforces the protocol's unload gate: no GPU compute processes remain and
available memory is no more than 2 GiB below its pre-launch baseline. Missing or malformed
memory readings and failed GPU queries also fail the session and remove its DONE marker.
`unload-status.txt` records both checks; more available memory is acceptable. The check observes
memory after process cleanup without flushing caches or killing unrelated processes. A failure
requires investigation before another arm starts. Fixture tests exercise the actual cleanup
function; real-model unload and shared-service restoration still require hardware evidence.

`bench/shared-window.py` integrates the guard, one arm session and same-configuration restoration.
It requires a fresh output directory, the saved process snapshot, exact file hashes in a JSON
mapping of absolute paths to SHA-256, and the checksum of BOARD after the coordinator has reviewed
live test activity. It appends intent and undo, then refuses if the board changes before stopping.
Required pins cover the controller, guard, launcher, snapshot, shared executable, runner and its
helpers and prompts, plus `arms.conf` if present. Prepare frozen copies and their pins before use.
The launcher contract is the existing 35B batch model with eight slots and context 49152; a different
snapshot is refused before stop. The controller does not authorize a model swap.

The controller holds the global lock through restoration, passes descriptor 9 to the runner, and
closes it for the restored service. Failed or interrupted tests still attempt restoration only
after the original and runner have exited, GPU inventory is empty and memory has recovered.
It refuses overlapping launches and never kills a foreign process. A runner that cannot finish
cleanup retains its inherited lock; a failed recovery gate leaves the model unrestored and records
the reason in `window-result.json`. Such a failure requires investigation, not another launch.
Events, runner output and restoration output remain in the fresh attempt directory.

Use `python3 bench/shared-window.py --help` for required arguments. Host-only controls exercise
sequencing and refusal paths, real fixture-child lock inheritance, and a fixture restoration
launcher. They do not establish real GPU unloading, actual shared-model recovery, or completion of
the larger-model comparison. Those hardware checks remain pending.

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

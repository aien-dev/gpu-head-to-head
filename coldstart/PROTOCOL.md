# PROTOCOL: cold start to first verified result, CUDApp vs NVIDIA user-space compile paths (GB10)

Predeclared before any measured GPU run. This file, the harness in `coldstart/` and amendment A7 of the
repo's [PROTOCOL.md](../PROTOCOL.md) are committed and pushed before the GPU window opens. Any later change
is a separate, dated commit listed in section 11, and results name the protocol commit they ran under.

This is a chip-level microbenchmark. The repo's PROTOCOL.md section 1 still holds: none of these numbers
enters the main results table (`results/RESULTS.md`). They live in `results/coldstart/` only.

## 1. Question

On one DGX Spark (GB10, sm_121), how long does a fresh process take from `exec` to its first verified
GPU result when the kernel reaches the GPU through CUDApp's own path (its kernel builder and encoder,
then its own `ioctl`/`mmap` calls on `/dev/nvidiactl` and `/dev/nvidia0`, no NVIDIA user-space library),
compared with the NVIDIA user-space paths that compile a kernel at run time (NVRTC, or the PTX compiler
library) and then load and launch it through `libcuda`?

## 2. Arms

| Arm | Binary | Kernel description in the process | Build step (timed as `build`) | GPU path |
|-----|--------|-----------------------------------|-------------------------------|----------|
| CUDAPP | `cs_cudapp` | CUDApp `examples/hello_gpu/hello_gpu.c` `build_vecadd` (IR program) | `build_vecadd` + `omega_blackwell_kernel_encode_into`: 512 B of machine code | CUDApp `libomega_gpu.a` session API (own nvrm ioctls) |
| PTXC | `cs_ptxc` | `coldstart/vecadd.ptx`, read into memory during prep | static `nvPTXCompiler` (`--gpu-name=sm_121`): 5600 B cubin | CUDA driver API, `libcuda` |
| NVRTC | `cs_nvrtc` | the CUDA C++ string in `cs_cuda.c` | `nvrtcCompileProgram` (`--gpu-architecture=sm_121`): 5600 B cubin | CUDA driver API, `libcuda` |

PTXC is the like-for-like arm against CUDAPP: both start from a device-level description and stop at
loadable machine code. `vecadd.ptx` is NVRTC's own `compute_121` PTX for the NVRTC source (build.sh checks
that it is still byte-identical to what NVRTC emits), and PTXC's cubin is byte-identical to NVRTC's
(sha256 `691edc87423da8a86c5403d332bb980aa20f8808c84a0ffbf79481081c7dd6e8`, 5600 B, observed with
`--build-only --emit-code` after the section 3 build). So PTXC and NVRTC load and run the same code;
they differ in the build step and in what the process links.

## 3. Pins and build

- CUDApp `cc0908bdc0a0377f2166e9328dcf1d985457e27f` (cloned by `coldstart/build.sh` into `OUTDIR/cudapp`,
  checked clean and at the pin; library and `hello_gpu` built with CUDApp's own Makefile and flags).
- gcc 13.3.0. NVIDIA arms: `-std=gnu11 -Wall -Wextra -Werror -O2`.
- CUDA 13.0 (`/usr/local/cuda-13.0`, `version.json` cuda 13.0.3): NVRTC `libnvrtc.so.13.0.88`
  (`nvrtcVersion` 13.0), PTX compiler `libnvptxcompiler_static.a` (`nvPTXCompilerGetVersion` 13.0).
- Driver 580.173.02 (`/proc/driver/nvidia/version`, `libcuda.so.580.173.02`), `cuDriverGetVersion` 13000.
- Build: `taskset -c 10-14 nice -n 19 ionice -c3 coldstart/build.sh /tmp/csb/out` printed `build: OK
  /tmp/csb/out` on 2026-10-29 (binaries written 22:08:17Z). `build.txt` sha256:

  | File | sha256 |
  |------|--------|
  | `cs_cudapp` | `ba4496b361a6f53ec65ccec1b8c999509e392bce7f439d2e4d1e86aa4d7d31cd` |
  | `cs_nvrtc` | `878bc185cd7a28f52d33253f217efd4a1773cdd1c5bf92669ed0168195e8f14e` |
  | `cs_ptxc` | `27ae182fe36dda55813ba96f89b7f416505010c2bb3d40805fc263d17c3a28e4` |
  | `cudapp/build/hello_gpu` | `af5e9f6640445cb4d02a857c63ca568f4da48d6c15212dab98631ad9ea6ce3aa` |
  | `cudapp/build/libomega_gpu.a` | `3c626939a7f599d929d30bc8c9f9f37e08a6bc02bdcd9755c98504bda9e06724` |
  | `coldstart/vecadd.ptx` | `9abb51a8a751ec3e2483ba7a57effbe6a387f282f5ffa1bf4a485048e66669a2` |

  The three arm shas equal those of the earlier build from 16:53Z, although the header comments of both
  C files changed in between: comments do not reach object code, and nothing is built with `-g`.
- Emitted code: CUDAPP `--emit-code` sha256 `9e28bd9c46f80584d77238a752e99cecddbc8d622ffbe320885ed2f499e1f075`
  (512 B), equal to the "machine code sha256" that `hello_gpu --sim` prints at the pin; PTXC and NVRTC as
  in section 2.

## 4. Job, phases, metrics

The job is `hello_gpu`'s job at the pin: `c = a + b` over 1,048,576 floats, `a[i] = i * 0.25`,
`b[i] = 1 / (1 + i % 7)`, `c` poisoned with `0xFFC0FFEE`, eight launches of 1024 blocks x 128 threads,
then every element compared bit for bit with the CPU's own `a[i] + b[i]`. A process is correct only with
`wrong == 0` and `unwritten == 0` (exit 0).

Phases, each closed by a `CLOCK_MONOTONIC` stamp inside the process:

| Phase | CUDAPP | PTXC / NVRTC |
|-------|--------|--------------|
| prep | host inputs | host inputs (PTXC: also reads `vecadd.ptx`) |
| build | kernel description to machine code in host memory | PTX or C++ to cubin in host memory |
| open | `omega_gpu_session_open` | `cuInit`, `cuDeviceGet`, `cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent` |
| load | code buffer allocated, words copied, synced to GPU | `cuModuleLoadData`, `cuModuleGetFunction` |
| data | three buffers allocated, inputs and poison copied in and synced | `cuMemAlloc` x3, `cuMemcpyHtoD` x3 |
| run | eight launches, each waits for its completion marker | eight `cuLaunchKernel`, `cuCtxSynchronize` |
| check | output synced to CPU, bit-exact check | `cuMemcpyDtoH`, bit-exact check |
| close | buffers freed, session closed | `cuMemFree` x3, `cuModuleUnload`, `cuDevicePrimaryCtxRelease` |

The driver (`run.py`) stamps `t0` just before `Popen` and `t1` when the child has exited and its pipes
are drained, on the same clock.

| ID | Metric | Definition (ns, reported in us) | Role |
|----|--------|---------------------------------|------|
| H1 | build | build phase | headline |
| H2 | ready | build + open + load | descriptive: CUDA 13 defaults to lazy module loading (`CUDA_MODULE_LOADING` unset), so part of the NVIDIA arms' load work can move into run |
| H3 | firstResult | exec + prep + build + open + load + data + run + check | headline: spawn to verified result |
| H4 | proc | `t1 - t0` | headline: spawn to process exit, close and teardown included |
| E | exec | `t_main - t0`: spawn, `exec`, dynamic linking, constructors | descriptive |
| W | warm build | per process, median of 10 further builds in a `--build-only` process | descriptive |

## 5. Schedule (`coldstart/run.py`)

1. Pre-checks (section 6), then the control processes of section 6, one per arm per control.
2. 2 warmup rounds (each arm once per round, order CUDAPP, PTXC, NVRTC), not analysed.
3. 30 measured rounds. Round `r` runs the arms in permutation `r mod 6` of the 6 orders, so every order
   runs 5 times. 1 s sleep before every measured spawn. Measured processes run with `--warm 0`, so no
   per-process warm build sits inside their proc time.
4. Warm phase W: 10 rounds under the same permutation rule and gap, each process `--build-only`
   (no GPU opened) with its 10 warm builds.
5. Cap: 15 minutes from driver start. When it is reached the driver stops, writes what it has and the
   session verdict is `INCOMPLETE_CAP`.

Output: `results/coldstart/<UTC run id>/{raw.jsonl,session.json,rows.wl}`, plus the control artefacts
`code.<ARM>.bin` and `maps.<ARM>.txt`. Every spawned process gets `CUDA_CACHE_DISABLE=1`.

Pinning: the driver runs on core 12 (Cortex-A725). Every arm process runs on one core, the first of
9, 8, 7 (Cortex-X925, A2) that hosts no nvidia-modeset kthread at driver start (field 39 of
`/proc/<pid>/stat`); the core and the kthread cpus go into `session.json`.

## 6. Pre-checks, controls, validity

The driver refuses to start, before any process runs, when any of these fails:

- not inside a spark-window that holds the chip lock (`SPARK_WINDOW` set and `AIEN_GB10_LOCK_HELD=1`);
- an arm binary or `vecadd.ptx` does not match its sha256 in `build.txt`;
- every candidate arm core hosts an nvidia-modeset kthread;
- `bench/quiet.py --window 60 --max-load 6` fails (a foreign process above 20% of one CPU over 60 s, or
  `load1 >= 6`; the omp harness is recorded, not gated, as in A2);
- `nvidia-smi --query-compute-apps` lists any compute process.

Controls, all before the first measured process. Any failure makes the session verdict
`CONTROLS_FAILED` and no measured process runs:

- code: each arm `--build-only --emit-code`; CUDAPP's code sha256 must be `9e28bd9c...` (section 3);
  PTXC's cubin must be byte-identical to NVRTC's.
- skip-run (negative control): each arm `--skip-run` must exit 1 with `unwritten == wrong == 1048576`.
- maps: each arm runs the full job and copies `/proc/self/maps`. CUDAPP's map must hold no
  `libcuda`, `libcudart`, `libnvrtc`, `libnvptx` or `libnvidia` mapping and at least one `/dev/nvidia`
  mapping; each NVIDIA arm's map must hold `libcuda` (the positive control of the library check).
- output: the maps runs must all be correct and give the same output FNV-1a hash across the three arms.

Validity of rows and arms (in `coldstart/analysis.wl`): a measured row is valid when its process was
correct and `load1 < 6` at its spawn. An arm is valid with at least 28 of its 30 measured rows valid.

Session validity: a session is VOID when its verdict is not `MEASURED` (`CONTROLS_FAILED`,
`INCOMPLETE_CAP`, a refused pre-check), when the spark-window lease ran out before close (spark-window
verdict NO VERDICT), or when the window's holder lost the chip lock during the run (for example a
chip-watchdog kill posted on the BOARD). A void session is never retried inside the same window. It is
re-run in a new window, and both sessions are published.

## 7. Statistics and verdicts

Computed by `coldstart/analysis.wl` (`csAnalyse`, with `h2hStats` of `results/wolfram/analysis.wl`) in the
Wolfram MCP kernel on the run's `rows.wl`; query and output are kept beside the rows. Per arm and metric:
n, median, Q1, Q3, IQR (Wolfram `Median`, `Quantile` with its default definition, `InterquartileRange`),
p99 as a descriptive figure, in microseconds over valid rows. W is reported as the statistics of the
10 per-process medians.

For each headline (H1, H3, H4) and each comparison CUDAPP vs PTXC, CUDAPP vs NVRTC:

- `CUDAPP_LOWER`: CUDAPP Q3 < other arm Q1;
- `CUDAPP_HIGHER`: other arm Q3 < CUDAPP Q1;
- `OVERLAPPING`: otherwise;
- `NO_VERDICT`: either arm is invalid (section 6), or the session is void.

No significance tests. Median ratios (other over CUDAPP) are descriptive. The same calls are printed for
H2, E and W, labelled descriptive. Results are published whichever way they come out, including every
case where CUDApp is higher.

## 8. Caveats and disclosures

- Different builds: `ptxas` (inside NVRTC and nvPTXCompiler) runs an optimizing assembler and register
  allocator over a general kernel description; CUDApp's builder lowers one fixed IR program for this one
  kernel to 32 instructions (512 B). H1 compares those two build steps as each stack does them, not two
  compilers of equal scope.
- Process spawn (`fork`/`exec` from Python, page-in) is in exec, firstResult and proc for every arm. The
  binaries differ in size (`cs_ptxc` links the static PTX compiler, 38,394,840 B; `cs_cudapp`
  1,405,168 B; `cs_nvrtc` 72,496 B plus the shared `libnvrtc` and `libcuda`), and that difference is part
  of what E measures.
- PTXC and NVRTC load and run an identical cubin, so differences between them after build come from the
  process (linked libraries), not from the code.
- The data and check phases are each stack's own transfers: CUDApp syncs host-mapped buffers, the
  NVIDIA arms copy with `cuMemcpyHtoD`/`cuMemcpyDtoH`. firstResult includes them for all arms.
- One session, one machine, one date. The study measures GB10 under the stated load only.
- Pre-protocol pilot (before this file existed; the numbers came with the lane assignment, no raw rows
  are filed): NVRTC fresh compile 290,679 / 138,026 / 137,686 us in three pilot runs, warm NVRTC builds
  about 10.8 to 11.9 ms; CUDApp `hello_gpu --sim` kernel build 145 / 62 / 74 us (host, niced core; see
  CUDApp branch `docs/hello-sim-receipt-date` aa7b8771, README section 9). They shaped the schedule
  and are not results.
- Host build-only dry run (`run.py --dry-run`, 2026-10-29 21:57:16Z to 21:57:31Z, gap 0.02 s, no GPU
  opened, scratch `/tmp/csb/dry`): EXERCISED, not results. Wolfram `csAnalyse[csRows, csWarm]` on its
  `rows.wl` gave build medians CUDAPP 72.6085 us, PTXC 5919.412 us, NVRTC 139870.6725 us and warm-build
  medians 5.184 / 3156.583 / 16345.90225 us. Its firstResult and proc stop at build (no GPU phases).

## 9. Publication

`results/coldstart/<run>/`: `raw.jsonl` (every process, controls and warmups included), `session.json`
(pins, build.txt, controls, kthread cpus, quiet output, GPU apps before and after), `rows.wl`,
`code.*.bin`, `maps.*.txt`, the Wolfram query and output, and `RESULTS.md` with window id and times,
pins, controls, valid row counts, per arm median and IQR per metric, verdicts and void status. Evidence
label: MEASURED on this Spark in that window. Chip-only; never copied into `results/RESULTS.md`.

## 10. Tooling

`cs_cuda.c`, `cs_cudapp.c`, `build.sh` and `run.py` are C, shell and Python measurement tooling, not
product code: the NVIDIA arms must call NVIDIA's C APIs, and `cs_cudapp.c` wraps CUDApp's own C example.
The Omega successor of the CUDApp arm is the native `.om` hello tracked in aien-dev/CUDApp#79.

## 11. Amendments

None yet.

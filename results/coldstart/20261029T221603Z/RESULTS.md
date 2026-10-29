# Cold-start study, session 20261029T221603Z: RESULTS

Evidence label: MEASURED on this DGX Spark (GB10) in one spark-window. Chip-level microbenchmark under
[coldstart/PROTOCOL.md](../../../coldstart/PROTOCOL.md) (PROTOCOL.md amendment A7). Not part of the main
results table `results/RESULTS.md`.

## Session

| Item | Value |
|------|-------|
| Protocol commit (predeclaration) | `63855ab4822eea6fec4e488f23b62bc976469c8c`, pushed to `protocol/a6-cold-start` 2026-10-29T22:15:13Z |
| spark-window | `20261029T221538Z-ed5493`, `--gb10 exclusive --quiet exclusive --lease 20`; opened 22:15:41Z (waited 3 s), closed 22:19:51Z, held 253 s, spark-window verdict `quiet machine` |
| run.py | started 22:16:03Z, finished 22:19:46Z, verdict `MEASURED`, `complete: true`, 90 measured processes |
| Quiet gate | `bench/quiet.py` 60 s ending 22:17:03Z: `quiet: true`, load1 1.2, busy `[]` |
| GPU compute apps | none before, none after |
| Cores | arm core 9; driver core 12; nvidia-modeset kthreads on cpu 4 |
| Measured load1 range | 0.92 to 3.38 (Wolfram `MinMax[csRows[[All, 4]]] -> {0.92, 3.38}`) |
| Environment | `CUDA_CACHE_DISABLE=1`, `CUDA_MODULE_LOADING` unset |

Pins as predeclared (coldstart/PROTOCOL.md section 3): CUDApp `cc0908bdc0a0377f2166e9328dcf1d985457e27f`,
gcc 13.3.0, CUDA 13.0 (NVRTC 13.0.88, nvPTXCompiler 13.0), driver 580.173.02 (`cuDriverGetVersion`
13000). `session.json` carries `build.txt` verbatim; the driver's sha256 check of all three arm binaries
and `vecadd.ptx` against it passed.

## Controls (all passed, `control_failures: []`)

| Control | CUDAPP | PTXC | NVRTC |
|---------|--------|------|-------|
| code sha256 (`code.<ARM>.bin`) | `9e28bd9c...` (512 B), equals `hello_gpu --sim` | `691edc87...` (5600 B) | `691edc87...` (5600 B), identical to PTXC |
| `--skip-run` (negative) | exit 1, wrong = unwritten = 1048576 | exit 1, 1048576 / 1048576 | exit 1, 1048576 / 1048576 |
| maps (`maps.<ARM>.txt`) | no NVIDIA user library, 12 `/dev/nvidia` mappings | `libcuda` present | `libcuda` present |
| output FNV-1a (maps runs) | `e8aa66742b796a3d` | `e8aa66742b796a3d` | `e8aa66742b796a3d` |

## Validity

Every arm: 30 measured rows, 30 valid, 0 failed, 0 excluded for load (`rows_valid` in
`wolfram.out.json`). All arms valid (at least 28 of 30 required). Session not void: verdict `MEASURED`,
lease not exceeded, no chip-watchdog action during the window.

## Headline metrics (microseconds, valid rows, n = 30 per arm)

Median (Q1 to Q3), IQR. Source: `wolfram.table.out.txt` (rounded to 0.1 us by Wolfram); unrounded in
`wolfram.out.json`.

| Metric | CUDAPP | PTXC | NVRTC |
|--------|--------|------|-------|
| H1 build | 77.7 (75.0 to 85.4), IQR 10.4 | 7029.6 (6027.1 to 7143.9), IQR 1116.8 | 119047.8 (98378.3 to 141661.8), IQR 43283.5 |
| H3 firstResult | 96774.1 (95517.6 to 102370.2), IQR 6852.6 | 263653.7 (259361.2 to 305322.0), IQR 45960.8 | 302786.4 (278044.6 to 325938.8), IQR 47894.3 |
| H4 proc | 154374.2 (144943.3 to 189373.1), IQR 44429.9 | 423561.1 (418338.3 to 466366.7), IQR 48028.4 | 481722.4 (441550.3 to 487635.4), IQR 46085.2 |

Verdicts (section 7, non-overlapping IQRs):

| Headline | CUDAPP vs PTXC | CUDAPP vs NVRTC |
|----------|----------------|-----------------|
| H1 build | CUDAPP_LOWER, median ratio 90.45 | CUDAPP_LOWER, median ratio 1531.74 |
| H3 firstResult | CUDAPP_LOWER, 2.72 | CUDAPP_LOWER, 3.13 |
| H4 proc | CUDAPP_LOWER, 2.74 | CUDAPP_LOWER, 3.12 |

Ratios are other arm median over CUDAPP median, descriptive only.

## Descriptive metrics

| Metric | CUDAPP | PTXC | NVRTC | Calls (vs PTXC / vs NVRTC) |
|--------|--------|------|-------|----------------------------|
| H2 ready | 87150.7 (85946.8 to 93097.3) | 249194.6 (245828.0 to 290059.2) | 284483.7 (261889.2 to 307408.3) | CUDAPP_LOWER 2.86 / CUDAPP_LOWER 3.26 |
| E exec | 2609.7 (2524.7 to 2801.1) | 3878.2 (3649.3 to 4012.7) | 6611.2 (6328.0 to 6828.9) | CUDAPP_LOWER 1.49 / CUDAPP_LOWER 2.53 |
| W warm build (10 per-process medians) | 5.16 (5.144 to 5.192) | 3162.57 (3137.27 to 3176.66) | 16353.48 (16296.44 to 16385.04) | CUDAPP_LOWER 612.90 / CUDAPP_LOWER 3169.28 |

W calls: `wolfram.warm-call.out.txt` (`{"PTXC" -> {"CUDAPP_LOWER", 612.9013081395348}, "NVRTC" ->
{"CUDAPP_LOWER", 3169.279360465116}}`).

Phase medians, not predeclared, added as a description of where each arm spends its time
(`wolfram.phases.out.txt`, Wolfram `Median` per phase over the 30 measured rows, us):

| Phase | CUDAPP | PTXC | NVRTC |
|-------|--------|------|-------|
| prep | 3537.2 | 3601.5 | 4670.8 |
| build | 77.7 | 7029.6 | 119047.8 |
| open | 86983.2 | 242130.8 | 163547.1 |
| load | 68.1 | 23.9 | 23.5 |
| data | 1675.8 | 900.6 | 898.3 |
| run | 844.7 | 60.1 | 61.7 |
| check | 519.8 | 5726.2 | 5480.1 |
| close | 50298.2 | 87451.5 | 87378.1 |

## Reading

- Every headline is CUDAPP_LOWER against both arms with non-overlapping IQRs, in this one session.
- The build gap is the largest; after build, opening the GPU dominates every arm. CUDApp's
  `omega_gpu_session_open` has a median of 86983.2 us, nearly all of its firstResult. The NVIDIA arms'
  open covers `cuInit` and primary context creation.
- Per phase, the NVIDIA arms are faster where the GPU itself works: run 60.1 and 61.7 us against
  CUDApp's 844.7 us, and data 900.6 and 898.3 us against 1675.8 us. Their load (23.9, 23.5 us) is
  near zero because CUDA 13 loads modules lazily; check holds their device to host copy.
- NVRTC's build median 119047.8 us sits between two groups (`wolfram.nvrtc-build-sorted.out.txt`):
  15 builds from 96662 to 101962 us and 15 from 136134 to 145481 us. Its IQR (43283.5) reflects that split.
- PTXC and NVRTC loaded the same cubin; their open medians differ (242130.8 against 163547.1 us). The
  phase table does not explain that difference.

## Caveats

As predeclared (coldstart/PROTOCOL.md section 8): the builds differ in scope (`ptxas` optimizes a
general description; CUDApp lowers one fixed IR program to 32 instructions); process spawn and binary
size are inside exec, firstResult and proc for every arm; one session, one machine, one date. The
pre-protocol pilot and the host dry run are disclosed there and are not results.

## Files

`raw.jsonl` (every process, controls and warmups included), `session.json`, `rows.wl`, `code.*.bin`,
`maps.*.txt`, `wolfram.query.wl` with `wolfram.out.json` (csAnalyse), `wolfram.table.*`,
`wolfram.warm-call.*`, `wolfram.phases.*`.

Zero-CUDA proof of `hello_gpu` taken in the same window before run.py (not part of the protocol):
`/home/drakestapleton/handoffs/cudapp-video-20261029/{hello.stdout,maps.txt,openat.txt,ldd.txt}`.

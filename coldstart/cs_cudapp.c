/* SPDX-License-Identifier: Apache-2.0 */
/* coldstart/cs_cudapp.c: the CUDAPP arm of the cold-start study (coldstart/PROTOCOL.md).
 *
 * It compiles CUDApp's own examples/hello_gpu/hello_gpu.c from the pinned commit into this file, with that
 * file's main renamed, so the kernel builder (build_vecadd), the inputs and the job are the ones hello_gpu
 * runs, byte for byte. Only the timing points and the JSON line are added here. The machine-code sha256 of
 * --emit-code must equal the "machine code sha256" hello_gpu prints at the same commit.
 *
 * Measurement tool, not product code. Its .om successor is the native .om hello of aien-dev/CUDApp#79.
 *
 * Output: one JSON line on stdout. Exit 0 = every element right, 1 = check failed, 2 = setup failed or no GPU.
 * Options: --skip-run (negative control: no launch, the check must fail), --emit-code FILE (write the
 * instruction words), --maps FILE (copy /proc/self/maps after the run), --build-only (build and time the warm
 * builds, no GPU opened), --warm N (0..10 warm builds after the cold path; run.py passes 0 to measured cold
 * processes). */
#define main cudapp_hello_main
#include "examples/hello_gpu/hello_gpu.c"
#undef main

#define WARM 10

static uint64_t cs_now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static uint64_t cs_fnv(const void *p, size_t n) { const uint8_t *q = p; uint64_t h = 0xcbf29ce484222325ull; for (size_t i = 0; i < n; i++) { h ^= q[i]; h *= 0x100000001b3ull; } return h; }
static int cs_write(const char *path, const void *p, size_t n) { FILE *f = fopen(path, "wb"); if (!f) return -1; size_t w = fwrite(p, 1, n, f); return (fclose(f) == 0 && w == n) ? 0 : -1; }
static int cs_maps(const char *path) {
    FILE *in = fopen("/proc/self/maps", "r"), *out = fopen(path, "w"); char buf[4096]; size_t k;
    if (!in || !out) return -1;
    while ((k = fread(buf, 1, sizeof buf, in)) > 0) fwrite(buf, 1, k, out);
    fclose(in); return fclose(out);
}
static int cs_fail(const char *what) { fprintf(stderr, "cs: %s (stage: %s)\n", what, omega_gpu_session_last_error()); return 2; }

static BlackwellIRProgram cs_prog;

/* build: kernel description -> machine code in host memory (what hello_gpu times as "kernel built in") */
static int cs_build(uint8_t **words, size_t *need, OmegaBlackwellKernel *k) {
    if (build_vecadd(&cs_prog) != 0) return -1;
    *need = omega_bw_program_code_bytes(&cs_prog);
    *words = calloc(1, *need);
    if (!*words || omega_blackwell_kernel_encode_into(&cs_prog, *words, *need, k) != 0) return -1;
    return 0;
}
static int nwarm = WARM;   /* --warm N (0..WARM): measured cold processes use 0 so process time holds the cold path only */
static int cs_warm(uint64_t *warm) {
    for (int r = 0; r < nwarm; r++) {
        uint8_t *w = NULL; size_t wn = 0; OmegaBlackwellKernel wk; uint64_t s = cs_now_ns();
        if (cs_build(&w, &wn, &wk)) return -1;
        warm[r] = cs_now_ns() - s; free(w);
    }
    return 0;
}
static void cs_print_warm(const uint64_t *warm) { for (int r = 0; r < nwarm; r++) printf("%s%llu", r ? "," : "", (unsigned long long)warm[r]); }


int main(int argc, char **argv) {
    const uint64_t t_main = cs_now_ns();
    int skip_run = 0, build_only = 0;
    const char *emit_code = NULL, *maps = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--skip-run")) skip_run = 1;
        else if (!strcmp(argv[i], "--build-only")) build_only = 1;
        else if (!strcmp(argv[i], "--warm") && i + 1 < argc) { nwarm = atoi(argv[++i]); if (nwarm < 0 || nwarm > WARM) return 2; }
        else if (!strcmp(argv[i], "--emit-code") && i + 1 < argc) emit_code = argv[++i];
        else if (!strcmp(argv[i], "--maps") && i + 1 < argc) maps = argv[++i];
        else { fprintf(stderr, "usage: cs_cudapp [--skip-run|--build-only] [--warm N] [--emit-code FILE] [--maps FILE]\n"); return 2; }
    }

    /* prep: host inputs in memory. Not compared between arms. */
    const size_t bytes = (size_t)N_TOTAL * sizeof(float);
    float *fa = aligned_alloc(64, bytes), *fb = aligned_alloc(64, bytes);
    uint32_t *wc0 = aligned_alloc(64, bytes);
    if (!fa || !fb || !wc0) return 2;
    inputs(fa, fb, wc0);
    const uint64_t t_prep = cs_now_ns();

    /* build */
    uint8_t *words = NULL; size_t need = 0; OmegaBlackwellKernel k;
    if (cs_build(&words, &need, &k)) { fprintf(stderr, "cs: build failed\n"); return 2; }
    const uint64_t t_build = cs_now_ns();

    /* --build-only: no GPU is opened; the build, WARM more builds, the code written if asked */
    if (build_only) {
        uint64_t warm[WARM];
        if (cs_warm(warm)) return 2;
        if (emit_code && cs_write(emit_code, words, k.insn_count * 16u)) return 2;
        printf("{\"arm\":\"CUDAPP\",\"build_only\":1,\"t_main_ns\":%llu,\"prep_ns\":%llu,\"build_ns\":%llu,\"warm_build_ns\":[",
               (unsigned long long)t_main, (unsigned long long)(t_prep - t_main), (unsigned long long)(t_build - t_prep));
        cs_print_warm(warm);
        printf("],\"code_bytes\":%zu,\"code_fnv\":\"%016llx\"}\n", k.insn_count * 16u, (unsigned long long)cs_fnv(words, k.insn_count * 16u));
        return 0;
    }

    /* open */
    omega_gpu_session_lock();
    if (!omega_gpu_session_open()) { fprintf(stderr, "cs: GPU not available (stage: %s)\nNOT_RUN\n", omega_gpu_session_last_error()); return 2; }
    const uint64_t t_open = cs_now_ns();

    /* load: the code buffer, the built words copied in and made visible to the GPU */
    OmegaGpuMem code = { 0 }, a = { 0 }, b = { 0 }, c = { 0 };
    if (omega_gpu_session_alloc_code(need, &code) != 0) return cs_fail("alloc_code");
    memcpy(code.cpu, words, need);
    if (omega_gpu_session_sync_to_gpu(&code) != 0) return cs_fail("sync code");
    const uint64_t t_load = cs_now_ns();

    /* data: three buffers, inputs and the poisoned output copied in */
    if (omega_gpu_session_alloc(bytes, &a) != 0 || omega_gpu_session_alloc(bytes, &b) != 0 || omega_gpu_session_alloc(bytes, &c) != 0)
        return cs_fail("alloc");
    memcpy(a.cpu, fa, bytes); memcpy(b.cpu, fb, bytes); memcpy(c.cpu, wc0, bytes);
    if (omega_gpu_session_sync_to_gpu(&a) || omega_gpu_session_sync_to_gpu(&b) || omega_gpu_session_sync_to_gpu(&c)) return cs_fail("sync data");
    const uint64_t t_data = cs_now_ns();

    /* run: the same eight launches hello_gpu makes; each returns when its completion marker is seen */
    uint32_t declared = omega_bw_gpr_declared(cs_prog.regalloc.peak_gpr_usage);
    uint64_t gpu_ns = 0;
    if (!skip_run) {
        for (uint32_t base = 0; base < N_TOTAL; base += CHUNK) {
            uint64_t off = (uint64_t)base * sizeof(float);
            uint32_t n = N_TOTAL - base < CHUNK ? N_TOTAL - base : CHUNK;
            uint32_t args[OMEGA_BW_CBANK_MATMUL_ARGS_WORDS] = { (uint32_t)(a.va + off), (uint32_t)((a.va + off) >> 32),
                (uint32_t)(b.va + off), (uint32_t)((b.va + off) >> 32), (uint32_t)(c.va + off), (uint32_t)((c.va + off) >> 32), n };
            OmegaGpuLaunch L = { .code_va = code.va, .gpr_count = declared < 64 ? 64 : declared, .threads_x = THREADS, .threads_y = 1,
                                 .grid_x = (n + THREADS - 1) / THREADS, .grid_y = 1, .num_elements = n,
                                 .args = args, .n_args = OMEGA_BW_CBANK_MATMUL_ARGS_WORDS, .timeout_ms = 10000 };
            uint64_t el = 0;
            if (omega_gpu_session_launch(&L, &el, NULL) != 0) return cs_fail("launch");
            gpu_ns += el;
        }
    }
    const uint64_t t_run = cs_now_ns();

    /* readback: the output made visible to the CPU, every element checked bit for bit */
    if (omega_gpu_session_sync_to_cpu(&c) != 0) return cs_fail("sync to cpu");
    const uint32_t *wc = c.cpu;
    uint32_t wrong = 0, unwritten = 0;
    for (uint32_t i = 0; i < N_TOTAL; i++) {
        float want = fa[i] + fb[i];
        if (wc[i] == POISON) unwritten++;
        if (memcmp(&want, &wc[i], sizeof want) != 0) wrong++;
    }
    const uint64_t t_check = cs_now_ns();

    /* after the cold path: WARM more builds in the same process */
    uint64_t warm[WARM];
    if (cs_warm(warm)) return 2;
    if (maps && cs_maps(maps)) return 2;
    if (emit_code && cs_write(emit_code, words, k.insn_count * 16u)) return 2;
    const uint64_t code_fnv = cs_fnv(words, k.insn_count * 16u), out_fnv = cs_fnv(wc, bytes);

    const uint64_t t_close0 = cs_now_ns();
    omega_gpu_session_free(&code); omega_gpu_session_free(&a); omega_gpu_session_free(&b); omega_gpu_session_free(&c);
    omega_gpu_session_unlock();
    omega_gpu_session_close();
    const uint64_t t_close = cs_now_ns();

    printf("{\"arm\":\"CUDAPP\",\"t_main_ns\":%llu,\"prep_ns\":%llu,\"build_ns\":%llu,\"open_ns\":%llu,\"load_ns\":%llu,"
           "\"data_ns\":%llu,\"run_ns\":%llu,\"check_ns\":%llu,\"close_ns\":%llu,\"warm_build_ns\":[",
           (unsigned long long)t_main, (unsigned long long)(t_prep - t_main), (unsigned long long)(t_build - t_prep),
           (unsigned long long)(t_open - t_build), (unsigned long long)(t_load - t_open), (unsigned long long)(t_data - t_load),
           (unsigned long long)(t_run - t_data), (unsigned long long)(t_check - t_run), (unsigned long long)(t_close - t_close0));
    cs_print_warm(warm);
    printf("],\"code_bytes\":%zu,\"code_fnv\":\"%016llx\",\"out_fnv\":\"%016llx\",\"wrong\":%u,\"unwritten\":%u,\"skip_run\":%d,"
           "\"gpu_ns\":%llu}\n", k.insn_count * 16u, (unsigned long long)code_fnv, (unsigned long long)out_fnv, wrong, unwritten,
           skip_run, (unsigned long long)gpu_ns);
    return wrong ? 1 : 0;
}

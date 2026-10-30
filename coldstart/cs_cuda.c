/* SPDX-License-Identifier: Apache-2.0 */
/* coldstart/cs_cuda.c: the two NVIDIA arms of the cold-start study (coldstart/PROTOCOL.md).
 *
 * Built twice by coldstart/build.sh:
 *   -DCS_NVRTC  arm NVRTC: the CUDA C++ source below -> sm_121 cubin with NVRTC (CUDA 13.0.88)
 *   -DCS_PTXC   arm PTXC:  the committed coldstart/vecadd.ptx -> sm_121 cubin with the static nvPTXCompiler
 * Both then open the GPU, load the cubin and launch through the CUDA driver API (libcuda 580.173.02).
 *
 * The job is the job of CUDApp's examples/hello_gpu at the pinned commit: c = a + b over 1048576 floats,
 * eight launches of 1024 blocks x 128 threads, inputs a[i] = i * 0.25, b[i] = 1 / (1 + i % 7), c poisoned
 * with 0xFFC0FFEE, every element checked bit for bit against the CPU's own a + b.
 *
 * Measurement tool (a referee for the NVIDIA side), not product code: it has to call NVIDIA's C APIs.
 *
 * Output: one JSON line on stdout. Exit 0 = every element right, 1 = check failed, 2 = setup failed.
 * Options: --skip-run (negative control: no launch, the check must fail), --emit-code FILE (write the
 * cubin), --maps FILE (copy /proc/self/maps after the run), --ptx FILE (PTXC: the PTX to compile),
 * --emit-ptx FILE (NVRTC: write compute_121 PTX for the same source and exit), --versions, --build-only (build and
 * time the warm builds, no GPU opened), --warm N (0..10 warm builds after the cold path; run.py passes 0 to
 * measured cold processes). */
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(CS_NVRTC)
#include <nvrtc.h>
#define ARM "NVRTC"
#elif defined(CS_PTXC)
#include <nvPTXCompiler.h>
#define ARM "PTXC"
#else
#error "build with -DCS_NVRTC or -DCS_PTXC"
#endif

#define N_TOTAL (1u << 20)
#define THREADS 128u
#define CHUNK   (THREADS * 1024u)
#define POISON  0xFFC0FFEEu
#define WARM    10

/* The CUDA C++ source of the NVRTC arm. coldstart/vecadd.ptx is NVRTC's compute_121 PTX for this same string. */
#if defined(CS_NVRTC)
static const char src[] =
    "extern \"C\" __global__ void vecadd(const float*a,const float*b,float*c,unsigned n)"
    "{unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]=a[i]+b[i];}";
#endif

static uint64_t now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static uint64_t fnv(const void *p, size_t n) { const uint8_t *b = p; uint64_t h = 0xcbf29ce484222325ull; for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ull; } return h; }
static int write_file(const char *path, const void *p, size_t n) { FILE *f = fopen(path, "wb"); if (!f) return -1; size_t w = fwrite(p, 1, n, f); return (fclose(f) == 0 && w == n) ? 0 : -1; }
static int copy_maps(const char *path) {
    FILE *in = fopen("/proc/self/maps", "r"), *out = fopen(path, "w"); char buf[4096]; size_t k;
    if (!in || !out) return -1;
    while ((k = fread(buf, 1, sizeof buf, in)) > 0) fwrite(buf, 1, k, out);
    fclose(in); return fclose(out);
}

#if defined(CS_NVRTC)
static int compile(const char *opt, int want_ptx, char **out, size_t *n) {
    nvrtcProgram p;
    if (nvrtcCreateProgram(&p, src, "vecadd.cu", 0, NULL, NULL) != NVRTC_SUCCESS) return -1;
    nvrtcResult e = nvrtcCompileProgram(p, 1, &opt);
    if (e != NVRTC_SUCCESS) { fprintf(stderr, "cs: nvrtcCompileProgram: %s\n", nvrtcGetErrorString(e)); nvrtcDestroyProgram(&p); return -1; }
    if ((want_ptx ? nvrtcGetPTXSize(p, n) : nvrtcGetCUBINSize(p, n)) != NVRTC_SUCCESS || *n == 0) { nvrtcDestroyProgram(&p); return -1; }
    *out = malloc(*n);
    if (!*out || (want_ptx ? nvrtcGetPTX(p, *out) : nvrtcGetCUBIN(p, *out)) != NVRTC_SUCCESS) { nvrtcDestroyProgram(&p); return -1; }
    nvrtcDestroyProgram(&p);
    return 0;
}
static int build(const char *ptx, char **out, size_t *n) { (void)ptx; return compile("--gpu-architecture=sm_121", 0, out, n); }
#else
static int build(const char *ptx, char **out, size_t *n) {
    nvPTXCompilerHandle h;
    const char *opt[] = { "--gpu-name=sm_121" };
    if (nvPTXCompilerCreate(&h, strlen(ptx), ptx) != NVPTXCOMPILE_SUCCESS) return -1;
    if (nvPTXCompilerCompile(h, 1, opt) != NVPTXCOMPILE_SUCCESS) { nvPTXCompilerDestroy(&h); return -1; }
    if (nvPTXCompilerGetCompiledProgramSize(h, n) != NVPTXCOMPILE_SUCCESS || *n == 0) { nvPTXCompilerDestroy(&h); return -1; }
    *out = malloc(*n);
    if (!*out || nvPTXCompilerGetCompiledProgram(h, *out) != NVPTXCOMPILE_SUCCESS) { nvPTXCompilerDestroy(&h); return -1; }
    nvPTXCompilerDestroy(&h);
    return 0;
}
static char *slurp(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1); if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); return NULL; }
    b[n] = 0; fclose(f); return b;
}
#endif

#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = NULL; cuGetErrorName(r_, &s_); \
    fprintf(stderr, "cs: %s: %s\n", #x, s_ ? s_ : "?"); return 2; } } while (0)

static int versions(void) {
    int drv = 0; cuDriverGetVersion(&drv);
#if defined(CS_NVRTC)
    int ma = 0, mi = 0; nvrtcVersion(&ma, &mi);
    printf("{\"arm\":\"%s\",\"cuDriverGetVersion\":%d,\"nvrtcVersion\":\"%d.%d\"}\n", ARM, drv, ma, mi);
#else
    unsigned ma = 0, mi = 0; nvPTXCompilerGetVersion(&ma, &mi);
    printf("{\"arm\":\"%s\",\"cuDriverGetVersion\":%d,\"nvPTXCompilerGetVersion\":\"%u.%u\"}\n", ARM, drv, ma, mi);
#endif
    return 0;
}

static int nwarm = WARM;   /* --warm N (0..WARM): measured cold processes use 0 so process time holds the cold path only */
static int warm_builds(const char *ptx, uint64_t *warm) {
    for (int r = 0; r < nwarm; r++) {
        char *w = NULL; size_t wn = 0; uint64_t s = now_ns();
        if (build(ptx, &w, &wn)) return -1;
        warm[r] = now_ns() - s; free(w);
    }
    return 0;
}
static void print_warm(const uint64_t *warm) { for (int r = 0; r < nwarm; r++) printf("%s%llu", r ? "," : "", (unsigned long long)warm[r]); }

int main(int argc, char **argv) {
    const uint64_t t_main = now_ns();
    int skip_run = 0, build_only = 0;
    const char *emit_code = NULL, *maps = NULL, *ptx_path = "coldstart/vecadd.ptx";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--skip-run")) skip_run = 1;
        else if (!strcmp(argv[i], "--build-only")) build_only = 1;
        else if (!strcmp(argv[i], "--warm") && i + 1 < argc) { nwarm = atoi(argv[++i]); if (nwarm < 0 || nwarm > WARM) return 2; }
        else if (!strcmp(argv[i], "--emit-code") && i + 1 < argc) emit_code = argv[++i];
        else if (!strcmp(argv[i], "--maps") && i + 1 < argc) maps = argv[++i];
        else if (!strcmp(argv[i], "--ptx") && i + 1 < argc) ptx_path = argv[++i];
        else if (!strcmp(argv[i], "--versions")) return versions();
#if defined(CS_NVRTC)
        else if (!strcmp(argv[i], "--emit-ptx") && i + 1 < argc) {
            char *p; size_t n;
            if (compile("--gpu-architecture=compute_121", 1, &p, &n)) return 2;
            return write_file(argv[i + 1], p, strlen(p)) ? 2 : 0;
        }
#endif
        else { fprintf(stderr, "usage: cs_%s [--skip-run|--build-only] [--warm N] [--emit-code FILE] [--maps FILE] [--ptx FILE] [--versions]\n", ARM); return 2; }
    }
    (void)ptx_path;

    /* prep: host inputs (and, for PTXC, the PTX text) in memory. Not compared between arms. */
    const size_t bytes = (size_t)N_TOTAL * sizeof(float);
    float *fa = aligned_alloc(64, bytes), *fb = aligned_alloc(64, bytes);
    uint32_t *wc0 = aligned_alloc(64, bytes), *wc = aligned_alloc(64, bytes);
    if (!fa || !fb || !wc0 || !wc) return 2;
    for (uint32_t i = 0; i < N_TOTAL; i++) { fa[i] = (float)i * 0.25f; fb[i] = 1.0f / (float)(1 + i % 7); wc0[i] = POISON; }
#if defined(CS_PTXC)
    char *ptx = slurp(ptx_path);
    if (!ptx) { fprintf(stderr, "cs: cannot read %s\n", ptx_path); return 2; }
#else
    const char *ptx = NULL;
#endif
    const uint64_t t_prep = now_ns();

    /* build: kernel description in memory -> loadable device code in host memory */
    char *cubin = NULL; size_t cubin_n = 0;
    if (build(ptx, &cubin, &cubin_n)) { fprintf(stderr, "cs: build failed\n"); return 2; }
    const uint64_t t_build = now_ns();

    /* --build-only: no GPU is opened; the build, WARM more builds, the code written if asked */
    if (build_only) {
        uint64_t warm[WARM];
        if (warm_builds(ptx, warm)) return 2;
        if (emit_code && write_file(emit_code, cubin, cubin_n)) return 2;
        printf("{\"arm\":\"%s\",\"build_only\":1,\"t_main_ns\":%llu,\"prep_ns\":%llu,\"build_ns\":%llu,\"warm_build_ns\":[", ARM,
               (unsigned long long)t_main, (unsigned long long)(t_prep - t_main), (unsigned long long)(t_build - t_prep));
        print_warm(warm);
        printf("],\"code_bytes\":%zu,\"code_fnv\":\"%016llx\"}\n", cubin_n, (unsigned long long)fnv(cubin, cubin_n));
        return 0;
    }

    /* open: no GPU state in this process -> a current context */
    CUdevice dev; CUcontext ctx;
    CK(cuInit(0)); CK(cuDeviceGet(&dev, 0)); CK(cuDevicePrimaryCtxRetain(&ctx, dev)); CK(cuCtxSetCurrent(ctx));
    const uint64_t t_open = now_ns();

    /* load: device code in host memory -> a launchable function */
    CUmodule mod; CUfunction fn;
    CK(cuModuleLoadData(&mod, cubin)); CK(cuModuleGetFunction(&fn, mod, "vecadd"));
    const uint64_t t_load = now_ns();

    /* data: three device buffers, inputs and the poisoned output copied in */
    CUdeviceptr a, b, c;
    CK(cuMemAlloc(&a, bytes)); CK(cuMemAlloc(&b, bytes)); CK(cuMemAlloc(&c, bytes));
    CK(cuMemcpyHtoD(a, fa, bytes)); CK(cuMemcpyHtoD(b, fb, bytes)); CK(cuMemcpyHtoD(c, wc0, bytes));
    const uint64_t t_data = now_ns();

    /* run: eight launches on the default stream, then wait for all of them */
    if (!skip_run) {
        for (uint32_t base = 0; base < N_TOTAL; base += CHUNK) {
            uint64_t off = (uint64_t)base * sizeof(float);
            CUdeviceptr pa = a + off, pb = b + off, pc = c + off;
            unsigned n = N_TOTAL - base < CHUNK ? N_TOTAL - base : CHUNK;
            void *params[] = { &pa, &pb, &pc, &n };
            CK(cuLaunchKernel(fn, (n + THREADS - 1) / THREADS, 1, 1, THREADS, 1, 1, 0, NULL, params, NULL));
        }
    }
    CK(cuCtxSynchronize());
    const uint64_t t_run = now_ns();

    /* readback: the output to host memory, every element checked bit for bit */
    CK(cuMemcpyDtoH(wc, c, bytes));
    uint32_t wrong = 0, unwritten = 0;
    for (uint32_t i = 0; i < N_TOTAL; i++) {
        float want = fa[i] + fb[i];
        if (wc[i] == POISON) unwritten++;
        if (memcmp(&want, &wc[i], sizeof want) != 0) wrong++;
    }
    const uint64_t t_check = now_ns();

    /* after the cold path: WARM more builds in the same process */
    uint64_t warm[WARM];
    if (warm_builds(ptx, warm)) return 2;
    if (maps && copy_maps(maps)) return 2;
    if (emit_code && write_file(emit_code, cubin, cubin_n)) return 2;
    const uint64_t code_fnv = fnv(cubin, cubin_n), out_fnv = fnv(wc, bytes);

    const uint64_t t_close0 = now_ns();
    CK(cuMemFree(a)); CK(cuMemFree(b)); CK(cuMemFree(c)); CK(cuModuleUnload(mod)); CK(cuDevicePrimaryCtxRelease(dev));
    const uint64_t t_close = now_ns();

    printf("{\"arm\":\"%s\",\"t_main_ns\":%llu,\"prep_ns\":%llu,\"build_ns\":%llu,\"open_ns\":%llu,\"load_ns\":%llu,"
           "\"data_ns\":%llu,\"run_ns\":%llu,\"check_ns\":%llu,\"close_ns\":%llu,\"warm_build_ns\":[",
           ARM, (unsigned long long)t_main, (unsigned long long)(t_prep - t_main), (unsigned long long)(t_build - t_prep),
           (unsigned long long)(t_open - t_build), (unsigned long long)(t_load - t_open), (unsigned long long)(t_data - t_load),
           (unsigned long long)(t_run - t_data), (unsigned long long)(t_check - t_run), (unsigned long long)(t_close - t_close0));
    print_warm(warm);
    printf("],\"code_bytes\":%zu,\"code_fnv\":\"%016llx\",\"out_fnv\":\"%016llx\",\"wrong\":%u,\"unwritten\":%u,\"skip_run\":%d}\n",
           cubin_n, (unsigned long long)code_fnv, (unsigned long long)out_fnv, wrong, unwritten, skip_run);
    return wrong ? 1 : 0;
}

#!/usr/bin/env bash
# coldstart/build.sh: build the three cold-start arms from pinned sources (coldstart/PROTOCOL.md section 3).
# CPU only, no GPU. Usage: coldstart/build.sh OUTDIR   (default /tmp/h2h-coldstart-build)
# Env: CUDAPP_REF=path of an existing CUDApp clone to borrow objects from (optional), CUDA=/usr/local/cuda-13.0.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-/tmp/h2h-coldstart-build}
pin=cc0908bdc0a0377f2166e9328dcf1d985457e27f
cuda=${CUDA:-/usr/local/cuda-13.0}
mkdir -p "$out"
out=$(cd "$out" && pwd)

if [ ! -d "$out/cudapp/.git" ]; then
  ref=()
  [ -n "${CUDAPP_REF:-}" ] && ref=(--reference-if-able "$CUDAPP_REF")
  git clone --quiet "${ref[@]}" https://github.com/aien-dev/CUDApp "$out/cudapp"
fi
git -C "$out/cudapp" -c advice.detachedHead=false checkout --quiet "$pin"
[ "$(git -C "$out/cudapp" rev-parse HEAD)" = "$pin" ] || { echo "build: CUDApp is not at $pin" >&2; exit 1; }
[ -z "$(git -C "$out/cudapp" status --porcelain)" ] || { echo "build: CUDApp tree is dirty" >&2; exit 1; }

# CUDApp's library and its own hello_gpu, with CUDApp's own Makefile and flags (default encoder: om).
make -C "$out/cudapp" -s -j5 build/libomega_gpu.a build/hello_gpu
# cs_cudapp with CUDApp's exact CFLAGS (including -Werror), linked like build/hello_gpu.
make -C "$out/cudapp" -s --eval "cs_cudapp: ; \$(CC) \$(CFLAGS) -I. -o $out/cs_cudapp $here/cs_cudapp.c \$(OUT_DIR)/libomega_gpu.a -lpthread -lm" cs_cudapp
rm -f "$out/cudapp/build/cs_cudapp.d" "$out/cs_cudapp.d"

common=(-std=gnu11 -Wall -Wextra -Werror -O2 -D_GNU_SOURCE -I"$cuda/include")
gcc "${common[@]}" -DCS_NVRTC -o "$out/cs_nvrtc" "$here/cs_cuda.c" -L"$cuda/lib64" -Wl,-rpath,"$cuda/lib64" -lnvrtc -lcuda
gcc "${common[@]}" -DCS_PTXC -o "$out/cs_ptxc" "$here/cs_cuda.c" "$cuda/lib64/libnvptxcompiler_static.a" -lcuda -lpthread -lm -ldl -lrt

# The committed PTX must be what NVRTC emits for the same source today (drift check).
"$out/cs_nvrtc" --emit-ptx "$out/vecadd.ptx"
cmp -s "$out/vecadd.ptx" "$here/vecadd.ptx" || { echo "build: coldstart/vecadd.ptx differs from NVRTC output ($out/vecadd.ptx)" >&2; exit 1; }

{
  echo "cudapp_pin $pin"
  echo "gcc $(gcc -dumpfullversion)"
  for f in cs_cudapp cs_nvrtc cs_ptxc cudapp/build/hello_gpu cudapp/build/libomega_gpu.a; do
    echo "sha256 $f $(sha256sum "$out/$f" | cut -d' ' -f1)"
  done
  echo "sha256 vecadd.ptx $(sha256sum "$here/vecadd.ptx" | cut -d' ' -f1)"
  echo "versions $("$out/cs_nvrtc" --versions)"
  echo "versions $("$out/cs_ptxc" --versions)"
} > "$out/build.txt"
cat "$out/build.txt"
echo "build: OK $out"

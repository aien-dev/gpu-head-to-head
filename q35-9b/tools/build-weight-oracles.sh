#!/bin/sh
# Verification-only. Requires an existing pinned container and Omega kernel object.
set -eu
[ "$#" -eq 2 ] || { echo 'usage: build-weight-oracles.sh NEW_OUT_DIR OMEGA_KERNEL_OBJECT' >&2; exit 2; }
SRC=$(cd "$(dirname "$0")" && pwd)
OUT=$(realpath -m "$1")
OBJECT=$(realpath "$2")
IMAGE=nvcr.io/nvidia/vllm@sha256:557747337846eecf34486866f3d992383c6f8af869fdb0e365b5093fcaee02c2
docker image inspect "$IMAGE" > /dev/null
mkdir "$OUT" # Refuse to replace earlier evidence or binaries.
cc --version > "$OUT/host-compiler.txt"
sha256sum "$OBJECT" "$SRC/weight-numeric-glue.c" "$SRC/torch-cpu-exp.cpp" > "$OUT/sources.sha256"
cc -std=gnu11 -O2 -Wall -Wextra -Werror -o "$OUT/weight-numeric-glue" "$SRC/weight-numeric-glue.c" "$OBJECT" -lm
docker run --rm --pull=never --runtime runc --network none --read-only \
  -e NVIDIA_VISIBLE_DEVICES=void --tmpfs /tmp -v "$SRC:/src:ro" -v "$OUT:/out" \
  --entrypoint /bin/sh "$IMAGE" -c '
    set -eu
    T=/usr/local/lib/python3.12/dist-packages/torch
    /usr/bin/g++ --version
    head -9 "$T/version.py"
    /usr/bin/g++ -std=c++20 -O2 -Wall -Wextra -Werror \
      -isystem "$T/include" -isystem "$T/include/torch/csrc/api/include" \
      /src/torch-cpu-exp.cpp -L"$T/lib" -Wl,-rpath,"$T/lib" -ltorch_cpu -lc10 -o /out/torch-cpu-exp
    ldd /out/torch-cpu-exp
  ' > "$OUT/container-build.txt" 2>&1
sha256sum "$OUT/weight-numeric-glue" "$OUT/torch-cpu-exp" > "$OUT/binaries.sha256"

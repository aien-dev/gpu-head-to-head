#!/bin/sh
# Verification oracle only: pinned ATen CPU library, no Python entrypoint or GPU model.
set -eu
ORACLE=${STACK_TORCH_ORACLE_DIR:-/home/drakestapleton/handoffs/stack-comparison-20261029/torch-cpu-oracle}
exec docker run --rm -i --runtime runc --network none --read-only \
  -e NVIDIA_VISIBLE_DEVICES=void -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -v "$ORACLE:/tool:ro" --entrypoint /tool/torch-cpu-exp \
  nvcr.io/nvidia/vllm@sha256:557747337846eecf34486866f3d992383c6f8af869fdb0e365b5093fcaee02c2 "$@"

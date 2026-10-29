# Ordered weight verification

The old sorted-value checks remain historical evidence. They cannot establish
the tensor layout: see `results/WEIGHTS.md.correction.json` and its Wolfram
counterexample. This check compares named tensors at their expected positions.

The converter reference is llama.cpp
`7aca0d2ea8d92b56911233b822db01f9fbb1ae50`, particularly
`conversion/qwen.py` and `gguf-py/gguf/tensor_mapping.py`. The checker validates
Q/K prefixes, grouped-to-tiled V-head rows, output-projection columns, conv1d
channels, scalar head parameters, widening and zero-centred norm folding.
Unknown names, types, shapes, malformed lengths and missing coverage refuse the
check. Model files are opened read-only.

`tools/weight-files.mjs` and `tools/ordered-weights.mjs` only catalogue and move
bytes. Arithmetic transformations call existing seeded-Omega `bin`/`sexp` units
through `weight-numeric-glue.c`. For the exact converter exponential, the final
check uses the pinned converter's ATen CPU `exp` through `torch-cpu-exp.cpp`.
This is an independent verification oracle, never an engine/build dependency.
No Python process or GPU model is started by that helper. The library does link
libcudart; this is not a claim that its process has no NVIDIA libraries.

The verification dependency is Torch `2.14.0a0+b2c75dd062.nv26.09` in image digest
`557747337846eecf34486866f3d992383c6f8af869fdb0e365b5093fcaee02c2`.
Its role is to reproduce the converter's CPU numerical function. No Omega
replacement is planned for this independent oracle. Production remains outside
this check. Build logs record the host/container compilers and binary hashes.

The initial Omega exponential reference differs from the converter's values by
at most one binary32 ULP in the observed inputs, as calculated by the retained
Wolfram query `exp-oracle-difference.wl`. That attempt remains a mismatch record.
No tolerance was widened: the final comparison uses exact bytes from the pinned
converter library. Both numeric helpers also match the retained independent
Wolfram control vectors and refuse malformed byte inputs.

## Reproduce

Use the same pinned model files and compiler/bridge/backend artifacts recorded
in `resumption.json`. Build the verifier against that Omega kernel object:

```sh
sh q35-9b/tools/build-weight-oracles.sh NEW_ORACLE_DIR /absolute/path/omega_ggml_kernels.o
node --test q35-9b/tools/test-ordered-weights.mjs
STACK_TORCH_ORACLE_DIR=/absolute/NEW_ORACLE_DIR node q35-9b/tools/ordered-weights.mjs \
  HF_DIR BF16_GGUF NEW_ORACLE_DIR/weight-numeric-glue NEW_REPORT.json \
  --negexp-helper="$PWD/q35-9b/tools/torch-cpu-exp.sh"
```

The build and report refuse existing outputs. The optional `--only=NAME` is a
partial diagnostic and never returns a whole-model verdict. `--mutant-order`
deliberately leaves the original head order in the expected buffer; it preserves
the values but must fail against a genuinely reordered target tensor.

## Observed scope

The source-bound local report `ordered-weight-check-3.json` compares all 427
named text tensors exactly, with no mismatches. The 15 GGUF MTP tensors are
explicitly excluded from plain decoding, as in the original protocol; image and
MTP source tensors are separately listed. The real-tensor wrong-order mutant
for `blk.0.attn_qkv.weight` fails the intended byte comparison. Synthetic parser,
mapping and corruption controls pass.

This verifies the named file transformations for the pinned checkpoint. It does
not prove every inference operation, full backend model qualification, output
correctness, deployment, or performance. Those gates remain separate.

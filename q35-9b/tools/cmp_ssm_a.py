#!/usr/bin/env python3
"""Throwaway: max ULP distance between GGUF ssm_a and -exp(A_log) from the HF safetensors, per layer."""
import glob, sys, re
from safetensors import safe_open
import numpy as np, torch, gguf
HF, GG = sys.argv[1], sys.argv[2]
alog = {}
for f in sorted(glob.glob(HF + "/model.safetensors-*.safetensors")):
    with safe_open(f, "pt") as s:
        for k in s.keys():
            m = re.search(r"layers\.(\d+)\.linear_attn\.A_log$", k)
            if m and "mtp" not in k:
                alog[int(m.group(1))] = s.get_tensor(k)
worst = 0
for t in gguf.GGUFReader(GG).tensors:
    m = re.match(r"blk\.(\d+)\.ssm_a$", t.name)
    if not m:
        continue
    g = np.sort(np.array(t.data, dtype=np.float32).ravel())  # converter may reorder value heads
    ref = np.sort((-torch.exp(alog[int(m.group(1))].float())).numpy().ravel())
    ulp = np.abs(g.view(np.int32).astype(np.int64) - ref.view(np.int32).astype(np.int64)).max()
    rel = np.abs(g - ref).max() / np.abs(ref).max()
    worst = max(worst, ulp)
    print(t.name, "max_ulp", ulp, "max_rel", "%.3g" % rel, "A_log dtype", alog[int(m.group(1))].dtype)
print("WORST_ULP", worst)

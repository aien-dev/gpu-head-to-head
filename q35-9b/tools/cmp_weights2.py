#!/usr/bin/env python3
"""Throwaway check, part 2: for GGUF tensors that are not byte-identical to an HF tensor, test whether the
values are the same up to (a) a reordering (sorted-value multiset equal after exact bf16->f32 widening),
(b) +1 (zero-centred RMSNorm stored as 1+w), or (c) -exp(A_log). Prints one line per class."""
import hashlib, json, sys, glob
from safetensors import safe_open
import torch, gguf
import numpy as np

HF, GG, PREV = sys.argv[1], sys.argv[2], sys.argv[3]
prev = json.load(open(PREV))
todo = {n for n, _, _ in prev["not_identical"] if not n.startswith("blk.32.") and not n.startswith("nextn")}

def key(a):
    return hashlib.sha256(np.sort(a.astype(np.float32).ravel()).tobytes()).hexdigest()

hf = {}
for f in sorted(glob.glob(HF + "/model.safetensors-*.safetensors")):
    with safe_open(f, "pt") as s:
        for k in s.keys():
            if "visual" in k or k.startswith("mtp."):
                continue
            t = s.get_tensor(k).float().numpy()
            for tag, v in (("same", t), ("plus1", t + np.float32(1)), ("negexp", -np.exp(t))):
                hf.setdefault(key(v), []).append((k, tag))

r = gguf.GGUFReader(GG)
res = {}
for t in r.tensors:
    if t.name not in todo:
        continue
    a = np.array(t.data)
    if t.tensor_type.name == "BF16":
        a = (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    m = hf.get(key(a))
    cls = (t.name.split(".", 2)[-1], t.tensor_type.name, m[0][1] if m else "NO_MATCH")
    res[cls] = res.get(cls, 0) + 1
for (n, ty, how), c in sorted(res.items()):
    print(n, ty, how, "x%d" % c)

#!/usr/bin/env python3
"""Throwaway check (host tool): are llama.cpp's BF16 GGUF tensors byte-identical to the HF safetensors?
Matches tensors by element count + bytes; reports identical, transformed (same count, different bytes),
and unmatched, by name. Run inside the vLLM container (torch, safetensors) with gguf-py installed."""
import hashlib, json, sys, glob
from safetensors import safe_open
import torch
import gguf

HF, GG = sys.argv[1], sys.argv[2]
hf = {}
for f in sorted(glob.glob(HF + "/model.safetensors-*.safetensors")):
    with safe_open(f, "pt") as s:
        for k in s.keys():
            if ".visual." in k or k.startswith("mtp.") or "model.visual" in k:
                continue
            t = s.get_tensor(k)
            hf[k] = t
by_hash = {}
for k, t in hf.items():
    b = t.contiguous().view(torch.uint8).numpy().tobytes() if t.dtype == torch.bfloat16 else None
    if b is not None:
        by_hash.setdefault(hashlib.sha256(b).hexdigest(), []).append(k)

r = gguf.GGUFReader(GG)
out = {"identical": [], "not_identical": []}
for t in r.tensors:
    h = hashlib.sha256(bytes(t.data.tobytes())).hexdigest()
    name = t.name
    typ = t.tensor_type.name
    if h in by_hash:
        out["identical"].append([name, typ, by_hash[h][0]])
    else:
        out["not_identical"].append([name, typ, [int(x) for x in t.shape]])
print(json.dumps({"gguf_tensors": len(r.tensors), "hf_text_tensors": len(hf),
                  "identical": len(out["identical"]), "not_identical": len(out["not_identical"])}))
kinds = {}
for name, typ, shape in out["not_identical"]:
    key = (name.split(".", 2)[-1] if name.startswith("blk.") else name, typ)
    kinds[key] = kinds.get(key, 0) + 1
for (k, typ), n in sorted(kinds.items()):
    print("NOT_IDENTICAL", k, typ, "x%d" % n)
json.dump(out, open(sys.argv[3], "w"), indent=0)

#!/usr/bin/env python3
"""Output agreement (PROTOCOL.md M6): token counts only, no rates.

Run inside the pinned vLLM container (needs `tokenizers`):
  agree.py <model_dir> <repo>  > results/agree-counts.json
For each pair of (arm, round) agreement runs and each prompt id, tokenizes both outputs with the
checkpoint's tokenizer.json and records len_a, len_b, common-prefix length and byte identity.
Rates are computed from these integers in Wolfram.
"""
import glob
import itertools
import json
import sys

from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1] + "/tokenizer.json")
runs = {}
for path in sorted(glob.glob(sys.argv[2] + "/results/raw/*.jsonl")):
    for line in open(path):
        r = json.loads(line)
        if r["metric"] != "agree" or r["warmup"] or r["arm"] == "STUB":
            continue
        if r["error"] is None and r["http_status"] == 200:
            runs.setdefault("%s-%s" % (r["arm"], r["round"]), {})[r["prompt_id"]] = r["text"]

out = []
for a, b in itertools.combinations(sorted(runs), 2):
    for pid in sorted(set(runs[a]) & set(runs[b])):
        ta = tok.encode(runs[a][pid], add_special_tokens=False).ids
        tb = tok.encode(runs[b][pid], add_special_tokens=False).ids
        m = 0
        for x, y in zip(ta, tb):
            if x != y:
                break
            m += 1
        out.append({"a": a, "b": b, "prompt": pid, "len_a": len(ta), "len_b": len(tb),
                    "common_prefix": m, "identical": runs[a][pid] == runs[b][pid]})
json.dump({"runs": {k: sorted(v) for k, v in runs.items()}, "pairs": out}, sys.stdout, indent=1)

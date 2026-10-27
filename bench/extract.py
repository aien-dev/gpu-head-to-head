#!/usr/bin/env python3
"""Turn results/raw/*.jsonl into one Wolfram Language data file, with no arithmetic.

Every derived number (rates, medians, IQRs) is computed in Wolfram from these raw fields
(PROTOCOL.md section 7). Rows are kept as {arm, round, metric, rep, batch, stream, load1_start,
t_send, t_first, t_last, prompt_tokens, completion_tokens, busy_retries, ok}. Warmup rows are
dropped here (they stay in the raw files). Timestamps are printed with full repr precision.
"""
import glob
import json
import sys

REPO = sys.argv[1]
rows = []
for path in sorted(glob.glob(REPO + "/results/raw/*.jsonl")):
    for line in open(path):
        r = json.loads(line)
        if r["warmup"] or r["arm"] == "STUB":
            continue
        u = r.get("usage") or {}
        ok = (r["error"] is None and r["http_status"] == 200 and r["t_first"] is not None
              and u.get("completion_tokens") is not None)
        def num(x):
            return "Missing[]" if x is None else repr(x).replace("e", "*^")
        rows.append("{%s}" % ", ".join([
            json.dumps(r["arm"]), json.dumps(r["round"]), json.dumps(r["metric"]), str(r["rep"]),
            json.dumps(r["batch_id"]), str(r["stream_index"]), num(r["load1_start"]),
            num(r["t_send"]), num(r["t_first"]), num(r["t_last"]),
            num(u.get("prompt_tokens")), num(u.get("completion_tokens")),
            str(r.get("busy_retries", 0)), "True" if ok else "False"]))
print("h2hRows = {\n" + ",\n".join(rows) + "\n};")

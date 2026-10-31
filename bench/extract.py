#!/usr/bin/env python3
"""Turn results/raw/*.jsonl into one Wolfram Language data file, with no arithmetic.

Every derived number (rates, medians, IQRs) is computed in Wolfram from these raw fields
(PROTOCOL.md section 7). Rows are kept as {arm, round, metric, rep, batch, stream, load1_start,
t_send, t_first, t_last, prompt_tokens, completion_tokens, busy_retries, ok}. Warmup rows are
dropped here (they stay in the raw files), and so are rows of cells voided under amendments A2/A4
(results/voids.jsonl; a record counts only if its reason names the kthread or a non-empty
offender list, A4(b)). Of several non-void tries of one cell only the lowest try is kept (A4(c)).
Timestamps are printed with full repr precision.
"""
import glob
import json
import os
import sys

REPO = sys.argv[1]


def honoured(v):
    why = v.get("reason") or "kthread"  # A2-era records without a reason were kthread voids
    return "kthread" in why or ": [" in why


void = set()
if os.path.exists(REPO + "/results/voids.jsonl"):
    void = {v["cell_run"] for v in map(json.loads, filter(str.strip, open(REPO + "/results/voids.jsonl")))
            if honoured(v)}
# New runs are admitted per cell. A missing, duplicate or non-PASS record excludes the cell;
# untouched historical runs without a ledger retain their original extraction semantics.
admission = None
admission_path = REPO + "/results/cell-status.jsonl"
if os.path.exists(admission_path):
    admission = {}
    for line in open(admission_path):
        entry = json.loads(line)
        cr = entry["cell_run"]
        admission[cr] = entry["status"] if cr not in admission else "DUPLICATE"

raw = [json.loads(line) for path in sorted(glob.glob(REPO + "/results/raw/*.jsonl")) for line in open(path)]
first = {}  # cell without its try suffix -> lowest non-void try number
for r in raw:
    cr = r.get("cell_run")
    if cr and cr not in void and (admission is None or admission.get(cr) == "PASS"):
        cell, _, n = cr.rpartition("-t")
        first[cell] = min(first.get(cell, int(n)), int(n))
rows = []
for r in raw:
    cr = r.get("cell_run")
    # A10: rows of a profiled arm (*_NSYS) carry tracer overhead and never enter timing results
    if (r["warmup"] or r["arm"] == "STUB" or r["arm"].endswith("_NSYS") or cr in void
            or (admission is not None and admission.get(cr) != "PASS")):
        continue
    if cr:
        cell, _, n = cr.rpartition("-t")
        if int(n) != first[cell]:
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

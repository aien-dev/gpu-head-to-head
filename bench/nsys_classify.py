#!/usr/bin/env python3
"""PROTOCOL.md A10/A8: classify the kernels of one vLLM decode trace into the l2 timeline buckets.

External test equipment, not part of any arm. Input: the CSV of
  nsys stats -r cuda_gpu_trace -f csv <report>.nsys-rep
(columns "Start (ns)", "Duration (ns)", "Name"). Output: integer nanosecond sums and kernel counts per
bucket as JSON on stdout plus a Wolfram association on the last line; every ratio is taken in Wolfram.
Exit 0 = classified, 2 = INCONCLUSIVE (unclassified time > 10 % of total), 1 = unusable input.

Kernel names are matched case-insensitively, first rule wins. Dense GEMMs carry no role in their
names, so their role comes from the position inside a layer:
  before the token mixer -> mix (QKV / in_proj), after the mixer -> o_proj, after the post-mixer norm
  -> router (gate linear), inside the MoE phase -> shared_gate_up / shared_down alternately;
  any GEMM of >= 1 ms is lm_head, and steps = number of lm_head kernels. Prefill steps inside the
  captured range are counted the same way; the trace covers one decode-c8 cell (A10).
"""
import csv
import json
import sys

BUCKETS = ['gate_up', 'down', 'lm_head', 'gdn', 'mix', 'attn', 'res', 'o_proj', 'router', 'perm',
           'shared_gate_up', 'shared_down', 'act', 'sample', 'copy', 'other']
RULES = [  # (bucket, substrings); first match wins
    ('copy', ['memcpy', 'memset']),
    ('moe', ['marlin_moe', 'moe_wna16', 'fused_moe', 'cutlass_moe', 'grouped']),
    ('router', ['topk', 'gating', 'routing']),
    ('perm', ['moe_align', 'moe_sum', 'permute', 'sort', 'count_and_sort', 'scatter', 'gather']),
    ('gdn', ['gated_delta', 'recurrent', 'causal_conv1d', 'gdn', 'l2norm', 'fused_sigmoid_gating']),
    ('attn', ['flashinfer', 'batchdecode', 'batchprefill', 'flash_fwd', 'fmha', 'reshape_and_cache', 'rotary']),
    ('res', ['rms_norm', 'rmsnorm', 'fused_add', 'layernorm']),
    ('act', ['silu', 'act_and_mul']),
    ('sample', ['argmax', 'sampl', 'softmax', 'logits']),
    ('gemm', ['gemm', 'gemv', 'cutlass', 'nvjet', 'cublas', 'matmul', 'splitk', 'scaled_mm', 'marlin', 'sm90_', 'sm100_']),
]
LM_HEAD_NS = 1_000_000


def rule(name):
    low = name.lower()
    for bucket, keys in RULES:
        if any(k in low for k in keys):
            return bucket
    return 'other'


def classify(rows):
    ns = dict.fromkeys(BUCKETS, 0)
    count = dict.fromkeys(BUCKETS, 0)
    other = {}
    phase, moe_gemms, shared = 'pre', 0, 0
    for start, dur, name in rows:
        kind = rule(name)
        if kind == 'moe':
            if phase != 'moe':
                phase, moe_gemms, shared = 'moe', 0, 0
            bucket = 'gate_up' if moe_gemms % 2 == 0 else 'down'
            moe_gemms += 1
        elif kind == 'router':
            bucket, phase = 'router', 'moe'
        elif kind in ('gdn', 'attn'):
            bucket, phase = kind, 'core'
        elif kind == 'res':
            bucket = 'res'
            if phase == 'moe':
                phase = 'pre'
        elif kind == 'gemm':
            if dur >= LM_HEAD_NS:
                bucket, phase = 'lm_head', 'pre'
            elif phase == 'pre':
                bucket = 'mix'
            elif phase == 'core':
                bucket, phase = 'o_proj', 'post'
            elif phase == 'post':
                bucket = 'router'
            else:
                bucket = 'shared_gate_up' if shared % 2 == 0 else 'shared_down'
                shared += 1
        else:
            bucket = kind
            if kind == 'other':
                other[name] = other.get(name, 0) + dur
        ns[bucket] += dur
        count[bucket] += 1
    return ns, count, other


def read(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        need = {'Start (ns)', 'Duration (ns)', 'Name'}
        if not reader.fieldnames or not need <= set(reader.fieldnames):
            raise ValueError('missing columns %s' % sorted(need - set(reader.fieldnames or [])))
        rows = [(int(r['Start (ns)']), int(r['Duration (ns)']), r['Name']) for r in reader]
    rows.sort()
    if not rows:
        raise ValueError('no kernels')
    return rows


def main(argv):
    if len(argv) != 2:
        print('usage: nsys_classify.py cuda_gpu_trace.csv', file=sys.stderr)
        return 1
    try:
        rows = read(argv[1])
    except (OSError, ValueError, KeyError) as e:
        print('unusable input: %s' % e, file=sys.stderr)
        return 1
    ns, count, other = classify(rows)
    total = sum(ns.values())
    top = sorted(other.items(), key=lambda kv: -kv[1])[:10]
    verdict = 'INCONCLUSIVE' if ns['other'] * 10 > total else 'CLASSIFIED'
    out = dict(verdict=verdict, kernels=len(rows), total_ns=total, steps=count['lm_head'],
               ns=ns, count=count, other_top=[dict(name=n, ns=v) for n, v in top])
    print(json.dumps(out, indent=1))
    assoc = ', '.join('"%s" -> %d' % (b, ns[b]) for b in BUCKETS)
    print('h2hNsys = <|"total_ns" -> %d, "steps" -> %d, "ns" -> <|%s|>|>;' % (total, count['lm_head'], assoc))
    return 2 if verdict == 'INCONCLUSIVE' else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

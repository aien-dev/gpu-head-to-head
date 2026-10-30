#!/usr/bin/env python3
"""Throwaway analysis (host tool): explain every greedy divergence against the HF transformers oracle.

For each arm's agree output and each prompt: retokenize the arm's text, find the first token index k where it
differs from the oracle's greedy ids, then run the oracle model on prompt + oracle_ids[:k] and read the logits
at that position. Report the oracle's logit gap between its own token and the arm's token. A small gap means the
two tokens were a near tie for the reference model too (a precision-level divergence); a large gap means the
arm picked a token the reference model considers clearly worse (a bug signal).
usage: tie_check.py MODEL ORACLE.jsonl OUT.jsonl ARM=raw.jsonl [ARM=raw.jsonl ...]
"""
import json, sys
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

MODEL, ORACLE, OUT = sys.argv[1:4]
arms = dict(a.split("=", 1) for a in sys.argv[4:])
tok = AutoTokenizer.from_pretrained(MODEL)
model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
prompts = {json.loads(l)["id"]: json.loads(l)["text"]
           for l in open("/home/drakestapleton/handoffs/HeadToHead/repo/prompts/decode.jsonl")}
oracle = {r["prompt_id"]: r for r in map(json.loads, open(ORACLE))}

with open(OUT, "w") as f:
    for arm, path in arms.items():
        rows = {}
        for r in map(json.loads, open(path)):
            if r.get("metric", "agree") == "agree" and not r.get("warmup") and not r.get("error"):
                rows.setdefault(r["prompt_id"], r["text"])
        for pid, text in sorted(rows.items()):
            o = oracle[pid]["ids"]
            a = tok(text, add_special_tokens=False)["input_ids"]
            k = next((i for i in range(min(len(a), len(o))) if a[i] != o[i]), None)
            rec = {"arm": arm, "prompt_id": pid, "oracle_len": len(o), "arm_len": len(a), "first_div": k}
            if k is not None:
                p = tok.apply_chat_template([{"role": "user", "content": prompts[pid]}], add_generation_prompt=True,
                                            enable_thinking=False, return_tensors="pt", return_dict=True)
                ids = torch.cat([p["input_ids"][0], torch.tensor(o[:k], dtype=torch.long)]).unsqueeze(0).to("cuda")
                with torch.no_grad():
                    lg = model(input_ids=ids).logits[0, -1].float()
                lp = torch.log_softmax(lg, -1)
                top2 = torch.topk(lg, 2)
                rec.update({"oracle_tok": o[k], "arm_tok": a[k],
                            "logit_gap": float(lg[o[k]] - lg[a[k]]),
                            "oracle_top1_top2_gap": float(top2.values[0] - top2.values[1]),
                            "arm_tok_rank": int((lg > lg[a[k]]).sum()),
                            "logprob_oracle_tok": float(lp[o[k]]), "logprob_arm_tok": float(lp[a[k]]),
                            "oracle_piece": tok.decode([o[k]]), "arm_piece": tok.decode([a[k]])})
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(json.dumps(rec), flush=True)

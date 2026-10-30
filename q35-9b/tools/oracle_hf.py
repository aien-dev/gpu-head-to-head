#!/usr/bin/env python3
"""Throwaway oracle (host tool): greedy BF16 decode of Qwen3.5-9B with Hugging Face transformers, the
model's reference implementation, independent of both llama.cpp and our engine. Same 16 prompts, same
chat template (thinking off), 128 new tokens. Writes one JSON row per prompt with token ids and text."""
import json, sys, time
import torch, transformers
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

MODEL, PROMPTS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(MODEL)
model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
meta = {"transformers": transformers.__version__, "torch": torch.__version__, "model": MODEL}
with open(OUT, "w") as f:
    for line in open(PROMPTS):
        p = json.loads(line)
        ids = tok.apply_chat_template([{"role": "user", "content": p["text"]}], add_generation_prompt=True,
                                      enable_thinking=False, return_tensors="pt", return_dict=True)
        ids = {k: v.to("cuda") for k, v in ids.items()}
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=128, do_sample=False, temperature=None, top_p=None, top_k=None)
        new = out[0, ids["input_ids"].shape[1]:].tolist()
        f.write(json.dumps({"prompt_id": p["id"], "prompt_tokens": ids["input_ids"].shape[1], "ids": new,
                            "text": tok.decode(new, skip_special_tokens=True), "seconds": time.time() - t0,
                            **meta}) + "\n")
        f.flush()
        print(p["id"], len(new), repr(tok.decode(new[:24])), flush=True)

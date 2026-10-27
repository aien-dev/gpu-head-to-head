#!/usr/bin/env python3
"""Generate prompts/decode.jsonl and prompts/prefill.jsonl deterministically.

Needs the `tokenizers` package (run inside the pinned vLLM container, see bench/README) and the
checkpoint's tokenizer.json. Prefill documents are calibrated so that the chat-rendered prompt
(Qwen3.6 template, thinking off) lands on the target token count.
"""
import json
import random
import sys

from tokenizers import Tokenizer

MODEL = sys.argv[1]
OUT = sys.argv[2]
tok = Tokenizer.from_file(MODEL + "/tokenizer.json")

TOPICS = [
    "the history of the printing press", "how a river delta forms", "the life cycle of a star",
    "building a wooden boat by hand", "the first transatlantic telegraph cable",
    "how bees choose a new hive site", "a day in a medieval market town", "the physics of a bicycle",
    "how glaciers shape valleys", "the invention of the mechanical clock",
    "a lighthouse keeper's winter", "how coral reefs grow", "the design of Roman aqueducts",
    "a journey along the Silk Road", "how volcanoes create new islands", "the story of paper money",
]


def chat_len(text):
    # Qwen3.6 chat template, thinking off: user turn, assistant header, empty think block.
    rendered = ("<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"
                "<think>\n\n</think>\n\n")
    return len(tok.encode(rendered, add_special_tokens=False).ids)


WORDS = ("river stone light market harbor engine winter garden letter mountain bridge village "
         "signal window forest copper lantern season voyage archive canal meadow orchard tower "
         "compass ledger quarry furnace beacon cellar harvest pilgrim chapel mill granary").split()
VERBS = "carried crossed watched built opened measured repaired followed recorded traded".split()


def sentence(rng):
    return "The %s near the %s %s the %s before the %s %s again." % (
        rng.choice(WORDS), rng.choice(WORDS), rng.choice(VERBS), rng.choice(WORDS),
        rng.choice(WORDS), rng.choice(VERBS))


def document(target, seed):
    rng = random.Random(seed)
    question = "\n\nQuestion: in one sentence, what is the document above about?"
    sents = []
    while True:
        sents.append(sentence(rng))
        if chat_len(" ".join(sents) + question) >= target:
            break
    # trim back to the closest count not above the target
    while chat_len(" ".join(sents) + question) > target and len(sents) > 1:
        sents.pop()
    text = " ".join(sents) + question
    return text, chat_len(text)


with open(OUT + "/decode.jsonl", "w") as f:
    for i, t in enumerate(TOPICS):
        text = ("Write a long, detailed essay of at least 600 words about %s. "
                "Use several paragraphs and do not stop early." % t)
        f.write(json.dumps({"id": "d%02d" % i, "text": text, "chat_tokens": chat_len(text)}) + "\n")

with open(OUT + "/prefill.jsonl", "w") as f:
    for name, target in (("p256", 256), ("p2k", 2048), ("p8k", 8192)):
        text, n = document(target, seed=target)
        f.write(json.dumps({"id": name, "text": text, "chat_tokens": n, "target": target}) + "\n")
        print(name, target, n, file=sys.stderr)

#!/usr/bin/env python3
"""Generate prompts/ttft.jsonl (PROTOCOL.md A10 TTFT rows) from pinned source texts; stdlib only.

usage: bench/make_ttft_prompts.py LONG2048 LONG8192 LONG32768 LONG256 OUT

The sources are the PrefillGate long prompts (sovereign evidence/merge-train/l369/long-{2048,8192,256}.txt
@ b129e1ea) and the DRA-85 32k prompt (long-8192 x3 + long-2048 x3 + long-256 x2). Every source is
checked against the sha256 below before anything is written; a mismatch exits 2 and writes nothing.
Token counts depend on each server's tokenizer and chat template, so none is claimed here: the run
records usage.prompt_tokens for every request.
"""
import hashlib
import json
import sys

SOURCES = {  # id -> (argv index, sha256 of the source file)
    "t2k": (1, "26a44c508412e213de876eb54bb90390426b086c887a68b90fa78f7470425ee3"),
    "t8k": (2, "2bcc84b25c418cba83df9546344fd4ae28962c314e03defe2aab62a8e6c8abb9"),
    "t32k": (3, "791ef12bc33c74cb23f3318e1543def0a07e6008aa38e8d815381687dd5098b2"),
    "turn2-new": (4, "153a5f88932393b2be56fd81f8914d44218169c614ebc5b2753069c9a09bc3a1"),
}
# The assistant turn between the primed 32k user turn and the new 1k user turn. It is a fixed text,
# not the model's own reply, so the second request is byte-identical across arms.
TURN2_ASSISTANT = "I have read the whole document above and I am ready for your question about it."
TURN2_NEW_REPEAT = 4  # four copies of the 256-token PrefillGate text: about 1k new tokens


def main(argv):
    if len(argv) != 6:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    rows = []
    for pid, (i, want) in SOURCES.items():
        data = open(argv[i], "rb").read()
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            print("%s: %s sha256 %s is not %s" % (pid, argv[i], got, want), file=sys.stderr)
            return 2
        text = data.decode("utf-8")
        if pid == "turn2-new":
            text = "\n".join([text.rstrip("\n")] * TURN2_NEW_REPEAT) + "\n"
        rows.append({"id": pid, "text": text, "source_sha256": want})
    rows.append({"id": "turn2-assistant", "text": TURN2_ASSISTANT, "source_sha256": None})
    with open(argv[5], "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

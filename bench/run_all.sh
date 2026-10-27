#!/usr/bin/env bash
# The interleaved schedule of PROTOCOL.md section 7: R1 = OURS VLLM LLAMA, R2 = LLAMA VLLM OURS.
# Each session waits for load1 < 4 and for /tmp/aien-gb10.lock itself; an arm whose result file
# already holds measured rows for that round is skipped (resume after interruption).
set -u
cd "$(dirname "$0")/.."
for s in OURS:R1 VLLM:R1 LLAMA:R1 LLAMA:R2 VLLM:R2 OURS:R2; do
  arm=${s%:*}; round=${s#*:}
  if [ -s "results/raw/$arm-$round.jsonl" ] && grep -q '"metric": "agree"' "results/raw/$arm-$round.jsonl"; then
    echo "skip $arm $round (done)"; continue
  fi
  bash bench/run_arm.sh "$arm" "$round"
done

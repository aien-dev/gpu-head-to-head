#!/usr/bin/env bash
# Throwaway: capture LLAMA16 (BF16 GGUF) greedy reference outputs for the correctness gate.
# Not a timed cell. Holds /tmp/aien-gb10.lock for its few minutes.
set -u
H=/home/drakestapleton/handoffs/H2H9B; REPO=/home/drakestapleton/handoffs/HeadToHead/repo
LLAMA_DIR=/home/drakestapleton/h2h/dl/llama-bin/llama-b11913
CUDART_DIR=/home/drakestapleton/h2h/dl/cudart-llama-b11913-bin-ubuntu-cuda-13.4-arm64
mkdir -p $H/results/ref
exec 9>/tmp/aien-gb10.lock; flock 9
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Main(H2H-9B): GPU-START LLAMA16 reference capture (lock held, <10 min)" >> /home/drakestapleton/handoffs/BOARD.md
LD_LIBRARY_PATH=$CUDART_DIR:$LLAMA_DIR taskset -c 5-9,15-19 $LLAMA_DIR/llama-server -m /home/drakestapleton/h2h/gguf/q35-9b-bf16.gguf \
  --host 127.0.0.1 --port 18293 -ngl 999 -fa on -c 32768 -np 1 --jinja < /dev/null > $H/results/ref/server.log 2>&1 &
SPID=$!
for _ in $(seq 1 300); do curl -s -m 2 127.0.0.1:18293/v1/models | grep -q q35 && break; sleep 1; done
taskset -c 0-1 python3 $REPO/bench/bench.py --port 18293 --model q35-9b-bf16 --arm LLAMA16 --round REF --metric agree \
  --prompts $REPO/prompts/decode.jsonl --concurrency 1 --max-tokens 128 --warmup 0 --reps 16 \
  --out $H/results/ref/LLAMA16-REF.jsonl --cell-run LLAMA16-REF-agree 2> $H/results/ref/client.log
echo "client exit $?"
kill $SPID; wait $SPID 2>/dev/null
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Main(H2H-9B): GPU-END LLAMA16 reference capture" >> /home/drakestapleton/handoffs/BOARD.md

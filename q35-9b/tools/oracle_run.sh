#!/usr/bin/env bash
# Throwaway: run oracle_hf.py in the pinned vLLM container while holding /tmp/aien-gb10.lock.
set -u
B=/home/drakestapleton/handoffs/BOARD.md
exec 9>/tmp/aien-gb10.lock
flock 9
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Main(H2H-9B): GPU-START HF transformers BF16 oracle (lock held, untimed)" >> $B
docker run --rm --name h2h9b-oracle --runtime nvidia --gpus all --ipc host --cpuset-cpus 5-9,15-19 --entrypoint bash \
  -v /home/drakestapleton/models:/home/drakestapleton/models:ro -v /home/drakestapleton/handoffs:/home/drakestapleton/handoffs \
  nvcr.io/nvidia/vllm:26.09-py3 -c "python3 /home/drakestapleton/handoffs/H2H9B/oracle_hf.py \
  /home/drakestapleton/models/qwen3.5-9b-c2022362 /home/drakestapleton/handoffs/HeadToHead/repo/prompts/decode.jsonl \
  /home/drakestapleton/handoffs/H2H9B/results/ref/HF-ORACLE.jsonl" > /home/drakestapleton/handoffs/H2H9B/results/ref/hf-oracle.log 2>&1
rc=$?
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Main(H2H-9B): GPU-END HF oracle rc=$rc" >> $B
echo "rc=$rc"

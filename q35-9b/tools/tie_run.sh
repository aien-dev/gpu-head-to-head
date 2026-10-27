#!/usr/bin/env bash
# Throwaway: run tie_check.py in the pinned vLLM container under /tmp/aien-gb10.lock.
# usage: tie_run.sh OUT.jsonl ARM=/abs/raw.jsonl [...]
set -u
B=/home/drakestapleton/handoffs/BOARD.md
OUT=$1; shift
exec 9>/tmp/aien-gb10.lock
flock 9
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Main(H2H-9B): GPU-START HF tie check $* (lock held, untimed)" >> $B
docker run --rm --name h2h9b-tie --runtime nvidia --gpus all --ipc host --cpuset-cpus 5-9,15-19 --entrypoint bash \
  -v /home/drakestapleton/models:/home/drakestapleton/models:ro -v /home/drakestapleton/handoffs:/home/drakestapleton/handoffs \
  nvcr.io/nvidia/vllm:26.09-py3 -c "python3 /home/drakestapleton/handoffs/H2H9B/tie_check.py \
  /home/drakestapleton/models/qwen3.5-9b-c2022362 /home/drakestapleton/handoffs/H2H9B/results/ref/HF-ORACLE.jsonl $OUT $*" \
  > "$OUT.log" 2>&1
rc=$?
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Main(H2H-9B): GPU-END HF tie check rc=$rc" >> $B
echo "rc=$rc"

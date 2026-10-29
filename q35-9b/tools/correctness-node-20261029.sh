#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/drakestapleton/handoffs/stack-comparison-20261029/correctness-full-1
REPO=/home/drakestapleton/workspace/stack-comparison-20261029
LLAMA=/home/drakestapleton/h2h/dl/llama-bin/llama-b11913
CUDART=/home/drakestapleton/h2h/dl/llama-bin/cudart-llama-b11913-bin-ubuntu-cuda-13.4-arm64
mkdir "$ROOT"
server_pid=
stop_server() {
    if [ -n "$server_pid" ]; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        server_pid=
    fi
}
cleanup() {
    stop_server
    printf 'CodexStackCompare: full correctness driver ended; all own servers stopped; resident18091 unchanged.\n' >> /home/drakestapleton/handoffs/BOARD.md
}
trap cleanup EXIT
export H2H_SRV_CPUS=5-9,15-19 H2H_CLI_CPUS=0-1
for arm in OURS16 LLAMA16 LLAMA8; do
    printf '%s %s start\n' "$(date -u +%FT%TZ)" "$arm" >> "$ROOT/status.log"
    port=18290
    if [ "$arm" = OURS16 ]; then
        model=q35-9b-ours-bf16
        (
            cd /home/drakestapleton/work/Ours9B/bin16
            exec env AIEN_GB10_LOCK_HELD=1 taskset -c "$H2H_SRV_CPUS" ./aien-serve-http \
                --engine omega --model-dir /home/drakestapleton/models/qwen3.5-9b-c2022362 \
                --model-id "$model" --max-context 32768 --max-tokens-cap 16384 \
                --thinking off --bind 127.0.0.1 --port "$port"
        ) > "$ROOT/$arm-server.log" 2>&1 &
    else
        if [ "$arm" = LLAMA16 ]; then format=bf16;port=18293;else format=q8_0;port=18292;fi
        model=q35-9b-$format
        LD_LIBRARY_PATH="$CUDART:$LLAMA" taskset -c "$H2H_SRV_CPUS" "$LLAMA/llama-server" \
            -m "/home/drakestapleton/h2h/gguf/q35-9b-$format.gguf" --host 127.0.0.1 \
            --port "$port" -ngl 999 -fa on -c 32768 -np 1 --jinja \
            > "$ROOT/$arm-server.log" 2>&1 &
    fi
    server_pid=$!
    ready=0
    for attempt in $(seq 1 120); do
        if curl --fail --silent --max-time 2 "http://127.0.0.1:$port/v1/models" > "$ROOT/$arm-models.json"; then ready=1;break;fi
        kill -0 "$server_pid" 2>/dev/null || { echo "$arm server exited";exit 1; }
        sleep 1
    done
    [ "$ready" -eq 1 ] || { echo "$arm startup timeout";exit 1; }
    ss -ltnH "sport = :$port" > "$ROOT/$arm-listen.txt"
    awk '$6 ~ /\.so/ {print $6}' /proc/"$server_pid"/maps | sort -u > "$ROOT/$arm-maps-so.txt"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$ROOT/$arm-gpu.csv"
    taskset -c "$H2H_CLI_CPUS" node "$REPO/bench/bench-node.mjs" \
        --port "$port" --model "$model" --arm "$arm" --round CORRECTNESS \
        --metric agree --prompts "$REPO/prompts/decode.jsonl" --concurrency 1 \
        --max-tokens 128 --warmup 0 --reps 16 --out "$ROOT/$arm.jsonl" \
        --cell-run correctness-only
    stop_server
    printf '%s %s complete\n' "$(date -u +%FT%TZ)" "$arm" >> "$ROOT/status.log"
done
printf 'COMPLETE\n' >> "$ROOT/status.log"

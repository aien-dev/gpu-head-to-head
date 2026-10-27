#!/usr/bin/env bash
# Throwaway measurement driver for PROTOCOL.md (host tool, not product code).
# usage: run9b.sh ARM ROUND        ARM in OURS LLAMA LLAMA16
# Holds /tmp/aien-gb10.lock for the whole session; one server at a time.
set -u
ARM=$1 ROUND=$2
H=/home/drakestapleton/handoffs/H2H9B
REPO=/home/drakestapleton/handoffs/HeadToHead/repo
LLAMA_DIR=/home/drakestapleton/h2h/dl/llama-bin/llama-b11913
CUDART_DIR=/home/drakestapleton/h2h/dl/cudart-llama-b11913-bin-ubuntu-cuda-13.4-arm64
SRV_CPUS=5-9,15-19 CLI_CPUS=0-1
export H2H_SRV_CPUS=$SRV_CPUS H2H_CLI_CPUS=$CLI_CPUS
LOGD=$H/results/logs/$ARM-$ROUND; OUT=$H/results/raw/$ARM-$ROUND.jsonl
mkdir -p "$LOGD" "$H/results/raw" "$H/results/llama-bench"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "$(ts) $*" | tee -a "$LOGD/session.log"; }
board() { echo "$(ts) Main(H2H-9B): $*" >> /home/drakestapleton/handoffs/BOARD.md; }

case $ARM in
  LLAMA16) PORT=18293; MID=q35-9b-bf16; G=/home/drakestapleton/h2h/gguf/q35-9b-bf16.gguf ;;
  LLAMA8)  PORT=18292; MID=q35-9b-q8_0; G=/home/drakestapleton/h2h/gguf/q35-9b-q8_0.gguf ;;
  OURS16)  PORT=18290; source "$H/ours16.env" ;;  # sets OURS_DIR, OURS_ENV, OURS_ARGS, MID
  OURS8)   PORT=18291; source "$H/ours8.env" ;;
  LLOMEGA8)  PORT=18294; source "$H/llomega8.env" ;;   # sets LLOMEGA_DIR, LLOMEGA_ENV, LLOMEGA_ARGS, MID
  LLOMEGA16) PORT=18295; source "$H/llomega16.env" ;;
  *) echo "bad arm"; exit 2 ;;
esac

start_server() {  # $1 = parallel slots (llama only)
  case $ARM in
    LLAMA*) LD_LIBRARY_PATH=$CUDART_DIR:$LLAMA_DIR taskset -c $SRV_CPUS "$LLAMA_DIR/llama-server" -m "$G" \
              --host 127.0.0.1 --port $PORT -ngl 999 -fa on -c 32768 -np $1 --jinja \
              < /dev/null >> "$LOGD/server.log" 2>&1 & ;;
    OURS*)  ( cd "$OURS_DIR" && exec env $OURS_ENV taskset -c $SRV_CPUS ./aien-serve-http $OURS_ARGS \
              --bind 127.0.0.1 --port $PORT ) < /dev/null >> "$LOGD/server.log" 2>&1 & ;;
    LLOMEGA*) ( cd "$LLOMEGA_DIR" && exec env $LLOMEGA_ENV taskset -c $SRV_CPUS ./llama-server $LLOMEGA_ARGS \
              --host 127.0.0.1 --port $PORT ) < /dev/null >> "$LOGD/server.log" 2>&1 & ;;
  esac
  SPID=$!
  local t0; t0=$(date +%s.%N)
  for _ in $(seq 1 600); do
    curl -s -m 2 -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && {
      log "ready np=$1 load_seconds $(echo "$(date +%s.%N) - $t0" | bc)"; return 0; }
    kill -0 $SPID 2>/dev/null || { log "server died during load"; return 1; }
    sleep 1
  done
  log "server not ready in 600 s"; return 1
}
stop_server() { kill $SPID 2>/dev/null; for _ in $(seq 1 60); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null; wait $SPID 2>/dev/null; }

cell() {  # metric concurrency max_tokens promptfile [extra]
  local metric=$1 c=$2 mt=$3 pf=$4; shift 4
  local q=0
  for _ in $(seq 1 10); do python3 "$REPO/bench/quiet.py" --window 60 --max-load 6 --exclude-tree $$ >> "$LOGD/quiet.jsonl" && { q=1; break; }; done
  [ $q = 1 ] || log "cell $metric LOADED (no quiet window in 10 tries)"
  log "cell $metric start"
  taskset -c $CLI_CPUS python3 "$REPO/bench/bench.py" --port $PORT --model "$MID" --arm $ARM --round $ROUND --metric "$metric" \
    --prompts "$REPO/prompts/$pf" --concurrency $c --max-tokens $mt --out "$OUT" --cell-run "$ARM-$ROUND-$metric" "$@" \
    2>> "$LOGD/client.log"
  log "cell $metric exit $?"
}

session() {
  board "GPU-START $ARM $ROUND (lock held)"
  nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv > "$LOGD/gpu-before.csv" 2>&1
  { nvidia-smi -q -d CLOCK,POWER,PERFORMANCE; cat /proc/loadavg; uname -r; ps -eo pcpu,pid,comm --sort=-pcpu | head -12; } > "$LOGD/machine-before.txt" 2>&1
  vmstat -t 1 > "$LOGD/vmstat.txt" & VMS=$!
  if start_server 1; then
    awk '$6 ~ /\.so/ {print $6}' /proc/$SPID/maps 2>/dev/null | sort -u > "$LOGD/maps-so.txt"
    cell agree 1 128 decode.jsonl --warmup 0 --reps 16
    cell decode-c1 1 256 decode.jsonl --warmup 2 --reps 3
    for p in p512 p2k p8k; do cell prefill-$p 1 1 prefill.jsonl --select $p --warmup 2 --reps 3 --nonce; done
  fi
  stop_server
  case $ARM in LLAMA*)
    if start_server 8; then cell decode-c8 8 256 decode.jsonl --warmup 1 --reps 3; fi
    stop_server
    LD_LIBRARY_PATH=$CUDART_DIR:$LLAMA_DIR "$LLAMA_DIR/llama-server" --version > "$LOGD/version.txt" 2>&1
    LD_LIBRARY_PATH=$CUDART_DIR:$LLAMA_DIR timeout 900 taskset -c $SRV_CPUS "$LLAMA_DIR/llama-bench" -m "$G" \
      -ngl 99 -fa on -p 512,2048,8192 -n 128 -r 5 -o jsonl > "$H/results/llama-bench/$ARM-$ROUND.jsonl" 2> "$LOGD/llama-bench.log" < /dev/null
    log "llama-bench exit $?" ;;
  esac
  kill $VMS 2>/dev/null
  nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv > "$LOGD/gpu-after.csv" 2>&1
  board "GPU-END $ARM $ROUND"
}

exec 9>/tmp/aien-gb10.lock
log "waiting for /tmp/aien-gb10.lock"
flock 9
log "lock held"
session
log "DONE"

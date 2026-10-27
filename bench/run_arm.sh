#!/usr/bin/env bash
# One arm session of the head-to-head (PROTOCOL.md sections 6-8).
# usage: bench/run_arm.sh OURS|VLLM|LLAMA R1|R2
# Holds /tmp/aien-gb10.lock for the whole session (at most 60 min), writes BOARD lines,
# starts exactly one server, runs every cell, stops the server and checks the unload.
set -u
ARM=$1; ROUND=$2
REPO=$(cd "$(dirname "$0")/.." && pwd)
H2H=${H2H:-/home/drakestapleton/h2h}
MODEL=/home/drakestapleton/models/qwen3.6-35b-a3b-fp8-95a723d0
VLLM_IMG=nvcr.io/nvidia/vllm@sha256:557747337846eecf34486866f3d992383c6f8af869fdb0e365b5093fcaee02c2
LLAMA_DIR=$H2H/dl/llama-bin/llama-b11913
CUDART_DIR=$H2H/dl/llama-bin/cudart-llama-b11913-bin-ubuntu-cuda-13.4-arm64
GGUF=$H2H/gguf/q36-35b-a3b-q8_0.gguf
BOARD=/home/drakestapleton/handoffs/BOARD.md
VLLM_UTIL=${VLLM_UTIL:-0.50}
OUT=$REPO/results/raw/$ARM-$ROUND.jsonl
LOGD=$REPO/results/logs/$ARM-$ROUND
mkdir -p "$LOGD" "$(dirname "$OUT")"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
board() { echo "$(ts) HeadToHead: $*" >> "$BOARD"; echo "$(ts) $*" >> "$LOGD/session.log"; }
memavail() { awk '/MemAvailable/{print $2}' /proc/meminfo; }
case $ARM in
  OURS)  PORT=18190; PAT='aien-serve-http.*--port 18190'; MID=qwen3.6-35b-a3b-fp8 ;;
  VLLM)  PORT=18191; PAT='vllm serve|VLLM::|EngineCore'; MID=q36 ;;
  LLAMA) PORT=18192; PAT='llama-server.*--port 18192'; MID=q36 ;;
  *) echo "bad arm"; exit 2 ;;
esac

wait_quiet() {  # PROTOCOL 8: load1 < 4 before each timing cell
  local waited=0
  while :; do
    l=$(cut -d' ' -f1 /proc/loadavg)
    if awk -v l="$l" 'BEGIN{exit !(l<4)}'; then return 0; fi
    [ $waited -eq 0 ] && echo "$(ts) waiting for load1<4 (now $l)" >> "$LOGD/session.log"
    sleep 15; waited=$((waited+15))
    if [ $((SECONDS)) -ge 3000 ]; then echo "$(ts) load gate timeout, load1=$l" >> "$LOGD/session.log"; return 1; fi
  done
}

sampler() {  # 1 Hz: epoch MemAvailable_kB summed_RSS_kB load1
  while :; do
    rss=$(for p in $(pgrep -f "$PAT"); do awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null; done | awk '{s+=$1}END{print s+0}')
    echo "$(date +%s) $(memavail) $rss $(cut -d' ' -f1 /proc/loadavg)"
    sleep 1
  done
}

start_server() {
  case $ARM in
    OURS)
      ( cd /home/drakestapleton/.local/state/aien-q36/bin-19173f9e && \
        env AIEN_Q36_STROKE=1 Q36_BATCH_BIN=bin-19173f9e AIEN_SERVE_GPU_LOCK=/tmp/aien-serve-18190.lock \
            AIEN_GB10_LOCK_HELD=1 \
        exec ./aien-serve-http --engine omega --model-dir "$MODEL" --port 18190 --max-context 49152 \
             --max-tokens-cap 16384 --thinking off --batch-slots 8 ) < /dev/null > "$LOGD/server.log" 2>&1 &
      SPID=$! ;;
    VLLM)
      docker run --rm --name h2h-vllm --runtime nvidia --gpus all --ipc host -p 127.0.0.1:18191:8000 \
        -v "$MODEL":/model:ro "$VLLM_IMG" vllm serve /model --served-model-name q36 --port 8000 \
        --max-model-len 49152 --max-num-seqs 16 --gpu-memory-utilization "$VLLM_UTIL" \
        --language-model-only < /dev/null > "$LOGD/server.log" 2>&1 &
      SPID=$! ;;
    LLAMA)
      LD_LIBRARY_PATH=$CUDART_DIR:$LLAMA_DIR "$LLAMA_DIR/llama-server" -m "$GGUF" --host 127.0.0.1 \
        --port 18192 -ngl 999 -fa on -c 196608 -np 16 --jinja < /dev/null > "$LOGD/server.log" 2>&1 &
      SPID=$! ;;
  esac
}

stop_server() {
  case $ARM in
    VLLM) docker stop -t 60 h2h-vllm >/dev/null 2>&1 ;;
    *) kill -TERM $SPID 2>/dev/null ;;
  esac
  for i in $(seq 1 120); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -KILL $SPID 2>/dev/null
  for i in $(seq 1 60); do pgrep -f "$PAT" >/dev/null || break; sleep 1; done
}

ready() {
  for i in $(seq 1 1200); do
    if ! kill -0 $SPID 2>/dev/null; then return 1; fi
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  return 1
}

cell() {  # metric concurrency max_tokens promptfile [extra]
  local metric=$1 c=$2 mt=$3 pf=$4; shift 4
  wait_quiet || { echo "$(ts) cell $metric ran past load gate timeout (LOADED)" >> "$LOGD/session.log"; }
  python3 "$REPO/bench/bench.py" --port $PORT --model $MID --arm $ARM --round $ROUND --metric "$metric" \
    --prompts "$REPO/prompts/$pf" --concurrency $c --max-tokens $mt --out "$OUT" "$@" 2>> "$LOGD/client.log"
}

# ---- session ----
# wait for a quiet host before taking the lock, so the 60-min hold is not spent waiting
while ! awk -v l="$(cut -d' ' -f1 /proc/loadavg)" 'BEGIN{exit !(l<4)}'; do sleep 30; done
exec 9>/tmp/aien-gb10.lock
echo "$(ts) quiet (load1 $(cut -d' ' -f1 /proc/loadavg)); waiting for /tmp/aien-gb10.lock" >> "$LOGD/session.log"
flock -w 900 9 || { board "$ARM $ROUND NOT STARTED: /tmp/aien-gb10.lock busy 900 s"; exit 3; }
SECONDS=0
board "START $ARM $ROUND gpu lock held (<=60 min; test port $PORT; live 18091 untouched)"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv > "$LOGD/gpu-before.csv" 2>&1
BASE=$(memavail); echo "baseline_memavail_kB $BASE" > "$LOGD/mem-baseline.txt"
sampler > "$LOGD/mem.tsv" & SAMP=$!
T0=$(date +%s.%N)
start_server
( sleep 3300; board "T-5 $ARM $ROUND: gpu lock released by 60 min" ; sleep 300; \
  touch "$LOGD/TIMEOUT"; board "$ARM $ROUND hit 60 min; harness stops its server"; stop_server ) & WD=$!
if ready; then
  echo "load_seconds $(echo "$(date +%s.%N) - $T0" | bc)" >> "$LOGD/session.log"
  [ "$ARM" = LLAMA ] && LD_LIBRARY_PATH=$CUDART_DIR:$LLAMA_DIR "$LLAMA_DIR/llama-server" --version > "$LOGD/version.txt" 2>&1
  [ "$ARM" = VLLM ] && docker exec h2h-vllm python3 -c "import vllm;print(vllm.__version__)" > "$LOGD/version.txt" 2>&1
  for c in 1 4 8 16; do
    [ -e "$LOGD/TIMEOUT" ] && break
    cell decode-c$c $c 256 decode.jsonl --warmup 2 --reps 3
  done
  for p in p256 p2k p8k; do
    [ -e "$LOGD/TIMEOUT" ] && break
    cell prefill-$p 1 1 prefill.jsonl --select $p --warmup 2 --reps 3 --nonce
  done
  [ -e "$LOGD/TIMEOUT" ] || cell agree 1 128 decode.jsonl --warmup 0 --reps 16
  board "$ARM $ROUND cells done at $SECONDS s"
else
  board "$ARM $ROUND NOT_RUN: server did not become ready (see results/logs/$ARM-$ROUND/server.log)"
fi
stop_server
kill $WD 2>/dev/null; pkill -P $WD 2>/dev/null
sleep 5
kill $SAMP 2>/dev/null
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv > "$LOGD/gpu-after.csv" 2>&1
AFTER=$(memavail); echo "after_memavail_kB $AFTER" >> "$LOGD/mem-baseline.txt"
board "END $ARM $ROUND gpu lock released after $SECONDS s; memavail base=${BASE}kB after=${AFTER}kB"
flock -u 9

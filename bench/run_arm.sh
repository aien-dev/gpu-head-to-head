#!/usr/bin/env bash
# One arm session of the head-to-head (PROTOCOL.md sections 6-8, amendments A1-A3).
# usage: bench/run_arm.sh OURS|VLLM|SGLANG|LLAMA|LLAMA16|VLLM4|SGLANG4|LLAMA4 R1|R2
# Holds /tmp/aien-gb10.lock for the whole session (at most 60 min), writes BOARD lines,
# starts exactly one server, runs every cell, stops the server and checks the unload.
# Ends with results/logs/<ARM>-<ROUND>/DONE, or NOT_RUN (with the reason) if the arm cannot start.
set -u
ARM=$1; ROUND=$2
REPO=${H2H_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
source "$REPO/bench/session-lifecycle.sh" || exit 2
RUN_ID="${ARM,,}-${ROUND,,}-$(date +%s)-$$"
H2H=${H2H:-/home/drakestapleton/h2h}
MODEL=/home/drakestapleton/models/qwen3.6-35b-a3b-fp8-95a723d0
VLLM_IMG=nvcr.io/nvidia/vllm@sha256:557747337846eecf34486866f3d992383c6f8af869fdb0e365b5093fcaee02c2
LLAMA_DIR=$H2H/dl/llama-bin/llama-b11913
CUDART_DIR=$H2H/dl/llama-bin/cudart-llama-b11913-bin-ubuntu-cuda-13.4-arm64
GGUF=$H2H/gguf/q36-35b-a3b-q8_0.gguf
GGUF16=$H2H/gguf/q36-35b-a3b-bf16.gguf
# A3 arms: SGLANG_IMG (image@digest), MODEL4 (vLLM/SGLang 4-bit checkpoint dir), GGUF4 (llama.cpp
# 4-bit GGUF) are set in bench/arms.conf once verified; unset -> the arm is recorded NOT_RUN.
SGLANG_IMG=; MODEL4=; GGUF4=
[ -f "$REPO/bench/arms.conf" ] && . "$REPO/bench/arms.conf"
BOARD=/home/drakestapleton/handoffs/BOARD.md
VLLM_UTIL=${VLLM_UTIL:-0.50}
SGLANG_MEM=${SGLANG_MEM:-0.50}
RESULTS=${H2H_RESULTS:-$REPO/results}
OUT=$RESULTS/raw/$ARM-$ROUND.jsonl
LOGD=$RESULTS/logs/$ARM-$ROUND
if [ -e "$OUT" ] || [ -e "$LOGD" ]; then
  echo "refusing to overwrite an existing session: $LOGD or $OUT" >&2
  exit 2
fi
mkdir -p "$LOGD" "$(dirname "$OUT")"
touch "$RESULTS/cell-status.jsonl" || exit 2
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
board() { echo "$(ts) HeadToHead: $*" >> "$BOARD"; echo "$(ts) $*" >> "$LOGD/session.log"; }
notrun() { board "$ARM $ROUND NOT_RUN: $*"; echo "$*" > "$LOGD/NOT_RUN"; exit 4; }
memavail() { awk '/MemAvailable/{print $2}' /proc/meminfo; }
CTR=
case $ARM in
  OURS)    PORT=18190; PAT='aien-serve-http.*--port 18190'; MID=qwen3.6-35b-a3b-fp8 ;;
  VLLM)    PORT=18191; PAT='vllm serve|VLLM::|EngineCore'; MID=q36; CTR=stack-h2h-$RUN_ID; MDIR=$MODEL ;;
  VLLM4)   PORT=18191; PAT='vllm serve|VLLM::|EngineCore'; MID=q36; CTR=stack-h2h-$RUN_ID; MDIR=$MODEL4 ;;
  SGLANG)  PORT=18194; PAT='sglang'; MID=q36; CTR=stack-h2h-$RUN_ID; MDIR=$MODEL ;;
  SGLANG4) PORT=18194; PAT='sglang'; MID=q36; CTR=stack-h2h-$RUN_ID; MDIR=$MODEL4 ;;
  LLAMA)   PORT=18192; PAT='llama-server.*--port 18192'; MID=q36; G=$GGUF; MMAP= ;;
  LLAMA16) PORT=18193; PAT='llama-server.*--port 18193'; MID=q36; G=$GGUF16; MMAP=--no-mmap ;;
  LLAMA4)  PORT=18192; PAT='llama-server.*--port 18192'; MID=q36; G=$GGUF4; MMAP= ;;
  *) echo "bad arm"; exit 2 ;;
esac

# A2: cpus 5-9,15-19 are Cortex-X925 (CPU part 0xd85), 0-4,10-14 Cortex-A725 (0xd87).
# Every server gets 9 of the 10 X925 cores; the client runs on two A725 cores.
# A4(e): the excluded X925 core is the one the nvidia-modeset kthread is on when the session takes
# the GPU lock (core 18 if it is on an A725 core), chosen by pick_srv before the server starts.
X925=" 5 6 7 8 9 15 16 17 18 19 "
pick_srv() {
  local k=$(ktpsr) c
  case $X925 in *" $k "*) ;; *) k=18 ;; esac
  SRV_LIST=" "; for c in $X925; do [ "$c" = "$k" ] || SRV_LIST="$SRV_LIST$c "; done
  SRV_CPUS=$(echo $SRV_LIST | tr ' ' ',')
  export H2H_SRV_CPUS=$SRV_CPUS
}
SRV_CPUS=5-9,15-17,19
SRV_LIST=" 5 6 7 8 9 15 16 17 19 "
CLI_CPUS=0-1   # A5(d): A725 cores outside the 10-14 build set other lanes use
MON_CPU=2      # coremon.py (A4), an A725 core outside the server and client sets
export H2H_SRV_CPUS=$SRV_CPUS H2H_CLI_CPUS=$CLI_CPUS
KT=1671   # nvidia-modeset kthread; never reniced or pinned during the study (A2)
ktpsr() { awk '{print $39}' /proc/$KT/stat 2>/dev/null || echo NA; }

mytree() {  # pids whose process trees belong to this session
  echo $$ $SPID
  [ -n "$CTR" ] && docker inspect -f '{{.State.Pid}}' $CTR 2>/dev/null
}

wait_quiet() {  # PROTOCOL A2: before each cell, no foreign process >20% CPU over 10 s
  while :; do
    if python3 "$REPO/bench/quiet.py" --window 10 --max-load 0 --exclude-tree $(mytree) >> "$LOGD/quiet.jsonl"; then
      return 0
    fi
    if [ $SECONDS -ge 3000 ]; then echo "$(ts) quiet gate timeout" >> "$LOGD/session.log"; return 1; fi
  done
}

sampler() {  # 1 Hz: epoch MemAvailable_kB summed_RSS_kB load1 kthread_psr
  while :; do
    rss=$(for p in $(server_pids); do awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null; done | awk '{s+=$1}END{print s+0}')
    echo "$(date +%s) $(memavail) $rss $(cut -d' ' -f1 /proc/loadavg) $(ktpsr)"
    sleep 1
  done
}

server_pids() {
  if [ -n "$CTR" ]; then
    [ "$(docker inspect -f '{{index .Config.Labels "aien.stack-comparison.run"}}' "$CTR" 2>/dev/null)" = "$RUN_ID" ] || return 1
    local root
    root=$(docker inspect -f '{{.State.Pid}}' "$CTR" 2>/dev/null) || return 1
    h2h_descendant_pids "$root"
  elif [ -s "$LOGD/server-identity.txt" ]; then
    local root expected identity state birth parent
    read -r root expected < "$LOGD/server-identity.txt" || return 1
    identity=$(h2h_identity "$root") || return 1
    read -r state birth parent <<< "$identity"
    [ "$birth" = "$expected" ] || return 1
    h2h_descendant_pids "$root"
  fi
}

start_server() {
  case $ARM in
    OURS)
      ( cd /home/drakestapleton/.local/state/aien-q36/bin-19173f9e && \
        exec env AIEN_Q36_STROKE=1 Q36_BATCH_BIN=bin-19173f9e AIEN_SERVE_GPU_LOCK=/tmp/aien-serve-18190.lock \
            AIEN_GB10_LOCK_HELD=1 \
        taskset -c $SRV_CPUS ./aien-serve-http --engine omega --model-dir "$MODEL" --bind 127.0.0.1 --port 18190 --max-context 49152 \
             --max-tokens-cap 16384 --thinking off --batch-slots 8 ) < /dev/null > "$LOGD/server.log" 2>&1 &
      SPID=$! ;;
    VLLM|VLLM4)
      timeout 30 docker create --pull never --rm --name "$CTR" --label "aien.stack-comparison.run=$RUN_ID" --runtime nvidia --gpus all --ipc host --network host \
        --cpuset-cpus $SRV_CPUS -v "$MDIR":/model:ro "$VLLM_IMG" vllm serve /model --served-model-name q36 \
        --host 127.0.0.1 --port 18191 \
        --max-model-len 49152 --max-num-seqs 32 --gpu-memory-utilization "$VLLM_UTIL" \
        --language-model-only < /dev/null > "$LOGD/container-id.txt" 2> "$LOGD/server.log" || return 1
      docker start --attach "$CTR" < /dev/null >> "$LOGD/server.log" 2>&1 &
      SPID=$! ;;
    SGLANG|SGLANG4)
      timeout 30 docker create --pull never --rm --name "$CTR" --label "aien.stack-comparison.run=$RUN_ID" --runtime nvidia --gpus all --ipc host --network host \
        --cpuset-cpus $SRV_CPUS -v "$MDIR":/model:ro "$SGLANG_IMG" python3 -m sglang.launch_server \
        --model-path /model --served-model-name q36 --host 127.0.0.1 --port 18194 \
        --context-length 49152 --max-running-requests 32 --mem-fraction-static "$SGLANG_MEM" \
        < /dev/null > "$LOGD/container-id.txt" 2> "$LOGD/server.log" || return 1
      docker start --attach "$CTR" < /dev/null >> "$LOGD/server.log" 2>&1 &
      SPID=$! ;;
    LLAMA*)
      LD_LIBRARY_PATH=$CUDART_DIR:$LLAMA_DIR taskset -c $SRV_CPUS "$LLAMA_DIR/llama-server" -m "$G" --host 127.0.0.1 \
        --port $PORT -ngl 999 -fa on -c 393216 -np 32 --jinja $MMAP < /dev/null > "$LOGD/server.log" 2>&1 &
      SPID=$! ;;
  esac
  h2h_track "$SPID" || return 1
  printf '%s %s\n' "$SPID" "${H2H_PID_START[$SPID]}" > "$LOGD/server-identity.txt"
}

stop_server() {
  if [ -n "$CTR" ] && docker inspect "$CTR" >/dev/null 2>&1; then
    local owner
    owner=$(docker inspect -f '{{index .Config.Labels "aien.stack-comparison.run"}}' "$CTR" 2>/dev/null)
    if [ "$owner" != "$RUN_ID" ]; then
      echo "refusing to stop container owned by another session: $CTR" >&2
      return 1
    fi
    timeout 15 docker stop -t 10 "$CTR" >/dev/null 2>&1 || {
      docker inspect "$CTR" >/dev/null 2>&1 && return 1
    }
  fi
  if [ -n "$SPID" ]; then h2h_stop "$SPID" || return 1; SPID=; fi
}

ready() {
  for i in $(seq 1 1200); do
    if ! kill -0 $SPID 2>/dev/null; then return 1; fi
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then return 0; fi
    sleep 1
  done
  return 1
}

loopback_only() {  # A5(f): every listener on the test port must be 127.0.0.1, else the arm FAILs
  ss -ltnH "sport = :$PORT" > "$LOGD/listen.txt"
  [ -s "$LOGD/listen.txt" ] && ! awk '{print $4}' "$LOGD/listen.txt" | grep -qv '^127\.0\.0\.1:'
}

cell_status() {
  python3 -c 'import json,sys; print(json.dumps({"cell_run":sys.argv[1],"status":sys.argv[2]}))' \
    "$1" "$2" >> "$RESULTS/cell-status.jsonl"
}

cell() {  # metric concurrency max_tokens promptfile [extra]
  # A2: a cell is VOID and re-run (<=3 tries) when the nvidia-modeset kthread was sampled on a server
  # core, or a process outside HeadToHead used >20% of one server core (bench/coremon.py)
  local metric=$1 c=$2 mt=$3 pf=$4; shift 4
  local try t0 t1 hit cm cmrc why off clientrc
  mkdir -p "$LOGD/coremon"
  for try in 1 2 3; do
    local id="$ARM-$ROUND-$metric-t$try"
    [ -e "$LOGD/TIMEOUT" ] && { cell_status "$id" NOT_RUN; return 1; }
    wait_quiet || { echo "$(ts) cell $metric NOT_RUN: no quiet window" >> "$LOGD/session.log"; cell_status "$id" NOT_RUN; return 1; }
    taskset -c $MON_CPU python3 "$REPO/bench/coremon.py" --cpus $SRV_CPUS --exclude-tree $(mytree) \
      --out "$LOGD/coremon/$id.jsonl" & cm=$!
    CMON_PID=$cm
    h2h_track "$cm" || return 1
    t0=$(date +%s)
    taskset -c $CLI_CPUS python3 "$REPO/bench/bench.py" --port $PORT --model $MID --arm $ARM --round $ROUND --metric "$metric" \
      --prompts "$REPO/prompts/$pf" --concurrency $c --max-tokens $mt --out "$OUT" --cell-run "$id" "$@" 2>> "$LOGD/client.log" & CPID=$!
    h2h_track "$CPID" || return 1
    wait "$CPID"; clientrc=$?
    unset 'H2H_PID_START[$CPID]'; CPID=
    h2h_matches "$cm" && kill -TERM "$cm" 2>/dev/null || true
    wait "$cm"; cmrc=$?
    unset 'H2H_PID_START[$cm]'; CMON_PID=
    if [ "$clientrc" -ne 0 ]; then
      echo "$(ts) $id FAIL: client exit $clientrc; raw rows retained" >> "$LOGD/session.log"
      cell_status "$id" FAIL
      return 1
    fi
    if [ "$cmrc" -ne 0 ] && [ "$cmrc" -ne 1 ]; then
      echo "$(ts) $id foreign-process check NOT_RUN: monitor exit $cmrc" >> "$LOGD/session.log"
      cell_status "$id" NOT_RUN
      return 1
    fi
    sleep 2; t1=$(date +%s)
    if ! awk -v a=$t0 -v b=$t1 '$1>=a && $1<=b {n++; if ($5 !~ /^[0-9]+$/) bad=1} END{exit (!n || bad)}' "$LOGD/mem.tsv"; then
      echo "$(ts) $id kthread check NOT_RUN: missing or invalid samples" >> "$LOGD/session.log"
      cell_status "$id" NOT_RUN
      return 1
    fi
    hit=$(awk -v a=$t0 -v b=$t1 -v s="$SRV_LIST" '$1>=a && $1<=b && index(s, " "$5" ") {print $5; exit}' "$LOGD/mem.tsv")
    why=
    [ -n "$hit" ] && why="nvidia-modeset kthread on server core $hit"
    # A4: only a coremon summary that names offenders voids the cell; a monitor that wrote no
    # summary is recorded as "foreign check NOT_RUN" for that cell, never as a void
    off=$(python3 -c 'import json,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
s=[r for r in rows if r.get("summary") is True]
if len(s)!=1 or not isinstance(s[0].get("elapsed_ticks"),int) or s[0]["elapsed_ticks"]<=0 or not isinstance(s[0].get("offenders"),list):
    print("NOSUMMARY")
else:
    print(json.dumps(s[0]["offenders"]))' "$LOGD/coremon/$id.jsonl" 2>/dev/null || echo NOSUMMARY)
    case $off in
      NOSUMMARY) echo "$(ts) $id foreign-process check NOT_RUN: coremon exit $cmrc, no summary" >> "$LOGD/session.log"; cell_status "$id" NOT_RUN; return 1 ;;
      '[]') [ "$cmrc" -eq 0 ] || { echo "$(ts) $id FAIL: monitor exit disagrees with clean summary" >> "$LOGD/session.log"; cell_status "$id" FAIL; return 1; } ;;
      *) why="${why:+$why; }foreign process >20% of a server core: $(echo "$off" | cut -c1-300)" ;;
    esac
    if [ -z "$why" ]; then
      cell_status "$id" PASS || return 1
      CELLS_COMPLETED=$((CELLS_COMPLETED + 1))
      return 0
    fi
    python3 -c 'import json,sys; print(json.dumps({"cell_run": sys.argv[1], "reason": sys.argv[2], "t0": int(sys.argv[3]), "t1": int(sys.argv[4])}))' \
      "$id" "$why" $t0 $t1 >> "$RESULTS/voids.jsonl"
    cell_status "$id" VOID || return 1
    echo "$(ts) VOID $id: $why" >> "$LOGD/session.log"
  done
  echo "$(ts) $metric NOT_RUN: all three attempts disturbed" >> "$LOGD/session.log"
  return 1
}

mark_complete() {
  if [ "$CELLS_COMPLETED" -ne 14 ] || [ -e "$LOGD/TIMEOUT" ]; then
    echo "incomplete: $CELLS_COMPLETED/14 accepted cells; see session.log" > "$LOGD/INCOMPLETE"
    return 1
  fi
  touch "$LOGD/DONE"
}

start_watchdog() {
  (
    if [ "$#" -ge 1 ]; then sleep "$1";
    else delay=$((3300 - SECONDS)); [ "$delay" -le 0 ] || sleep "$delay"; fi
    board "T-5 $ARM $ROUND: cleanup reserved within 60-minute limit"
    if [ "$#" -ge 2 ]; then sleep "$2";
    else delay=$((3540 - SECONDS)); [ "$delay" -le 0 ] || sleep "$delay"; fi
    touch "$LOGD/TIMEOUT"
    board "$ARM $ROUND reached 59 min; stopping own session with cleanup time reserved"
    kill -TERM $$
  ) & WD=$!
  h2h_track "$WD"
}

cleanup_session() {
  local rc=$? pid
  trap - EXIT
  trap '' INT TERM
  H2H_STOP_DEADLINE=$((SECONDS + 30))
  for pid in "$WD" "$CPID" "$CMON_PID"; do
    [ -z "$pid" ] || h2h_stop "$pid" || rc=1
  done
  stop_server || rc=1
  for pid in "$SAMP" "$VMS"; do
    [ -z "$pid" ] || h2h_stop "$pid" || rc=1
  done
  if [ "$GPU_LOCK_HELD" = 1 ]; then
    timeout 5 nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv > "$LOGD/gpu-after.csv" 2>&1 || rc=1
    AFTER=$(memavail)
    echo "after_memavail_kB $AFTER" >> "$LOGD/mem-baseline.txt"
    # Recording the inventory alone does not prove that the model unloaded.
    local remaining_gpu
    if ! remaining_gpu=$(timeout 5 nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); then
      echo "FAIL GPU inventory unavailable after unload" >> "$LOGD/unload-status.txt"
      rc=1
    elif [ -n "$remaining_gpu" ]; then
      echo "FAIL GPU processes remain after unload" >> "$LOGD/unload-status.txt"
      rc=1
    else
      echo "PASS no GPU compute processes remain" >> "$LOGD/unload-status.txt"
    fi
    # /proc/meminfo reports KiB. Refuse missing/malformed/overflow-sized samples.
    if [[ ! "$BASE" =~ ^[0-9]{1,12}$ || ! "$AFTER" =~ ^[0-9]{1,12}$ ]]; then
      echo "FAIL unavailable or invalid memory baseline/sample" >> "$LOGD/unload-status.txt"
      rc=1
    elif (( 10#$BASE == 0 || 10#$AFTER == 0 || 10#$BASE - 10#$AFTER > 2 * 1024 * 1024 )); then
      echo "FAIL memory did not recover within 2 GiB of baseline" >> "$LOGD/unload-status.txt"
      rc=1
    else
      echo "PASS memory recovered within 2 GiB of baseline" >> "$LOGD/unload-status.txt"
    fi
    echo "kthread_psr_end $(ktpsr)" >> "$LOGD/session.log"
    board "END $ARM $ROUND own session stopped at $SECONDS s; memavail base=${BASE}kB after=${AFTER}kB; exit=$rc"
    if [ "$GPU_LOCK_RELEASE" = 1 ]; then flock -u 9 || rc=1; fi
  fi
  if [ "$rc" -ne 0 ]; then
    rm -f "$LOGD/DONE"
    echo "session exit $rc; see session.log" >> "$LOGD/INCOMPLETE"
  fi
  exit "$rc"
}

# ---- session ----
# PROTOCOL A2 (Main's gate): start only when no foreign process used >20% CPU for 60 s and load1 < 6,
# before taking the lock, so the 60-min hold is not spent waiting
SPID= CPID= CMON_PID= WD= SAMP= VMS= BASE= GPU_LOCK_HELD=0 GPU_LOCK_RELEASE=1
trap cleanup_session EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
CELLS_COMPLETED=0
SESSION_RC=1
case $ARM in
  SGLANG*) [ -n "$SGLANG_IMG" ] || notrun "no SGLang image verified for aarch64/sm_121 (bench/arms.conf)" ;;
esac
case $ARM in
  VLLM4|SGLANG4) [ -n "$MODEL4" ] && [ -f "$MODEL4/config.json" ] || notrun "no 4-bit checkpoint (bench/arms.conf MODEL4)" ;;
  LLAMA4) [ -n "$GGUF4" ] && [ -f "$GGUF4" ] || notrun "no 4-bit GGUF (bench/arms.conf GGUF4)" ;;
esac
until python3 "$REPO/bench/quiet.py" --window 60 --max-load 6 --exclude-tree $$ >> "$LOGD/quiet.jsonl"; do :; done
for pair in LLAMA:convert.log LLAMA16:convert-bf16.log; do  # the GGUF must be fully written
  [ "$ARM" = "${pair%%:*}" ] || continue
  CL=$H2H/gguf/${pair#*:}
  until grep -q '^EXIT 0$' "$CL" 2>/dev/null; do
    grep -q '^EXIT [1-9]' "$CL" 2>/dev/null && notrun "GGUF conversion failed ($CL)"
    sleep 30
  done
done
if [ "${H2H_GPU_LOCK_FD:-}" = 9 ]; then
  GPU_LOCK_RELEASE=0
  [ "$(readlink /proc/$$/fd/9)" = /tmp/aien-gb10.lock ] || notrun "inherited GPU-lock descriptor names another file"
  flock -n 9 || notrun "inherited GPU-lock descriptor cannot acquire the lock"
else
  exec 9>/tmp/aien-gb10.lock
  echo "$(ts) quiet: $(tail -1 "$LOGD/quiet.jsonl"); waiting for /tmp/aien-gb10.lock" >> "$LOGD/session.log"
  flock -w 900 9 || { board "$ARM $ROUND NOT STARTED: /tmp/aien-gb10.lock busy 900 s"; exit 3; }
fi
GPU_LOCK_HELD=1
SECONDS=0
board "START $ARM $ROUND gpu lock held (<=60 min; test port $PORT; live 18091 untouched)"
echo "kthread_psr_start $(ktpsr)" >> "$LOGD/session.log"
pick_srv
echo "server_cpus $SRV_CPUS client_cpus $CLI_CPUS monitor_cpu $MON_CPU" >> "$LOGD/session.log"
timeout 5 nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv > "$LOGD/gpu-before.csv" 2>&1 || notrun "GPU process inventory failed"
GPU_ROWS=$(timeout 5 nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) || notrun "GPU process inventory failed"
[ -z "$GPU_ROWS" ] || notrun "another model/process is resident; shared-service window must be coordinated before launch"
BASE=$(memavail); echo "baseline_memavail_kB $BASE" > "$LOGD/mem-baseline.txt"
sampler > "$LOGD/mem.tsv" & SAMP=$!
h2h_track "$SAMP" || exit 1
vmstat -t 1 > "$LOGD/vmstat.txt" & VMS=$!
h2h_track "$VMS" || exit 1
start_watchdog || exit 1
T0=$(date +%s.%N)
start_server || { board "$ARM $ROUND NOT_RUN: launch failed"; exit 4; }
if ready && loopback_only; then
  echo "load_seconds $(echo "$(date +%s.%N) - $T0" | bc)" >> "$LOGD/session.log"
  case $ARM in LLAMA*) LD_LIBRARY_PATH=$CUDART_DIR:$LLAMA_DIR "$LLAMA_DIR/llama-server" --version > "$LOGD/version.txt" 2>&1 ;; esac
  [ -n "$CTR" ] && docker exec $CTR pip list 2>/dev/null | grep -i -E '^(vllm|sglang|torch|transformers|flashinfer)' > "$LOGD/version.txt"
  for c in 1 4 8 16; do
    [ -e "$LOGD/TIMEOUT" ] && break
    cell decode-c$c $c 256 decode.jsonl --warmup 2 --reps 3 || break
  done
  for p in p256 p2k p8k; do
    [ "$CELLS_COMPLETED" -ge 4 ] || break
    [ -e "$LOGD/TIMEOUT" ] && break
    cell prefill-$p 1 1 prefill.jsonl --select $p --warmup 2 --reps 3 --nonce || break
  done
  if [ "$CELLS_COMPLETED" -eq 7 ] && [ ! -e "$LOGD/TIMEOUT" ]; then
    cell agree 1 128 decode.jsonl --warmup 0 --reps 16 || true
  fi
  # A3 grid: N = 1, 8, 32 streams x prompt 512 / 8k tokens, 256 new tokens, 1 warmup + 3 runs
  for p in p512 p8k; do
    [ "$CELLS_COMPLETED" -ge 8 ] || break
    for n in 1 8 32; do
      [ -e "$LOGD/TIMEOUT" ] && break
      cell grid-$p-n$n $n 256 prefill.jsonl --select $p --warmup 1 --reps 3 --nonce || break 2
    done
  done
  board "$ARM $ROUND cells done at $SECONDS s"
  if mark_complete; then SESSION_RC=0; fi
else
  if [ -s "$LOGD/listen.txt" ]; then why="FAIL: listener on port $PORT is not loopback-only (listen.txt); server stopped"
  else why="server did not become ready (see results/logs/$ARM-$ROUND/server.log)"; fi
  echo "$why" > "$LOGD/NOT_RUN"
  board "$ARM $ROUND NOT_RUN: $why"
fi
exit "$SESSION_RC"

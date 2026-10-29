#!/usr/bin/env bash
# External benchmark process ownership. Source from a Bash runner; no model/GPU access here.
# Linux proc_pid_stat(5): after comm, state is index0, ppid index1 and starttime index19.
declare -A H2H_PID_START=()

h2h_identity() {
  local pid=$1 line rest
  local -a fields
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  { IFS= read -r line < "/proc/$pid/stat"; } 2>/dev/null || return 1
  rest=${line##*) }
  read -r -a fields <<< "$rest"
  [ "${#fields[@]}" -ge 20 ] || return 1
  printf '%s %s %s\n' "${fields[0]}" "${fields[19]}" "${fields[1]}"
}

h2h_track() {
  local pid=$1 owner=$BASHPID identity state birth parent
  identity=$(h2h_identity "$pid") || { H2H_PID_START[$pid]=gone; wait "$pid" 2>/dev/null || true; return 0; }
  read -r state birth parent <<< "$identity"
  [ "$parent" = "$owner" ] || { echo "refusing to track non-child process $pid" >&2; return 1; }
  H2H_PID_START[$pid]=$birth
}

h2h_matches() {
  local pid=$1 identity state birth parent
  [ -n "${H2H_PID_START[$pid]:-}" ] || return 1
  identity=$(h2h_identity "$pid") || return 1
  read -r state birth parent <<< "$identity"
  [ "$birth" = "${H2H_PID_START[$pid]}" ] && [ "$state" != Z ] && [ "$state" != X ]
}

h2h_tree_snapshot() {
  local root=$1 pid identity state birth parent changed=1
  local -A parents=() births=() keep=()
  keep[$root]=1
  while read -r pid parent; do parents[$pid]=$parent; done < <(ps -e -o pid=,ppid=)
  while [ "$changed" = 1 ]; do
    changed=0
    for pid in "${!parents[@]}"; do
      if [ -n "${keep[${parents[$pid]}]:-}" ] && [ -z "${keep[$pid]:-}" ]; then
        keep[$pid]=1; changed=1
      fi
    done
  done
  h2h_matches "$root" || return 0
  for pid in "${!keep[@]}"; do
    identity=$(h2h_identity "$pid") || continue
    read -r state birth parent <<< "$identity"
    if [ "$pid" = "$root" ]; then
      [ "$birth" = "${H2H_PID_START[$root]}" ] || return 0
    else
      [ "$parent" = "${parents[$pid]}" ] || continue
    fi
    births[$pid]=$birth
  done
  h2h_matches "$root" || return 0
  for pid in "${!births[@]}"; do printf '%s %s\n' "$pid" "${births[$pid]}"; done
}

h2h_descendant_pids() {
  local root=$1
  [[ "$root" =~ ^[1-9][0-9]*$ ]] || return 1
  ps -e -o pid=,ppid= | awk -v root="$root" '
    {parent[$1]=$2; present[$1]=1}
    END {
      if (!present[root]) exit
      keep[root]=1; changed=1
      while (changed) {changed=0; for (p in parent) if (keep[parent[p]] && !keep[p]) {keep[p]=1;changed=1}}
      for (p in keep) if (keep[p]) print p
    }'
}

h2h_stop() {
  local pid=$1 attempts=${2:-600} child birth attempt alive
  local -a tree=()
  [ -n "${H2H_PID_START[$pid]:-}" ] || { echo "refusing to stop untracked process $pid" >&2; return 1; }
  if ! h2h_matches "$pid"; then
    local identity state current parent
    if identity=$(h2h_identity "$pid"); then
      read -r state current parent <<< "$identity"
      if [ "$state" != Z ] && [ "$state" != X ]; then
        echo "refusing stale process identity $pid" >&2
        return 1
      fi
    fi
    wait "$pid" 2>/dev/null || true
    unset 'H2H_PID_START[$pid]'
    return 0
  fi
  if h2h_matches "$pid"; then
    while read -r child birth; do
      H2H_PID_START[$child]=$birth
      tree+=("$child")
    done < <(h2h_tree_snapshot "$pid")
    for child in "${tree[@]}"; do
      h2h_matches "$child" && kill -TERM "$child" 2>/dev/null || true
    done
    for ((attempt=0; attempt<attempts; attempt++)); do
      if [ -n "${H2H_STOP_DEADLINE:-}" ] && [ "$SECONDS" -ge "$H2H_STOP_DEADLINE" ]; then break; fi
      alive=0
      for child in "${tree[@]}"; do h2h_matches "$child" && alive=1; done
      [ "$alive" = 0 ] && break
      sleep 0.1
    done
    for child in "${tree[@]}"; do
      h2h_matches "$child" && kill -KILL "$child" 2>/dev/null || true
    done
  fi
  wait "$pid" 2>/dev/null || true
  for ((attempt=0; attempt<20; attempt++)); do
    if [ -n "${H2H_STOP_DEADLINE:-}" ] && [ "$SECONDS" -ge "$H2H_STOP_DEADLINE" ]; then break; fi
    alive=0
    for child in "${tree[@]}"; do h2h_matches "$child" && alive=1; done
    [ "$alive" = 0 ] && break
    sleep 0.1
  done
  for child in "${tree[@]}"; do
    if h2h_matches "$child"; then echo "owned process $child still live after shutdown" >&2; return 1; fi
    unset 'H2H_PID_START[$child]'
  done
  unset 'H2H_PID_START[$pid]'
}

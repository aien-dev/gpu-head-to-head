#!/usr/bin/env bash
# Runs the sessions listed in bench/schedule.txt (one ARM:ROUND per line, '#' comments), in order.
# PROTOCOL.md section 7 + A2/A3: the 8-bit arms of every engine interleaved first (R1 forward, R2
# reversed), then the optional reference and 4-bit arms. The schedule file is re-read before every
# session, so queued arms can be appended while a session runs. Each session runs from a snapshot
# of run_arm.sh (editing the script cannot disturb a running session). A session whose
# results/logs/<ARM>-<ROUND>/DONE exists is skipped (resume after interruption).
set -u
cd "$(dirname "$0")/.."
REPO=$PWD
while :; do
  next=
  while read -r s; do
    s=${s%%#*}; s=${s// /}; [ -n "$s" ] || continue
    [ -e "results/logs/${s%:*}-${s#*:}/DONE" ] && continue
    [ -e "results/logs/${s%:*}-${s#*:}/NOT_RUN" ] && continue
    next=$s; break
  done < bench/schedule.txt
  [ -n "$next" ] || { echo "$(date -u +%FT%TZ) schedule complete"; break; }
  snap=$(mktemp /tmp/h2h-run_arm.XXXXXX.sh)
  cp bench/run_arm.sh "$snap"
  echo "$(date -u +%FT%TZ) session $next (run_arm.sh sha256 $(sha256sum < "$snap" | cut -c1-16))"
  H2H_REPO=$REPO bash "$snap" "${next%:*}" "${next#*:}"
  rc=$?
  rm -f "$snap"
  # a session that neither finished nor recorded NOT_RUN would loop forever: record it and move on
  if [ ! -e "results/logs/${next%:*}-${next#*:}/DONE" ] && [ ! -e "results/logs/${next%:*}-${next#*:}/NOT_RUN" ]; then
    echo "exit $rc without DONE" > "results/logs/${next%:*}-${next#*:}/NOT_RUN"
  fi
done

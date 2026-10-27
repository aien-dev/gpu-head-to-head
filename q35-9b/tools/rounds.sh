#!/usr/bin/env bash
# Throwaway: PROTOCOL.md "Design". 3 rounds; per round every ready arm once, order from random.Random(9).
# usage: rounds.sh ARM [ARM ...]     (arms whose env/GGUF is ready; missing arms are NOT_RUN, listed in the log)
set -u
H=/home/drakestapleton/handoffs/H2H9B
mkdir -p $H/results
ORDER=$(python3 -c "
import random, sys
arms = sys.argv[1:]; r = random.Random(9)
for k in (1, 2, 3):
    a = list(arms); r.shuffle(a); print('R%d %s' % (k, ' '.join(a)))" "$@")
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) arms: $* ; order:" | tee -a $H/results/rounds.log
echo "$ORDER" | tee -a $H/results/rounds.log
while read -r R arms; do
  for A in $arms; do
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start $A $R" >> $H/results/rounds.log
    $H/run9b.sh $A $R > /dev/null 2>&1
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end $A $R rc=$?" >> $H/results/rounds.log
  done
done <<< "$ORDER"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) ALL ROUNDS DONE" >> $H/results/rounds.log

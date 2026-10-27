#!/usr/bin/env python3
"""Quiet-host gate (PROTOCOL.md amendment A2).

  quiet.py --window S --max-load L [--exclude-tree PID ...]

Exit 0 when, over the last S seconds, no process outside the excluded trees used more than 20% of
one CPU, and (if L > 0) load1 < L at the end of the window. The nvidia-modeset kernel thread is
excluded (it always spins one core on this host). One JSON line on stdout: the verdict, load1 and
the busiest foreign processes, in clock ticks (integers; 20% of S seconds is S * CLK_TCK / 5 ticks).
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--window", type=int, default=60)
ap.add_argument("--max-load", type=float, default=6.0)
ap.add_argument("--exclude-tree", type=int, nargs="*", default=[])
a = ap.parse_args()
TCK = os.sysconf("SC_CLK_TCK")


def snap():
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            s = open("/proc/%s/stat" % d).read()
        except OSError:
            continue
        comm = s[s.index("(") + 1:s.rindex(")")]
        f = s[s.rindex(")") + 2:].split()
        out[int(d)] = (comm, int(f[1]), int(f[11]) + int(f[12]))  # comm, ppid, utime+stime
    return out


def tree(procs, roots):
    keep, grew = set(roots), True
    while grew:
        grew = False
        for pid, (_, ppid, _) in procs.items():
            if ppid in keep and pid not in keep:
                keep.add(pid)
                grew = True
    return keep


s0 = snap()
time.sleep(a.window)
s1 = snap()
mine = tree(s1, a.exclude_tree + [os.getpid()])
busy, harness = [], []
for pid, (comm, _, t1) in s1.items():
    if pid in mine or comm.startswith("nvidia-modeset") or pid not in s0:
        continue
    d = t1 - s0[pid][2]
    if comm == "omp":  # the agent harness hosting this lane: recorded, not gated (amendment A2)
        harness.append({"pid": pid, "ticks": d})
        continue
    if d * 5 > a.window * TCK:
        busy.append({"pid": pid, "comm": comm, "ticks": d})
load1 = float(open("/proc/loadavg").read().split()[0])
ok = not busy and (a.max_load <= 0 or load1 < a.max_load)
busy.sort(key=lambda b: -b["ticks"])
print(json.dumps({"quiet": ok, "load1": load1, "window_s": a.window, "clk_tck": TCK,
                  "busy": busy[:5], "omp_harness": harness,
                  "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
sys.exit(0 if ok else 1)

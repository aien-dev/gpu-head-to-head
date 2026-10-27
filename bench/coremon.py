#!/usr/bin/env python3
"""Per-core monitor for one timed cell (PROTOCOL.md amendment A2, void rule).

  coremon.py --cpus 5-9,15-17,19 --exclude-tree PID ... --out FILE   (runs until SIGTERM)

Every second it appends one JSON line to FILE: the busy share of each watched core (from
/proc/stat, integer jiffies) and, per foreign process, the CPU ticks its threads spent on each
watched core in that second (a thread's ticks are attributed to the core it is on at the end of the
second, field 39 of /proc/<pid>/task/<tid>/stat). Foreign = not in the excluded process trees
(re-resolved every second, so server children count as ours) and not the nvidia-modeset kthread,
which the session log tracks separately. On SIGTERM it appends a summary line and exits:
  {"summary": true, "elapsed_ticks": E, "offenders": [{pid, comm, cpu, ticks}, ...]}
An offender is a foreign process with more than 20% of one watched core over the run
(ticks * 5 > elapsed_ticks). Exit status: 0 clean, 1 offenders (the cell is VOID).
"""
import argparse
import json
import os
import signal
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--cpus", required=True)
ap.add_argument("--exclude-tree", type=int, nargs="*", default=[])
ap.add_argument("--out", required=True)
a = ap.parse_args()
TCK = os.sysconf("SC_CLK_TCK")
CPUS = set()
for part in a.cpus.split(","):
    lo, _, hi = part.partition("-")
    CPUS.update(range(int(lo), int(hi or lo) + 1))


def cpustat():
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            c = int(f[0][3:])
            if c in CPUS:
                v = [int(x) for x in f[1:]]
                out[c] = (sum(v), v[3] + v[4])  # total, idle + iowait
    return out


def threads():
    """{(pid, tid): (comm, ppid, ticks, cpu)} for every thread on the host."""
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            tids = os.listdir("/proc/%s/task" % d)
        except OSError:
            continue
        for t in tids:
            try:
                s = open("/proc/%s/task/%s/stat" % (d, t)).read()
            except OSError:
                continue
            comm = s[s.index("(") + 1:s.rindex(")")]
            f = s[s.rindex(")") + 2:].split()
            out[(int(d), int(t))] = (comm, int(f[1]), int(f[11]) + int(f[12]), int(f[36]))
    return out


def tree(th, roots):
    parent = {pid: v[1] for (pid, tid), v in th.items() if pid == tid}
    keep, grew = set(roots), True
    while grew:
        grew = False
        for pid, ppid in parent.items():
            if ppid in keep and pid not in keep:
                keep.add(pid)
                grew = True
    return keep


stop = False


def on_term(*_):
    global stop
    stop = True


signal.signal(signal.SIGTERM, on_term)
signal.signal(signal.SIGINT, on_term)
roots = a.exclude_tree + [os.getpid()]
names, total = {}, {}  # (pid, cpu) -> ticks over the run
c0, t0 = cpustat(), threads()
start = c0
with open(a.out, "a") as out:
    while not stop:
        time.sleep(1)
        c1, t1 = cpustat(), threads()
        mine = tree(t1, roots)
        util = {str(c): round(100 * (1 - (c1[c][1] - c0[c][1]) / max(1, c1[c][0] - c0[c][0])), 1)
                for c in sorted(CPUS)}
        foreign = {}
        for key, (comm, _, ticks, cpu) in t1.items():
            pid = key[0]
            if pid in mine or cpu not in CPUS or comm.startswith("nvidia-modeset") or key not in t0:
                continue
            d = ticks - t0[key][2]
            if d <= 0:
                continue
            names[pid] = t1.get((pid, pid), (comm,))[0]
            foreign[(pid, cpu)] = foreign.get((pid, cpu), 0) + d
            total[(pid, cpu)] = total.get((pid, cpu), 0) + d
        out.write(json.dumps({"t": round(time.time(), 3), "core_busy_pct": util,
                              "foreign": [{"pid": p, "comm": names[p], "cpu": c, "ticks": n}
                                          for (p, c), n in sorted(foreign.items())]}) + "\n")
        out.flush()
        c0, t0 = c1, t1
    end = cpustat()
    # elapsed ticks of one core, from the kernel's own per-core counters
    elapsed = max(end[c][0] - start[c][0] for c in CPUS)
    off = [{"pid": p, "comm": names[p], "cpu": c, "ticks": n}
           for (p, c), n in sorted(total.items()) if n * 5 > elapsed]
    out.write(json.dumps({"summary": True, "elapsed_ticks": elapsed, "clk_tck": TCK,
                          "offenders": off}) + "\n")
sys.exit(1 if off else 0)

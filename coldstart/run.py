#!/usr/bin/env python3
"""coldstart/run.py: driver of the cold-start chip microbenchmark (coldstart/PROTOCOL.md section 5).

Measurement tooling, not first-party product code. It runs inside one spark-window and writes
results/coldstart/{raw.jsonl,session.json,rows.wl}. --dry-run runs the same schedule on the host with
--build-only (no GPU opened) into a scratch directory; dry-run output is never a result.
"""
import argparse, datetime, hashlib, itertools, json, os, subprocess, sys, time

ARMS = ("CUDAPP", "PTXC", "NVRTC")
BIN = {"CUDAPP": "cs_cudapp", "PTXC": "cs_ptxc", "NVRTC": "cs_nvrtc"}
ORDERS = list(itertools.permutations(ARMS))            # the 6 orders, round r uses ORDERS[r % 6]
ARM_CORES = (9, 8, 7)                                  # X925 (A2), first one free of the nvidia-modeset kthreads
DRIVER_CORE = 12                                       # A725 (A2)
N_TOTAL = 1 << 20
CUDAPP_CODE_SHA = "9e28bd9c46f80584d77238a752e99cecddbc8d622ffbe320885ed2f499e1f075"  # hello_gpu --sim at cc0908bd
FORBIDDEN = ("libcuda", "libcudart", "libnvrtc", "libnvptx", "libnvidia")
CAP_S = 15 * 60

here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.dirname(here)


def mono():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load1():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def kthread_cpus():
    """CPU (stat field 39) of every nvidia-modeset kernel thread."""
    cpus = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/comm") as f:
                comm = f.read().strip()
            if not comm.startswith("nvidia-modeset"):
                continue
            with open(f"/proc/{d}/stat") as f:
                rest = f.read().rsplit(")", 1)[1].split()
            cpus[f"{d}:{comm}"] = int(rest[36])
        except (OSError, IndexError, ValueError):
            continue
    return cpus


def gpu_apps():
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=30)
    return {"rc": r.returncode, "apps": [l.strip() for l in r.stdout.splitlines() if l.strip()]}


def spawn(path, args, core, cwd):
    env = dict(os.environ, CUDA_CACHE_DISABLE="1")
    l1 = load1()
    t0 = mono()
    p = subprocess.Popen([path] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env,
                         preexec_fn=lambda: os.sched_setaffinity(0, {core}))
    out, err = p.communicate(timeout=120)
    t1 = mono()
    row = {"exit": p.returncode, "load1": l1, "spawn_ns": t0, "proc_ns": t1 - t0}
    try:
        rep = json.loads(out.decode().strip().splitlines()[-1])
        row.update(rep)
        row["exec_ns"] = rep["t_main_ns"] - t0
    except (ValueError, IndexError, KeyError):
        row["stdout"] = out.decode(errors="replace")[-400:]
    if p.returncode != 0:
        row["stderr"] = err.decode(errors="replace")[-400:]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="/tmp/h2h-coldstart-build", help="output directory of coldstart/build.sh")
    ap.add_argument("--out", default=os.path.join(repo, "results", "coldstart"))
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--warm-rounds", type=int, default=10)
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true", help="host only: --build-only, scratch output, no GPU")
    a = ap.parse_args()

    t_start = mono()
    if a.dry_run:
        if a.out == os.path.join(repo, "results", "coldstart"):
            sys.exit("run: --dry-run needs --out outside results/")
    elif os.environ.get("AIEN_GB10_LOCK_HELD") != "1" or not os.environ.get("SPARK_WINDOW"):
        sys.exit("run: not inside a spark-window holding the chip lock (spark-window run coldstart/run.py ...)")
    os.makedirs(a.out, exist_ok=True)
    os.sched_setaffinity(0, {DRIVER_CORE})

    session = {"started_utc": utc(), "dry_run": a.dry_run, "spark_window": os.environ.get("SPARK_WINDOW"),
               "rounds": a.rounds, "warmup": a.warmup, "warm_rounds": a.warm_rounds, "gap_s": a.gap, "driver_core": DRIVER_CORE,
               "uname": " ".join(os.uname()), "CUDA_CACHE_DISABLE": "1"}
    with open(os.path.join(a.build, "build.txt")) as f:
        build = f.read()
    session["build_txt"] = build
    want = {l.split()[1]: l.split()[2] for l in build.splitlines() if l.startswith("sha256 ")}
    paths = {arm: os.path.join(a.build, BIN[arm]) for arm in ARMS}
    for arm, p in paths.items():
        got = sha256(p)
        if got != want.get(BIN[arm]):
            sys.exit(f"run: {p} sha256 {got} is not the one in build.txt")
    ptx = os.path.join(here, "vecadd.ptx")
    if sha256(ptx) != want.get("vecadd.ptx"):
        sys.exit("run: coldstart/vecadd.ptx is not the one in build.txt")
    extra = {"CUDAPP": [], "PTXC": ["--ptx", ptx], "NVRTC": []}

    kt = kthread_cpus()
    session["nvidia_modeset_kthread_cpus"] = kt
    core = next((c for c in ARM_CORES if c not in kt.values()), None)
    if core is None:
        sys.exit(f"run: every arm core {ARM_CORES} hosts an nvidia-modeset kthread {kt}")
    session["arm_core"] = core

    if not a.dry_run:
        q = subprocess.run([sys.executable, os.path.join(repo, "bench", "quiet.py"), "--window", "60", "--max-load", "6",
                            "--exclude-tree", str(os.getpid())], capture_output=True, text=True)
        session["quiet"] = {"rc": q.returncode, "out": q.stdout.strip()}
        if q.returncode != 0:
            sys.exit(f"run: machine not quiet: {q.stdout.strip()}")
        session["gpu_apps_before"] = gpu_apps()
        if session["gpu_apps_before"]["apps"]:
            sys.exit(f"run: GPU compute apps present: {session['gpu_apps_before']}")

    raw = open(os.path.join(a.out, "raw.jsonl"), "w")

    def record(row):
        raw.write(json.dumps(row, sort_keys=True) + "\n")
        raw.flush()

    # Section 6 controls. Any failure stops the study before a measured process runs.
    fails = []
    codes = {}
    for arm in ARMS:
        cf = os.path.join(a.out, f"code.{arm}.bin")
        r = spawn(paths[arm], extra[arm] + ["--build-only", "--emit-code", cf], core, repo)
        r.update(arm=arm, phase="control_code")
        record(r)
        codes[arm] = sha256(cf) if r["exit"] == 0 else None
        if r["exit"] != 0:
            fails.append(f"{arm} --build-only exit {r['exit']}")
    session["code_sha256"] = codes
    if codes["CUDAPP"] != CUDAPP_CODE_SHA:
        fails.append(f"CUDAPP code sha {codes['CUDAPP']} != hello_gpu --sim {CUDAPP_CODE_SHA}")
    if codes["PTXC"] is None or codes["PTXC"] != codes["NVRTC"]:
        fails.append(f"PTXC cubin {codes['PTXC']} != NVRTC cubin {codes['NVRTC']} (like-for-like broken)")
    if not a.dry_run:
        outs = {}
        for arm in ARMS:
            r = spawn(paths[arm], extra[arm] + ["--skip-run"], core, repo)
            r.update(arm=arm, phase="control_skip_run")
            record(r)
            if not (r["exit"] == 1 and r.get("unwritten") == N_TOTAL and r.get("wrong") == N_TOTAL):
                fails.append(f"{arm} --skip-run not caught: exit {r['exit']} unwritten {r.get('unwritten')}")
            mf = os.path.join(a.out, f"maps.{arm}.txt")
            r = spawn(paths[arm], extra[arm] + ["--maps", mf], core, repo)
            r.update(arm=arm, phase="control_maps")
            record(r)
            if r["exit"] != 0 or r.get("wrong") != 0 or r.get("unwritten") != 0:
                fails.append(f"{arm} maps run exit {r['exit']}")
            outs[arm] = r.get("out_fnv")
            maps = ""
            if os.path.exists(mf):
                with open(mf) as f:
                    maps = f.read()
            libs = sorted({l.split()[-1] for l in maps.splitlines() if any(x in l.split()[-1] for x in FORBIDDEN)} if maps else [])
            dev = [l for l in maps.splitlines() if "/dev/nvidia" in l]
            if arm == "CUDAPP" and (libs or not dev):
                fails.append(f"CUDAPP maps: NVIDIA user libraries {libs}, /dev/nvidia mappings {len(dev)}")
            if arm != "CUDAPP" and not any("libcuda" in l for l in libs):
                fails.append(f"{arm} maps: libcuda absent, the library check checks nothing")
        session["out_fnv"] = outs
        if len(set(outs.values())) != 1 or None in outs.values():
            fails.append(f"output differs between arms {outs}")
    session["control_failures"] = fails
    if fails:
        session["verdict"] = "CONTROLS_FAILED"
        _finish(a, session, raw)
        sys.exit("run: controls failed: " + "; ".join(fails))

    # Cold processes run the cold path only (--warm 0), so proc_ns holds no warm builds.
    mode = (["--build-only"] if a.dry_run else []) + ["--warm", "0"]
    n = 0
    for w in range(a.warmup):
        for arm in ARMS:
            r = spawn(paths[arm], extra[arm] + mode, core, repo)
            r.update(arm=arm, phase="warmup", round=w)
            record(r)
    complete = True

    def timed_out():
        return (mono() - t_start) / 1e9 > CAP_S

    for rnd in range(a.rounds):
        order = ORDERS[rnd % len(ORDERS)]
        for slot, arm in enumerate(order):
            if timed_out():
                complete = False
                break
            time.sleep(a.gap)
            r = spawn(paths[arm], extra[arm] + mode, core, repo)
            r.update(arm=arm, phase="measured", round=rnd, slot=slot, order="".join(x[0] for x in order))
            record(r)
            n += 1
        if not complete:
            break
    # Warm builds (descriptive W): separate --build-only processes, 10 builds each after the cold one.
    for rnd in range(a.warm_rounds if complete else 0):
        order = ORDERS[rnd % len(ORDERS)]
        for slot, arm in enumerate(order):
            if timed_out():
                complete = False
                break
            time.sleep(a.gap)
            r = spawn(paths[arm], extra[arm] + ["--build-only"], core, repo)
            r.update(arm=arm, phase="warm", round=rnd, slot=slot, order="".join(x[0] for x in order))
            record(r)
        if not complete:
            break
    session["measured_processes"] = n
    session["complete"] = complete
    session["verdict"] = "DRY_RUN" if a.dry_run else ("MEASURED" if complete else "INCOMPLETE_CAP")
    if not a.dry_run:
        session["gpu_apps_after"] = gpu_apps()
    _finish(a, session, raw)
    print(json.dumps({"verdict": session["verdict"], "measured": n, "out": a.out}))


def _finish(a, session, raw):
    raw.close()
    session["finished_utc"] = utc()
    with open(os.path.join(a.out, "session.json"), "w") as f:
        json.dump(session, f, indent=1, sort_keys=True)
    # Wolfram input (coldstart/analysis.wl), every time in ns. csRows: one row per measured cold process.
    # csWarm: one row per warm-phase process with its 10 warm build times; analysis.wl takes each one's median.
    rows, warm = [], []
    with open(os.path.join(a.out, "raw.jsonl")) as f:
        for l in f:
            r = json.loads(l)
            g = lambda k: r.get(k, 0)
            if r.get("phase") == "warm":
                warm.append('{"%s",%d,%s,{%s}}' % (r["arm"], r["round"], "True" if r["exit"] == 0 else "False",
                                                  ",".join(str(x) for x in r.get("warm_build_ns", []))))
            if r.get("phase") != "measured":
                continue
            ok = r["exit"] == 0 and (r.get("build_only") == 1 or (r.get("wrong") == 0 and r.get("unwritten") == 0))
            first = g("exec_ns") + g("prep_ns") + g("build_ns") + g("open_ns") + g("load_ns") + g("data_ns") + g("run_ns") + g("check_ns")
            rows.append('{"%s",%d,%s,%s,%d,%d,%d,%d,%d}' % (
                r["arm"], r["round"], "True" if ok else "False", repr(r["load1"]), g("build_ns"),
                g("build_ns") + g("open_ns") + g("load_ns"), first, r["proc_ns"], g("exec_ns")))
    with open(os.path.join(a.out, "rows.wl"), "w") as f:
        f.write("(* csRows: {arm, round, ok, load1, build, ready, firstResult, proc, exec}; ns *)\n")
        f.write("csRows = {" + ",\n ".join(rows) + "};\n")
        f.write("(* csWarm: {arm, round, ok, {warm build ns ...}} *)\n")
        f.write("csWarm = {" + ",\n ".join(warm) + "};\n")


if __name__ == "__main__":
    main()

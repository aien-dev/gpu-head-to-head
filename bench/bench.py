#!/usr/bin/env python3
"""End-to-end OpenAI-chat streaming client for the gpu-head-to-head protocol.

Host measurement tool (Python 3 stdlib only). It is not product code; see README "Host tools".
Every request goes over HTTP to /v1/chat/completions with stream=true, temperature=0, thinking off.
One JSON row per request is appended to --out. Nothing is aggregated here; aggregation is done in
Wolfram (see PROTOCOL.md section 7) so that the raw rows stay the evidence.
"""
import argparse
import hashlib
import http.client
import json
import os
import sys
import threading
import time
import uuid


def load1():
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def one_request(host, port, model, prompt, max_tokens, timeout=1800, messages=None):
    body = {
        "model": model,
        "messages": messages if messages is not None else [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()
    row = {"t_send": None, "t_first": None, "t_last": None, "chunks": 0, "text": "",
           "usage": None, "finish_reason": None, "error": None, "http_status": None,
           "busy_retries": 0}
    conn = None
    done = False
    try:
        row["t_send"] = time.monotonic()
        while True:
            # PROTOCOL amendment A1: a server that refuses with "busy" (HTTP 429/503) is retried
            # every 50 ms; the clock keeps running from the first send.
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
            conn.request("POST", "/v1/chat/completions", body=data,
                         headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
            resp = conn.getresponse()
            row["http_status"] = resp.status
            if resp.status in (429, 503) and time.monotonic() - row["t_send"] < timeout:
                resp.read()
                conn.close()
                row["busy_retries"] += 1
                time.sleep(0.05)
                continue
            break
        if resp.status != 200:
            row["error"] = resp.read().decode(errors="replace")[:2000]
            return row
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                done = True
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                row["error"] = "bad json chunk: " + payload[:200].decode(errors="replace")
                continue
            if ev.get("error"):
                row["error"] = "server error event: " + str(ev["error"])[:1000]
                continue
            now = time.monotonic()
            if ev.get("usage"):
                row["usage"] = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    if row["t_first"] is None:
                        row["t_first"] = now
                    row["t_last"] = now
                    row["chunks"] += 1
                    row["text"] += piece
                if ch.get("finish_reason"):
                    row["finish_reason"] = ch["finish_reason"]
    except Exception as e:  # recorded, never hidden
        row["error"] = repr(e)
    finally:
        if conn is not None:
            conn.close()
    if row["error"] is None:
        if not done:
            row["error"] = "stream ended without DONE"
        elif not row["text"] or row["t_first"] is None or row["t_last"] is None:
            row["error"] = "no content received"
        elif not row["finish_reason"]:
            row["error"] = "missing finish reason"
    return row


# PROTOCOL A10 TTFT rows. The nonce depends only on the prompt id, the repetition and whether the
# request is a warmup, never on the arm or the clock: every arm receives byte-identical requests,
# and no prefix cache can carry one repetition into the next.
NONCE_LINE = "Request id %s. Ignore this line.\n"


def ttft_nonce(prompt_id, rep, warm):
    key = "h2h-ttft|%s|%s%d" % (prompt_id, "w" if warm else "r", rep)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def ttft_requests(kind, byid, prompt_id, nonce):
    """[(phase, messages)] for one repetition: fresh -> one request; turn2 -> prime, then turn 2."""
    first = {"role": "user", "content": NONCE_LINE % nonce + byid[prompt_id]["text"]}
    if kind == "fresh":
        return [("fresh", [first])]
    full = [dict(first), {"role": "assistant", "content": byid["turn2-assistant"]["text"]},
            {"role": "user", "content": byid["turn2-new"]["text"]}]
    return [("prime", [first]), ("turn2", full)]


def messages_sha256(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def prefix_cache_lines(host, port):
    """vLLM /metrics prefix-cache counter lines (raw text; deltas are taken in Wolfram)."""
    try:
        conn = http.client.HTTPConnection(host, port, timeout=10)
        conn.request("GET", "/metrics")
        resp = conn.getresponse()
        text = resp.read().decode(errors="replace")
        conn.close()
        if resp.status != 200:
            return {"error": "HTTP %d" % resp.status}
        return {"lines": [l for l in text.splitlines() if "prefix_cache" in l and not l.startswith("#")]}
    except Exception as e:  # recorded, never hidden
        return {"error": repr(e)}


def run_ttft(args, byid, f):
    """One request at a time; rows carry phase, nonce and the sha256 of the exact message list."""
    failed = False
    for k in range(args.warmup + args.reps):
        warm = k < args.warmup
        rep = args.rep_base + (k if warm else k - args.warmup)
        # a warmup may use a cheaper prompt (--warm-select) and is always a single fresh request
        pid = (args.warm_select or args.select) if warm else args.select
        nonce = ttft_nonce(pid, rep, warm)
        for phase, messages in ttft_requests("fresh" if warm else args.ttft, byid, pid, nonce):
            before = prefix_cache_lines(args.host, args.port) if args.ttft == "turn2" else None
            l1 = load1()
            ts = time.time()
            r = one_request(args.host, args.port, args.model, None, args.max_tokens, messages=messages)
            r.update({"arm": args.arm, "round": args.round,
                      "metric": args.metric + ("-prime" if phase == "prime" else ""), "rep": rep,
                      "warmup": warm, "concurrency": 1, "stream_index": 0,
                      "batch_id": uuid.uuid4().hex[:12], "load1_start": l1, "load1_end": load1(),
                      "wall_utc": ts, "max_tokens": args.max_tokens, "prompt_id": pid,
                      "cell_run": args.cell_run, "ttft_phase": phase, "nonce": nonce,
                      "messages_sha256": messages_sha256(messages),
                      "prefix_sha256": messages_sha256(messages[:1]),
                      "server_cpus": os.environ.get("H2H_SRV_CPUS"),
                      "client_cpus": os.environ.get("H2H_CLI_CPUS")})
            if before is not None:
                r["prefix_cache_before"] = before
                r["prefix_cache_after"] = prefix_cache_lines(args.host, args.port)
            f.write(json.dumps(r) + "\n")
            f.flush()
            failed = failed or bool(r["error"])
            u = r["usage"] or {}
            print("%s %s %s %s rep=%d warm=%s pt=%s ct=%s err=%s" % (
                args.arm, args.round, r["metric"], phase, rep, warm, u.get("prompt_tokens"),
                u.get("completion_tokens"), (r["error"] or "")[:120]), file=sys.stderr)
    return failed


def run_batch(args, prompts, rep, warm):
    rows = [None] * len(prompts)
    barrier = threading.Barrier(len(prompts))

    def worker(i):
        barrier.wait()
        rows[i] = one_request(args.host, args.port, args.model, prompts[i], args.max_tokens)

    l1 = load1()
    ts = time.time()
    th = [threading.Thread(target=worker, args=(i,)) for i in range(len(prompts))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    batch_id = uuid.uuid4().hex[:12]
    out = []
    for i, r in enumerate(rows):
        r.update({"arm": args.arm, "round": args.round, "metric": args.metric, "rep": rep,
                  "warmup": warm, "concurrency": len(prompts), "stream_index": i,
                  "batch_id": batch_id, "load1_start": l1, "load1_end": load1(),
                  "wall_utc": ts, "max_tokens": args.max_tokens,
                  "prompt_id": args.prompt_ids[i] if args.prompt_ids else None,
                  "cell_run": args.cell_run,
                  "server_cpus": os.environ.get("H2H_SRV_CPUS"),
                  "client_cpus": os.environ.get("H2H_CLI_CPUS")})
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--round", required=True)
    ap.add_argument("--metric", required=True)
    ap.add_argument("--prompts", required=True, help="JSONL with {id, text}")
    ap.add_argument("--select", default="", help="comma-separated prompt ids (default: all, in order)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--nonce", action="store_true",
                    help="prefix each prompt with a unique nonce line so no prefix cache can hit")
    ap.add_argument("--ttft", choices=["fresh", "turn2"],
                    help="A10 TTFT rows: one --select id, deterministic per-rep nonce, concurrency 1")
    ap.add_argument("--rep-base", type=int, default=0, help="first repetition number (--ttft only)")
    ap.add_argument("--warm-select", default="", help="prompt id for --ttft warmup requests")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cell-run", default="", help="id of this cell invocation (voiding unit, A2)")
    args = ap.parse_args()
    if args.concurrency < 1 or args.reps < 1 or args.warmup < 0 or args.max_tokens < 1:
        ap.error("concurrency, reps and max-tokens must be positive; warmup must be nonnegative")
    if args.ttft:
        byid = {p["id"]: p for p in (json.loads(l) for l in open(args.prompts) if l.strip())}
        need = [args.select] + (["turn2-assistant", "turn2-new"] if args.ttft == "turn2" else [])
        need += [args.warm_select] if args.warm_select else []
        if (args.nonce or args.concurrency != 1 or args.rep_base < 0 or "," in args.select
                or not all(n in byid for n in need)):
            ap.error("--ttft takes exactly one --select id present in --prompts (turn2 also needs "
                     "turn2-assistant and turn2-new), concurrency 1, no --nonce, rep-base >= 0")
        with open(args.out, "a") as f:
            return 1 if run_ttft(args, byid, f) else 0
    if args.rep_base or args.warm_select:
        ap.error("--rep-base and --warm-select need --ttft")

    allp = [json.loads(l) for l in open(args.prompts) if l.strip()]
    if args.select:
        want = args.select.split(",")
        byid = {p["id"]: p for p in allp}
        allp = [byid[w] for w in want]
    if not allp:
        ap.error("prompt selection is empty")
    failed = False
    with open(args.out, "a") as f:
        total = args.warmup + args.reps
        for k in range(total):
            warm = k < args.warmup
            rep = k if warm else k - args.warmup
            # each batch takes `concurrency` prompts, cycling through the list
            sel = [allp[(k * args.concurrency + j) % len(allp)] for j in range(args.concurrency)]
            texts = []
            for p in sel:
                t = p["text"]
                if args.nonce:
                    t = NONCE_LINE % uuid.uuid4().hex + t
                texts.append(t)
            args.prompt_ids = [p["id"] for p in sel]
            for r in run_batch(args, texts, rep, warm):
                f.write(json.dumps(r) + "\n")
                f.flush()
                err = r["error"]
                failed = failed or bool(err)
                u = r["usage"] or {}
                print("%s %s %s rep=%d warm=%s c=%d i=%d pt=%s ct=%s err=%s" % (
                    args.arm, args.round, args.metric, rep, warm, r["concurrency"], r["stream_index"],
                    u.get("prompt_tokens"), u.get("completion_tokens"), (err or "")[:120]),
                    file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

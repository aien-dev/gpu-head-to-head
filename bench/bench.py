#!/usr/bin/env python3
"""End-to-end OpenAI-chat streaming client for the gpu-head-to-head protocol.

Host measurement tool (Python 3 stdlib only). It is not product code; see README "Host tools".
Every request goes over HTTP to /v1/chat/completions with stream=true, temperature=0, thinking off.
One JSON row per request is appended to --out. Nothing is aggregated here; aggregation is done in
Wolfram (see PROTOCOL.md section 7) so that the raw rows stay the evidence.
"""
import argparse
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


def one_request(host, port, model, prompt, max_tokens, timeout=1800):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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
           "usage": None, "finish_reason": None, "error": None, "http_status": None}
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        row["t_send"] = time.monotonic()
        conn.request("POST", "/v1/chat/completions", body=data,
                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        resp = conn.getresponse()
        row["http_status"] = resp.status
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
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                row["error"] = "bad json chunk: " + payload[:200].decode(errors="replace")
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
        conn.close()
    return row


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
                  "prompt_id": args.prompt_ids[i] if args.prompt_ids else None})
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
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    allp = [json.loads(l) for l in open(args.prompts) if l.strip()]
    if args.select:
        want = args.select.split(",")
        byid = {p["id"]: p for p in allp}
        allp = [byid[w] for w in want]
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
                    t = "Request id %s. Ignore this line.\n" % uuid.uuid4().hex + t
                texts.append(t)
            args.prompt_ids = [p["id"] for p in sel]
            for r in run_batch(args, texts, rep, warm):
                f.write(json.dumps(r) + "\n")
                f.flush()
                err = r["error"]
                u = r["usage"] or {}
                print("%s %s %s rep=%d warm=%s c=%d i=%d pt=%s ct=%s err=%s" % (
                    args.arm, args.round, args.metric, rep, warm, r["concurrency"], r["stream_index"],
                    u.get("prompt_tokens"), u.get("completion_tokens"), (err or "")[:120]),
                    file=sys.stderr)


if __name__ == "__main__":
    main()

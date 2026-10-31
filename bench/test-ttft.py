#!/usr/bin/env python3
"""PROTOCOL.md A10 TTFT rows: identical requests across arms, per-rep nonce, turn-2 prefix identity.

Local synthetic HTTP only; no model, no GPU."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROMPTS = HERE.parent / 'prompts' / 'ttft.jsonl'
spec = importlib.util.spec_from_file_location('bench_client', HERE / 'bench.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
gspec = importlib.util.spec_from_file_location('make_ttft', HERE / 'make_ttft_prompts.py')
make_ttft = importlib.util.module_from_spec(gspec)
gspec.loader.exec_module(make_ttft)


def event(value):
    return 'data: ' + json.dumps(value) + '\n\n'


STREAM = (event({'choices': [{'delta': {'content': 'x'}}]})
          + event({'choices': [{'finish_reason': 'length'}], 'usage': {'prompt_tokens': 7, 'completion_tokens': 1}})
          + 'data: [DONE]\n\n')
METRICS = ('# HELP vllm:prefix_cache_hits_total hits\n'
           'vllm:prefix_cache_queries_total{model_name="q36"} 10.0\n'
           'vllm:prefix_cache_hits_total{model_name="q36"} 4.0\nvllm:other_total 1.0\n')


class FakeServer:
    """Records every request body in arrival order; answers /metrics and the chat stream."""
    def __enter__(self):
        bodies, gets = self.bodies, self.gets = [], []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                gets.append(self.path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(METRICS.encode())

            def do_POST(self):
                bodies.append(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(STREAM.encode())
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


def run_bench(arm, *extra, prompts=PROMPTS):
    """Run bench.py against a fresh fake server; returns (rc, raw request bodies, rows, GET paths, stderr)."""
    with FakeServer() as srv, tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp, 'rows.jsonl')
        r = subprocess.run([sys.executable, str(HERE / 'bench.py'), '--port', str(srv.server.server_port),
                            '--model', 'q36', '--arm', arm, '--round', 'T1', '--metric', 'ttft-x',
                            '--prompts', str(prompts), '--max-tokens', '1', '--out', str(out)] + list(extra),
                           capture_output=True, text=True, timeout=60, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        rows = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
        return r.returncode, list(srv.bodies), rows, list(srv.gets), r.stderr


def first_user(body):
    return json.loads(body)['messages'][0]


class NonceTests(unittest.TestCase):
    def test_nonce_is_deterministic_and_arm_free(self):
        self.assertEqual(bench.ttft_nonce('t2k', 0, False), bench.ttft_nonce('t2k', 0, False))
        self.assertEqual(bench.ttft_nonce('t2k', 0, False),
                         hashlib.sha256(b'h2h-ttft|t2k|r0').hexdigest()[:16])

    def test_nonce_changes_per_rep_warmup_and_prompt(self):
        seen = {bench.ttft_nonce(p, rep, w) for p in ('t2k', 't8k', 't32k') for rep in range(4) for w in (False, True)}
        self.assertEqual(len(seen), 3 * 4 * 2)


class FreshRowTests(unittest.TestCase):
    ARGS = ('--ttft', 'fresh', '--select', 't8k', '--warmup', '1', '--warm-select', 't2k', '--reps', '3')

    def test_requests_are_byte_identical_across_arms(self):
        rc_a, bodies_a, rows_a, _, err = run_bench('OURS', *self.ARGS)
        self.assertEqual(rc_a, 0, err)
        rc_b, bodies_b, rows_b, _, err = run_bench('VLLM_FP8', *self.ARGS)
        self.assertEqual(rc_b, 0, err)
        self.assertEqual(len(bodies_a), 4)
        self.assertEqual(bodies_a, bodies_b)
        self.assertEqual([r['messages_sha256'] for r in rows_a], [r['messages_sha256'] for r in rows_b])

    def test_nonce_changes_per_rep_and_text_is_the_pinned_prompt(self):
        rc, bodies, rows, gets, err = run_bench('OURS', *self.ARGS)
        self.assertEqual(rc, 0, err)
        byid = {p['id']: p for p in map(json.loads, PROMPTS.read_text().splitlines())}
        users = [first_user(b)['content'] for b in bodies]
        self.assertEqual(len(set(u.split('\n', 1)[0] for u in users)), 4)        # 1 warmup + 3 reps
        self.assertTrue(users[0].endswith(byid['t2k']['text']))                  # cheap warmup prompt
        for u, row in zip(users[1:], rows[1:]):
            self.assertEqual(u, bench.NONCE_LINE % row['nonce'] + byid['t8k']['text'])
            self.assertEqual(row['nonce'], bench.ttft_nonce('t8k', row['rep'], False))
        self.assertEqual([(r['warmup'], r['rep'], r['prompt_id']) for r in rows],
                         [(True, 0, 't2k'), (False, 0, 't8k'), (False, 1, 't8k'), (False, 2, 't8k')])
        self.assertEqual(gets, [], 'fresh rows never read /metrics')

    def test_rep_base_continues_the_nonce_sequence(self):
        _, one, _, _, _ = run_bench('OURS', '--ttft', 'fresh', '--select', 't2k', '--warmup', '0', '--reps', '3')
        _, split, _, _, _ = run_bench('OURS', '--ttft', 'fresh', '--select', 't2k', '--warmup', '0', '--reps', '2',
                                      '--rep-base', '1')
        self.assertEqual(one[1:], split)

    def test_negative_control_random_nonce_differs_across_arms(self):
        # the legacy prefill path (uuid nonce) must fail the same identity comparison
        a = run_bench('OURS', '--select', 't2k', '--warmup', '0', '--reps', '1', '--nonce')[1]
        b = run_bench('VLLM_FP8', '--select', 't2k', '--warmup', '0', '--reps', '1', '--nonce')[1]
        self.assertEqual((len(a), len(b)), (1, 1))
        self.assertNotEqual(a, b)


class TurnTwoTests(unittest.TestCase):
    ARGS = ('--ttft', 'turn2', '--select', 't32k', '--warmup', '1', '--warm-select', 't2k', '--reps', '2')

    def test_turn2_prefix_is_byte_identical_to_the_primed_prompt(self):
        rc, bodies, rows, gets, err = run_bench('VLLM_FP8', *self.ARGS)
        self.assertEqual(rc, 0, err)
        self.assertEqual([r['ttft_phase'] for r in rows], ['fresh', 'prime', 'turn2', 'prime', 'turn2'])
        self.assertEqual([r['metric'] for r in rows[1:3]], ['ttft-x-prime', 'ttft-x'])
        byid = {p['id']: p for p in map(json.loads, PROMPTS.read_text().splitlines())}
        for prime, full in ((bodies[1], bodies[2]), (bodies[3], bodies[4])):
            pm, fm = json.loads(prime)['messages'], json.loads(full)['messages']
            self.assertEqual(len(pm), 1)
            self.assertEqual(json.dumps(pm[0]).encode(), json.dumps(fm[0]).encode())
            self.assertEqual([m['role'] for m in fm], ['user', 'assistant', 'user'])
            self.assertEqual(fm[1]['content'], byid['turn2-assistant']['text'])
            self.assertEqual(fm[2]['content'], byid['turn2-new']['text'])
            self.assertTrue(pm[0]['content'].endswith(byid['t32k']['text']))
        self.assertNotEqual(first_user(bodies[1]), first_user(bodies[3]), 'nonce must change per rep')
        self.assertEqual(rows[1]['prefix_sha256'], rows[2]['prefix_sha256'])
        self.assertNotEqual(rows[1]['prefix_sha256'], rows[3]['prefix_sha256'])

    def test_turn2_requests_identical_across_arms(self):
        a = run_bench('OURS', *self.ARGS)
        b = run_bench('VLLM_NVFP4', *self.ARGS)
        self.assertEqual((a[0], b[0]), (0, 0))
        self.assertEqual(a[1], b[1])

    def test_prefix_cache_counters_are_recorded_around_each_request(self):
        rc, _, rows, gets, err = run_bench('VLLM_FP8', *self.ARGS)
        self.assertEqual(rc, 0, err)
        self.assertEqual(gets, ['/metrics'] * 10)
        want = ['vllm:prefix_cache_queries_total{model_name="q36"} 10.0', 'vllm:prefix_cache_hits_total{model_name="q36"} 4.0']
        self.assertEqual(rows[2]['prefix_cache_before'], {'lines': want})
        self.assertEqual(rows[2]['prefix_cache_after'], {'lines': want})

    def test_negative_control_tampered_prefix_is_detected(self):
        byid = {p['id']: p for p in map(json.loads, PROMPTS.read_text().splitlines())}
        prime = bench.ttft_requests('turn2', byid, 't32k', bench.ttft_nonce('t32k', 0, False))[0][1]
        other = bench.ttft_requests('turn2', byid, 't32k', bench.ttft_nonce('t32k', 1, False))[1][1]
        self.assertNotEqual(bench.messages_sha256(prime[:1]), bench.messages_sha256(other[:1]))


class ArgumentTests(unittest.TestCase):
    def test_refusals(self):
        for extra in (('--ttft', 'fresh', '--select', 't2k', '--nonce'),
                      ('--ttft', 'fresh', '--select', 't2k,t8k'),
                      ('--ttft', 'fresh', '--select', 'p2k'),
                      ('--ttft', 'fresh', '--select', 't2k', '--concurrency', '2'),
                      ('--ttft', 'fresh', '--select', 't2k', '--warm-select', 'nope'),
                      ('--select', 't2k', '--rep-base', '1'),
                      ('--select', 't2k', '--warm-select', 't2k')):
            with self.subTest(extra=extra):
                rc, bodies, _, _, _ = run_bench('OURS', *extra)
                self.assertEqual((rc, bodies), (2, []))

    def test_turn2_needs_its_conversation_texts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, 'p.jsonl')
            p.write_text(json.dumps({'id': 't32k', 'text': 'x'}) + '\n')
            rc, bodies, _, _, _ = run_bench('OURS', '--ttft', 'turn2', '--select', 't32k', prompts=p)
            self.assertEqual((rc, bodies), (2, []))


class PromptFileTests(unittest.TestCase):
    def test_committed_prompts_carry_the_pinned_source_shas(self):
        rows = {r['id']: r for r in map(json.loads, PROMPTS.read_text().splitlines())}
        self.assertEqual(sorted(rows), ['t2k', 't32k', 't8k', 'turn2-assistant', 'turn2-new'])
        for pid, (_, sha) in make_ttft.SOURCES.items():
            self.assertEqual(rows[pid]['source_sha256'], sha)
            if pid != 'turn2-new':
                self.assertEqual(hashlib.sha256(rows[pid]['text'].encode()).hexdigest(), sha)
        self.assertEqual(rows['turn2-assistant']['text'], make_ttft.TURN2_ASSISTANT)

    def test_generator_refuses_a_wrong_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = [Path(tmp, n) for n in ('a', 'b', 'c', 'd')]
            for s in src:
                s.write_text('wrong\n')
            out = Path(tmp, 'out.jsonl')
            self.assertEqual(make_ttft.main(['x'] + [str(s) for s in src] + [str(out)]), 2)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()

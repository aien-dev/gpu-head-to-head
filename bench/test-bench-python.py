#!/usr/bin/env python3
"""External Python measurement client controls; local synthetic HTTP only."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bench_client', HERE / 'bench.py')
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)

def event(value):
    return 'data: ' + json.dumps(value) + '\n\n'
CONTENT = event({'choices': [{'delta': {'content': 'fixture'}}]})
FINISH = event({'choices': [{'finish_reason': 'length'}],
                'usage': {'prompt_tokens': 5, 'completion_tokens': 2}})
DONE = 'data: [DONE]\n\n'

class ClientTests(unittest.TestCase):
    def serve(self, payload, callback, status=200):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(status)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(payload.encode())
        with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                return callback(server.server_port)
            finally:
                server.shutdown()
                thread.join()
    def request(self, payload, status=200):
        return self.serve(payload, lambda port: client.one_request(
            '127.0.0.1', port, 'fixture', 'test', 2), status)
    def test_complete_stream_passes(self):
        row = self.request(CONTENT + FINISH + DONE)
        self.assertIsNone(row['error'])
        self.assertEqual(row['text'], 'fixture')
    def test_truncated_stream_refused(self):
        self.assertIsNotNone(self.request(CONTENT + FINISH)['error'])
    def test_missing_finish_refused(self):
        self.assertIsNotNone(self.request(CONTENT + DONE)['error'])
    def test_empty_stream_refused(self):
        self.assertIsNotNone(self.request(FINISH + DONE)['error'])
    def test_server_error_event_refused(self):
        self.assertIsNotNone(self.request(event({'error': {'message': 'fixture'}}) + DONE)['error'])
    def test_malformed_json_refused(self):
        self.assertIsNotNone(self.request('data: {bad}\n\n' + CONTENT + FINISH + DONE)['error'])
    def test_http_error_refused(self):
        self.assertIsNotNone(self.request('fixture', 400)['error'])
    def test_missing_usage_remains_retokenizable(self):
        row = self.request(CONTENT + event({'choices': [{'finish_reason': 'stop'}]}) + DONE)
        self.assertIsNone(row['error'])
        self.assertIsNone(row['usage'])
    def test_cli_preserves_failed_row_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts, output = root / 'prompts.jsonl', root / 'rows.jsonl'
            prompts.write_text('{"id":"p0","text":"test"}\n')
            def run(port):
                args = [sys.executable, str(HERE / 'bench.py'), '--port', str(port), '--model', 'fixture',
                        '--arm', 'CONTROL', '--round', 'R0', '--metric', 'decode', '--prompts', str(prompts),
                        '--max-tokens', '2', '--warmup', '0', '--reps', '1', '--out', str(output)]
                result = subprocess.run(args, capture_output=True, text=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                rows = [json.loads(x) for x in output.read_text().splitlines()]
                self.assertEqual(len(rows), 1)
                self.assertTrue(rows[0]['error'])
            self.serve('fixture', run, status=400)

if __name__ == '__main__':
    unittest.main()

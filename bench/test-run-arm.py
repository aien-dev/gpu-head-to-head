#!/usr/bin/env python3
"""Exercise the real shell cell function with CPU-only instrument substitutes."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent

class CellGateTests(unittest.TestCase):
    def run_cell(self, *, quiet=0, client=0, monitor=0, summary=True, offenders=False,
                 contaminated=False, timeout=False, missing_samples=False, invalid_samples=False, elapsed=100):
        source = (HERE / 'run_arm.sh').read_text()
        function = source[source.index('cell_status() {'):source.index('# ---- session ----')]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'mem.tsv').write_text('' if missing_samples else '100 1000 100 0 %s\n' % ('NA' if invalid_samples else '5' if contaminated else '99'))
            if timeout:
                (root / 'TIMEOUT').touch()
            summary_row = {'summary': True, 'elapsed_ticks': elapsed, 'offenders':
                           [{'pid': 42, 'cpu': 5, 'ticks': 99}] if offenders else []}
            monitor_text = json.dumps(summary_row) + '\n' if summary else ''
            prefix = 'source %s\n' % shlex.quote(str(HERE / 'session-lifecycle.sh')) + '''set -u
ROOT=%s
LOGD=$ROOT REPO=$ROOT RESULTS=$ROOT OUT=$ROOT/rows.jsonl
ARM=CONTROL ROUND=R0 PORT=1 MID=fixture
MON_CPU=2 CLI_CPUS=0-1 CLI_NICE=0 SRV_CPUS=5 SRV_LIST=' 5 '
CELLS_COMPLETED=0
wait_quiet() { return %d; }
mytree() { echo $$; }
date() { echo 100; }
sleep() { :; }
kill() { :; }
ts() { echo fixture-time; }
taskset() {
  case "$*" in
    *coremon.py*) printf '%%s' %s > "$LOGD/coremon/CONTROL-R0-decode-c1-t${try}.jsonl"; return %d ;;
    *) echo client >> "$ROOT/client-calls"; return %d ;;
  esac
}
''' % (shlex.quote(tmp), quiet, shlex.quote(monitor_text), monitor, client)
            result = subprocess.run(['bash', '-c', prefix + function + '\ncell decode-c1 1 2 decode.jsonl\n'],
                                    capture_output=True, text=True, timeout=5)
            calls = (root / 'client-calls').read_text().splitlines() if (root / 'client-calls').exists() else []
            return result.returncode, calls, result.stderr

    def test_clean_cell_passes(self):
        rc, calls, err = self.run_cell()
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(calls), 1)

    def test_quiet_failure_never_runs_client(self):
        rc, calls, err = self.run_cell(quiet=1)
        self.assertNotEqual(rc, 0, err)
        self.assertEqual(calls, [])

    def test_client_failure_is_not_accepted(self):
        self.assertNotEqual(self.run_cell(client=7)[0], 0)

    def test_missing_monitor_summary_is_not_accepted(self):
        self.assertNotEqual(self.run_cell(summary=False)[0], 0)

    def test_monitor_crash_is_not_accepted(self):
        self.assertNotEqual(self.run_cell(monitor=2)[0], 0)

    def test_exhausted_foreign_process_retries_fail(self):
        rc, calls, _ = self.run_cell(monitor=1, offenders=True)
        self.assertNotEqual(rc, 0)
        self.assertEqual(len(calls), 3)

    def test_exhausted_kthread_retries_fail(self):
        rc, calls, _ = self.run_cell(contaminated=True)
        self.assertNotEqual(rc, 0)
        self.assertEqual(len(calls), 3)

    def test_missing_kthread_samples_fail(self):
        self.assertNotEqual(self.run_cell(missing_samples=True)[0], 0)

    def test_invalid_kthread_samples_fail(self):
        self.assertNotEqual(self.run_cell(invalid_samples=True)[0], 0)

    def test_zero_monitor_duration_fails(self):
        self.assertNotEqual(self.run_cell(elapsed=0)[0], 0)

    def test_completion_requires_all_cells_and_no_timeout(self):
        source = (HERE / 'run_arm.sh').read_text()
        function = source[source.index('mark_complete() {'):source.index('# ---- session ----')]
        for count, timed_out, expected in [(0, False, False), (13, False, False),
                                            (14, True, False), (14, False, True)]:
            with self.subTest(count=count, timed_out=timed_out), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                if timed_out:
                    (root / 'TIMEOUT').touch()
                script = 'LOGD=%s\nCELLS_COMPLETED=%d\n' % (shlex.quote(tmp), count)
                result = subprocess.run(['bash', '-c', script + function + '\nmark_complete'],
                                        capture_output=True, timeout=5)
                self.assertEqual(result.returncode == 0, expected)
                self.assertEqual((root / 'DONE').exists(), expected)

    def test_existing_session_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / 'logs/OURS-R1'
            log.mkdir(parents=True)
            sentinel = log / 'DONE'
            sentinel.write_text('historical')
            env = dict(os.environ, H2H_RESULTS=tmp)
            result = subprocess.run(['bash', str(HERE / 'run_arm.sh'), 'OURS', 'R1'],
                                    env=env, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(sentinel.read_text(), 'historical')

    def test_timeout_never_runs_client(self):
        rc, calls, _ = self.run_cell(timeout=True)
        self.assertNotEqual(rc, 0)
        self.assertEqual(calls, [])

if __name__ == '__main__':
    unittest.main()

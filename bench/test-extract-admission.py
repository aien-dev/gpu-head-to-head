#!/usr/bin/env python3
"""Prove that rejected, unobserved and ambiguous cells cannot reach report data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent

class AdmissionTests(unittest.TestCase):
    def extract(self, statuses):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / 'results/raw'
            raw.mkdir(parents=True)
            row = dict(cell_run='CONTROL-R0-decode-t1', warmup=False, arm='CONTROL',
                       round='R0', metric='decode', rep=0, batch_id='fixture', stream_index=0,
                       load1_start=0, t_send=1, t_first=2, t_last=3, error=None, http_status=200,
                       usage=dict(prompt_tokens=5, completion_tokens=2))
            (raw / 'CONTROL-R0.jsonl').write_text(json.dumps(row) + '\n')
            if statuses is not None:
                entries = [dict(cell_run=row['cell_run'], status=status) for status in statuses]
                (root / 'results/cell-status.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in entries))
            result = subprocess.run([sys.executable, os.environ.get('H2H_EXTRACT_SCRIPT', str(HERE / 'extract.py')), tmp],
                                    text=True, capture_output=True, timeout=5, check=True)
            return '"CONTROL"' in result.stdout

    def test_only_single_pass_is_admitted(self):
        self.assertTrue(self.extract(['PASS']))
    def test_failed_or_unobserved_cells_are_excluded(self):
        for statuses in [[], ['FAIL'], ['NOT_RUN'], ['VOID'], ['UNKNOWN']]:
            with self.subTest(statuses=statuses):
                self.assertFalse(self.extract(statuses))
    def test_duplicate_or_conflicting_admissions_are_excluded(self):
        for statuses in [['PASS', 'PASS'], ['FAIL', 'PASS'], ['PASS', 'FAIL']]:
            with self.subTest(statuses=statuses):
                self.assertFalse(self.extract(statuses))
    def test_historical_runs_keep_explicit_legacy_behavior(self):
        self.assertTrue(self.extract(None))

if __name__ == '__main__':
    unittest.main()

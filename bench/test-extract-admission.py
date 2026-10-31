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
    def extract(self, statuses, arm='CONTROL', chunks=None, completion_tokens=2, stdout=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / 'results/raw'
            raw.mkdir(parents=True)
            row = dict(cell_run=arm + '-R0-decode-t1', warmup=False, arm=arm,
                       round='R0', metric='decode', rep=0, batch_id='fixture', stream_index=0,
                       load1_start=0, t_send=1, t_first=2, t_last=3, error=None, http_status=200,
                       usage=dict(prompt_tokens=5, completion_tokens=completion_tokens))
            if chunks is not None:
                row['chunks'] = chunks
            (raw / (arm + '-R0.jsonl')).write_text(json.dumps(row) + '\n')
            if statuses is not None:
                entries = [dict(cell_run=row['cell_run'], status=status) for status in statuses]
                (root / 'results/cell-status.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in entries))
            result = subprocess.run([sys.executable, os.environ.get('H2H_EXTRACT_SCRIPT', str(HERE / 'extract.py')), tmp],
                                    text=True, capture_output=True, timeout=5, check=True)
            return result.stdout if stdout else json.dumps(arm) in result.stdout

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
    def test_profiled_arm_rows_never_enter_timing(self):
        # A10: *_NSYS rows are excluded; the same row under the unprofiled arm name is the control
        self.assertFalse(self.extract(['PASS'], arm='VLLM_NVFP4_NSYS'))
        self.assertTrue(self.extract(['PASS'], arm='VLLM_NVFP4'))
    def test_mtp_rows_count_usage_tokens_not_stream_chunks(self):
        # A10: an MTP stream delivers several tokens per chunk; the row carries usage.completion_tokens
        out = self.extract(['PASS'], arm='VLLM_FP8_MTP', chunks=3, completion_tokens=9, stdout=True)
        fields = out[out.index('{"VLLM_FP8_MTP"'):].split('}')[0].split(', ')
        self.assertEqual(fields[11], '9')

if __name__ == '__main__':
    unittest.main()

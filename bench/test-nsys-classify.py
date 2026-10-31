#!/usr/bin/env python3
"""PROTOCOL.md A10: nsys_classify.py on a synthetic decode step, with a negative control."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent

# One synthetic step: a GDN layer, an attention layer, final norm, lm_head, sampling.
LAYER_GDN = [('fused_add_rms_norm_kernel', 100), ('cutlass_scaled_mm_sm100', 300),
             ('causal_conv1d_update_kernel', 50), ('fused_recurrent_gated_delta_rule_fwd', 400),
             ('rms_norm_kernel', 30), ('cutlass_scaled_mm_sm100', 200),
             ('fused_add_rms_norm_kernel', 100), ('cutlass_scaled_mm_sm100', 40),
             ('topkGatingSoftmax', 60), ('moe_align_block_size_kernel', 20),
             ('marlin_moe_wna16::Marlin<4>', 2000), ('act_and_mul_kernel', 30), ('marlin_moe_wna16::Marlin<4>', 1500),
             ('cutlass_scaled_mm_sm100', 120), ('cutlass_scaled_mm_sm100', 90), ('moe_sum_kernel', 10)]
LAYER_ATTN = [('fused_add_rms_norm_kernel', 100), ('cutlass_scaled_mm_sm100', 250),
              ('rotary_embedding_kernel', 20), ('reshape_and_cache_flash_kernel', 15),
              ('BatchDecodeWithPagedKVCacheKernel', 500), ('cutlass_scaled_mm_sm100', 210),
              ('fused_add_rms_norm_kernel', 100), ('cutlass_scaled_mm_sm100', 40),
              ('topkGatingSoftmax', 60), ('marlin_moe_wna16::Marlin<4>', 2100),
              ('marlin_moe_wna16::Marlin<4>', 1600), ('cutlass_scaled_mm_sm100', 130), ('cutlass_scaled_mm_sm100', 95)]
TAIL = [('rms_norm_kernel', 40), ('nvjet_tst_bf16_gemm', 4_000_000), ('argmax_kernel', 25),
        ('Memcpy DtoH (Device -> Pageable)', 8)]


def run(kernels):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp, 'trace.csv')
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Start (ns)', 'Duration (ns)', 'CorrId', 'Name'])
            t = 1000
            for i, (name, dur) in enumerate(kernels):
                w.writerow([t, dur, i, name])
                t += dur + 5
        r = subprocess.run([sys.executable, str(HERE / 'nsys_classify.py'), str(path)],
                           capture_output=True, text=True, timeout=10)
    out = json.loads(r.stdout[:r.stdout.rindex('}') + 1]) if r.stdout.strip() else None
    return r.returncode, out, r.stdout


class ClassifyTests(unittest.TestCase):
    def test_synthetic_step(self):
        rc, out, stdout = run(LAYER_GDN + LAYER_ATTN + TAIL)
        self.assertEqual(rc, 0, stdout)
        ns = out['ns']
        self.assertEqual(out['steps'], 1)
        self.assertEqual(ns['gate_up'], 2000 + 2100)
        self.assertEqual(ns['down'], 1500 + 1600)
        self.assertEqual(ns['lm_head'], 4_000_000)
        self.assertEqual(ns['gdn'], 450)
        self.assertEqual(ns['attn'], 535)
        self.assertEqual(ns['mix'], 300 + 250)
        self.assertEqual(ns['o_proj'], 200 + 210)
        self.assertEqual(ns['router'], 40 + 60 + 40 + 60)
        self.assertEqual(ns['shared_gate_up'], 120 + 130)
        self.assertEqual(ns['shared_down'], 90 + 95)
        self.assertEqual(ns['perm'], 30)
        self.assertEqual(ns['other'], 0)
        self.assertEqual(out['total_ns'], sum(d for _, d in LAYER_GDN + LAYER_ATTN + TAIL))
        self.assertIn('h2hNsys = <|"total_ns" -> %d' % out['total_ns'], stdout.splitlines()[-1])

    def test_unknown_kernels_are_inconclusive(self):
        # negative control: same step with most time in kernels the rules do not know
        rc, out, _ = run(LAYER_GDN + [('mystery_kernel_a', 9_000_000), ('mystery_kernel_b', 10)] + TAIL)
        self.assertEqual(rc, 2)
        self.assertEqual(out['verdict'], 'INCONCLUSIVE')
        self.assertEqual(out['other_top'][0], dict(name='mystery_kernel_a', ns=9_000_000))

    def test_missing_columns_are_unusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'bad.csv')
            path.write_text('Start,Name\n1,x\n')
            r = subprocess.run([sys.executable, str(HERE / 'nsys_classify.py'), str(path)],
                               capture_output=True, text=True, timeout=10)
        self.assertEqual(r.returncode, 1)


if __name__ == '__main__':
    unittest.main()

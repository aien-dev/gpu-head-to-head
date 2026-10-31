#!/usr/bin/env python3
"""PROTOCOL.md A10 (DRA-74): print-launch argv, OURS binary pin, in-window lock, deadline, cell sets."""
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
SOURCE = (HERE / 'run_arm.sh').read_text()
LEGACY_VARGV = ['vllm', 'serve', '/model', '--served-model-name', 'q36', '--host', '127.0.0.1', '--port', '18191',
                '--max-model-len', '49152', '--max-num-seqs', '32', '--gpu-memory-utilization', '0.50',
                '--language-model-only']


def piece(start, end):
    return SOURCE[SOURCE.index(start):SOURCE.index(end)]


def print_launch(arm, **env):
    with tempfile.TemporaryDirectory() as tmp:
        full = dict(os.environ, H2H_PRINT_LAUNCH='1', H2H_RESULTS=tmp + '/results', H2H_BOARD=tmp + '/BOARD')
        full.update(env)
        r = subprocess.run(['bash', str(HERE / 'run_arm.sh'), arm, 'D1'], env=full, capture_output=True,
                           text=True, timeout=30)
        touched = os.listdir(tmp)
    return r, touched


class PrintLaunchTests(unittest.TestCase):
    def test_legacy_vllm_argv_is_unchanged(self):
        r, touched = print_launch('VLLM')
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.splitlines()
        self.assertEqual(lines[0:2], ['-v', '/home/drakestapleton/models/qwen3.6-35b-a3b-fp8-95a723d0:/model:ro'])
        self.assertEqual(lines[3:], LEGACY_VARGV)
        self.assertEqual(touched, [], 'print mode must not create the session')

    def test_comparator_arms_append_their_hooks(self):
        cases = {
            'VLLM_FP8': ['--kv-cache-dtype', 'auto', '--attention-backend', 'FLASH_ATTN', '--moe-backend', 'deep_gemm',
                         '--no-enable-prefix-caching'],
            'VLLM_FP8_MTP': ['--kv-cache-dtype', 'auto', '--attention-backend', 'FLASH_ATTN', '--moe-backend', 'deep_gemm',
                             '--speculative-config', '{"method":"mtp","num_speculative_tokens":3}',
                             '--no-enable-prefix-caching'],
            'VLLM_NVFP4': ['--quantization', 'modelopt', '--kv-cache-dtype', 'fp8', '--attention-backend', 'FLASHINFER',
                           '--moe-backend', 'marlin', '--no-enable-prefix-caching'],
        }
        for arm, tail in cases.items():
            with self.subTest(arm=arm):
                r, _ = print_launch(arm)
                self.assertEqual(r.returncode, 0, r.stderr)
                lines = r.stdout.splitlines()
                self.assertEqual(lines[3:3 + len(LEGACY_VARGV)], LEGACY_VARGV)
                self.assertEqual(lines[3 + len(LEGACY_VARGV):], tail)

    def test_profile_arm_runs_only_its_cell_set(self):
        r, _ = print_launch('VLLM_NVFP4_NSYS', H2H_CELLS='dra74')
        self.assertEqual(r.returncode, 4)
        r, _ = print_launch('VLLM_NVFP4_NSYS', H2H_CELLS='nsys-c8')
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.splitlines()
        self.assertIn('/nsys/bin/nsys', lines)
        self.assertIn('--capture-range=cudaProfilerApi', lines)
        self.assertEqual(lines[-2:], ['--profiler-config', '{"profiler":"cuda"}'])
        r, _ = print_launch('VLLM_FP8', H2H_CELLS='nsys-c8')
        self.assertEqual(r.returncode, 4, 'an unprofiled arm must refuse the profile cell set')

    def test_ours_binary_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'aien-serve-http').write_bytes(b'fixture server\n')
            sha = hashlib.sha256(b'fixture server\n').hexdigest()
            r, _ = print_launch('OURS', OURS_BIN_DIR=tmp, OURS_BIN_SHA256=sha)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.splitlines(), ['AIEN_OMEGA_FP8_GROUPED_WAVE=0', tmp + '/aien-serve-http'])
            r, _ = print_launch('OURS', OURS_BIN_DIR=tmp, OURS_BIN_SHA256='0' * 64)
            self.assertEqual(r.returncode, 4)
            self.assertIn('sha256', r.stderr)
            r, _ = print_launch('OURS', OURS_BIN_DIR=tmp, OURS_BIN_SHA256=sha, OURS_WAVE='2')
            self.assertEqual(r.returncode, 4)

    def test_deadline_and_cell_set_are_checked(self):
        now = int(time.time())
        self.assertEqual(print_launch('VLLM_FP8', H2H_DEADLINE_EPOCH=str(now + 2000))[0].returncode, 0)
        for bad in [str(now + 100), 'soon']:
            with self.subTest(deadline=bad):
                self.assertEqual(print_launch('VLLM_FP8', H2H_DEADLINE_EPOCH=bad)[0].returncode, 4)
        self.assertEqual(print_launch('VLLM_FP8', H2H_CELLS='bogus')[0].returncode, 4)

    def test_client_core_guard(self):
        r, _ = print_launch('VLLM_FP8', H2H_CLIENT_CPUS='5', H2H_CLIENT_NICE='19')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('client_cpus 5 client_nice 19', r.stderr)
        self.assertNotIn(',5,', ',' + r.stderr.split()[1] + ',')
        self.assertEqual(print_launch('VLLM_FP8', H2H_CLIENT_CPUS='12')[0].returncode, 4)


class VllmArgvTests(unittest.TestCase):
    def argv(self, prefix):
        script = ('declare -A ARM_QUANT ARM_KV ARM_ATTN ARM_MOE ARM_SPEC ARM_PREFIX ARM_PROFILE\n'
                  'ARM=VLLM_X VLLM_UTIL=0.50 ARM_PREFIX[VLLM_X]=%s\n' % shlex.quote(prefix)
                  + piece('vllm_argv() {', 'stop_server() {') + '\nvllm_argv && printf "%s\\n" "${VARGV[@]}"\n')
        return subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=5)

    def test_prefix_values(self):
        self.assertEqual(self.argv('on').stdout.splitlines()[-1], '--enable-prefix-caching')
        self.assertEqual(self.argv('').stdout.splitlines(), LEGACY_VARGV)
        bad = self.argv('maybe')
        self.assertEqual(bad.returncode, 1)
        self.assertIn('must be on, off or empty', bad.stderr)


class WindowLockTests(unittest.TestCase):
    def held(self, *, held_id='w1', env_id='w1', gb10='exclusive', lock_held='1'):
        with tempfile.TemporaryDirectory() as tmp:
            sw = Path(tmp, 'spark-window')
            sw.write_text('#!/bin/sh\n[ "$1" = id ] && echo %s\n' % held_id)
            sw.chmod(0o755)
            Path(tmp, 'gb10').write_text(gb10 + '\n')
            env = dict(os.environ, SPARK_WINDOW_BIN=str(sw), SPARK_WINDOW=env_id, SPARK_WINDOW_DIR=tmp,
                       AIEN_GB10_LOCK_HELD=lock_held)
            script = piece('window_held() {', 'arm_preconditions() {') + '\nwindow_held\n'
            return subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True, timeout=15)

    def test_exclusive_window_is_accepted(self):
        self.assertEqual(self.held().returncode, 0)

    def test_negative_controls_refuse(self):
        for kw, why in [(dict(held_id='w2'), 'is not SPARK_WINDOW'), (dict(gb10='shared'), 'exclusively'),
                        (dict(lock_held=''), 'incomplete')]:
            with self.subTest(**kw):
                r = self.held(**kw)
                self.assertEqual(r.returncode, 1)
                self.assertIn(why, r.stdout)

    def test_window_mode_never_waits_on_flock(self):
        block = piece('if [ "${AIEN_GB10_LOCK_HELD:-}" = 1 ] || [ -n "${SPARK_WINDOW:-}" ]; then', 'GPU_LOCK_HELD=1')
        window_branch = block[:block.index('elif')]
        self.assertNotIn('flock', window_branch.replace('no flock taken', ''))
        self.assertIn('GPU_LOCK_RELEASE=0', window_branch)
        self.assertIn('flock -w 900 9', block[block.index('else'):])


class DeadlineAndCellSetTests(unittest.TestCase):
    def epochs(self, deadline):
        block = piece('LOCK_EPOCH=$(date +%s)', 'echo "lock_epoch')
        script = 'date() { echo 1000; }\nDEADLINE=%s\n%s\necho $STOP_AT $WARN_AT $QUIET_AT\n' % (deadline, block)
        return subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=5).stdout.split()

    def test_epochs(self):
        self.assertEqual(self.epochs(''), ['4540', '4300', '4000'])        # no deadline: the 60-min hold
        self.assertEqual(self.epochs('2140'), ['2080', '1960', '1960'])    # slice deadline wins
        self.assertEqual(self.epochs('9999'), ['4540', '4300', '4000'])    # later deadline never extends

    def cells(self, cellset):
        block = piece('  case $CELLSET in\n    full)\n      EXPECTED_CELLS=14', 'else\n  if [ -s "$LOGD/listen.txt" ]')
        with tempfile.TemporaryDirectory() as tmp:
            script = ('LOGD=%s PORT=1 CELLSET=%s CELLS_COMPLETED=0\n'
                      'cell() { echo "$1" >> "$LOGD/cells"; CELLS_COMPLETED=$((CELLS_COMPLETED+1)); }\n'
                      'curl() { echo "$@" >> "$LOGD/curl"; }\nsleep() { :; }\n'
                      'ts() { echo T; }\n%s\necho "$EXPECTED_CELLS"\n') % (shlex.quote(tmp), cellset, block)
            r = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=10)
            cells = Path(tmp, 'cells').read_text().split() if Path(tmp, 'cells').exists() else []
            curls = Path(tmp, 'curl').read_text() if Path(tmp, 'curl').exists() else ''
            return r.stdout.split()[-1], cells, curls

    def test_dra74_set(self):
        expected, cells, _ = self.cells('dra74')
        self.assertEqual(expected, '7')
        self.assertEqual(cells, ['decode-c1', 'decode-c4', 'decode-c8', 'decode-c16',
                                 'prefill-p256', 'prefill-p2k', 'prefill-p8k'])

    def test_full_set_unchanged(self):
        expected, cells, _ = self.cells('full')
        self.assertEqual((expected, len(cells)), ('14', 14))

    def test_nsys_set_brackets_one_cell_with_the_profiler(self):
        expected, cells, curls = self.cells('nsys-c8')
        self.assertEqual((expected, cells), ('1', ['decode-c8']))
        self.assertEqual([l.split()[-1].rsplit('/', 1)[-1] for l in curls.splitlines()], ['start_profile', 'stop_profile'])


if __name__ == '__main__':
    unittest.main()

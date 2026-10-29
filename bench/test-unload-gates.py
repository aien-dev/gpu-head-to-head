#!/usr/bin/env python3
"""Host-only regression tests of the runner's actual EXIT cleanup acceptance."""
import pathlib, subprocess, tempfile, unittest, shlex
HERE=pathlib.Path(__file__).resolve().parent
class UnloadTests(unittest.TestCase):
    def run_cleanup(self, baseline='5000000', after='5000000', gpu='', query_exit=0):
        source=(HERE/'run_arm.sh').read_text()
        cleanup=source[source.index('cleanup_session() {'):source.index('# ---- session ----')]
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp); (root/'DONE').touch()
            script='''set -u
LOGD=%s
BASE=%s AFTER_FIXTURE=%s GPU_FIXTURE=%s QUERY_EXIT=%s
WD= CPID= CMON_PID= SAMP= VMS= GPU_LOCK_HELD=1 GPU_LOCK_RELEASE=0 ARM=fixture ROUND=R1
stop_server() { return 0; }
timeout() { shift; "$@"; }
nvidia-smi() { printf '%%s' "$GPU_FIXTURE"; return "$QUERY_EXIT"; }
memavail() { printf '%%s' "$AFTER_FIXTURE"; }
ktpsr() { echo 0; }
board() { :; }
''' % tuple(map(shlex.quote,[tmp,baseline,after,gpu,str(query_exit)]))
            proc=subprocess.run(['bash','-c',script+cleanup+'\ntrap cleanup_session EXIT\nexit 0\n'],capture_output=True,text=True,timeout=3)
            return proc.returncode,(root/'DONE').exists(),(root/'INCOMPLETE').exists(),proc.stderr
    def assert_rejected(self, **kwargs):
        rc,done,incomplete,err=self.run_cleanup(**kwargs)
        self.assertNotEqual(rc,0,err); self.assertFalse(done); self.assertTrue(incomplete)
    def test_recovered_memory_and_empty_gpu_pass(self):
        self.assertEqual(self.run_cleanup()[:3],(0,True,False))
    def test_boundary_two_gib_deficit_passes(self):
        self.assertEqual(self.run_cleanup(after=str(5000000-2*1024*1024))[:3],(0,True,False))
    def test_one_kib_beyond_boundary_fails(self):
        self.assert_rejected(after=str(5000000-2*1024*1024-1))
    def test_more_available_memory_passes(self):
        self.assertEqual(self.run_cleanup(after='6000000')[:3],(0,True,False))
    def test_missing_memory_fails(self):
        self.assert_rejected(after='')
    def test_invalid_memory_fails(self):
        for value in ['unknown','-1','0','999999999999999999999999','1\n2']:
            with self.subTest(value=value): self.assert_rejected(after=value)
    def test_missing_baseline_fails(self):
        self.assert_rejected(baseline='')
    def test_remaining_gpu_process_fails(self):
        self.assert_rejected(gpu='12345\n')
    def test_inventory_failure_fails(self):
        self.assert_rejected(query_exit=1)
if __name__=='__main__': unittest.main()

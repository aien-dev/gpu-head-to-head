#!/usr/bin/env python3
"""Host-only sequencing, recovery and real-child controls; no live shared service access."""
import contextlib,importlib.util,pathlib,tempfile,unittest,types,os,fcntl,subprocess,threading
from unittest.mock import patch
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('window',HERE/'shared-window.py')
window=importlib.util.module_from_spec(spec);spec.loader.exec_module(window)
class Fake:
    def __init__(self,fail=None,code=0): self.fail=fail;self.code=code;self.events=[];self.held=False
    def step(self,name):
        if not self.held: raise AssertionError('operation without lock')
        self.events.append(name)
        if self.fail==name: raise RuntimeError('fixture '+name)
    @contextlib.contextmanager
    def lock(self):
        self.held=True;self.events.append('lock')
        try: yield
        finally:self.held=False;self.events.append('unlock')
    def preflight(self):self.step('preflight')
    def stop(self):self.step('stop')
    def wait_original_exit(self):self.step('wait')
    def before_test(self):self.step('before')
    def run(self):self.step('run');return self.code
    def recovery_gate(self):self.step('recovery')
    def restore(self):self.step('restore')
    def verify_restored(self):self.step('verify')
class Sequencing(unittest.TestCase):
    def test_success_holds_lock_through_verified_restore(self):
        f=Fake();r=window.execute(f)
        self.assertEqual(r['status'],'PASS')
        self.assertEqual(f.events,['lock','preflight','stop','wait','before','run','recovery','restore','verify','unlock'])
    def test_busy_preflight_never_stops_or_restores(self):
        f=Fake('preflight')
        with self.assertRaises(RuntimeError):window.execute(f)
        self.assertEqual(f.events,['lock','preflight','unlock'])
    def test_failed_test_still_restores_and_fails(self):
        f=Fake(code=7);r=window.execute(f)
        self.assertEqual(r['test_exit'],7);self.assertTrue(r['restored']);self.assertEqual(r['status'],'INCOMPLETE')
    def test_exception_after_stop_always_attempts_gated_recovery(self):
        for step in ['stop','wait','before','run']:
            with self.subTest(step=step):
                f=Fake(step);r=window.execute(f)
                self.assertIn('recovery',f.events);self.assertTrue(r['restored']);self.assertEqual(r['status'],'INCOMPLETE')
    def test_failed_recovery_gate_never_launches_original(self):
        f=Fake('recovery');r=window.execute(f)
        self.assertNotIn('restore',f.events);self.assertFalse(r['restored'])
    def test_failed_launcher_never_claims_restored(self):
        f=Fake('restore');r=window.execute(f)
        self.assertNotIn('verify',f.events);self.assertFalse(r['restored'])
    def test_failed_identity_never_claims_restored(self):
        f=Fake('verify');r=window.execute(f)
        self.assertFalse(r['restored']);self.assertIn('recovery_error',r)
class LiveAdapter(unittest.TestCase):
    def bare(self,root):
        obj=object.__new__(window.Live)
        obj.a=types.SimpleNamespace(out=root,repo=root,arm='OURS',round='R1')
        obj.child=None;obj.interrupted=False;obj.baseline=5000000
        return obj
    def test_real_child_borrows_lock_and_failed_exit_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=pathlib.Path(temp);(root/'bench').mkdir()
            (root/'bench/run_arm.sh').write_text('test -e /proc/$$/fd/9 || exit 22\nflock -n 9 || exit 23\necho "$H2H_GPU_LOCK_FD"\nexit 7\n')
            obj=self.bare(root)
            fd=os.open(root/'lock',os.O_CREAT|os.O_RDWR,0o600)
            saved=None
            try:
                try:saved=os.dup(9)
                except OSError:pass
                os.dup2(fd,9);fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertEqual(obj.run(),7)
                probe=subprocess.run(['flock','-n',str(root/'lock'),'true'])
                self.assertNotEqual(probe.returncode,0,'child released parent lock')
                self.assertEqual((root/'runner.log').read_text().strip(),'9')
            finally:
                os.close(9)
                if saved is not None:os.dup2(saved,9);os.close(saved)
                if fd!=9:os.close(fd)
    def test_interruption_before_launch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=self.bare(pathlib.Path(tmp));obj.interrupted=True
            with self.assertRaises(window.guard.Refused):obj.run()
            self.assertIsNone(obj.child)
    def test_success_without_done_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=self.bare(pathlib.Path(tmp))
            child=types.SimpleNamespace(poll=lambda:0,returncode=0)
            with patch.object(window.subprocess,'Popen',return_value=child):
                with self.assertRaisesRegex(window.guard.Refused,'completed-session marker'):obj.run()
    def test_interruption_stops_and_reaps_real_fixture_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);(root/'bench').mkdir()
            (root/'bench/run_arm.sh').write_text('exec sleep 30\n')
            obj=self.bare(root);fd=os.open(root/'lock',os.O_CREAT|os.O_RDWR,0o600);saved=None
            timer=threading.Timer(0.1,lambda:setattr(obj,'interrupted',True))
            try:
                try:saved=os.dup(9)
                except OSError:pass
                os.dup2(fd,9);timer.start()
                with self.assertRaisesRegex(window.guard.Refused,'interrupted'):obj.run()
                self.assertIsNotNone(obj.child.returncode)
                with self.assertRaises(ChildProcessError):os.waitpid(obj.child.pid,os.WNOHANG)
            finally:
                timer.cancel();timer.join()
                if obj.child is not None and obj.child.poll() is None:obj.child.kill();obj.child.wait()
                os.close(9)
                if saved is not None:os.dup2(saved,9);os.close(saved)
                if fd!=9:os.close(fd)
    def test_active_runner_refuses_restoration_without_gpu_query(self):
        obj=self.bare(pathlib.Path('/unused'));obj.child=types.SimpleNamespace(poll=lambda:None)
        with patch.object(obj,'gpu_empty') as gpu:
            with self.assertRaises(window.guard.Refused):obj.recovery_gate()
            gpu.assert_not_called()
    def test_original_still_alive_refuses_duplicate(self):
        obj=self.bare(pathlib.Path('/unused'));obj.original_fd=42
        with patch.object(window.select,'select',return_value=([],[],[])),patch.object(obj,'gpu_empty') as gpu:
            with self.assertRaises(window.guard.Refused):obj.recovery_gate()
            gpu.assert_not_called()
    def test_memory_deficit_refuses_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj=self.bare(pathlib.Path(tmp));obj.original_fd=42
            with patch.object(window.select,'select',return_value=([42],[],[])),patch.object(obj,'gpu_empty'),patch.object(obj,'memory',return_value=1):
                with self.assertRaises(window.guard.Refused):obj.recovery_gate()
    def test_residual_gpu_refuses_restore(self):
        obj=self.bare(pathlib.Path('/unused'));obj.original_fd=42
        with patch.object(window.select,'select',return_value=([42],[],[])),patch.object(obj,'gpu_empty',side_effect=window.guard.Refused('foreign GPU')),patch.object(obj,'memory') as memory:
            with self.assertRaises(window.guard.Refused):obj.recovery_gate()
            memory.assert_not_called()
    def test_restore_launcher_does_not_inherit_gpu_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);obj=self.bare(root)
            obj.a.launcher=root/'restore.sh'
            obj.a.launcher.write_text('#!/bin/bash\n[ ! -e /proc/$$/fd/9 ] || exit 21\n[ "$Q36_BATCH_BIN" = bin-fixture ] || exit 22\n[ "$Q36_BATCH_SLOTS" = 8 ] || exit 23\n[ "$Q36_MAX_CONTEXT" = 49152 ] || exit 24\necho RESTORED_FIXTURE\n')
            obj.a.launcher.chmod(0o700)
            obj.snapshot={'cwd':str(root/'bin-fixture'),'env':{'AIEN_Q36_STROKE':'1'}}
            fd=os.open(root/'lock',os.O_CREAT|os.O_RDWR,0o600);saved=None
            try:
                try:saved=os.dup(9)
                except OSError:pass
                os.dup2(fd,9);fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch.object(obj,'validate_pins') as pins:
                    obj.restore();pins.assert_called_once_with(restoration=True)
                self.assertEqual((root/'restore.log').read_text().strip(),'RESTORED_FIXTURE')
                self.assertNotEqual(subprocess.run(['flock','-n',str(root/'lock'),'true']).returncode,0)
            finally:
                os.close(9)
                if saved is not None:os.dup2(saved,9);os.close(saved)
                if fd!=9:os.close(fd)
    def test_changed_snapshot_contract_refused_before_stop(self):
        obj=self.bare(pathlib.Path('/unused'));obj.snapshot={'argv':['different-model']}
        with self.assertRaises(window.guard.Refused):obj.validate_restore_contract()
    def test_gpu_query_error_is_not_empty_inventory(self):
        obj=self.bare(pathlib.Path('/unused'))
        with patch.object(window.subprocess,'check_output',side_effect=subprocess.CalledProcessError(1,'fixture')):
            with self.assertRaises(subprocess.CalledProcessError):obj.gpu_empty()
if __name__=='__main__':unittest.main()

#!/usr/bin/env python3
"""External guard controls. No shared port, GPU query or shared model is touched."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('shared_guard',HERE/'shared-model-guard.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
HEALTH=dict(status='ok',model='qwen3.6-35b-a3b-fp8',engine='omega-qwen36-batch',
            concurrency=8,busy=False,in_flight=0)

class ObservationTests(unittest.TestCase):
    def test_expected_idle_observation_passes(self):
        guard.check_observation(42,HEALTH,'',[42],'reviewed','reviewed')
    def test_active_or_changed_health_refused(self):
        for key,value in [('busy',True),('in_flight',1),('in_flight',False),('concurrency',1),
                          ('model','other'),('engine','other'),('status','loading')]:
            health=dict(HEALTH);health[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(guard.Refused):
                guard.check_observation(42,health,'',[42],'reviewed','reviewed')
    def test_established_client_refused(self):
        with self.assertRaises(guard.Refused):guard.check_observation(42,HEALTH,'fixture client',[42],'reviewed','reviewed')
    def test_foreign_missing_or_duplicate_gpu_pid_refused(self):
        for pids in [[],[41],[42,43],[42,42],['42'],[True]]:
            with self.subTest(pids=pids),self.assertRaises(guard.Refused):
                guard.check_observation(42,HEALTH,'',pids,'reviewed','reviewed')
    def test_board_change_refused(self):
        with self.assertRaises(guard.Refused):guard.check_observation(42,HEALTH,'',[42],'new','reviewed')

class IdentityTests(unittest.TestCase):
    def fixture(self,root):
        binary=root/'fixture-executable';binary.write_bytes(b'harmless identity fixture')
        proc=root/'proc/42';proc.mkdir(parents=True)
        (proc/'exe').symlink_to(binary);(proc/'cwd').symlink_to(root)
        argv=['./fixture','--engine','omega'];(proc/'cmdline').write_bytes(b'\0'.join(x.encode() for x in argv)+b'\0')
        (proc/'environ').write_bytes(b'AIEN_Q36_STROKE=1\0UNRELATED=not-recorded\0')
        fields=['S','1']+['0']*17+['12345']+['0']*5
        (proc/'stat').write_text('42 (fixture name) '+ ' '.join(fields))
        snapshot=dict(exe=str(binary),cwd=str(root),argv=argv,env={'AIEN_Q36_STROKE':'1'})
        return snapshot,hashlib.sha256(binary.read_bytes()).hexdigest(),proc
    def test_saved_identity_matches_without_recording_unrelated_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);snapshot,digest,proc=self.fixture(root)
            actual=guard.process_identity(42,snapshot,digest,root/'proc')
            self.assertEqual(actual['starttime_ticks'],'12345')
            self.assertNotIn('not-recorded',json.dumps(actual))
    def test_configuration_drift_refused(self):
        for key,value in [('argv',['changed']),('cwd','/different'),('exe','/different')]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);snapshot,digest,proc=self.fixture(root);snapshot[key]=value
                with self.assertRaises(guard.Refused):guard.process_identity(42,snapshot,digest,root/'proc')
    def test_binary_and_required_environment_changes_refused(self):
        for kind in ['binary','environment']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);snapshot,digest,proc=self.fixture(root)
                if kind=='binary':Path(snapshot['exe']).write_bytes(b'changed')
                else:(proc/'environ').write_bytes(b'AIEN_Q36_STROKE=0\0')
                with self.assertRaises(guard.Refused):guard.process_identity(42,snapshot,digest,root/'proc')

class SequencingTests(unittest.TestCase):
    def run_fake(self,stop=False,busy=False,changed=False,exited=False):
        identity=dict(pid=42,starttime_ticks='1',exe_sha256='fixture')
        observations=[dict(HEALTH)]*4
        with patch.object(guard.os,'pidfd_open',return_value=999),patch.object(guard.os,'close') as close,\
             patch.object(guard,'process_identity',side_effect=[identity,dict(identity,starttime_ticks='2') if changed else identity]),\
             patch.object(guard,'observe',side_effect=guard.Refused('busy') if busy else observations) as observe,\
             patch.object(guard.time,'sleep'),patch.object(guard.select,'select',return_value=([999] if exited else [],[],[])),\
             patch.object(guard,'send_term') as send:
            try:
                result=guard.guard(42,{},'fixture',Path('/unused'),'reviewed',stop)
                return result,send.call_count,observe.call_count
            finally:
                self.assertEqual(close.call_args_list,[( (999,), {} )])
                if busy or changed or exited:self.assertEqual(send.call_count,0)
    def test_check_only_never_signals(self):
        result,sent,observed=self.run_fake()
        self.assertFalse(result['stop_sent']);self.assertEqual(sent,0);self.assertEqual(observed,4)
    def test_stop_requires_all_samples_and_final_identity_check(self):
        result,sent,observed=self.run_fake(stop=True)
        self.assertTrue(result['stop_sent']);self.assertEqual(sent,1);self.assertEqual(observed,4)
    def test_busy_identity_drift_or_exited_handle_never_signals(self):
        for options in [dict(busy=True),dict(changed=True),dict(exited=True)]:
            with self.subTest(options=options),self.assertRaises(guard.Refused):self.run_fake(stop=True,**options)

class ReceiptTests(unittest.TestCase):
    def test_cli_refuses_existing_receipt_before_any_process_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'existing.json';output.write_text('preserve evidence')
            result=subprocess.run([sys.executable,str(HERE/'shared-model-guard.py'),
                '--pid','0','--snapshot',str(Path(tmp)/'absent.json'),
                '--binary-sha256','invalid','--reviewed-board-sha256','invalid',
                '--stop','--out',str(output)],capture_output=True,text=True,timeout=5)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(output.read_text(),'preserve evidence')
            self.assertIn('FileExistsError',result.stderr)

class PidfdTests(unittest.TestCase):
    def test_bound_signal_stops_only_fixture_child(self):
        child=subprocess.Popen(['sleep','60']);decoy=subprocess.Popen(['sleep','60'])
        fd=os.pidfd_open(child.pid)
        try:
            guard.send_term(fd);self.assertEqual(child.wait(timeout=2),-signal.SIGTERM)
            self.assertIsNone(decoy.poll())
        finally:
            os.close(fd)
            for process in [child,decoy]:
                if process.poll() is None:process.terminate()
                process.wait(timeout=2)
    def test_exited_bound_handle_refused(self):
        child=subprocess.Popen(['sleep','60']);fd=os.pidfd_open(child.pid)
        try:
            child.terminate();child.wait(timeout=2)
            with self.assertRaises(guard.Refused):guard.send_term(fd)
        finally:
            os.close(fd)
            if child.poll() is None:child.terminate();child.wait(timeout=2)

if __name__=='__main__':unittest.main()

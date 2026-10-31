#!/usr/bin/env python3
"""External test controller. Requires reviewed BOARD and pinned inputs; never swaps models."""
import argparse
import contextlib
import datetime
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time
import urllib.request

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('shared_guard',HERE/'shared-model-guard.py')
guard=importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)


def execute(ops):
    """Keep the lock through all recovery. Refusal before stop never launches restoration."""
    result={'test_exit':None,'restored':False,'status':'INCOMPLETE'}
    with ops.lock():
        ops.preflight()
        stop_attempted=False
        try:
            stop_attempted=True
            ops.stop()
            ops.wait_original_exit()
            ops.before_test()
            result['test_exit']=ops.run()
        except Exception as error:
            result['error']=str(error)
        finally:
            if stop_attempted:
                try:
                    ops.recovery_gate()
                    ops.restore()
                    ops.verify_restored()
                    result['restored']=True
                except Exception as error:
                    result['recovery_error']=str(error)
        if result['test_exit']==0 and result['restored'] and 'error' not in result:
            result['status']='PASS'
    return result


class Live:
    def __init__(self,args):
        self.a=args; self.original_fd=None; self.child=None; self.interrupted=False
        self.baseline=None; self.snapshot=json.loads(args.snapshot.read_text())
        self.pins=json.loads(args.pins.read_text()); self.original=None
    def event(self,kind,**fields):
        with (self.a.out/'events.jsonl').open('a') as out:
            out.write(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  'event':kind,**fields})+'\n')
    @contextlib.contextmanager
    def lock(self):
        fd=os.open('/tmp/aien-gb10.lock',os.O_WRONLY|os.O_CREAT,0o600)
        try:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            # Runner contract requires descriptor 9. Never overwrite an unknown descriptor.
            if fd!=9:
                try: os.fstat(9)
                except OSError: pass
                else: raise guard.Refused('descriptor 9 already in use')
                os.dup2(fd,9); os.close(fd); fd=9
            self.event('lock_acquired')
            yield
        finally:
            if self.original_fd is not None: os.close(self.original_fd)
            os.close(fd)
    def validate_pins(self,restoration=False):
        required=[self.a.snapshot,self.a.launcher,Path(self.snapshot['exe']),
                  HERE/'shared-model-guard.py',Path(__file__).resolve()]
        if not restoration:
            required += [self.a.repo/p for p in ['bench/run_arm.sh','bench/session-lifecycle.sh',
            'bench/bench.py','bench/quiet.py','bench/coremon.py','prompts/decode.jsonl','prompts/prefill.jsonl']]
        conf=self.a.repo/'bench/arms.conf'
        if not restoration and conf.exists(): required.append(conf)
        for path in required:
            if str(path.resolve()) not in self.pins: raise guard.Refused('missing pin: '+str(path))
        checks={str(p.resolve()):self.pins[str(p.resolve())] for p in required} if restoration else self.pins
        for name,expected in checks.items():
            if guard.digest(name)!=expected: raise guard.Refused('pinned file changed: '+name)
    def validate_restore_contract(self):
        expected=['./aien-serve-http','--engine','omega','--model-dir',
            str(Path.home()/'models/qwen3.6-35b-a3b-fp8-95a723d0'),'--port','18091',
            '--max-context','49152','--max-tokens-cap','16384','--thinking','off','--batch-slots','8']
        if self.snapshot['argv']!=expected:
            raise guard.Refused('snapshot differs from supported same-model restoration contract')
        cwd=Path(self.snapshot['cwd'])
        if cwd.parent!=self.a.launcher.parent or str(cwd/'aien-serve-http')!=self.snapshot['exe']:
            raise guard.Refused('snapshot executable/working directory does not match launcher layout')
        expected_env={'AIEN_Q36_STROKE':'1','AIEN_SERVE_GPU_LOCK':'/tmp/aien-serve-18091.lock'}
        if self.snapshot['env']!=expected_env:
            raise guard.Refused('snapshot environment differs from supported restoration contract')
    def gpu_empty(self):
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
            '--format=csv,noheader,nounits'],text=True,timeout=5)
        if rows.strip(): raise guard.Refused('GPU processes remain; refusing overlapping model launch')
    def memory(self):
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                parts=line.split()
                if len(parts)==3 and parts[2]=='kB' and parts[1].isdigit() and int(parts[1])>0:
                    return int(parts[1])
        raise guard.Refused('valid MemAvailable unavailable')
    def preflight(self):
        self.validate_pins()
        self.validate_restore_contract()
        self.sha=self.pins[str(Path(self.snapshot['exe']).resolve())]
        self.original=int(self.a.pidfile.read_text().strip())
        self.original_fd=os.pidfd_open(self.original)
        self.identity=guard.process_identity(self.original,self.snapshot,self.sha)
        # Review is a coordinator action, not something inferred from file age or locks.
        old=self.a.board.read_bytes()
        import hashlib
        if hashlib.sha256(old).hexdigest()!=self.a.reviewed_board_sha256:
            raise guard.Refused('BOARD changed since coordinator review')
        entry=('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+
            ' CodexStackCompare EXCLUSIVE35B '+self.a.arm+' '+self.a.round+
            ': Drake said "you may stop the model" and "Only you can\'t stop it if it\'s doing a test." '+
            'Coordinator reviewed BOARD; idle/identity checks required before stop. '+
            'One session capped at60min plus restoration. UNDO: same saved executable/model/flags via '+
            str(self.a.launcher)+' with explicit saved batch settings. Evidence '+str(self.a.out)+'\n').encode()
        with self.a.board.open('ab') as out: out.write(entry)
        reviewed=hashlib.sha256(old+entry).hexdigest()
        observation=guard.guard(self.original,self.snapshot,self.sha,self.a.board,reviewed,False)
        if observation['identity']!=self.identity: raise guard.Refused('original process identity changed')
        self.reviewed=reviewed
        self.event('preflight',observation=observation)
    def stop(self):
        if self.interrupted: raise guard.Refused('interrupted before stopping shared model')
        guard.observe(self.original,self.a.board,self.reviewed)
        guard.send_term(self.original_fd)
        self.event('shared_stop_sent')
    def wait_original_exit(self):
        if not select.select([self.original_fd],[],[],60)[0]:
            raise guard.Refused('original model did not exit; no force kill')
        self.event('shared_exited')
    def before_test(self):
        self.gpu_empty(); self.baseline=self.memory(); self.validate_pins()
        self.event('baseline',memavailable_kib=self.baseline)
    def run(self):
        if self.interrupted: raise guard.Refused('interrupted before test launch')
        env=os.environ.copy()
        env.update(H2H_GPU_LOCK_FD='9',H2H_REPO=str(self.a.repo),H2H_RESULTS=str(self.a.out/'results'))
        with (self.a.out/'runner.log').open('xb') as log:
            self.child=subprocess.Popen(['bash',str(self.a.repo/'bench/run_arm.sh'),self.a.arm,self.a.round],
                env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(9,))
            # The inner runner owns descendants, containers and its 60-minute watchdog.
            # Its pre-lock quiet wait can be longer; bound the complete invocation as well.
            deadline=time.monotonic()+4500
            while self.child.poll() is None:
                if self.interrupted or time.monotonic()>deadline:
                    self.child.terminate()
                    try: self.child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        raise guard.Refused('runner did not finish cleanup; retain its inherited lock, no restore')
                    raise guard.Refused('test interrupted or outer deadline reached')
                time.sleep(0.25)
            self.event('test_exit',code=self.child.returncode)
            done=self.a.out/'results/logs'/('%s-%s' % (self.a.arm,self.a.round))/'DONE'
            if self.child.returncode==0 and not done.is_file():
                raise guard.Refused('runner returned success without a completed-session marker')
            return self.child.returncode
    def recovery_gate(self):
        if self.child is not None and self.child.poll() is None:
            raise guard.Refused('runner still active; restoration refused')
        if not select.select([self.original_fd],[],[],0)[0]:
            raise guard.Refused('original still present; duplicate restoration refused')
        self.gpu_empty()
        after=self.memory()
        self.event('unloaded',memavailable_kib=after,baseline_kib=self.baseline)
        if self.baseline is not None and self.baseline-after>2*1024*1024:
            raise guard.Refused('memory has not recovered within 2 GiB; restoration refused')
    def restore(self):
        # An unrelated benchmark-input change must not prevent restoring the saved service.
        self.validate_pins(restoration=True)
        env=os.environ.copy()
        env.update(self.snapshot['env'])
        env.update(Q36_BATCH_BIN=Path(self.snapshot['cwd']).name,Q36_BATCH_SLOTS='8',Q36_MAX_CONTEXT='49152')
        with (self.a.out/'restore.log').open('xb') as log:
            # close_fds prevents the restored service from retaining the global GPU lock.
            subprocess.run([str(self.a.launcher)],env=env,stdout=log,stderr=subprocess.STDOUT,
                           close_fds=True,timeout=30,check=True)
        self.event('restore_launched')
    def verify_restored(self):
        deadline=time.monotonic()+180
        last='no health response'
        while time.monotonic()<deadline:
            try:
                pid=int(self.a.pidfile.read_text().strip())
                identity=guard.process_identity(pid,self.snapshot,self.sha)
                with urllib.request.urlopen('http://127.0.0.1:18091/health',timeout=3) as response:
                    health=json.load(response)
                expected={'status':'ok','model':'qwen3.6-35b-a3b-fp8','engine':'omega-qwen36-batch','concurrency':8}
                if any(type(health.get(k)) is not type(v) or health.get(k)!=v for k,v in expected.items()):
                    raise guard.Refused('restored health configuration mismatch')
                self.event('restored_verified',identity=identity,health=health)
                with self.a.board.open('a') as out:
                    out.write('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat()+
                        ' CodexStackCompare EXCLUSIVE35B ended: saved shared-model identity and health verified; '+
                        'test verdict retained in '+str(self.a.out)+'\n')
                return
            except (OSError,ValueError,guard.Refused) as error: last=str(error)
            time.sleep(1)
        raise guard.Refused('restoration not verified: '+last)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['repo','snapshot','pins','launcher','pidfile','board','out']:
        p.add_argument('--'+name,required=True,type=lambda s:Path(s).resolve())
    p.add_argument('--reviewed-board-sha256',required=True)
    p.add_argument('--arm',required=True,choices=['OURS','VLLM','LLAMA','LLAMA16','SGLANG','VLLM4','SGLANG4','LLAMA4'])
    p.add_argument('--round',required=True,choices=['R1','R2'])
    args=p.parse_args(); args.out.mkdir(parents=False,exist_ok=False)
    try:
        ops=Live(args)
        def interrupted(signum,frame): ops.interrupted=True
        signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
        result=execute(ops)
    except Exception as error:
        result={'status':'REFUSED','error':str(error),'restored':False}
    (args.out/'window-result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    return 0 if result['status']=='PASS' else 1
if __name__=='__main__': raise SystemExit(main())

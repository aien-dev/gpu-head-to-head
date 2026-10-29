#!/usr/bin/env python3
"""External benchmark guard. Default checks only; --stop requires all idle/identity gates.

This is test equipment, not part of Aien architecture or model-serving dependencies.
Never invoke against the shared service during an active test. The enclosing controller must
record its intent/undo on BOARD before calling, retain the GPU lock, and restore the same setup.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time
import urllib.request


class Refused(RuntimeError):
    pass


def digest(path):
    with open(path,'rb') as source:
        return hashlib.file_digest(source,'sha256').hexdigest()


def process_identity(pid, snapshot, expected_sha256, proc_root=Path('/proc')):
    root=proc_root/str(pid)
    actual={'exe':os.readlink(root/'exe'),'cwd':os.readlink(root/'cwd'),
            'argv':(root/'cmdline').read_bytes().decode().rstrip('\0').split('\0')}
    for key in ['exe','cwd','argv']:
        if actual[key]!=snapshot[key]:
            raise Refused('shared model '+key+' changed')
    if digest(root/'exe')!=expected_sha256 or digest(snapshot['exe'])!=expected_sha256:
        raise Refused('shared model executable digest changed')
    env={}
    for entry in (root/'environ').read_bytes().split(b'\0'):
        if b'=' in entry:
            key,value=entry.split(b'=',1)
            env[key.decode(errors='replace')]=value.decode(errors='replace')
    for key,value in snapshot.get('env',{}).items():
        if env.get(key)!=value:
            raise Refused('shared model required environment changed: '+key)
    fields=(root/'stat').read_text().rsplit(') ',1)[1].split()
    if fields[0] in ['Z','X']:
        raise Refused('shared model already exited')
    return {'pid':pid,'starttime_ticks':fields[19],'exe_sha256':expected_sha256,
            'configuration':'matches saved executable/cwd/argv/required environment'}


def check_observation(pid, health, connections, gpu_pids, board_sha256, reviewed_board_sha256):
    if board_sha256!=reviewed_board_sha256:
        raise Refused('shared board changed; review current activity before retrying')
    expected={'status':'ok','model':'qwen3.6-35b-a3b-fp8','engine':'omega-qwen36-batch',
              'concurrency':8,'busy':False,'in_flight':0}
    for key,value in expected.items():
        if type(health.get(key)) is not type(value) or health.get(key)!=value:
            raise Refused('shared model is not the expected healthy idle configuration: '+key)
    if connections.strip():
        raise Refused('shared model has established connections')
    if type(gpu_pids) is not list or any(type(p) is not int or p<1 for p in gpu_pids) or gpu_pids!=[pid]:
        raise Refused('GPU inventory is not solely the expected shared model')


def observe(pid, board, reviewed_sha256):
    request=urllib.request.Request('http://127.0.0.1:18091/health',headers={'Connection':'close'})
    with urllib.request.urlopen(request,timeout=3) as response:
        health=json.load(response)
    connections=subprocess.check_output(['ss','-tnH','state','established',
        '( sport = :18091 or dport = :18091 )'],text=True,timeout=5)
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
        '--format=csv,noheader,nounits'],text=True,timeout=5)
    gpu_pids=[int(line.strip()) for line in raw.splitlines() if line.strip()]
    check_observation(pid,health,connections,gpu_pids,digest(board),reviewed_sha256)
    return health


def send_term(pidfd):
    if select.select([pidfd],[],[],0)[0]:
        raise Refused('bound process exited before stop; no signal sent')
    signal.pidfd_send_signal(pidfd,signal.SIGTERM)


def guard(pid, snapshot, expected_sha256, board, reviewed_sha256, stop=False):
    # Open first, then inspect. A numeric PID reused after this point cannot redirect this signal.
    fd=os.pidfd_open(pid)
    try:
        identity=process_identity(pid,snapshot,expected_sha256)
        samples=[]
        for _ in range(3):
            samples.append(observe(pid,board,reviewed_sha256))
            time.sleep(1)
        if process_identity(pid,snapshot,expected_sha256)!=identity:
            raise Refused('process identity changed during idle checks')
        samples.append(observe(pid,board,reviewed_sha256))
        if select.select([fd],[],[],0)[0]:
            raise Refused('bound process exited during checks')
        result={'identity':identity,'idle_samples':len(samples),'stop_sent':False,
                'reviewed_board_sha256':reviewed_sha256,'scope':'shared-model preflight only'}
        if stop:
            send_term(fd)
            result['stop_sent']=True
        return result
    finally:
        os.close(fd)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid',required=True,type=int)
    parser.add_argument('--snapshot',required=True,type=Path)
    parser.add_argument('--binary-sha256',required=True)
    parser.add_argument('--board',type=Path,default=Path('/home/drakestapleton/handoffs/BOARD.md'))
    parser.add_argument('--reviewed-board-sha256',required=True)
    parser.add_argument('--stop',action='store_true')
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    # Refuse a reused receipt before touching a process.
    with args.out.open('x') as output:
        try:
            result=guard(args.pid,json.loads(args.snapshot.read_text()),args.binary_sha256,
                         args.board,args.reviewed_board_sha256,args.stop)
            status=0
        except (Refused,OSError,ValueError,TypeError,KeyError,IndexError,AttributeError,subprocess.SubprocessError) as error:
            result={'status':'REFUSED','reason':str(error),'stop_sent':False}
            status=75
        json.dump(result,output,indent=2)
        output.write('\n')
    return status


if __name__=='__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""CPU-only ownership and interruption controls for external benchmark lifecycle."""
from pathlib import Path
import shlex
import os
import signal
import time
import subprocess
import tempfile
import unittest
HERE=Path(__file__).resolve().parent

class ContainerOwnershipTests(unittest.TestCase):
    def test_foreign_container_is_never_stopped(self):
        source=(HERE/'run_arm.sh').read_text()
        function=source[source.index('stop_server() {'):source.index('ready() {')]
        with tempfile.TemporaryDirectory() as tmp:
            script='''set -u
ARM=VLLM CTR=fixture-unowned SPID=999999999 PAT=fixture-no-match
RUN_ID=our-session LOGD=%s
board() { :; }
docker() { case "$1" in inspect) echo someone-else ;; stop) echo stopped > "$LOGD/foreign-stopped" ;; esac; }
kill() { return 1; }
sleep() { :; }
pgrep() { return 1; }
''' % shlex.quote(tmp)
            result=subprocess.run(['bash','-c',script+function+'\nstop_server\n'],capture_output=True,text=True,timeout=5)
            self.assertFalse((Path(tmp)/'foreign-stopped').exists(),result.stderr)
            self.assertNotEqual(result.returncode,0)

    def test_owned_container_launch_records_label_and_shutdown(self):
        source=(HERE/'run_arm.sh').read_text()
        functions=source[source.index('start_server() {'):source.index('ready() {')]
        with tempfile.TemporaryDirectory() as tmp:
            script="""set -u
source %s
LOGD=%s ARM=VLLM CTR=stack-fixture RUN_ID=fixture SPID=
MDIR=/fixture VLLM_IMG=fixture-image SRV_CPUS=0-1 VLLM_UTIL=0.50
trap 'for p in $(jobs -pr); do kill -TERM "$p" 2>/dev/null || true; done; wait 2>/dev/null || true' EXIT
timeout() { shift; "$@"; }
docker() {
  case "$1" in
    create) printf '%%s\n' "$@" > "$LOGD/create-args"; echo fixture-id ;;
    start) echo started > "$LOGD/start-called"; exec sleep 60 ;;
    inspect) echo fixture ;;
    stop) echo stopped > "$LOGD/stop-called" ;;
  esac
}
""" % (shlex.quote(str(HERE/'session-lifecycle.sh')),shlex.quote(tmp))
            result=subprocess.run(['bash','-c',script+functions+'\nstart_server && stop_server\n'],
                                    capture_output=True,text=True,timeout=5)
            self.assertEqual(result.returncode,0,result.stderr)
            args=(Path(tmp)/'create-args').read_text().splitlines()
            self.assertIn('aien.stack-comparison.run=fixture',args)
            self.assertIn('never',args)
            self.assertTrue((Path(tmp)/'server-identity.txt').exists())
            self.assertTrue((Path(tmp)/'stop-called').exists())

    def test_failed_container_creation_never_starts_server(self):
        source=(HERE/'run_arm.sh').read_text()
        function=source[source.index('start_server() {'):source.index('stop_server() {')]
        with tempfile.TemporaryDirectory() as tmp:
            script="""set -u
source %s
LOGD=%s ARM=VLLM CTR=stack-fixture RUN_ID=fixture SPID=
MDIR=/fixture VLLM_IMG=fixture-image SRV_CPUS=0-1 VLLM_UTIL=0.50
timeout() { shift; "$@"; }
docker() { if [ "$1" = create ]; then return 7; fi; echo wrongly-started > "$LOGD/start-called"; }
""" % (shlex.quote(str(HERE/'session-lifecycle.sh')),shlex.quote(tmp))
            result=subprocess.run(['bash','-c',script+function+'\nstart_server\n'],capture_output=True,text=True,timeout=5)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((Path(tmp)/'start-called').exists())

class ProcessLifecycleTests(unittest.TestCase):
    def shell(self, body):
        prefix="""set -u
trap 'for p in $(jobs -pr); do kill -TERM "$p" 2>/dev/null || true; done; wait 2>/dev/null || true' EXIT
source %s
""" % shlex.quote(str(HERE/'session-lifecycle.sh'))
        result=subprocess.run(['bash','-c',prefix+body],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        return result.stdout

    def test_tracked_child_stops_and_untracked_decoy_survives(self):
        self.shell("""sleep 60 & own=$!; h2h_track "$own"
sleep 60 & decoy=$!
h2h_stop "$own" 10 || exit 1
kill -0 "$decoy" || exit 2
kill -TERM "$decoy"; wait "$decoy" 2>/dev/null || true
[ -z "${H2H_PID_START[$own]:-}" ]
""")

    def test_stale_birth_does_not_signal_process(self):
        self.shell("""sleep 60 & own=$!; h2h_track "$own"
H2H_PID_START[$own]=stale
if h2h_stop "$own" 1; then exit 1; fi
kill -0 "$own" || exit 2
kill -TERM "$own"; wait "$own" 2>/dev/null || true
""")

    def test_cleanup_deadline_forces_own_term_ignoring_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready=Path(tmp)/'ready'
            body="""bash -c 'trap "" TERM; echo ready > "$1"; exec sleep 60' fixture %s & own=$!
h2h_track "$own"
for attempt in {1..100}; do [ -e %s ] && break; sleep 0.01; done
H2H_STOP_DEADLINE=$SECONDS
h2h_stop "$own" || exit 1
if kill -0 "$own" 2>/dev/null; then exit 2; fi
""" % (shlex.quote(str(ready)),shlex.quote(str(ready)))
            self.shell(body)

    def test_non_child_and_untracked_process_refused(self):
        self.shell("""if h2h_track $$; then exit 1; fi
sleep 60 & other=$!
if h2h_stop "$other" 1; then kill "$other"; exit 2; fi
kill -0 "$other" || exit 3
kill -TERM "$other"; wait "$other" 2>/dev/null || true
""")

    def test_already_exited_child_can_be_reaped(self):
        self.shell("""true & own=$!; wait "$own"
h2h_track "$own" || exit 1
h2h_stop "$own" 1
""")

    def test_term_ignoring_child_is_killed_after_grace(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready=Path(tmp)/'ready'
            body="""bash -c 'trap "" TERM; echo ready > "$1"; exec sleep 60' fixture %s & own=$!
h2h_track "$own"
for attempt in {1..100}; do [ -e %s ] && break; sleep 0.01; done
h2h_stop "$own" 2 || exit 1
if kill -0 "$own" 2>/dev/null; then exit 2; fi
""" % (shlex.quote(str(ready)),shlex.quote(str(ready)))
            self.shell(body)

    def test_descendant_is_stopped_with_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready=Path(tmp)/'child'
            body="""( sleep 60 & child=$!
trap 'kill -TERM "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; exit 0' TERM
printf '%%s' "$child" > %s
wait "$child"
) & own=$!
h2h_track "$own"
for attempt in {1..100}; do [ -s %s ] && break; sleep 0.01; done
child=$(cat %s)
h2h_stop "$own" 10 || exit 1
identity=$(h2h_identity "$child") || exit 0
read -r state birth parent <<< "$identity"
[ "$state" = Z ] || [ "$state" = X ]
""" % tuple(shlex.quote(str(ready)) for _ in range(3))
            self.shell(body)

    def test_memory_sampler_resolves_published_server_identity(self):
        source=(HERE/'run_arm.sh').read_text()
        function=source[source.index('server_pids() {'):source.index('start_server() {')]
        with tempfile.TemporaryDirectory() as tmp:
            script='LOGD=%s\nCTR= SPID= RUN_ID=fixture\n' % shlex.quote(tmp)
            script+=function+"""
sleep 60 & actual=$!; h2h_track "$actual"
sleep 60 & decoy=$!
printf '%s %s\n' "$actual" "${H2H_PID_START[$actual]}" > "$LOGD/server-identity.txt"
observed=$(server_pids)
[ "$observed" = "$actual" ] || exit 1
h2h_stop "$actual" 10 || exit 2
kill -TERM "$decoy"; wait "$decoy" 2>/dev/null || true
"""
            self.shell(script)

    def test_cleanup_preserves_parent_lock_and_releases_own_lock(self):
        source=(HERE/'run_arm.sh').read_text()
        cleanup=source[source.index('cleanup_session() {'):source.index('# ---- session ----')]
        for release in [0,1]:
            with self.subTest(release=release),tempfile.TemporaryDirectory() as tmp:
                prefix="""set -u
LOGD=%s
SPID= CPID= CMON_PID= WD= SAMP= VMS= BASE=100 GPU_LOCK_HELD=1 GPU_LOCK_RELEASE=%d
ARM=CONTROL ROUND=R0
timeout() { shift; "$@"; }
nvidia-smi() { case "$*" in *noheader*) : ;; *) echo fixture-empty-inventory-header ;; esac; }
memavail() { echo 100; }
ktpsr() { echo 99; }
board() { :; }
stop_server() { :; }
""" % (shlex.quote(tmp),release)
                script=prefix+cleanup+"""
export LOGD SPID CPID CMON_PID WD SAMP VMS BASE GPU_LOCK_HELD GPU_LOCK_RELEASE ARM ROUND
export -f cleanup_session timeout nvidia-smi memavail ktpsr board stop_server
exec 9> "$LOGD/lock"
flock -n 9 || exit 1
bash -c 'trap cleanup_session EXIT; exit 0' || exit 2
if flock -n "$LOGD/lock" true; then acquired=1; else acquired=0; fi
[ "$acquired" = "$GPU_LOCK_RELEASE" ] || exit 3
flock -u 9
flock -n "$LOGD/lock" true
"""
                result=subprocess.run(['bash','-c',script],capture_output=True,text=True,timeout=5)
                self.assertEqual(result.returncode,0,result.stderr)

    def lifecycle_fixture(self, mode):
        source=(HERE/'run_arm.sh').read_text()
        stop=source[source.index('stop_server() {'):source.index('ready() {')]
        cleanup=source[source.index('start_watchdog() {'):source.index('# ---- session ----')]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            prefix='''set -u
source %s
LOGD=%s
SPID= CPID= CMON_PID= WD= SAMP= VMS= BASE= GPU_LOCK_HELD=0 CTR=
ARM=OURS ROUND=R0 RUN_ID=fixture
board() { :; }
''' % (shlex.quote(str(HERE/'session-lifecycle.sh')),shlex.quote(tmp))
            body="""
trap cleanup_session EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
sleep 60 & SPID=$!; h2h_track "$SPID"
sleep 60 & CPID=$!; h2h_track "$CPID"
sleep 60 & CMON_PID=$!; h2h_track "$CMON_PID"
sleep 60 & WD=$!; h2h_track "$WD"
sleep 60 & SAMP=$!; h2h_track "$SAMP"
sleep 60 & VMS=$!; h2h_track "$VMS"
touch "$LOGD/DONE"
printf '%s\n' "$SPID" "$CPID" "$CMON_PID" "$WD" "$SAMP" "$VMS" > "$LOGD/children"
"""
            if mode=='timeout':
                body += 'h2h_stop "$WD" 10; WD=; start_watchdog 0.02 0.02; wait "$CPID"\n'
            else:
                body += 'wait "$CPID"\n' if mode=='term' else ('exit 7\n' if mode=='fail' else 'exit 0\n')
            process=subprocess.Popen(['bash','-c',prefix+stop+cleanup+body],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                if mode=='term':
                    deadline=time.monotonic()+5
                    while not (root/'children').exists() and time.monotonic()<deadline:
                        if process.poll() is not None: break
                        time.sleep(0.01)
                    self.assertTrue((root/'children').exists())
                    process.send_signal(signal.SIGTERM)
                stdout,stderr=process.communicate(timeout=12)
                self.assertEqual(process.returncode,143 if mode in ['term','timeout'] else 7 if mode=='fail' else 0,stderr)
                pids=(root/'children').read_text().splitlines()
                for pid in pids:
                    self.assertFalse(Path('/proc/'+pid).exists(),'owned child survived cleanup: '+pid)
                self.assertEqual((root/'DONE').exists(),mode=='normal')
            finally:
                if process.poll() is None:
                    process.kill();process.communicate(timeout=5)

    def test_normal_exit_cleans_own_processes(self): self.lifecycle_fixture('normal')
    def test_failed_exit_cleans_own_processes_and_removes_done(self): self.lifecycle_fixture('fail')
    def test_term_interrupt_cleans_own_processes_and_removes_done(self): self.lifecycle_fixture('term')
    def test_watchdog_interrupt_cleans_own_processes_and_removes_done(self): self.lifecycle_fixture('timeout')

if __name__=='__main__':unittest.main()

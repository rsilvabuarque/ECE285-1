"""Monitor only until the frozen R2 generation queue runs, then replace its owned CPU driver."""
from pathlib import Path
import hashlib,json,os,signal,socket,subprocess,time
r=Path(__file__).parent;oldpid=1535630;deadline=1794578068

def b(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
gen=r/'taskfiles/round2_acquisition_generation.json';control=r.parent.parent/'controls/uma.json'
while True:
 if time.time()>deadline:raise TimeoutError('Goal deadline before handoff')
 if gen.exists() and json.loads(control.read_text())['task_file']==str(gen) and list((gen.parent/(gen.stem+'.claims')).glob('*/owner.json')):break
 if not Path(f'/proc/{oldpid}/cmdline').exists():raise RuntimeError('Expected olddriver exited before designatedhandoff')
 time.sleep(2)
cmd=Path(f'/proc/{oldpid}/cmdline').read_bytes().decode().replace('\0',' ')
assert 'srun' in cmd and str(r/'resume_with_phase2.py') in cmd and Path(f'/proc/{oldpid}').stat().st_uid==os.getuid()
assert not list((r/'round2_acquisition').glob('oracle_*/registry.json'))
write(r/'queue_handoff_request.json',dict(status='AUTHORIZED_QUEUE_ONLY_CPU_HANDOFF_REQUESTED',created_unix=time.time(),old_srun_launcher_pid=oldpid,host=socket.gethostname(),old_source=b(r/'resume_with_phase2.py'),new_source=b(r/'resume_r2_optimized.py'),frozen_generation_tasks=b(gen),worker_override=b(r/'oracle_worker_override.json'),watcher_source=b(Path(__file__)),reason='Root approved validated queue-only worker snapshot from first R2SCF; original science source and raw proposals unchanged',only_owned_cpu_launcher_signaled=True))
os.kill(oldpid,signal.SIGTERM)
while Path(f'/proc/{oldpid}/stat').exists():
 state=Path(f'/proc/{oldpid}/stat').read_text().split()[2]
 if state=='Z':break
 time.sleep(.2)
write(r/'queue_handoff_completion.json',dict(status='OLD_CPU_DRIVER_TERMINAL_BEFORE_NEW_R2_DRIVER',created_unix=time.time(),old_srun_launcher_pid=oldpid,request=b(r/'queue_handoff_request.json'),old_log=b(r/'resume_with_phase2.log'),only_one_submitting_driver=True,gpu_tasks_untouched=True))
command=['srun','--jobid=61215970','--overlap','--exact','-N1','-n1','-c8','env','PYTHONPATH='+str(r.parent/'source/src'),'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','/pscratch/sd/r/rsb/ece285_conditioning_20261107/venvs/class_core/learning/bin/python','-u',str(r/'resume_r2_optimized.py')]
print('HANDOFF_COMPLETE_NEW_CPU_DRIVER',time.time(),flush=True)
with (r/'resume_r2_optimized.log').open('x') as output:result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT)
write(r/'optimized_driver_exit.json',dict(status='CPU_DRIVER_EXITED',created_unix=time.time(),exit_code=result.returncode,command=command,source=b(r/'resume_r2_optimized.py')))
raise SystemExit(result.returncode)

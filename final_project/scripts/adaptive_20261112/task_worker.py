"""One visible GPU executes prospectively frozen task files under an external allocation."""
import argparse,hashlib,json,os,socket,subprocess,time,traceback
from pathlib import Path


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);t=p.with_name(p.name+f'.{os.getpid()}.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)


def main():
 p=argparse.ArgumentParser();p.add_argument('--control',type=Path,required=True);p.add_argument('--stop',type=Path,required=True);a=p.parse_args()
 identity=f'{os.environ.get("SLURM_JOB_ID","local")}_{socket.gethostname()}_{os.getpid()}'
 while not a.stop.exists():
  if not a.control.exists(): time.sleep(2);continue
  control=json.loads(a.control.read_text()); taskfile=Path(control['task_file'])
  if not taskfile.exists(): time.sleep(2);continue
  tasks=json.loads(taskfile.read_text())['tasks'];claimroot=taskfile.parent/(taskfile.stem+'.claims');claimroot.mkdir(exist_ok=True)
  available=[t for t in tasks if not (claimroot/t['id']).exists()]
  if not available: time.sleep(2);continue
  task=available[int.from_bytes(os.urandom(4),'little')%len(available)];claim=claimroot/task['id']
  try:claim.mkdir()
  except FileExistsError:continue
  receipt={'status':'RUNNING','worker':identity,'task':task,'task_file':str(taskfile),'task_file_sha256':sha(taskfile),'started_unix':time.time(),'source_sha256':sha(__file__)};write(claim/'owner.json',receipt)
  try:
   for item in task.get('bindings',[]):
    if sha(item['path'])!=item['sha256']:raise ValueError('Frozen task input mutated: '+item['path'])
   env=os.environ.copy();env.update(task.get('env',{}))
   with (claim/'stdout.log').open('x') as log: result=subprocess.run(task['argv'],cwd=task.get('cwd'),env=env,stdout=log,stderr=subprocess.STDOUT)
   if result.returncode:raise RuntimeError(f'Child exit code {result.returncode}; see stdout.log')
   for check in task.get('checks',[]):
    path=Path(check['path'])
    if not path.is_file():raise ValueError('Missing expected output '+str(path))
    if check.get('equals'):
     output=json.loads(path.read_text())
     for k,v in check['equals'].items():
      if output.get(k)!=v:raise ValueError(f'Unexpected {k}: {output.get(k)!r}')
   receipt.update(status='COMPLETE',finished_unix=time.time());write(claim/'receipt.json',receipt);print('COMPLETE',task['id'],flush=True)
  except Exception as e:
   receipt.update(status='FAILED',finished_unix=time.time(),error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());write(claim/'receipt.json',receipt);print('FAILED',task['id'],str(e),flush=True)
 print('WORKER_STOP',identity,flush=True)

if __name__=='__main__':main()

"""Resident single-GPU oracle, dynamic atomic claims, raw evidence per shard."""
import argparse,hashlib,json,os,socket,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from ase import Atoms
from calculators import load,evaluate
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);a=ap.parse_args();assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
 calc,receipt=load(a.model);identity=f"{os.environ['SLURM_JOB_ID']}_{socket.gethostname()}_{os.getpid()}";receipt.update(worker=identity,source_sha256=sha(__file__),calculator_source_sha256=sha(ROOT/'calculators.py'))
 prior=json.loads((ROOT/'validation'/f'scf_{a.model}.json').read_text())
 # Exact pinned environments/checkpoints are certified by the previous physical pilot.
 old=json.loads((Path(json.loads((ROOT/'config.json').read_text())['fixed_pilot'])/'validation'/f'{a.model}.json').read_text())['receipt']
 for key in ['checkpoint_hashes','dtype','tf32','package_version','torch']:assert receipt[key]==old[key]
 dump(HERE/'workers'/f'{identity}.json',receipt);print('ORACLE_LOADED',a.model,identity,flush=True)
 parent_path=ROOT/'data/parents/O3_N48_pilot.npz'
 with np.load(parent_path) as z:parent={k:z[k] for k in ['numbers','positions_A','cell_A','li_indices']}
 cache={}
 while not (HERE/'STOP_SCORING').exists():
  queue=HERE/'data/score_tasks.jsonl'
  tasks=[json.loads(s) for s in queue.read_text().splitlines() if s] if queue.exists() else []
  available=[t for t in tasks if t['model']==a.model and not (HERE/'claims'/t['task_id']).exists()]
  if not available:time.sleep(2);continue
  task=available[int.from_bytes(os.urandom(2),'little')%len(available)];claim=HERE/'claims'/task['task_id']
  try:claim.mkdir()
  except FileExistsError:continue
  dump(claim/'owner.json',{'worker':identity,'started':time.time(),'job_id':os.environ['SLURM_JOB_ID']})
  path=Path(task['input_npz'])
  if str(path) not in cache:
   assert sha(path)==task['input_sha256']
   with np.load(path) as z:cache[str(path)]={k:z[k] for k in z.files}
  d=cache[str(path)];ii=np.arange(task['start'],task['stop']);E=[];F=[];S=[];seconds=[]
  for i in ii:
   v=d['masks'][i].astype(bool);keep=np.ones(192,bool);keep[parent['li_indices'][v]]=False
   atoms=Atoms(numbers=parent['numbers'][keep],positions=parent['positions_A'][keep],cell=parent['cell_A'],pbc=True)
   calc.reset();started=time.perf_counter();e,f,s=evaluate(calc,atoms);seconds.append(time.perf_counter()-started)
   padded=np.zeros((192,3),np.float32);padded[keep]=f;E.append(e);F.append(padded);S.append(s)
  out=HERE/'scores'/a.model;out.mkdir(exist_ok=True);base=out/task['task_id'];temp=Path(str(base)+'.tmp.npz')
  np.savez_compressed(temp,structure_ids=d['structure_ids'][ii],indices=ii,energy_eV=E,forces_parent_eV_A=F,stress_eV_A3=S,seconds=seconds)
  temp.replace(base.with_suffix('.npz'));dump(base.with_suffix('.json'),dict(task,worker=identity,status='complete',finished=time.time(),output_sha256=sha(base.with_suffix('.npz'))));print('SCORED',task['task_id'],flush=True)
 print('ORACLE_STOP',identity,flush=True)
if __name__=='__main__':main()

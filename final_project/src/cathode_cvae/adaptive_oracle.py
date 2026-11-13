"""Generic-parent scoring queue for the November 12 adaptive/entropy campaign.

Schedulers are external. A physical key includes parent bytes, canonical mask,
geometry protocol, teacher checkpoint/head and precision. Only scalar energies
are imported from certified exact-geometry caches; force vectors are not aliased.
"""
import argparse,json,os,socket,time,hashlib
from pathlib import Path
import numpy as np
from .data import sha256,deletion_atoms
from .mlip.registry import MODELS

PROTOCOL='pristine-derived deletion; unrelaxed; exact canonical parent coordinates'

def atomic_json(path,value):
 path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def physical_key(parent_sha,mask,model):
 s=MODELS[model];body=dict(parent_sha256=parent_sha,mask=str(mask),geometry_protocol=PROTOCOL,model=model,head=s['head'],checkpoint_hashes=s['files'],package_version=s['version'],dtype='float32',tf32=False)
 return hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()


def cache_context(model,parent_sha):
 s=MODELS[model]
 return dict(parent_sha256=parent_sha,model=model,head=s['head'],checkpoint_hashes=s['files'],package_version=s['version'],dtype='float32',tf32=False,geometry_protocol=PROTOCOL,orientation='canonical')

def certify_cache(root,model,parent_sha):
 root=Path(root);p=root/f'cache_{model}.npz';certificate=root/f'cache_{model}.json'
 if not certificate.exists():raise ValueError('Cache requires an exact geometry/teacher/precision certificate')
 c=json.loads(certificate.read_text())
 if c.get('context')!=cache_context(model,parent_sha):raise ValueError('Cache context differs from parent/teacher/protocol/precision')
 if c.get('npz_sha256')!=sha256(p):raise ValueError('Cache bytes differ from certified source')
 return c

def register_pool(root,input_path,parent_path,models=('uma','mace'),batch_size=64,as_written=False):
 if batch_size < 1:raise ValueError('batch_size must be positive')
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 parent_sha=sha256(parent_path)
 with np.load(parent_path,allow_pickle=False) as z: sites=len(z['target_indices'] if 'target_indices' in z else z['li_indices'])
 if not 1 <= sites <= 62: raise ValueError('Canonical integer codes require 1..62 mobile sites')
 registry_path=root/'registry.json';registry=json.loads(registry_path.read_text()) if registry_path.exists() else {'parent_sha256':parent_sha,'physical_keys':{},'tasks':[],'as_written':as_written}
 if registry['parent_sha256']!=parent_sha:raise ValueError('One queue must have one exact parent')
 if registry.get('as_written',False)!=as_written:raise ValueError('Do not mix canonical and as-written audit queues')
 registry['as_written']=as_written
 with np.load(input_path,allow_pickle=False) as z:
  ids=z['structure_ids'].astype(str);masks=z['masks'] if 'masks' in z else 1-z['occupancy'];embedded=str(z['parent_sha256'].item()) if 'parent_sha256' in z else None
 if embedded and embedded!=parent_sha:raise ValueError('Pool parent hash differs')
 if len(set(ids))!=len(ids) or masks.shape!=(len(ids),sites) or not np.isin(masks,[0,1]).all():raise ValueError('Pool must contain unique binary geometries matching the parent')
 with np.load(parent_path,allow_pickle=False) as z: inv=np.argsort(z['permutations'],axis=1)
 powers=1<<np.arange(sites-1,-1,-1,dtype=np.int64)
 for start in range(0,len(masks),128):
  v=masks[start:start+128].astype(np.int64)
  if not as_written and not np.array_equal((v*powers).sum(-1),(v[:,inv]*powers).sum(-1).min(1)):raise ValueError('Pool masks must be in canonical orientation')
 input_sha=sha256(input_path);added=[]
 for model in models:
  cache_path=root/f'cache_{model}.npz';cached=set()
  if cache_path.exists():
   certify_cache(root,model,parent_sha)
   with np.load(cache_path,allow_pickle=False) as z:cached=set(z['structure_ids'].astype(str))
  pending=[]
  for i,(sid,mask) in enumerate(zip(ids,masks)):
   key=''.join(map(str,mask));expected=f'mask{sites}_{int(key,2):0{(sites+3)//4}x}'
   if not as_written and sid!=expected:raise ValueError('Scoring ID disagrees with canonical mask code')
   pk=physical_key(parent_sha,key,model)
   if pk not in registry['physical_keys']:
    registry['physical_keys'][pk]=dict(model=model,structure_id=sid,input_sha256=input_sha,provenance='exact_geometry_cache' if sid in cached else 'new_scoring');
    if sid not in cached:pending.append(i)
  for start in range(0,len(pending),batch_size):
   ii=pending[start:start+batch_size];tid=f'{model}_{input_sha[:12]}_{start:07d}'
   row=dict(task_id=tid,model=model,input_npz=str(Path(input_path).resolve()),input_sha256=input_sha,parent_npz=str(Path(parent_path).resolve()),parent_sha256=parent_sha,indices=ii,geometry_protocol=PROTOCOL,expected_structure_ids=ids[ii].tolist())
   registry['tasks'].append(row);added.append(row)
 atomic_json(registry_path,registry)
 # Publish queue atomically after the complete registry is durable.
 queue=root/'tasks.jsonl';tmp=queue.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(t)+'\n' for t in registry['tasks']));tmp.replace(queue)
 return dict(added_tasks=len(added),added_geometry_model_pairs=sum(len(t['indices']) for t in added),total_tasks=len(registry['tasks']),queue=str(queue))

def worker(root,model,checkpoint,device='cuda',threads=4):
 from .mlip.calculators import make_calculator,evaluate
 root=Path(root);(root/'claims').mkdir(exist_ok=True);(root/'scores'/model).mkdir(parents=True,exist_ok=True);(root/'workers').mkdir(exist_ok=True)
 identity=f'{os.environ.get("SLURM_JOB_ID","local")}_{socket.gethostname()}_{os.getpid()}'
 print('WORKER_LOADING',model,identity,flush=True);calc,receipt=make_calculator(model,checkpoint,device,threads)
 receipt.update(worker=identity,worker_source_sha256=sha256(__file__));atomic_json(root/'workers'/f'{identity}.json',receipt);print('WORKER_READY',model,identity,flush=True)
 parents={};inputs={}
 while not (root/'STOP').exists():
  queue=root/'tasks.jsonl';tasks=[json.loads(s) for s in queue.read_text().splitlines() if s] if queue.exists() else []
  candidates=[t for t in tasks if t['model']==model and not (root/'claims'/t['task_id']).exists()]
  if not candidates:time.sleep(1);continue
  # Randomized task choice reduces lock contention; task identity fixes all inputs.
  t=candidates[int.from_bytes(os.urandom(4),'little')%len(candidates)];claim=root/'claims'/t['task_id']
  try:claim.mkdir()
  except FileExistsError:continue
  atomic_json(claim/'owner.json',dict(worker=identity,started_unix=time.time()))
  try:
   path=t['input_npz'];parentfile=t['parent_npz']
   if path not in inputs:
    if sha256(path)!=t['input_sha256']:raise ValueError('Input mutated after task publication')
    with np.load(path,allow_pickle=False) as z:inputs[path]=dict(structure_ids=z['structure_ids'],masks=z['masks'] if 'masks' in z else 1-z['occupancy'])
   if parentfile not in parents:
    if sha256(parentfile)!=t['parent_sha256']:raise ValueError('Parent mutated after task publication')
    with np.load(parentfile,allow_pickle=False) as z:parents[parentfile]={k:z[k] for k in z.files}
   d=inputs[path];parent=parents[parentfile];rows=[];forces=[]
   for i in t['indices']:
    sid=str(d['structure_ids'][i]);a,slots=deletion_atoms(parent,1-d['masks'][i]);begin=time.perf_counter()
    try:
     E,F,S=evaluate(calc,a);padded=np.zeros((len(parent['numbers']),3),np.float32);padded[slots]=F;forces.append(padded)
     rows.append(dict(structure_id=sid,status='success',energy_eV=E,n_atoms=len(a),n_Li=int(len(d['masks'][i])-d['masks'][i].sum()),seconds=time.perf_counter()-begin,force_component_rms_eV_A=float(np.sqrt(np.mean(F**2))),force_max_eV_A=float(np.linalg.norm(F,axis=1).max()),stress_eV_A3=S.tolist()))
    except Exception as err:
     rows.append(dict(structure_id=sid,status='failed',error_type=type(err).__name__,error=str(err),seconds=time.perf_counter()-begin));forces.append(np.full((len(parent['numbers']),3),np.nan,np.float32))
   output=root/'scores'/model/(t['task_id']+'.npz');tmp=output.with_suffix('.tmp.npz')
   np.savez_compressed(tmp,structure_ids=np.array([r['structure_id'] for r in rows]),status=np.array([r['status'] for r in rows]),energy_eV=np.array([r.get('energy_eV',np.nan) for r in rows]),n_atoms=np.array([r.get('n_atoms',0) for r in rows]),n_Li=np.array([r.get('n_Li',-1) for r in rows]),forces_parent_eV_A=np.array(forces),force_component_rms_eV_A=np.array([r.get('force_component_rms_eV_A',np.nan) for r in rows]),force_max_eV_A=np.array([r.get('force_max_eV_A',np.nan) for r in rows]),stress_eV_A3=np.array([r.get('stress_eV_A3',[np.nan]*6) for r in rows]),seconds=np.array([r['seconds'] for r in rows]));tmp.replace(output)
   atomic_json(output.with_suffix('.json'),dict(task=t,worker=identity,rows=rows,output_sha256=sha256(output),finished_unix=time.time(),status='complete' if all(r['status']=='success' for r in rows) else 'complete_with_failures'));print('SCORED',t['task_id'],len(rows),flush=True)
  except Exception as err:
   atomic_json(claim/'failure.json',dict(status='worker_task_failed',error_type=type(err).__name__,error=str(err),worker=identity));print('TASK_FAILED',t['task_id'],str(err),flush=True)
 print('WORKER_STOP',identity,flush=True)

def collect(root,output):
 root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False);registry=json.loads((root/'registry.json').read_text());by_model={};missing=[];failures=[]
 for model in ['uma','mace']:
  cache=root/f'cache_{model}.npz'
  wanted={v['structure_id'] for v in registry['physical_keys'].values() if v['model']==model and v.get('provenance')=='exact_geometry_cache'}
  if cache.exists() and wanted:
   certify_cache(root,model,registry['parent_sha256'])
   with np.load(cache,allow_pickle=False) as z:
    cached={k:z[k] for k in ['structure_ids','energy_eV','n_atoms','n_Li']}
   for i,sid in enumerate(cached['structure_ids'].astype(str)):
    if sid in wanted:by_model.setdefault(model,{})[sid]=dict(energy_eV=cached['energy_eV'][i],n_atoms=cached['n_atoms'][i],n_Li=cached['n_Li'][i],force_component_rms_eV_A=np.nan,force_max_eV_A=np.nan,stress_eV_A3=np.full(6,np.nan),seconds=0.,provenance='cached',status='cached')
 for task in registry['tasks']:
  p=root/'scores'/task['model']/(task['task_id']+'.npz')
  if not p.exists():missing.append(task['task_id']);continue
  receipt=json.loads(p.with_suffix('.json').read_text())
  if sha256(p)!=receipt['output_sha256']:raise ValueError('Score output hash changed')
  if receipt['task']!=task:raise ValueError('Score receipt task differs from registry')
  with np.load(task['input_npz'],allow_pickle=False) as original:expected=original['structure_ids'][task['indices']].astype(str)
  with np.load(p,allow_pickle=False) as z:
   # Compressed NPZ access decompresses a whole member. Load each member once,
   # rather than once per geometry; this leaves all scientific values unchanged.
   scored={k:z[k] for k in ['structure_ids','status','energy_eV','n_atoms','n_Li','force_component_rms_eV_A','force_max_eV_A','stress_eV_A3','seconds']}
  if not np.array_equal(scored['structure_ids'].astype(str),expected):raise ValueError('Scored identities/coverage differ from registered indices')
  for i,sid in enumerate(scored['structure_ids'].astype(str)):
   if scored['status'][i]!='success':failures.append(dict(task=task['task_id'],structure_id=sid));continue
   dest=by_model.setdefault(task['model'],{})
   if sid in dest:raise ValueError('One physical geometry/model scored twice in queue')
   dest[sid]={k:scored[k][i] for k in ['energy_eV','n_atoms','n_Li','force_component_rms_eV_A','force_max_eV_A','stress_eV_A3','seconds']}
   dest[sid].update(provenance='new',status='success')
 expected_by_model={}
 for item in registry['physical_keys'].values():expected_by_model.setdefault(item['model'],set()).add(item['structure_id'])
 for model,wanted in expected_by_model.items():
  for sid in sorted(wanted-set(by_model.get(model,{}))):failures.append(dict(model=model,structure_id=sid,error='registered geometry has no successful energy'))
 for model,items in by_model.items():
  ids=sorted(items);np.savez_compressed(output/f'{model}.npz',structure_ids=np.array(ids),**{k:np.array([items[s][k] for s in ids]) for k in next(iter(items.values()))})
 receipt=dict(status='PASS' if not missing and not failures else 'INCOMPLETE',tasks=len(registry['tasks']),missing_tasks=missing,failed_structures=failures,model_counts={m:len(v) for m,v in by_model.items()},registry_sha256=sha256(root/'registry.json'),scope='Exact canonical deletion geometries, fixed parent/teacher/precision; raw force vectors retain evaluated orientation')
 atomic_json(output/'receipt.json',receipt);return receipt

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 r=sub.add_parser('register');r.add_argument('--root',type=Path,required=True);r.add_argument('--input',type=Path,required=True);r.add_argument('--parent',type=Path,required=True);r.add_argument('--models',nargs='+',default=['uma','mace']);r.add_argument('--batch-size',type=int,default=64);r.add_argument('--as-written',action='store_true',help='Audit distinct actual orientations without canonical aliasing')
 w=sub.add_parser('worker');w.add_argument('--root',type=Path,required=True);w.add_argument('--model',required=True);w.add_argument('--checkpoint',required=True);w.add_argument('--device',default='cuda');w.add_argument('--threads',type=int,default=4)
 c=sub.add_parser('collect');c.add_argument('--root',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
 a=p.parse_args(argv)
 from .runtime import require_compute_allocation
 require_compute_allocation()
 if a.command=='register':value=register_pool(a.root,a.input,a.parent,a.models,a.batch_size,a.as_written)
 elif a.command=='collect':value=collect(a.root,a.output)
 else:return worker(a.root,a.model,a.checkpoint,a.device,a.threads)
 print(json.dumps(value,indent=2));return value

if __name__=='__main__':main()

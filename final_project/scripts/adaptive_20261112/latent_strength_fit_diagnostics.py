"""Launch predeclared strength fits only after required phase1 scoring is complete."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from cathode_cvae.runtime import require_compute_allocation
from cathode_cvae.iterative_acquisition import _canonical_vacancy
require_compute_allocation();import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args();p=args.run_root.resolve();b=p.parent;r=b/'latent_diagnostics';sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
start=time.time();limit=10*3600/(18/4)
control=b/'controls/mace.json'
def publish(path):
 t=control.with_suffix('.latent_phase2.tmp');t.write_text(json.dumps({'task_file':str(path)},indent=2)+'\n');t.replace(control)
def wait_tasks(path,n):
 while True:
  claims=path.parent/(path.stem+'.claims');done=list(claims.glob('*/receipt.json'));bad=[]
  for f in done:
   obj=json.loads(f.read_text())
   if obj['status']!='COMPLETE':bad.append(str(f))
  if bad:raise RuntimeError('Preserved failed tasks: '+str(bad))
  if len(done)==n:return
  if time.time()-start>limit:raise TimeoutError('Phase2 declared10GPU-node-hour bound exhausted; preserve completed work and do not launch later phases')
  time.sleep(5)
while not (r/'oracle_monitor_receipt.json').exists():time.sleep(5)
if json.loads((r/'oracle_monitor_receipt.json').read_text())['status']!='PASS':raise ValueError('Phase1 did not finish')
publish(p/'fit_tasks.json');print('PHASE2_FITS_PUBLISHED',time.time(),flush=True);wait_tasks(p/'fit_tasks.json',30)
base=json.loads((r/'diagnostic_tasks_without_entropy.json').read_text())['tasks'];tasks=[]
for family in ['ar','d2']+[f'lambda{w}_{mode}_d2' for w in [10,100] for mode in ['latent','condition','shuffled']]:
 for j,seed in enumerate(range(2026111301,2026111306)):
  ref=family if family in ['ar','d2'] else 'd2';t=json.loads(json.dumps(next(x for x in base if x['id']==f'diagnostic_{ref}_{seed}')));t['id']=f'diagnostic_{family}_{seed}';argv=t['argv'];previous=Path(argv[argv.index('--checkpoint')+1]);ck=b/'initial_fits'/f'{family}_{seed}'/'last_update.pt' if family in ['ar','d2'] else p/'fits'/f'{family}_{seed}'/'last_update.pt';argv[argv.index('--checkpoint')+1]=str(ck);argv[argv.index('--seed')+1]=str(2026111401+j);out=p/'diagnostics'/f'{family}_{seed}';argv[argv.index('--output-dir')+1]=str(out)
  t['bindings']=[x for x in t['bindings'] if x['path']!=str(previous)];t['bindings'].extend([dict(path=str(x),sha256=sha(x)) for x in [ck,p/'protocol.json']]);t['checks']=[dict(path=str(out/'receipt.json'),equals={'status':'PASS'})];tasks.append(t)
path=p/'diagnostic_tasks.json'
with path.open('x') as f:json.dump({'tasks':tasks},f,indent=2);f.write('\n')
publish(path);print('PHASE2_DIAGNOSTICS_PUBLISHED',time.time(),flush=True);wait_tasks(path,40)
parent=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project/artifacts/current/core/data/parents/O3_N48_pilot.npz');parent_sha=sha(parent)
with np.load(parent,allow_pickle=False) as z:perms=z['permutations']
allids=[];allmasks=[];sources=[];attempts=0
for task in tasks:
 out=Path(task['argv'][task['argv'].index('--output-dir')+1]);receipt=json.loads((out/'receipt.json').read_text());raw=out/'raw_intervention_proposals.npz'
 if receipt['status']!='PASS' or receipt['parent_sha256']!=parent_sha or receipt['generation']['attempts']!=12672:raise ValueError('Bad rawgeneration')
 with np.load(raw,allow_pickle=False) as z:
  ids=z['structure_ids'].astype(str);masks=z['masks'];n=z['n_target']
  if str(z['parent_sha256'].item())!=parent_sha or masks.shape!=(len(ids),48) or not np.isin(masks,[0,1]).all() or not np.array_equal(48-masks.sum(1),n):raise ValueError('Invalid mask/count/parent')
  code=(masks.astype(np.int64)*(1<<np.arange(47,-1,-1,dtype=np.int64))).sum(1)
  if not np.array_equal(ids,np.array([f'mask48_{int(v):012x}' for v in code])):raise ValueError('Bad ID')
  allids.append(ids);allmasks.append(masks);attempts+=len(ids)
 sources.append(dict(task_id=task['id'],output=str(raw),output_sha256=sha(raw),receipt_sha256=sha(out/'receipt.json')))
ids=np.concatenate(allids);masks=np.concatenate(allmasks);unique,first=np.unique(ids,return_index=True);masks=masks[first]
if not np.array_equal(_canonical_vacancy(1-masks,perms),np.array([''.join(map(str,m)) for m in masks])):raise ValueError('Noncanonical mask')
pool=p/'raw_valid_canonical_union.npz'
with pool.open('xb') as f:np.savez_compressed(f,structure_ids=unique,masks=masks,occupancy=1-masks,parent_sha256=np.asarray(parent_sha))
receipt=dict(status='ALL40_RAW_DIAGNOSTICS_FROZEN_BEFORE_ORACLE',parent_sha256=parent_sha,raw_attempts=attempts,unique_valid_orbits=len(unique),pool=str(pool),pool_sha256=sha(pool),task_file_sha256=sha(path),protocol_sha256=sha(p/'protocol.json'),sources=sources,no_energy_filtering=True,no_protected_energies_read=True,source_sha256=sha(__file__),phase2_elapsed_seconds=time.time()-start)
with (p/'raw_union_receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print('PHASE2_ALL_RAW_FROZEN',json.dumps({k:v for k,v in receipt.items() if k!='sources'}),flush=True)

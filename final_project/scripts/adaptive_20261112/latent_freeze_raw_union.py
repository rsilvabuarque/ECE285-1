from pathlib import Path
import json,hashlib,time
import numpy as np
from cathode_cvae.runtime import require_compute_allocation
from cathode_cvae.iterative_acquisition import _canonical_vacancy

require_compute_allocation()
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);parser.add_argument('--task-file',type=Path,required=True);args=parser.parse_args();root=args.run_root.resolve();base=root.parent;taskfile=args.task_file.resolve()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
tasks=json.loads((taskfile).read_text())['tasks'];tasks=[t for t in tasks if t['id'].startswith('diagnostic_')]
if len(tasks)!=40:raise ValueError('Need40diagnostic task definitions')
parent=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project/artifacts/current/core/data/parents/O3_N48_pilot.npz')
with np.load(parent,allow_pickle=False) as z:perms=z['permutations']
parent_sha=sha(parent);all_ids=[];all_masks=[];records=[];attempts=0
for t in tasks:
 claim=taskfile.parent/(taskfile.stem+'.claims')/t['id']/'receipt.json';receipt=json.loads(claim.read_text())
 if receipt['status']!='COMPLETE' or receipt['task']!=t:raise ValueError('Incomplete/changed diagnostic task')
 out=Path(t['argv'][t['argv'].index('--output-dir')+1]);result=json.loads((out/'receipt.json').read_text())
 if result['status']!='PASS' or result['parent_sha256']!=parent_sha or result['generation']['attempts']!=12672:raise ValueError('Unexpected raw generation')
 raw=out/'raw_intervention_proposals.npz'
 with np.load(raw,allow_pickle=False) as z:
  ids=z['structure_ids'].astype(str);masks=z['masks'];n=z['n_target']
  if str(z['parent_sha256'].item())!=parent_sha or masks.shape!=(len(ids),48) or not np.isin(masks,[0,1]).all() or not np.array_equal(48-masks.sum(1),n):raise ValueError('Invalid raw binary parent/geometry/count')
  codes=(masks.astype(np.int64)*(1<<np.arange(47,-1,-1,dtype=np.int64))).sum(1)
  expected=np.array([f'mask48_{int(v):012x}' for v in codes])
  if not np.array_equal(ids,expected):raise ValueError('Raw IDs differ from canonical mask code')
  all_ids.append(ids);all_masks.append(masks);attempts+=len(ids)
 records.append({'task_id':t['id'],'task_claim_sha256':sha(claim),'output':str(raw),'output_sha256':sha(raw),'receipt_sha256':sha(out/'receipt.json')})
ids=np.concatenate(all_ids);masks=np.concatenate(all_masks);unique,first=np.unique(ids,return_index=True);masks=masks[first]
keys=_canonical_vacancy(1-masks,perms)
expected=np.array([''.join(map(str,m)) for m in masks])
if not np.array_equal(keys,expected):raise ValueError('Raw diagnostic masks not canonical')
pool=root/'raw_valid_canonical_union.npz'
with pool.open('xb') as f:np.savez_compressed(f,structure_ids=unique,masks=masks,occupancy=1-masks,parent_sha256=np.asarray(parent_sha))
receipt={'status':'ALL40_RAW_DIAGNOSTICS_FROZEN_BEFORE_ORACLE','created_unix':time.time(),'parent_sha256':parent_sha,'raw_attempts':attempts,'unique_valid_orbits':len(unique),'tasks':records,'pool':str(pool),'pool_sha256':sha(pool),'task_file_sha256':sha(taskfile),'source_sha256':sha(__file__),'no_energy_filtering':True,'no_protected_energies_read':True}
with (root/'raw_union_receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ['tasks']},indent=2))

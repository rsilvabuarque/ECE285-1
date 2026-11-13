from pathlib import Path
import json,hashlib,time,csv
import numpy as np
from cathode_cvae.runtime import require_compute_allocation
from cathode_cvae.adaptive_oracle import register_pool
require_compute_allocation()
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);parser.add_argument('--boundary-dir',type=Path,required=True);args=parser.parse_args();r=args.run_root.resolve();b=r.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
boundary=args.boundary_dir.resolve();br=json.loads((boundary/'receipt.json').read_text())
if br['status']!='FROZEN_ADAPTIVE_CANONICAL_BOUNDARY':raise ValueError('Unexpected boundary')
for k in ('parent','history','protected'):
 if sha(br[k]['path'])!=br[k]['sha256']:raise ValueError('Boundary binding changed')
known=set(json.loads(Path(br['history']['path']).read_text()));protected=set(json.loads(Path(br['protected']['path']).read_text()))
if not protected<=known:raise ValueError('Protected not within known')
rr=json.loads((r/'raw_union_receipt.json').read_text());raw=Path(rr['pool'])
if rr['status']!='ALL40_RAW_DIAGNOSTICS_FROZEN_BEFORE_ORACLE' or sha(raw)!=rr['pool_sha256'] or rr['parent_sha256']!=br['parent']['sha256']:raise ValueError('Raw freeze mismatch')
with np.load(raw,allow_pickle=False) as z:
 ids=z['structure_ids'].astype(str);masks=z['masks'];occ=z['occupancy'];parent_sha=str(z['parent_sha256'].item())
keys=np.array([''.join(map(str,m)) for m in masks]);elig=np.array([k not in known for k in keys]);eligible_ids=set(ids[elig]);idstatus={s:('protected' if k in protected else 'known' if k in known else 'eligible') for s,k in zip(ids,keys)}
pool=r/'eligible_diagnostic_union.npz'
with pool.open('xb') as f:np.savez_compressed(f,structure_ids=ids[elig],masks=masks[elig],occupancy=occ[elig],parent_sha256=np.asarray(parent_sha))
rows=[]
for t in json.loads((r/('diagnostic_tasks.json' if (r/'diagnostic_tasks.json').exists() else 'diagnostic_tasks_without_entropy.json')).read_text())['tasks']:
 out=Path(t['argv'][t['argv'].index('--output-dir')+1]);family_seed=out.name;family,seed=family_seed.rsplit('_',1)
 with np.load(out/'raw_intervention_proposals.npz',allow_pickle=False) as z:
  statuses=np.array([idstatus[x] for x in z['structure_ids'].astype(str)])
  for n in np.unique(z['n_target']):
   for bin_ in np.unique(z['requested_bin']):
    for intervention in np.unique(z['intervention']):
     take=(z['n_target']==n)&(z['requested_bin']==bin_)&(z['intervention']==intervention)
     sid=z['structure_ids'][take].astype(str);status=statuses[take]
     rows.append(dict(family=family,seed=int(seed),n_target=int(n),requested_bin=str(bin_),intervention=str(intervention),all_attempts=int(take.sum()),protected_attempts=int((status=='protected').sum()),known_nonprotected_attempts=int((status=='known').sum()),eligible_attempts=int((status=='eligible').sum()),unique_eligible_orbits=len(set(sid[status=='eligible']))))
with (r/'attempt_exclusions.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
result=dict(status='FROZEN_ELIGIBLE_UNION_BEFORE_ORACLE',raw_union_receipt_sha256=sha(r/'raw_union_receipt.json'),boundary_receipt_sha256=sha(boundary/'receipt.json'),parent_sha256=parent_sha,raw_attempts=sum(x['all_attempts'] for x in rows),raw_unique=len(ids),eligible_unique=int(elig.sum()),known_excluded_unique=int((~elig).sum()),protected_excluded_unique=sum(k in protected for k in keys),eligible_input=str(pool),eligible_input_sha256=sha(pool),denominator_table_sha256=sha(r/'attempt_exclusions.csv'),source_sha256=sha(__file__),policy='Unique outputs outside immutable base known/protected scored; every raw attempt retained in denominator; no energies used for eligibility')
with (r/'eligibility_receipt.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
for model in ('uma','mace'):
 q=r/f'oracle_{model}';q.mkdir(exist_ok=False)
 reg=register_pool(q,pool,br['parent']['path'],models=(model,),batch_size=32)
 print(model,reg,flush=True)
print(json.dumps(result,indent=2),flush=True)

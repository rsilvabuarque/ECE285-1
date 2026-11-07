import argparse,csv,json,time
from pathlib import Path
import numpy as np
from cathode_cvae.data import sha256
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();b=a.root/'runs/overnight_20261107/benchmark';rows=[];keys={};receipts=[]
for fit in [20261106,20261107,20261108]:
 d=b/f'replicate_{fit}';r=json.loads((d/'receipt.json').read_text());receipts.append(r)
 with (d/'attempts.csv').open() as f:rows.extend(csv.DictReader(f))
 with np.load(d/'candidates.npz') as z:
  for sid,m in zip(z['structure_ids'].astype(str),z['masks']):keys[sid]=m
assert len(rows)==41472 and len({r['attempt_id'] for r in rows})==len(rows)
with (b/'attempts.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
ids=sorted(keys);masks=np.array([keys[s] for s in ids]);np.savez_compressed(b/'candidates.npz',structure_ids=np.array(ids),masks=masks,occupancy=1-masks,n_Li=48-masks.sum(1),n_atoms=192-masks.sum(1),parent_sha256=np.asarray(sha256(a.root/'artifacts/current/core/data/parents/O3_N48_pilot.npz')))
(b/'freeze_receipt.json').write_text(json.dumps(dict(status='FROZEN_BEFORE_MEASURED_RESULT_SELECTION',attempts=len(rows),unique_geometries=len(ids),replicate_receipts=receipts,combined_unix=time.time(),outputs={n:sha256(b/n) for n in ['attempts.csv','candidates.npz']}),indent=2)+'\n');print('COMBINED',len(rows),len(ids))

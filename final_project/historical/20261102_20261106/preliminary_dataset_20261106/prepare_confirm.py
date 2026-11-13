"""Freeze optimizer choices and materialize new independent confirmation inputs."""
import csv,hashlib,json,os
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write,read
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
out=HERE/'data/confirm_inputs';out.mkdir(exist_ok=True);rng=np.random.default_rng(20261107)
records=[json.loads(f.read_text()) for f in (HERE/'data/calibration').glob('*/*.json')];assert len(records)==100 and all(r['status']!='exception' for r in records)
models=['uma','mace','sevennet','chgnet','m3gnet'];choices={}
for m in models:
    arms=[]
    for optimizer,factor in [('FIRE',1.),('FIRE',.25),('LBFGS',1.),('LBFGS',.25)]:
        rs=[r for r in records if r['model']==m and r['optimizer']==optimizer and r['factor']==factor];assert len(rs)==5
        successes=[r for r in rs if r['converged']]
        arms.append(dict(optimizer=optimizer,factor=factor,successes=len(successes),median_calls=float(np.median([r['calculator_calls'] for r in successes])) if successes else 1e30))
    best=sorted(arms,key=lambda a:(-a['successes'],a['median_calls']))[0];choices[m]=dict(selected=best,arms=arms)
rows=list(csv.DictReader((HERE/'data/mask_manifest.csv').open()));core=[r for r in rows if r['parent']=='O3_N48_pilot'];chosen=[]
counts=[2,3,4,6,8,10,12,14,16,20,24,28,30,32,34,36,38,40,42,46]
with np.load(HERE/'data/parents/O3_N48_pilot.npz') as z:p={k:z[k] for k in z.files}
tasks=[]
for n in counts:
    pool=[r for r in core if int(r['n_Li'])==n and r['pilot_lineage_holdout']!='True'];assert pool
    r=pool[int(rng.integers(len(pool)))];i=int(r['parent_index']);v=p['masks'][i];keep=np.ones(len(p['numbers']),bool);keep[p['li_indices'][v.astype(bool)]]=False
    a=Atoms(numbers=p['numbers'][keep],positions=p['positions_A'][keep],cell=p['cell_A'],pbc=True);a.arrays['parent_slot']=np.flatnonzero(keep)
    path=out/(r['structure_id']+'.extxyz');write(path,a);restored=read(path);assert np.array_equal(restored.arrays['parent_slot'],np.flatnonzero(keep))
    digest=hashlib.sha256(path.read_bytes()).hexdigest();chosen.append(r)
    for m in models:
        arm=choices[m]['selected'];tasks.append(dict(task_id=f"confirm__{m}__{r['structure_id']}",stage='confirm',model=m,parent=r['parent'],structure_id=r['structure_id'],n_Li=n,input_file=str(path),input_sha256=digest,cell=True,optimizer=arm['optimizer'],factor=arm['factor'],max_steps=500,stride=5))
with (HERE/'data/confirm_tasks.jsonl').open('w') as f:
    for t in tasks:f.write(json.dumps(t)+'\n')
(HERE/'data/optimizer_selection.json').write_text(json.dumps(dict(choices=choices,confirmation_parents=chosen,criterion='Physicalsuccess thenmediancallsamong5calibrationcases; confirmationbeforevariableproduction',old_pilot='Preserved; freshFIRE/Ncontrolmakesphysical-only stoppingcomparable'),indent=2)+'\n')
print('CONFIRM_PREPARED',len(tasks),{m:choices[m]['selected'] for m in models},flush=True)

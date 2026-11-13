"""Select independent parents by composition and declared acquisition strata."""
import csv,hashlib,json,os,time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write,read
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
cfg=json.loads((HERE/'config.json').read_text());models=cfg['models'];source={}
for m in models:
    gate=HERE/'validation'/f'scf_{m}.json'
    while not gate.exists():time.sleep(20)
    assert json.loads(gate.read_text())['status']=='PASS'
    with np.load(HERE/'data'/f'learning_O3_N48_pilot_{m}.npz') as z:source[m]={k:z[k] for k in z.files}
with np.load(HERE/'data/parents/O3_N48_pilot.npz') as z:p={k:z[k] for k in z.files}
rows=sorted([r for r in csv.DictReader((HERE/'data/mask_manifest.csv').open()) if r['parent']=='O3_N48_pilot'],key=lambda r:int(r['parent_index']))
splits={r['structure_id']:r['split'] for r in csv.DictReader((HERE/'splits/lineage_splits.csv').open())}
rng=np.random.default_rng(20261108);nLi=source['uma']['n_Li'];disagreement=np.zeros(len(rows));quotas={}
for n in sorted(set(nLi)):
    ids=np.flatnonzero(nLi==n);ranks=[]
    for m in models:
        order=np.argsort(source[m]['energy_eV'][ids],kind='stable');rank=np.empty(len(ids));rank[order]=np.arange(len(ids))/max(1,len(ids)-1);ranks.append(rank)
    disagreement[ids]=np.std(ranks,axis=0)
    avail=[i for i in ids if rows[i]['pilot_lineage_holdout']!='True']
    if len(avail)>=12:quotas[int(n)]=min(22,len(avail))
while sum(quotas.values())<1000:
    for n in rng.permutation(list(quotas)):
        capacity=sum(int(r['n_Li'])==n and r['pilot_lineage_holdout']!='True' for r in rows)
        if quotas[int(n)]<capacity:quotas[int(n)]+=1
        if sum(quotas.values())==1000:break
selected=[]
for n,quota in sorted(quotas.items()):
    available=np.array([i for i in range(len(rows)) if int(rows[i]['n_Li'])==n and rows[i]['pilot_lineage_holdout']!='True']);rng.shuffle(available)
    random_count=quota//2;low_count=(quota-random_count+1)//2
    picked=[(int(i),'uniform_random') for i in available[:random_count]];remaining=np.setdiff1d(available,[i for i,_ in picked])
    low=remaining[np.argsort(source['uma']['energy_eV'][remaining],kind='stable')[:low_count]];picked.extend((int(i),'low_UMA_energy') for i in low)
    remaining=np.setdiff1d(remaining,low);high=remaining[np.argsort(-disagreement[remaining],kind='stable')[:quota-len(picked)]];picked.extend((int(i),'high_teacher_rank_disagreement') for i in high)
    selected.extend(picked)
assert len(selected)==1000 and len({i for i,_ in selected})==1000
out=HERE/'data/relax_inputs';out.mkdir(exist_ok=True);tasks=[];manifest=[]
for i,acquisition in selected:
    r=rows[i];v=p['masks'][i];keep=np.ones(len(p['numbers']),bool);keep[p['li_indices'][v.astype(bool)]]=False
    a=Atoms(numbers=p['numbers'][keep],positions=p['positions_A'][keep],cell=p['cell_A'],pbc=True);a.arrays['parent_slot']=np.flatnonzero(keep)
    a.info.update(structure_id=r['structure_id'],n_Li=int(r['n_Li']),canonical_mask=r['canonical_mask'])
    path=out/(r['structure_id']+'.extxyz');write(path,a);restored=read(path);assert np.array_equal(restored.arrays['parent_slot'],np.flatnonzero(keep))
    digest=hashlib.sha256(path.read_bytes()).hexdigest();entry=dict(r,acquisition=acquisition,rank_disagreement=float(disagreement[i]),split=splits[r['structure_id']],input_file=str(path),input_sha256=digest);manifest.append(entry)
    for m in models:
        tasks.append(dict(task_id=f"fixed__{m}__{r['structure_id']}",stage='fixed',model=m,parent=r['parent'],structure_id=r['structure_id'],canonical_mask=r['canonical_mask'],n_Li=int(r['n_Li']),parent_index=i,split=entry['split'],acquisition=acquisition,input_file=str(path),input_sha256=digest,cell=False,optimizer='FIRE',factor=.25,max_steps=500,stride=5))
with (HERE/'data/relaxation_parent_manifest.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
path=HERE/'data/fixed_tasks.jsonl';tmp=path.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(t)+'\n' for t in tasks));tmp.replace(path)
(HERE/'data/relaxation_design.json').write_text(json.dumps({'independent_parents':1000,'acquisition':'approximately50%uniform,25%lowUMA,25%highcross-modelrankdisagreement,balancedcomposition','namespaces':'Rankdisagreementcalculatedwithincomposition; modelenergyzerosnotpooled','lineage':'Samefrozenparentgroup/splitforall5modelsandallframes; existingpilotmatchedorbits excluded','scope':'NewindependentgeometrieswithfullEFS; noDFT, groundstateorMDstabilityclaim','production_fixed':'FIRE .05eV/A,500steps,.15globalgeneralizedstepcap,snapshotstride5'},indent=2)+'\n')
print('RELAXATIONS_PREPARED',len(tasks),'tasks',len(manifest),'parents',flush=True)

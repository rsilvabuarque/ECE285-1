"""Freeze equal-attempt proposals before new oracle calls; retain every duplicate."""
import csv,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from model import MaskCVAE
from sampling import top_count,conditional_count
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid') and torch.cuda.is_available()
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
with np.load(ROOT/'data/parents/O3_N48_pilot.npz') as z:parent={k:z[k] for k in z.files}
with np.load(ROOT/'data/learning_O3_N48_pilot_uma.npz') as z:learning={k:z[k] for k in z.files}
catalog=list(csv.DictReader((ROOT/'data/mask_manifest.csv').open()));catalog=[r for r in catalog if r['parent']=='O3_N48_pilot']
known={r['canonical_mask']:r['structure_id'] for r in catalog};split={r['structure_id']:r['split'] for r in csv.DictReader((ROOT/'splits/lineage_splits.csv').open())}
train_canon={r['canonical_mask'] for r in catalog if split[r['structure_id']]=='train'}
inv=np.argsort(parent['permutations'],axis=1);powers=1<<np.arange(47,-1,-1,dtype=np.int64);new={};attempts=[];B=512
def seed_for(text):return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'little')%(2**32)
def record(occupancy,method,seed,n,run='',condition='',mode='topk',source_seeds=None,logits=None):
    if logits is not None:
        probability=1/(1+np.exp(-logits));expected=probability.sum(axis=1);threshold=(logits>=0).sum(axis=1)
        draws=(np.random.default_rng(seed_for(f'rawcounts_{run}_{seed}_{n}')).random(logits.shape)<probability).sum(axis=1)
    for j,o in enumerate(occupancy):
        v=1-o;code=int(np.min(v[inv].astype(np.int64)@powers));c=format(code,'048b');sid=known.get(c,f'gen48_{code:012x}')
        if c not in known and c not in new:new[c]=(sid,v.copy())
        attempts.append(dict(method=method,training_run=run,seed=seed,n_Li=n,attempt=j,condition=condition,sampling=mode,canonical_mask=c,scoring_structure_id=sid,novel_to_training=c not in train_canon,absent_from_catalog=c not in known,source_training_parent='' if source_seeds is None else source_seeds[j],raw_expected_Li_count=None if logits is None else float(expected[j]),raw_threshold_Li_count=None if logits is None else int(threshold[j]),raw_Bernoulli_Li_count=None if logits is None else int(draws[j])))
configs=sorted((HERE/'runs').glob('*/configuration.json'));assert len(configs)==16
for path in configs:
    cfg=json.loads(path.read_text());run=path.parent.name;ckpt=torch.load(path.parent/'best.pt',map_location='cuda',weights_only=False)
    model=MaskCVAE(condition_dim=cfg['condition_dim']).cuda();model.load_state_dict(ckpt['state_dict']);model.eval()
    counts=[12,24,36] if cfg['split']=='main' else [9,15,21,27,33,39]
    for n in counts:
        seed=cfg['seed'];rng=np.random.default_rng(seed_for(f'{run}_{n}'));z=torch.tensor(rng.standard_normal((B,8)),dtype=torch.float32,device='cuda')
        options=[('low',1.)] if cfg['kind']=='cvae' else [('none',0.)]
        if cfg['kind']=='cvae' and cfg['teacher']=='uma' and 'beta01' not in run and cfg['split']=='main':options.append(('high',0.))
        for condition,value in options:
            c=np.tile([n/48,value],(B,1)).astype(np.float32)
            if cfg['kind']=='composition':c=c[:,:1]
            if cfg['kind']=='vae':c=c[:,:0]
            with torch.no_grad():logits=model.decode(z,torch.tensor(c,device='cuda')).cpu().numpy()
            method=f"{cfg['teacher']}_{cfg['kind']}"+('_beta01' if 'beta01' in run else '')+('_composition_holdout' if cfg['split']!='main' else '')+('_high' if condition=='high' else '')
            record(top_count(logits,n,np.random.default_rng(seed_for(run+str(n)+'ties'))),method,seed,n,run,condition,'topk',logits=logits)
            if cfg['teacher']=='uma' and cfg['kind']=='cvae' and cfg['split']=='main' and condition=='low':
                record(conditional_count(logits,n,np.random.default_rng(seed_for(run+str(n)+'countDP'))),method+'_countDP',seed,n,run,condition,'exact_count_Bernoulli',logits=logits)
for seed in [20261106,20261107,20261108]:
    for n in [12,24,36]:
        rng=np.random.default_rng(seed_for(f'baseline{seed}_{n}'));u=np.zeros((B,48),np.uint8)
        for o in u:o[rng.choice(48,n,replace=False)]=1
        record(u,'uniform',seed,n,mode='uniform_fixed_count')
        train_ids=np.array([i for i,s in enumerate(learning['structure_ids']) if split[s]=='train' and learning['n_Li'][i]==n]);assert len(train_ids)
        values=learning['energy_eV'][train_ids]/learning['n_atoms'][train_ids];q=np.sort(values)[int(np.ceil(.2*len(values)))-1];low=train_ids[values<=q]
        ii=rng.choice(low,B);mutated=learning['occupancy'][ii].copy()
        for o in mutated:
            swaps=int(rng.integers(1,4));ones=rng.choice(np.flatnonzero(o),swaps,replace=False);zeros=rng.choice(np.flatnonzero(1-o),swaps,replace=False);o[ones]=0;o[zeros]=1
        record(mutated,'low_train_mutation',seed,n,mode='one_to_three_swaps',source_seeds=learning['structure_ids'][ii])
        p=np.clip(learning['occupancy'][low].mean(axis=0),1e-5,1-1e-5);logits=np.tile(np.log(p/(1-p)),(B,1));record(conditional_count(logits,n,rng),'low_train_site_frequency',seed,n,mode='exact_count_Bernoulli')
# Separate extrapolation uniform control, matching the single held-composition fit.
for n in [9,15,21,27,33,39]:
    rng=np.random.default_rng(seed_for(f'holdout_uniform_{n}'));u=np.zeros((B,48),np.uint8)
    for o in u:o[rng.choice(48,n,replace=False)]=1
    record(u,'uniform_composition_holdout',20261106,n,mode='uniform_fixed_count')
with (HERE/'proposal_attempts.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(attempts[0]));w.writeheader();w.writerows(attempts)
items=sorted(new.items());masks=np.array([v[1][1] for v in items],np.uint8);ids=np.array([v[1][0] for v in items])
path=ROOT/'data/generated_candidates.npz';np.savez_compressed(path,masks=masks,structure_ids=ids,numbers=parent['numbers'],positions_A=parent['positions_A'],cell_A=parent['cell_A'],li_indices=parent['li_indices'])
digest=hashlib.sha256(path.read_bytes()).hexdigest();tasks=[]
for model in ['uma','mace','sevennet','chgnet','m3gnet']:
    for start in range(0,len(masks),128):tasks.append(dict(task_id=f'generated__{model}__{start:06d}',stage='generated',model=model,input_npz=str(path),input_sha256=digest,start=start,stop=min(start+128,len(masks)),base_parent='O3_N48_pilot'))
(ROOT/'data/generated_tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
(HERE/'generation_receipt.json').write_text(json.dumps({'attempts':len(attempts),'new_unique_orbits':len(masks),'catalog_orbit_hits':sum(not r['absent_from_catalog'] for r in attempts),'novel_to_training_attempts':sum(r['novel_to_training'] for r in attempts),'attempt_quota':B,'primary_compositions':[12,24,36],'conditions':'low/highsamepriorlatentdrawswithinrun; coefficients/splits/checkpointsfrozenbeforescoring','count_projection':'Top-nfixedcountMAPvsDPexactconditionalBernoulli; allmethodsretaininvalid/duplicateaccounting','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'new_candidate_sha256':digest,'cache':'Existingcatalogenergiesusedforcataloghits; onefreshcommongeometrypernewcanonicalorbit, allfivefrozenteachers'},indent=2)+'\n')
print('GENERATED',len(attempts),'attempts',len(masks),'neworbits',flush=True)

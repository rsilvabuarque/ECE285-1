"""Scoreable new-mask geometry proposals; desired properties use training rows only."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_masses
from geometry_model import DisplacementModel
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid') and torch.cuda.is_available()
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;rng=np.random.default_rng(20261402)
with np.load(ROOT/'data/parents/O3_N48_pilot.npz') as z:parent={k:z[k] for k in z.files}
with np.load(ROOT/'data/generated_candidates.npz') as z:generated={k:z[k] for k in z.files}
nLi=48-generated['masks'].sum(1);selected=[]
for n,count in [(12,43),(24,42),(36,43)]:selected.extend(rng.choice(np.flatnonzero(nLi==n),count,replace=False).tolist())
selected=np.array(selected);occupancy=1-generated['masks'][selected];numbers=parent['numbers'];N=len(numbers);B=len(selected)
presence=np.ones((B,N),bool);presence[:,parent['li_indices']]=occupancy.astype(bool);ctx_models={};predictions={'deletion':np.zeros((B,N,3))};configs={}
for kind in ['regressor','cvae']:
    path=HERE/'geometry_runs'/f'{kind}_20261400'/'best.pt';ck=torch.load(path,map_location='cuda',weights_only=False)
    # JSON normalizes integer composition keys consistently across serializers.
    cfg=json.loads((path.parent/'configuration.json').read_text());configs[kind]=cfg
    props=np.array([cfg['target_conditions'][str(int(n))]['standardized'] for n in nLi[selected]])
    ctx=np.concatenate([occupancy,(nLi[selected]/48)[:,None],np.tile([0,0,1],(B,1)),np.tile(np.eye(3).reshape(1,9),(B,1)),props],1).astype(np.float32)
    model=DisplacementModel(kind=kind).cuda();model.load_state_dict(ck['state_dict']);model.eval();c=torch.tensor(ctx,device='cuda')
    draws=range(3) if kind=='cvae' else [0]
    for draw in draws:
        z=torch.tensor(rng.normal(size=(B,16)),dtype=torch.float32,device='cuda')
        with torch.no_grad():u=model.decode(z if kind=='cvae' else None,c).cpu().numpy().reshape(B,N,3)*cfg['displacement_scale_A']
        mass=atomic_masses[numbers][None,:]*presence;translation=(u*mass[:,:,None]).sum(1)/mass.sum(1)[:,None];u=u-translation[:,None,:];u[~presence]=0
        predictions[f'cvae_draw{draw}' if kind=='cvae' else kind]=u
positions=[];masks=[];ids=[];methods=[];source_ids=[];valid=[];minimum=[]
for i,idx in enumerate(selected):
    for method,u in predictions.items():
        xyz=parent['positions_A']+u[i];keep=presence[i];a=Atoms(numbers=numbers[keep],positions=xyz[keep],cell=parent['cell_A'],pbc=True)
        distances=a.get_all_distances(mic=True);np.fill_diagonal(distances,np.inf);dmin=float(distances.min());ok=bool(np.isfinite(xyz).all() and dmin>=.6)
        positions.append(xyz);masks.append(keep);ids.append(f"decoded_{generated['structure_ids'][idx]}__{method}");methods.append(method);source_ids.append(str(generated['structure_ids'][idx]));valid.append(ok);minimum.append(dmin)
path=ROOT/'data/decoded_geometry_candidates.npz';np.savez_compressed(path,positions_parent_A=positions,atom_presence=masks,cell_A=np.tile(parent['cell_A'],(len(ids),1,1)),numbers_parent=numbers,structure_ids=ids,source_mask_ids=source_ids,methods=methods,valid=valid,minimum_separation_A=minimum)
digest=hashlib.sha256(path.read_bytes()).hexdigest();tasks=[]
for model in ['uma','mace','sevennet','chgnet','m3gnet']:
    for start in range(0,len(ids),32):tasks.append(dict(task_id=f'decoded_score__{model}__{start:05d}',stage='decoded_score',model=model,input_npz=str(path),input_sha256=digest,start=start,stop=min(start+32,len(ids))))
(ROOT/'data/decoded_score_tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
(HERE/'geometry_generation_receipt.json').write_text(json.dumps({'masks':B,'geometries':len(ids),'methods':list(predictions),'valid':sum(valid),'input_sha256':digest,'selection':'128freshnovelmaskorbits, stratifiedLi12/24/36, chosenbeforetheirteacherenergieswereused.','conditions':'Attainedtrainingendpointpropertytuplepercomposition; noheldoutoraclecondition. Suppliedoriginalparentcell; no cellprediction.','gauge':'Predictedglobalmassweightedtranslationremoved; noforce/energygradientprojection.','evaluation':'640geometriesunderall5teachers; 3CVAEdrawsarepriorstochasticity, not3trainingseeds. Oneprototypefitperkind.'},indent=2)+'\n')
print('GEOMETRIES_GENERATED',len(ids),'valid',sum(valid),flush=True)

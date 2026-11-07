"""Prespecified equal-attempt benchmark; energy evaluation happens afterwards."""
import argparse,csv,json,hashlib,os,time
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(True)
from cathode_cvae.learning.generate_mask import load_model,resolve_request,attempt_records
from cathode_cvae.learning.train_mask import canonicalize
from cathode_cvae.conditions import load_schema,classify
from cathode_cvae.data import sha256
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--replicates',type=int,nargs='+',default=[0,1,2]);p.add_argument('--count',type=int,default=256);p.add_argument('--device',default='cuda');a=p.parse_args();root=a.root;run=root/'runs/overnight_20261107';schema=load_schema(run/'data/condition_schema_uma.json')
parentfile=root/'artifacts/current/core/data/parents/O3_N48_pilot.npz';parent_sha=sha256(parentfile)
with np.load(parentfile) as z:inv=torch.tensor(np.argsort(z['permutations'],axis=1),device=a.device)
powers=1<<torch.arange(47,-1,-1,device=a.device,dtype=torch.long)
with np.load(root/'artifacts/current/improvement_v1/training_round0.npz') as z:d={k:z[k] for k in z.files}
train=d['split']=='train';labels=classify(schema,d['n_Li'],d['energy_eV_atom']);original_source=root/'artifacts/current/improvement_v1/training_round0.npz'
def rng_seed(*tokens):return int.from_bytes(hashlib.sha256('|'.join(map(str,tokens)).encode()).digest()[:4],'little')
def codes_for(x):
 t=torch.tensor(x,device=a.device,dtype=torch.float32)
 return canonicalize(t,inv)[1].tolist()

for rep in a.replicates:
 fit=20261106+rep;out=run/f'benchmark/replicate_{fit}';out.mkdir(parents=True,exist_ok=False);records=[];frozen=[];all_codes=[]
 for method in ['multibin','binary','composition','uniform','mutation_bin','mutation_low']:
  ck=(run/f'fits/multibin_v2_{fit}/best.pt' if method=='multibin' else root/f'artifacts/current/improvement_v1/runs/{"ar_adapt_cvae" if method=="binary" else "ar_adapt_composition"}_{fit}/best.pt') if method in ['multibin','binary','composition'] else None
  if ck:
   model,cfg=load_model(ck,a.device);cksha=sha256(ck);frozen.append(dict(method=method,checkpoint=str(ck),sha256=cksha));model.eval()
  else:cksha='not-applicable';cfg={}
  for n in [12,24,36]:
   for requested in ['low','middle','high']:
    seed=rng_seed('new-assessment-v1',method,fit,n,requested);rng=np.random.default_rng(seed)
    if ck:
     vector,_,effective,schemaid,_=resolve_request(cfg,n,requested,None,schema)
     c=torch.tensor(vector,device=a.device).repeat(a.count,1);g=torch.Generator(device=a.device).manual_seed(seed)
     with torch.no_grad():x=model.sample(c,n,generator=g).cpu().numpy().astype(np.uint8)
    elif method=='uniform':
     x=np.zeros((a.count,48),np.uint8)
     for row in x:row[rng.choice(48,n,replace=False)]=1
     effective='unconditioned';schemaid=schema['identity']
    else:
     seed_bin=requested if method=='mutation_bin' else 'low';source=d['occupancy'][train&(d['n_Li']==n)&(labels==seed_bin)]
     if len(source)==0:raise ValueError('Requested mutation seed bin has no protected training support')
     x=source[rng.integers(len(source),size=a.count)].copy()
     for row in x:
      swaps=int(rng.integers(1,4));ones=rng.choice(np.flatnonzero(row),swaps,replace=False);zeros=rng.choice(np.flatnonzero(1-row),swaps,replace=False);row[ones]=0;row[zeros]=1
     effective=f'{seed_bin}-seed; one-to-three swaps';schemaid=schema['identity']
    if not np.all(x.sum(1)==n):raise ValueError('Generator violated exact target count')
    codes=codes_for(x);keys=[format(int(k),'048b') for k in codes];ids=[f'mask48_{int(k):012x}' for k in codes];all_codes.extend(codes)
    rr=attempt_records(a.count,ids,keys,method=method,fit_seed=fit if ck else '',sampling_seed=seed,n_target=n,requested_bin=requested,effective_model_class=effective,schema_id=schemaid,checkpoint_sha256=cksha,parent_sha256=parent_sha,geometry_protocol='pristine-derived deletion; unrelaxed; exact canonical parent coordinates')
    for i,row in enumerate(rr):row.update(attempt_id=f'benchmark-{fit}-{method}-{n}-{requested}-{i}',replicate_id=fit,allocation_run_id=f'fixed-balanced-{fit}-{method}-{n}',allocation_policy='fixed-balanced-bin-budget',allocation_weight=1/3,seed_source_sha256=sha256(original_source) if method.startswith('mutation') else '')
    records.extend(rr);print('GENERATED',fit,method,n,requested,flush=True)
  if ck:
   if sha256(ck)!=cksha:raise ValueError('Checkpoint mutated during assessmentgeneration')
   del model
 unique=np.unique(all_codes);masks=((unique[:,None]>>np.arange(47,-1,-1))&1).astype(np.uint8);ids=np.array([f'mask48_{int(k):012x}' for k in unique]);np.savez_compressed(out/'candidates.npz',structure_ids=ids,masks=masks,occupancy=1-masks,n_Li=48-masks.sum(1),n_atoms=192-masks.sum(1),parent_sha256=np.asarray(parent_sha))
 with (out/'attempts.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 receipt=dict(status='GENERATED_AND_FROZEN_BEFORE_SCORING',fit_seed=fit,attempts=len(records),unique_geometries=len(unique),count_per_group=a.count,conditions=schema['identity'],generator_sha256=sha256(__file__),frozen_checkpoints=frozen,generated_unix=time.time(),outputs={n:sha256(out/n) for n in ['attempts.csv','candidates.npz']},no_energy_filtering=True,benchmark_scope='Sixmethods equalattempts, new multibin vs existing H0 controls; fixed pretraineddata and precision; newassesshistoryincludesindependentreference')
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('REPLICATE_COMPLETE',fit,len(records),len(unique),flush=True)

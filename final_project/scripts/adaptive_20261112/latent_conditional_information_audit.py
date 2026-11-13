"""Posthoc conditional latent probe, declared after fixed-condition visualization.

This reads TRAIN/VAL only. Rescaling deterministic mu may recover information
that is completely buried under sampled posterior noise; this is not generation.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from cathode_cvae.runtime import require_compute_allocation
from cathode_cvae.conditions import classify
from cathode_cvae.graph import PeriodicGraph
from cathode_cvae.learning.generate_mask import load_model,resolve_request
from cathode_cvae.learning.models import make_model
from cathode_cvae.learning.train_mask import canonicalize
from cathode_cvae.learning.physical_auxiliary import prepare_targets
require_compute_allocation();torch.set_num_threads(4);import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args();p=args.run_root.resolve();b=p.parent;out=p/'conditional_information_audit';out.mkdir(exist_ok=False);sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest()
protocol=dict(status='POSTHOC_DECLARED_AFTER_FIXED_C_VISUAL_BEFORE_COMPUTATION',created_unix=time.time(),observation='Fixed-condition scatter shows tiny but C_s-correlated posterior-mean variation; pooled TRAIN scaling can suppress its fitted ridge coefficient',trained='All8mean-phasefamilies x5seeds, all9composition/energycells; TRAIN-only centering/scaling of mu separately within eachcell; ridgealpha1(interceptunpenalized), genuineandwithin-cell shuffled TRAINtargets, genuineVAL; no tuning',random_initialization='SamegraphD2architecture and savedTorchfitseed,5seeds, fixedn24/middle TRAIN/VALcell. Random initialization is not trained or selected',structural_audit='Fixedn24/middleVALpooledfeature/muvariability inFP32 and read-only FP64 forward; random5seeds and firstseedtrained D2/sampleλ10/meanλ10; not primaryprecision',noise='Apply mu-fitted probe to paired posterior samples and independent standardNormalprior only as mismatchdiagnostics; actualgeneration/oracleyields remainprimary',no_new_training=True,no_new_oracle=True,no_TEST=True,source_sha256=sha(__file__))
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
families=['ar','d2']+[f'lambda10_{m}_d2' for m in ['latent','condition','shuffled']]+[f'mean_lambda10_{m}_d2' for m in ['latent','condition','shuffled']]
data=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project/artifacts/current/improvement_v1/training_round0.npz');parent=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project/artifacts/current/core/data/parents/O3_N48_pilot.npz')
reference=b/'initial_fits/d2_2026111301/last_update.pt';_,cfg0=load_model(reference,'cpu');graph=PeriodicGraph.from_dict(cfg0['graph'])
with np.load(data,allow_pickle=False) as z:
 active=np.flatnonzero(np.isin(z['split'].astype(str),['train','val']));roles=z['split'][active].astype(str);x=z['occupancy'][active];counts=(z['n_target'] if 'n_target' in z else z['n_Li'])[active];energy=z['energy_eV_atom'][active];ids=z['structure_ids'][active]
bins=classify(cfg0['condition_schema'],counts,energy)
with np.load(parent,allow_pickle=False) as z:perms=z['permutations']
xt,_=canonicalize(torch.tensor(x,dtype=torch.float32),torch.tensor(np.argsort(perms,axis=1)))
raw,target,_,targetmeta=prepare_targets(x,counts,bins,roles,graph,mode='latent',seed=2026111701)
rows=[];struct=[];bindings=[]
def ridge(f,y):
 xx=np.column_stack((np.ones(len(f)),f));pen=np.eye(xx.shape[1]);pen[0,0]=0;return np.linalg.solve(xx.T@xx+pen,xx.T@y)
def pred(coef,f):return np.column_stack((np.ones(len(f)),f))@coef
def metrics(y,yp):
 mse=float(np.mean((y-yp)**2));zero=float(np.mean(y*y));return dict(MSE=mse,skill_vs_TRAIN_zero=1-mse/zero if zero else None,Spearman=float(spearmanr(y,yp).statistic) if np.ptp(yp)>0 and np.ptp(y)>0 else None)
def encode(model,ix,c):
 mu=[];lv=[]
 with torch.no_grad():
  for start in range(0,len(ix),512):
   jj=ix[start:start+512];m,l=model.encode(xt[jj],c[jj]);mu.append(m.numpy());lv.append(l.numpy())
 return np.concatenate(mu),np.concatenate(lv)
def structural(model,ix,c,name,seed,stage):
 if not model.latent_dim:return
 for dtype in [torch.float32,torch.float64]:
  model=model.to(dtype=dtype);captured=[]
  handle=model.encoder.output.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().numpy().astype(np.float64)))
  with torch.no_grad():m,l=model.encode(xt[ix].to(dtype),c[ix].to(dtype))
  handle.remove();pooled=np.concatenate(captured);m=m.numpy().astype(np.float64);l=l.numpy().astype(np.float64);mm=m.std(0);pp=pooled.std(0)
  struct.append(dict(family=name,seed=seed,stage=stage,dtype=str(dtype),rows=len(ix),mu_SD=mm.tolist(),mu_distinct_rows=len(np.unique(m,axis=0)),maximum_pooled_feature_SD=float(pp.max()),pooled_nonconstant_features=int((pp>0).sum()),pooled_feature_SD=pp.tolist(),RMS_posterior_sigma=np.sqrt(np.exp(l).mean(0)).tolist(),mu_Cs_Pearson=[float(np.corrcoef(m[:,k],raw[ix])[0,1]) if np.ptp(m[:,k]) else None for k in range(m.shape[1])]))
 return model.to(dtype=torch.float32)
def probe(name,seed,model,c,mu,lv,indices,stage):
 pos={int(j):k for k,j in enumerate(indices)}
 for n,label in sorted({(int(counts[j]),str(bins[j])) for j in indices}):
  tr=np.array([j for j in indices if counts[j]==n and bins[j]==label and roles[j]=='train']);va=np.array([j for j in indices if counts[j]==n and bins[j]==label and roles[j]=='val']);ti=np.array([pos[int(j)] for j in tr]);vi=np.array([pos[int(j)] for j in va])
  center=mu[ti].mean(0,dtype=np.float64);scale=mu[ti].std(0,dtype=np.float64);scale=np.where(scale>0,scale,1);ftr=(mu[ti].astype(float)-center)/scale;fv=(mu[vi].astype(float)-center)/scale;y=target[tr].astype(float);yv=target[va].astype(float)
  coef=ridge(ftr,y);rng=np.random.default_rng(2026111701+(seed-2026111301)+1009*n+97*['low','middle','high'].index(label));null=ridge(ftr,y[rng.permutation(len(y))]);eps=rng.standard_normal(mu[vi].shape);post=mu[vi]+np.exp(.5*lv[vi])*eps
  predictions={'conditional_mu':pred(coef,fv),'shuffled_TRAIN_probe':pred(null,fv),'TRAIN_zero':np.zeros(len(yv)),'mu_probe_on_posterior_sample':pred(coef,(post-center)/scale),'mu_probe_on_independent_prior':pred(coef,(eps-center)/scale)}
  for key,value in predictions.items():rows.append(dict(family=name,seed=seed,stage=stage,n_target=n,energy_bin=label,probe=key,TRAIN_rows=len(tr),VAL_rows=len(va),**metrics(yv,value),TRAIN_mu_SD_min=float(mu[ti].std(0,dtype=np.float64).min()) if model.latent_dim else 0.,TRAIN_mu_SD_max=float(mu[ti].std(0,dtype=np.float64).max()) if model.latent_dim else 0.,mean_posterior_variance=float(np.exp(lv[vi]).mean()) if model.latent_dim else None))
for family in families:
 for seed in range(2026111301,2026111306):
  ck=(b/'initial_fits'/f'{family}_{seed}'/'last_update.pt' if family in ['ar','d2'] else b/'latent_phase2/fits'/f'{family}_{seed}'/'last_update.pt' if family.startswith('lambda10_') else p/'fits'/f'{family}_{seed}'/'last_update.pt');model,cfg=load_model(ck,'cpu')
  if cfg['data_sha256']!=sha(data) or cfg['parent_sha256']!=sha(parent):raise ValueError('Lineage changed')
  condition=np.zeros((len(x),4),np.float32);condition[:,0]=counts/48
  for j,label in enumerate(['low','middle','high']):condition[:,j+1]=bins==label
  ct=torch.tensor(condition);ii=np.arange(len(x));mu,lv=encode(model,ii,ct);probe(family,seed,model,ct,mu,lv,ii,'trained');bindings.append(dict(path=str(ck),sha256=sha(ck)))
  if seed==2026111301 and family in ['d2','lambda10_latent_d2','mean_lambda10_latent_d2']:
   jj=np.flatnonzero((roles=='val')&(counts==24)&(bins=='middle'));structural(model,jj,ct,family,seed,'trained')
  print('PROBED',family,seed,flush=True)
for seed in range(2026111301,2026111306):
 ck=b/'initial_fits'/f'd2_{seed}'/'last_update.pt';_,cfg=load_model(ck,'cpu');torch.manual_seed(seed)
 model=make_model(cfg['kind'],n_sites=cfg['n_sites'],condition_dim=cfg['condition_dim'],latent_dim=cfg['latent_dim'],hidden_dim=cfg['hidden_dim'],encoder_kind=cfg['encoder_kind'],graph_data=cfg['graph'],graph_context=cfg['graph_context'],context_spec=cfg['decoder_context_spec'],remaining_count_features=cfg['remaining_count_features']).eval()
 ii=np.flatnonzero((counts==24)&(bins=='middle'));mu,lv=encode(model,ii,ct);probe('random_graph_D2',seed,model,ct,mu,lv,ii,'random_init');jj=ii[roles[ii]=='val'];structural(model,jj,ct,'random_graph_D2',seed,'random_init');print('RANDOM_PROBED',seed,flush=True)
df=pd.DataFrame(rows);df.to_csv(out/'conditional_probes_cells.csv',index=False);balanced=df[df.stage=='trained'].groupby(['family','seed','probe'])[['MSE','skill_vs_TRAIN_zero','Spearman']].mean().reset_index();balanced.to_csv(out/'conditional_probes_seed_balanced.csv',index=False);df[(df.n_target==24)&(df.energy_bin=='middle')].to_csv(out/'fixed_condition_random_comparison.csv',index=False)
(out/'structural_sensitivity.json').write_text(json.dumps(struct,indent=2,allow_nan=False)+'\n');result=dict(status='PASS',protocol_sha256=sha(out/'protocol.json'),source_sha256=sha(__file__),checkpoint_bindings=bindings,TRAIN_rows=int((roles=='train').sum()),VAL_rows=int((roles=='val').sum()),TEST_processed=False,scope='Posthoc conditional deterministic-mu information audit; unweighted balanced means over9cells,5fitseeds; randominitialization5seeds at fixedn24/middle. Does not change primary generation/oracle endpoints or demonstrate useful noisy/prior z.')
(out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print('CONDITIONAL_AUDIT_PASS',flush=True)

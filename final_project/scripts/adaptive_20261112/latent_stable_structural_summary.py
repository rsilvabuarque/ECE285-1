"""Correct only illustrative moment calculations using float64 accumulation."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cathode_cvae.runtime import require_compute_allocation
from cathode_cvae.learning.generate_mask import load_model,resolve_request
from cathode_cvae.learning.models import make_model
from cathode_cvae.learning.train_mask import canonicalize
require_compute_allocation();torch.set_num_threads(4);import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args();p=args.run_root.resolve();b=p.parent;out=p/'structural_precision_correction';out.mkdir(exist_ok=False);sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest()
reference=p/'fixed_condition_embedding/embedding_0.npz'
with np.load(reference,allow_pickle=False) as z:ids=z['structure_ids'].astype(str);Cs=z['raw_Cs']
data=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project/artifacts/current/improvement_v1/training_round0.npz');parent=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project/artifacts/current/core/data/parents/O3_N48_pilot.npz')
with np.load(data,allow_pickle=False) as z:
 lookup={s:i for i,s in enumerate(z['structure_ids'].astype(str))};ii=np.array([lookup[s] for s in ids]);x=z['occupancy'][ii]
 if not np.all(z['split'][ii].astype(str)=='val'):raise ValueError('VAL-only structural audit')
with np.load(parent,allow_pickle=False) as z:perms=z['permutations']
xt,_=canonicalize(torch.tensor(x,dtype=torch.float32),torch.tensor(np.argsort(perms,axis=1)));cases=[];rows=[]
for name,path in [('d2',b/'initial_fits/d2_2026111301/last_update.pt'),('sampled_lambda10',b/'latent_phase2/fits/lambda10_latent_d2_2026111301/last_update.pt'),('mean_lambda10',p/'fits/mean_lambda10_latent_d2_2026111301/last_update.pt')]:
 model,cfg=load_model(path,'cpu');cases.append((name,2026111301,'trained',model,cfg,path))
for seed in range(2026111301,2026111306):
 path=b/'initial_fits'/f'd2_{seed}'/'last_update.pt';_,cfg=load_model(path,'cpu');torch.manual_seed(seed);model=make_model(cfg['kind'],n_sites=cfg['n_sites'],condition_dim=cfg['condition_dim'],latent_dim=cfg['latent_dim'],hidden_dim=cfg['hidden_dim'],encoder_kind=cfg['encoder_kind'],graph_data=cfg['graph'],graph_context=cfg['graph_context'],context_spec=cfg['decoder_context_spec'],remaining_count_features=cfg['remaining_count_features']).eval();cases.append(('random_graph_D2',seed,'random_init',model,cfg,path))
for name,seed,stage,model,cfg,path in cases:
 cv,*_=resolve_request(cfg,24,requested_bin='middle')
 for dtype in [torch.float32,torch.float64]:
  model=model.to(dtype);captured=[];hook=model.encoder.output.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().numpy().astype(np.float64)))
  with torch.no_grad():mu,lv=model.encode(xt.to(dtype),torch.tensor(np.tile(cv,(len(x),1)),dtype=dtype))
  hook.remove();m=mu.numpy().astype(np.float64);l=lv.numpy().astype(np.float64);features=np.concatenate(captured)
  rows.append(dict(family=name,seed=seed,stage=stage,forward_dtype=str(dtype),statistics_dtype='float64',rows=len(x),mu_mean=m.mean(0).tolist(),mu_SD=m.std(0).tolist(),mu_distinct_rows=len(np.unique(m,axis=0)),maximum_pooled_feature_SD=float(features.std(0).max()),RMS_posterior_sigma=np.sqrt(np.exp(l).mean(0)).tolist(),mu_Cs_Pearson=[float(np.corrcoef(m[:,k],Cs)[0,1]) for k in range(m.shape[1])],checkpoint_sha256=sha(path)))
(out/'structural_sensitivity_stable.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
# Re-render from preserved FP32 raw embeddings; there is no new inference here.
fig,ax=plt.subplots(2,3,figsize=(12,6.4),constrained_layout=True)
for j,title in enumerate(['D2','Sampled auxiliary λ10','Mean auxiliary λ10']):
 with np.load(p/f'fixed_condition_embedding/embedding_{j}.npz',allow_pickle=False) as z:m=z['posterior_mu'].astype(float);lv=z['posterior_logvar'].astype(float);cs=z['raw_Cs']
 sc=ax[0,j].scatter(m[:,0],m[:,1],c=cs,vmin=Cs.min(),vmax=Cs.max(),s=7,alpha=.6,rasterized=True);ax[0,j].set(title=title,xlabel='Posterior mean μ₁',ylabel='Posterior mean μ₂');ax[0,j].ticklabel_format(style='sci',axis='both',scilimits=(-2,2));ax[0,j].tick_params(labelsize=7);ax[0,j].text(.02,.98,f'SD(μ₁)={m[:,0].std():.3g}\nRMS posterior σ₁={np.sqrt(np.exp(lv[:,0]).mean()):.3g}\nPrior SD=1',transform=ax[0,j].transAxes,va='top',fontsize=7,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
 grid=np.linspace(-4,4,401);sigma=np.exp(.5*lv[:,0]);density=(np.exp(-.5*((grid[:,None]-m[None,:,0])/sigma[None])**2)/(np.sqrt(2*np.pi)*sigma[None])).mean(1);ax[1,j].plot(grid,density,label='VAL posterior mixture');ax[1,j].plot(grid,np.exp(-.5*grid**2)/np.sqrt(2*np.pi),'--',label='Unit Normal prior');ax[1,j].axvspan(m[:,0].min(),m[:,0].max(),alpha=.2,color='orange',label='Range of μ₁');ax[1,j].set(xlabel='Latent coordinate z₁ (dimensionless)',ylabel='Probability density');ax[1,j].legend(fontsize=6)
fig.colorbar(sc,ax=ax[0,:],label='Actual vacancy-pair fraction C_s',shrink=.7);fig.suptitle('Fixed condition:24 Li,middle energy class; predeclared fit seed2026111301\nActual dimensionless μ scales; moment statistics accumulated in float64',fontsize=10);fig.savefig(out/'fixed_condition_latent_stable.pdf');fig.savefig(out/'fixed_condition_latent_stable.png',dpi=180);plt.close(fig)
# Main posthoc probe table used float64 fitting throughout; its numbers are unchanged.
d=pd.read_csv(p/'conditional_information_audit/conditional_probes_seed_balanced.csv');fixed=pd.read_csv(p/'conditional_information_audit/fixed_condition_random_comparison.csv');families=['ar','d2','lambda10_latent_d2','lambda10_shuffled_d2','mean_lambda10_latent_d2','mean_lambda10_shuffled_d2'];fig,ax=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
for k,probe in enumerate(['conditional_mu','shuffled_TRAIN_probe']):
 v=d[d.probe==probe].groupby('family').MSE.agg(['mean','std']);ax[0].errorbar(np.arange(len(families))+(k-.5)*.16,[v.loc[f,'mean'] for f in families],yerr=[v.loc[f,'std'] for f in families],fmt='o',capsize=3,label=probe.replace('_',' '))
ax[0].axhline(1.021184,c='grey',ls='--',lw=.8,label='TRAIN-zero reference');ax[0].set(xticks=np.arange(len(families)),xticklabels=[f.replace('mean_lambda10_','Mean ').replace('lambda10_','Sampled ').replace('_d2','') for f in families],ylabel='VAL target MSE, balanced over9conditions',title='Posthoc conditional μ probe:5fits');ax[0].tick_params(axis='x',rotation=35);ax[0].legend(fontsize=7)
ff=['random_graph_D2','d2','lambda10_latent_d2','mean_lambda10_latent_d2'];z=fixed[fixed.probe=='conditional_mu'].groupby('family').MSE.agg(['mean','std']);ax[1].errorbar(np.arange(4),[z.loc[f,'mean'] for f in ff],yerr=[z.loc[f,'std'] for f in ff],fmt='o',capsize=3);ax[1].set(xticks=np.arange(4),xticklabels=['Random graph','D2 trained','Sampled λ10','Mean λ10'],ylabel='VAL target MSE at24 Li,middle',title='Random architectural features already contain C_s');ax[1].tick_params(axis='x',rotation=30);fig.suptitle('TRAIN-only within-condition scaling and ridgeα1; error bars=SD over5seeds\nRecoverability from rescaled μ does not imply controllable sampled or prior z',fontsize=10);fig.savefig(out/'conditional_information_comparison.pdf');fig.savefig(out/'conditional_information_comparison.png',dpi=180);plt.close(fig)
receipt=dict(status='PASS',source_sha256=sha(__file__),scope='Numericalmetadata correction only; no newtraining, targets, conditions, oracle scores or probe fits. Prior conditional ridge center/std and regression used float64, so MSE/R² results remainvalid.',corrected='Illustrative np.std/mean on FP32 arrays can lose tinyvariation due largeconstantoffset. Recomputed μ/pooledfeaturemoments in float64; FP32 and FP64 forward passes explicitlydistinguished.',supersedes='structural_sensitivity.json momentfields and original embedding momentannotations. OriginalTRAIN_mu_SD_min/max columns used float32 and should not beused; coreprobe MSE/skill/Spearman untouched.',TEST_processed=False)
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('STABLE_STRUCTURAL_CORRECTION_PASS',flush=True)

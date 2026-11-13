"""Predeclared first-fit VAL illustration at n=24, measured energy class=middle."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cathode_cvae.runtime import require_compute_allocation
from cathode_cvae.learning.generate_mask import load_model,resolve_request
from cathode_cvae.learning.train_mask import canonicalize
require_compute_allocation();torch.set_num_threads(4);import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args();p=args.run_root.resolve();b=p.parent;sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest();seed=2026111301
families=[('D2',b/'initial_fits'/f'd2_{seed}'/'last_update.pt'),('Sampled auxiliary λ10',b/'latent_phase2/fits'/f'lambda10_latent_d2_{seed}'/'last_update.pt'),('Mean auxiliary λ10',p/'fits'/f'mean_lambda10_latent_d2_{seed}'/'last_update.pt')]
while not families[-1][1].exists():time.sleep(5)
reference=b/'latent_phase2/diagnostics'/f'd2_{seed}'/'VAL_probe_predictions.npz'
with np.load(reference,allow_pickle=False) as z:
 take=(z['n_target']==24)&(z['measured_energy_bin']=='middle');ids=z['structure_ids'][take].astype(str);target=z['genuine_standardized_target'][take];Cs=z['raw_vacancy_pair_fraction'][take]
model,cfg=load_model(families[0][1],'cpu');data=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project/artifacts/current/improvement_v1/training_round0.npz');parent=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project/artifacts/current/core/data/parents/O3_N48_pilot.npz')
with np.load(data,allow_pickle=False) as z:
 mapping={s:i for i,s in enumerate(z['structure_ids'].astype(str))};ii=np.array([mapping[s] for s in ids]);roles=z['split'][ii].astype(str)
 if not np.all(roles=='val'):raise ValueError('Only VAL rows may enter illustration')
 occupancy=z['occupancy'][ii]
with np.load(parent,allow_pickle=False) as z:perms=z['permutations']
xt,_=canonicalize(torch.tensor(occupancy,dtype=torch.float32),torch.tensor(np.argsort(perms,axis=1)));out=p/'fixed_condition_embedding';out.mkdir(exist_ok=False);fig,ax=plt.subplots(2,3,figsize=(11.5,6.5),constrained_layout=True);rows=[];bindings=[]
for col,(name,ck) in enumerate(families):
 model,cfg=load_model(ck,'cpu')
 if cfg['data_sha256']!=sha(data) or cfg['parent_sha256']!=sha(parent) or cfg['seed']!=seed:raise ValueError('Model lineage differs')
 c,*_=resolve_request(cfg,24,requested_bin='middle');ct=torch.tensor(np.tile(c,(len(ids),1)),dtype=torch.float32)
 with torch.no_grad():mu,lv=model.encode(xt,ct)
 mu=mu.numpy().astype(np.float64);variance=np.exp(lv.numpy().astype(np.float64));sigma=np.sqrt(variance);std_mu=mu.std(0);rms_sigma=np.sqrt(variance.mean(0))
 sc=ax[0,col].scatter(mu[:,0],mu[:,1],c=Cs,vmin=float(Cs.min()),vmax=float(Cs.max()),s=8,alpha=.6,rasterized=True,cmap='viridis');ax[0,col].set(title=name,xlabel='Posterior mean μ₁ (actual dimensionless scale)',ylabel='Posterior mean μ₂');ax[0,col].ticklabel_format(style='sci',axis='both',scilimits=(-2,2));ax[0,col].text(.02,.98,f'SD(μ₁)={std_mu[0]:.3g}\nRMS posterior σ₁={rms_sigma[0]:.3g}\nPrior SD=1',ha='left',va='top',transform=ax[0,col].transAxes,fontsize=7,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
 grid=np.linspace(-4,4,401);density=(np.exp(-.5*((grid[:,None]-mu[None,:,0])/sigma[None,:,0])**2)/(np.sqrt(2*np.pi)*sigma[None,:,0])).mean(1);prior=np.exp(-.5*grid**2)/np.sqrt(2*np.pi);ax[1,col].plot(grid,density,label='VAL posterior mixture');ax[1,col].plot(grid,prior,'--',label='Standard Normal prior');ax[1,col].axvspan(mu[:,0].min(),mu[:,0].max(),alpha=.15,color='orange',label='Range of posterior means');ax[1,col].set(xlabel='Latent coordinate z₁',ylabel='Probability density');ax[1,col].legend(fontsize=6)
 np.savez_compressed(out/f'embedding_{col}.npz',structure_ids=ids,posterior_mu=mu,posterior_logvar=lv.numpy(),raw_Cs=Cs,standardized_Cs=target)
 rows.append(dict(model=name,rows=len(ids),mean_mu=mu.mean(0).tolist(),SD_mu=std_mu.tolist(),RMS_posterior_sigma=rms_sigma.tolist(),prior_sigma=1,variance_mu_over_mean_posterior_variance=(std_mu**2/variance.mean(0)).tolist()));bindings.append(dict(path=str(ck),sha256=sha(ck)))
fig.colorbar(sc,ax=ax[0,:],label='Actual nearest-shell vacancy-pair fraction C_s',shrink=.7);fig.suptitle('Fixed condition: 24 Li, middle energy class; first predeclared fit seed2026111301\nArbitrary latent axes; this illustration does not replace five-fit prediction and generation tests',fontsize=10);fig.savefig(out/'fixed_condition_latent.pdf');fig.savefig(out/'fixed_condition_latent.png',dpi=180);plt.close(fig)
receipt=dict(status='PASS',source_sha256=sha(__file__),predeclared_condition=dict(n_target=24,measured_energy_bin='middle',fit_seed=seed),checkpoint_bindings=bindings,VAL_descriptor_source=dict(path=str(reference),sha256=sha(reference)),TEST_processed=False,TRAIN_targets_refit=False,rows=rows,scope='Firstseed illustrative fixedcondition VAL-only latent scatter and unweighted VAL encoder Gaussian mixture; dimensionless latent units; prior variance1. Mean/variance ratio is a descriptive signal/noise diagnostic, not mutual information or physical convergence.')
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)

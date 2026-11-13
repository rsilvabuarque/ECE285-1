"""Render fixed-condition saved encoder outputs; no new model inference."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cathode_cvae.runtime import require_compute_allocation
require_compute_allocation();import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args();p=args.run_root.resolve();out=p/'structural_precision_correction/presentation_v2_largefonts';out.mkdir(exist_ok=False);sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest();plt.rcParams.update({'font.size':13,'axes.titlesize':14,'axes.labelsize':13,'xtick.labelsize':13,'ytick.labelsize':13,'legend.fontsize':13});fig,ax=plt.subplots(3,2,figsize=(12,12),constrained_layout=True);rows=[]
for j,title in enumerate(['D2 without auxiliary supervision','Sampled auxiliary, λ=10','Posterior-mean auxiliary, λ=10']):
 path=p/f'fixed_condition_embedding/embedding_{j}.npz'
 with np.load(path,allow_pickle=False) as z:mu=z['posterior_mu'].astype(np.float64);lv=z['posterior_logvar'].astype(np.float64);Cs=z['raw_Cs'].astype(np.float64)
 mean=mu.mean(0);centered=mu-mean;std=mu.std(0);sigma=np.sqrt(np.exp(lv).mean(0));left=ax[j,0];right=ax[j,1];sc=left.scatter(centered[:,0],centered[:,1],c=Cs,vmin=float(Cs.min()),vmax=float(Cs.max()),s=14,alpha=.65,rasterized=True,cmap='viridis');left.set(title=title,xlabel='δμ₁ = μ₁ − mean(μ₁)',ylabel='δμ₂ = μ₂ − mean(μ₂)');left.ticklabel_format(style='sci',useOffset=False,scilimits=(0,0),axis='both');left.text(.025,.975,f'mean μ₁ = {mean[0]:.6g}\nmean μ₂ = {mean[1]:.6g}\nSD(μ₁) = {std[0]:.3g}\nRMS σ₁ = {sigma[0]:.3g}; prior σ₁ = 1',transform=left.transAxes,va='top',fontsize=13,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
 grid=np.linspace(-4,4,401);ss=np.exp(.5*lv[:,0]);density=(np.exp(-.5*((grid[:,None]-mu[None,:,0])/ss[None])**2)/(np.sqrt(2*np.pi)*ss[None])).mean(1);right.plot(grid,density,label='VAL posterior mixture',lw=2);right.plot(grid,np.exp(-.5*grid**2)/np.sqrt(2*np.pi),'--',label='Unit Normal prior',lw=2);right.axvspan(mu[:,0].min(),mu[:,0].max(),alpha=.3,color='orange',label='Range of posterior means');right.set(xlabel='Latent coordinate z₁',ylabel='Probability density',title='Posterior noise and prior scale');right.legend(loc='upper right')
 rows.append(dict(model=title,rows=len(mu),mu_mean=mean.tolist(),mu_SD_float64=std.tolist(),RMS_posterior_sigma=sigma.tolist(),source_embedding_sha256=sha(path)))
fig.colorbar(sc,ax=ax[:,0],label='Nearest-shell vacancy-pair fraction C_s',shrink=.6,pad=.02);fig.suptitle('Fixed condition: 24 Li, middle energy class; fit seed 2026111301\nCentered μ axes retain actual dimensionless units; offsets and noise scales are shown',fontsize=14);fig.savefig(out/'fixed_condition_latent_v2.pdf');fig.savefig(out/'fixed_condition_latent_v2.png',dpi=180);plt.close(fig)
receipt=dict(status='PASS',source_sha256=sha(__file__),rows=rows,scope='Presentation-only rerender from preserved FP32 encoder-output arrays with float64 moment calculations; no newinference, targets, training ororacle. δμ subtracts empiricalVALmean only, without rescaling; originalv1plots preserved.')
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PRESENTATION_V2_PASS',flush=True)

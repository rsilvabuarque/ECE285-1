"""Small, descriptive plots of already completed experiments; no new fits or labels."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE/'reports/adaptation_explanation';OUT.mkdir(exist_ok=False)
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
methods=[('ar_adapt_cvae','AR CVAE before extra update','tab:blue'),
         ('adaptive_ar_adapt_cvae_beta01',r'Extra update, $\beta=0.1$','tab:orange'),
         ('adaptive_ar_adapt_cvae_beta1',r'Extra update, $\beta=1$','tab:green')]
metrics=rows(HERE/'reports/final/metrics.csv');counts=[12,24,36];xx=np.array(counts)/48*100
fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
for ax,teacher in zip(axs,['uma','mace']):
    for method,label,color in methods:
        groups=[[float(r['novel_low_yield_per_attempt'])*100 for r in metrics if r['teacher']==teacher and r['method']==method and int(r['n_Li'])==n] for n in counts]
        assert all(len(g)==3 for g in groups)
        ax.plot(xx,[np.mean(g) for g in groups],'o-',color=color,label=label)
        for x,g in zip(xx,groups):ax.scatter([x]*len(g),g,color=color,s=18,alpha=.4)
    uniform=[np.mean([float(r['novel_low_yield_per_attempt'])*100 for r in metrics if r['teacher']==teacher and r['method']=='uniform' and int(r['n_Li'])==n]) for n in counts]
    ax.plot(xx,uniform,'--',color='gray',label='Uniform baseline')
    ax.set(title=teacher.upper(),xlabel='Lithiation (%)',xticks=xx,ylim=(0,55));ax.grid(alpha=.2)
axs[0].set_ylabel('Distinct novel low-tail orbits / attempts (%)')
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=9)
fig.suptitle('Final assessment: 512 attempts per composition and seed\nLines: means of three seeds; faint points: individual seeds',fontsize=11)
fig.tight_layout(rect=(0,.16,1,.89))
for ext in ['pdf','png']:fig.savefig(OUT/f'adaptation_yield_by_composition.{ext}',dpi=180)
plt.close(fig)

quality=rows(HERE/'reports/final/quality/fresh_group_quality.csv')
fig,axs=plt.subplots(1,3,figsize=(11.5,3.8),sharex=True)
for method,label,color in methods:
    for ax,key,title in zip(axs,['fresh_minimum_delta_q20_meV_atom','fresh_p10_delta_q20_meV_atom','fresh_median_delta_q20_meV_atom'],['Best fresh energy','10th-percentile fresh energy','Median fresh energy']):
        groups=[[float(r[key]) for r in quality if r['teacher']=='uma' and r['method']==method and int(r['n_Li'])==n] for n in counts]
        assert all(len(g)==3 for g in groups)
        ax.plot(xx,[np.mean(g) for g in groups],'o-',color=color,label=label)
        ax.set(title=title,xlabel='Lithiation (%)',xticks=xx);ax.axhline(0,color='gray',ls='--',lw=.8);ax.grid(alpha=.2)
axs[0].set_ylabel(r'Energy minus original UMA $q_{20}$ (meV/atom)')
handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
fig.suptitle('Energy quality of distinct fresh proposals: lower is better\nEach point averages the three seed-specific statistics; known and duplicate masks excluded',fontsize=10)
fig.tight_layout(rect=(0,.12,1,.87))
for ext in ['pdf','png']:fig.savefig(OUT/f'adaptation_energy_quality.{ext}',dpi=180)
plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(10,3.9))
sources=[]
for method,label,color in methods[1:]:
    for k,seed in enumerate([20261106,20261107,20261108]):
        folder=HERE/'runs'/f'{method}_{seed}';history=rows(folder/'history.csv');summary=json.loads((folder/'summary.json').read_text());best=summary['best_epoch']
        sources.append(folder/'history.csv')
        for ax,key,title in zip(axs,['val_reconstruction','val_KL'],['Validation reconstruction loss','Validation latent KL']):
            ax.plot([int(r['epoch']) for r in history],[float(r[key]) for r in history],color=color,alpha=.6,lw=1,label=label if k==0 else None)
            r=next(r for r in history if int(r['epoch'])==best);ax.scatter(best,float(r[key]),color=color,s=28,zorder=4)
            ax.set(title=title,xlabel='Additional training epoch');ax.grid(alpha=.2)
axs[0].set_ylabel('Negative log likelihood (nats/mask)');axs[1].set_ylabel('KL divergence (nats/mask)')
axs[0].legend(fontsize=9)
fig.suptitle('Two adaptive variants on the SAME expanded validation set\nOne line per fitted seed; dots are the selected checkpoints',fontsize=11)
fig.tight_layout(rect=(0,0,1,.87))
for ext in ['pdf','png']:fig.savefig(OUT/f'adaptive_learning_diagnostics.{ext}',dpi=180)
plt.close(fig)

receipt={'scope':'Descriptive plots from existing completed runs. No new simulation, training, inference, or data split. Curves do not isolate effects of the coupled adaptive changes.','sources':{str(p.relative_to(HERE)):sha(p) for p in [HERE/'reports/final/metrics.csv',HERE/'reports/final/quality/fresh_group_quality.csv',*sources]},'outputs':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}
(OUT/'sources.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Created three focused PDF/PNG figures from existing CSVs.',flush=True)

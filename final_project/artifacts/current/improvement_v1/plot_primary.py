"""Compact final comparison; dots show seed macro-averages, not confidence intervals."""
import csv,json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
rows=list(csv.DictReader((HERE/'reports/final/seed_metrics.csv').open()))
methods=[('uniform','Uniform'),('legacy_uma_cvae_topk','Original CVAE'),('ar_adapt_cvae','AR CVAE, round0'),('adaptive_ar_adapt_cvae_beta01','Adaptive AR CVAE, beta0.1'),('adaptive_ar_adapt_cvae_beta1','Adaptive AR CVAE, beta1'),('mutation_original','Mutation, original data'),('mutation_adapt','Mutation, round0 data'),('mutation_round1','Mutation, round1 data')]
fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
for ax,teacher in zip(axes,['uma','mace']):
 for i,(method,label) in enumerate(methods):
  values=[100*float(r['macro_low_yield']) for r in rows if r['teacher']==teacher and r['method']==method]
  assert len(values)==3
  ax.barh(i,np.mean(values),alpha=.65,color='steelblue' if 'cvae' in method else 'gray');ax.scatter(values,[i]*3,s=22,color='black',zorder=3)
 ax.set_title(teacher.upper());ax.set_xlabel('Distinct novel low-tail orbits / attempts (%)');ax.set_xlim(0,100);ax.grid(axis='x',alpha=.2)
axes[0].set_yticks(range(len(methods)),[label for _,label in methods]);axes[0].invert_yaxis()
fig.suptitle('Fresh final assessment:3 Li counts x3 seeds;512 attempts/group\nOriginal training q20; novelty excludes all earlier scored/development masks',fontsize=11)
fig.tight_layout();fig.savefig(HERE/'reports/final/primary_comparison.pdf');fig.savefig(HERE/'reports/final/primary_comparison.png',dpi=180);plt.close(fig)

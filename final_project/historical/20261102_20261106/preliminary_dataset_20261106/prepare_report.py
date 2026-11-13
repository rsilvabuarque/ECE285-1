"""Populate report tables from validated measurements; draft until MD export exists."""
import csv,json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE/'reports'
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
r=list(csv.DictReader((OUT/'generation_metrics.csv').open()));methods=['uniform','uma_cvae','uma_cvae_beta01','low_train_mutation'];labels=['Uniform','CVAE','CVAE beta=0.1','Mutation']
fig,axs=plt.subplots(1,3,figsize=(9,3.1),sharey=True)
for ax,n in zip(axs,[12,24,36]):
    for seed,color in zip([20261106,20261107,20261108],['tab:blue','tab:orange','tab:green']):
        yy=[float(next(a['novel_low_yield_per_attempt'] for a in r if a['teacher']=='uma' and a['method']==m and int(a['n_Li'])==n and int(a['seed'])==seed)) for m in methods]
        ax.plot(range(4),yy,'o-',color=color,label=str(seed),lw=.8,ms=4)
    ax.set_title(f'Li count {n}/48',fontsize=10);ax.set_xticks(range(4),labels,rotation=25,ha='right',fontsize=8);ax.set_ylim(0,.6);ax.grid(alpha=.2)
axs[0].set_ylabel('Novel low-tail orbits / 512 attempts',fontsize=9);axs[2].legend(title='Model / sampling seed',fontsize=6,title_fontsize=7);fig.tight_layout();fig.savefig(OUT/'generation_seed_panels.pdf');fig.savefig(OUT/'generation_seed_panels.png',dpi=180);plt.close(fig)
rel=list(csv.DictReader((OUT/'relaxation_summary.csv').open()));table=[]
for m in ['uma','mace','sevennet','chgnet','m3gnet']:
    a=next(x for x in rel if x['model']==m and x['cohort']=='fixed');b=next(x for x in rel if x['model']==m and x['cohort']=='variable');table.append(f"{m} & {a['converged']}/1000 & {b['converged']}/1000 & {float(a['median_steps']):.0f} & {float(b['median_steps']):.0f} \\\\")
md=HERE/'validation/md_export.json';complete=md.exists();d=json.loads(md.read_text()) if complete else {};tasks=str(d.get('tasks','PENDING'));frames=str(sum(x['frames'] for x in d.get('exports',[]))) if complete else 'PENDING'
hours=json.loads((HERE/'resources.json').read_text())['last_observed']['allocated_four_gpu_node_equivalent_hours']
facts='\n'.join([r'\newcommand{\MDTasks}{'+tasks+'}',r'\newcommand{\MDFrames}{'+frames+'}',r'\newcommand{\NodeHours}{'+f'{hours:.2f}'+'}',r'\newcommand{\ReportStatus}{'+('' if complete else r'\textcolor{red}{DRAFT: thermal export pending; do not distribute.}')+'}',r'\newcommand{\RelaxationTable}{'+'\n'.join(table)+'}'])+'\n'
(OUT/'facts.tex').write_text(facts);(OUT/'report_build_inputs.json').write_text(json.dumps(dict(MD_complete=complete,MD_tasks=tasks,MD_frames=frames,resource_snapshot_four_gpu_node_hours=hours),indent=2)+'\n');print('REPORT_FACTS',complete,tasks,frames,flush=True)

"""Descriptive paired-fit plots for the frozen, explicitly post-hoc causal follow-up."""
from pathlib import Path
import csv,hashlib,json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
C=Path(__file__).parent

def rows(p):
 with p.open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert (C/'completion.json').exists();out=C/'analysis';out.mkdir(exist_ok=False)
 extra=json.loads((C/'preparation_receipt.json').read_text())['extra_H0_using_R1edges'];same='h0_range_r1' if extra else 'h0_range_h0';data=rows(C/'aggregate_metrics.csv');paired=rows(C/'paired_summary.csv')
 fig,axs=plt.subplots(2,2,figsize=(8.5,6),sharey=True)
 for i,family in enumerate(['ar','d2']):
  for j,teacher in enumerate(['uma','mace']):
   ax=axs[i,j]
   for label,cells,color in [('H0 rows',['h0_fixed',same],'#377eb8'),('H0 + acquired rows',['aug_fixed','aug_range'],'#e41a1c')]:
    means=[];sds=[]
    for cell in cells:
     v=[100*float(r['balanced_fixed_yield']) for r in data if r['family']==family and r['teacher']==teacher and r['cell']==cell];assert len(v)==3;means.append(np.mean(v));sds.append(np.std(v,ddof=1))
    ax.errorbar([0,1],means,yerr=sds,marker='o',capsize=3,color=color,label=label)
   ax.set_title(f'{family.upper()}: {teacher.upper()}');ax.set_xticks([0,1],['Original fixed bins','Observed-range bins\nmatched R1 edges']);ax.grid(alpha=.2)
   if j==0:ax.set_ylabel('Balanced H0 fixed-bin novel yield (%)')
 axs[0,0].legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(out/'data_by_conditioning_factorial.pdf');fig.savefig(out/'data_by_conditioning_factorial.png',dpi=180);plt.close(fig)
 contrast_names=[('H0_relabeling','Relabel H0\nwithout acquisition'),('added_rows_fixed','Add rows\nfixed bins'),('added_rows_identical_range_edges','Add rows\nmatched range bins')]
 fig,axs=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
 for j,teacher in enumerate(['uma','mace']):
  ax=axs[j]
  for family,offset,color in [('ar',-.08,'#377eb8'),('d2',.08,'#e41a1c')]:
   rr=[next(r for r in paired if r['teacher']==teacher and r['family']==family and r['contrast']==name and r['metric']=='balanced_fixed_yield') for name,_ in contrast_names]
   ax.errorbar(np.arange(3)+offset,[float(r['gain_pp_mean']) for r in rr],yerr=[float(r['gain_pp_sample_sd']) for r in rr],marker='o',linestyle='none',capsize=3,color=color,label=family.upper())
  ax.axhline(0,color='k',lw=.8);ax.set_title(teacher.upper());ax.set_xticks(range(3),[s for _,s in contrast_names],fontsize=8);ax.grid(axis='y',alpha=.2)
 axs[0].set_ylabel('Paired balanced H0 yield difference (pp)');axs[0].legend(frameon=False);fig.tight_layout();fig.savefig(out/'causal_paired_gains.pdf');fig.savefig(out/'causal_paired_gains.png',dpi=180);plt.close(fig)
 (out/'interpretation.md').write_text('This is a post-hoc mechanistic follow-up prompted by the first-round range-bin gain, not a new blind confirmation. The factorial plot holds physical condition edges identical across H0 versus augmented data: range edges come from each lineage\'s R1 acquired dataset. The separate H0 relabeling contrast instead fits range edges from H0 TRAIN alone and therefore needs no acquired labels. All metrics use original H0 teacher-specific fixed schemas; H0 fixed-low and H0 fixed-high are distinct from the pilot409 threshold diagnostic. Error bars are sample SD across three paired fit/draw replicates, not confidence intervals. No assessment labels entered any fit.\n')
 receipt=dict(status='POSTHOC_CAUSAL_ANALYSIS_COMPLETE',created_unix=time.time(),source_sha256=sha(Path(__file__)),completion_sha256=sha(C/'completion.json'),outputs={p.name:sha(p) for p in out.iterdir()});(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()

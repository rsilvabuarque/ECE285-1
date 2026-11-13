"""Presentation-only redraws from frozen tables; original analysis figures remain unchanged."""
from pathlib import Path
import csv,hashlib,json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).parent

def read(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=R/'presentation_v2';out.mkdir(exist_ok=False);summary=read(R/'analysis/paired_fit_summary.csv');pairs=read(R/'analysis/paired_fit_results.csv');support=read(R/'analysis/training_bin_support.csv')
 colors={'fixed':'#377eb8','adaptive':'#ff7f00','range':'#4daf4a'};labels={'fixed':'Fixed H0 bins','adaptive':'TRAIN quantiles','range':'Observed range'}
 plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11})
 fig,axs=plt.subplots(2,2,figsize=(11.5,7.5),sharey=True)
 for row,roundi in enumerate([1,2]):
  for col,teacher in enumerate(['uma','mace']):
   ax=axs[row,col]
   for k,family in enumerate(['ar','d2']):
    for j,mode in enumerate(['fixed','adaptive','range']):
     rr=next(r for r in summary if int(r['round'])==roundi and r['teacher']==teacher and r['family']==family and r['mode']==mode and r['metric']=='balanced_fixed_yield');xx=k+(j-1)*.23
     pp=[r for r in pairs if int(r['round'])==roundi and r['teacher']==teacher and r['family']==family and r['mode']==mode and r['metric']=='balanced_fixed_yield' and r['control_id'].startswith(f'r{roundi}_no_new_')];assert len(pp)==3
     values=[float(r['paired_gain_pp']) for r in sorted(pp,key=lambda r:r['seed'])]
     ax.scatter(xx+np.array([-.04,0,.04]),values,s=24,color=colors[mode],alpha=.58,zorder=3)
     ax.errorbar(xx,float(rr['paired_gain_pp']),yerr=float(rr['paired_gain_sd_pp']),color=colors[mode],marker='D',markersize=6,capsize=4,lw=1.8,label=labels[mode] if row==0 and col==0 and k==0 else None,zorder=4)
   ax.axhline(0,color='k',lw=1);ax.set_title(f'Round {roundi}: {teacher.upper()}');ax.set_xticks([0,1],['AR','D2 CVAE']);ax.grid(axis='y',alpha=.22)
   if col==0:ax.set_ylabel('Yield difference (pp)')
 fig.suptitle('Paired effect versus same-lineage no-new continuation',fontsize=15,y=.995)
 handles,leg=axs[0,0].get_legend_handles_labels();fig.legend(handles,leg,loc='upper center',bbox_to_anchor=(.53,.955),ncol=3,frameon=False,fontsize=11)
 fig.text(.5,.025,'Each point is one paired fit/draw seed; error bars show sample SD across three pairs. Each round has its own frozen novelty history.',ha='center',fontsize=10)
 fig.subplots_adjust(left=.08,right=.99,bottom=.11,top=.85,hspace=.28,wspace=.12);fig.savefig(out/'paired_policy_effects.pdf',bbox_inches='tight',pad_inches=.18);fig.savefig(out/'paired_policy_effects.png',dpi=200,bbox_inches='tight',pad_inches=.18);plt.close(fig)
 h0path=R.parent/'causal_range/h0_range_schema/condition_schema.json';h0=json.loads(h0path.read_text());styles={'ar':'-','d2':'--'}
 fig,axs=plt.subplots(2,3,figsize=(12,6.7),sharex=True,sharey='row')
 for col,count in enumerate([12,24,36]):
  ref=h0['compositions'][str(count)]
  for row,tail in enumerate(['low','high']):
   ax=axs[row,col];baseline=ref['observed_train_minimum_eV_atom' if tail=='low' else 'observed_train_maximum_eV_atom']
   for family in ['ar','d2']:
    for mode in ['fixed','adaptive','range']:
     means=[0.];sds=[0.]
     for roundi in [1,2]:
      rr=[r for r in support if int(r['round'])==roundi and int(r['n_target'])==count and r['fit_id'].startswith(f'r{roundi}_{family}_{mode}_')];assert len(rr)==3
      vv=[1000*(baseline-float(r['train_minimum_eV_atom'])) if tail=='low' else 1000*(float(r['train_maximum_eV_atom'])-baseline) for r in rr];means.append(float(np.mean(vv)));sds.append(float(np.std(vv,ddof=1)))
     ax.errorbar([0,1,2],means,yerr=sds,color=colors[mode],linestyle=styles[family],marker='o',markersize=3.5,capsize=2,lw=1.6,label=f'{family.upper()} / '+labels[mode])
   ax.set_title(f'nLi = {count}');ax.set_xticks([0,1,2],['H0','After R1','After R2']);ax.grid(alpha=.2);ax.axhline(0,color='k',lw=.7)
   if col==0:ax.set_ylabel(('Low-end gain' if tail=='low' else 'High-end gain')+' (meV/atom)')
 fig.suptitle('Observed TRAIN energy-range expansion',fontsize=15,y=1.01)
 handles,leg=axs[0,0].get_legend_handles_labels();fig.legend(handles,leg,loc='upper center',bbox_to_anchor=(.54,.97),ncol=3,frameon=False,fontsize=10)
 fig.text(.5,.015,'Observed surrogate-energy extrema only; no claim of global minima, maxima, or configurational-space coverage.',ha='center',fontsize=10)
 fig.subplots_adjust(left=.09,right=.99,bottom=.11,top=.79,hspace=.28,wspace=.12);fig.savefig(out/'observed_TRAIN_extrema.pdf',bbox_inches='tight',pad_inches=.2);fig.savefig(out/'observed_TRAIN_extrema.png',dpi=200,bbox_inches='tight',pad_inches=.2);plt.close(fig)
 (out/'captions.md').write_text('Paired policy effects: acquired-data/label policies versus the same-lineage no-new continuation, with matched update budgets and fit/draw seeds within a round. R1 controls retain H0 and its original schema; R2 controls retain each previous lineage data/schema. Bars are sample SD, and individual seed differences are shown. All metrics use original teacher-specific H0 fixed bins. Absolute levels from different assessment rounds have different frozen novelty histories and should not be interpreted as a paired learning gain.\n\nObserved extrema: expansion of each lineage\'s nested TRAIN archive relative to original H0 TRAIN extrema. Values are unchanged from the original plot; only labels/layout are redrawn. Low-end gain means a more negative observed energy, high-end gain means a more positive observed energy, always within one fixed Li count. These are observed MLIP extrema, not physical/global convergence or coverage claims.\n')
 inputs={str(p):sha(p) for p in [R/'analysis/receipt.json',R/'analysis/paired_fit_summary.csv',R/'analysis/paired_fit_results.csv',R/'analysis/training_bin_support.csv',h0path]}
 (out/'receipt.json').write_text(json.dumps(dict(status='PRESENTATION_ONLY_REDRAW_COMPLETE',created_unix=time.time(),source_sha256=sha(Path(__file__)),inputs=inputs,outputs={p.name:sha(p) for p in out.iterdir()},original_figures_and_tables_unchanged=True,no_new_fits_or_oracles=True),indent=2)+'\n')
if __name__=='__main__':main()

"""Existing-result diagnostics; numerical rate decomposition is not a causal failure mechanism."""
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
def write(p,rows):
 with Path(p).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean_sd(x):return float(np.mean(x)),float(np.std(x,ddof=1))
def model_rows(rows,id,teacher):return [r for r in rows if r['stage_id']==id and r['teacher']==teacher]
def rates(rr):
 n=sum(int(r['attempts']) for r in rr);s=sum(int(r['scored_unique']) for r in rr);h=sum(int(round(float(r['fixed_inbin_yield'])*int(r['attempts']))) for r in rr)
 return h/n,s/n,h/s if s else None,n,s,h

def main():
 assert (R/'analysis/receipt.json').exists();out=R/'interpretation_diagnostics';out.mkdir(exist_ok=False)
 cell2=read(R/'round2_evaluation/all_cell_metrics.csv');allcells=[];overall=[];percell_summary=[]
 for teacher in ['uma','mace']:
  for family in ['ar','d2']:
   for seed in [2026111301,2026111302,2026111303]:
    aa=model_rows(cell2,f'r2_{family}_range_{seed}',teacher);bb=model_rows(cell2,f'r2_no_new_{family}_range_{seed}',teacher);assert len(aa)==len(bb)==9;lookup={(r['n_target'],r['requested_bin']):r for r in bb};nov=[];acc=[]
    for a in aa:
     b=lookup[(a['n_target'],a['requested_bin'])];ya,va,pa,_,_,_=rates([a]);yb,vb,pb,_,_,_=rates([b]);nc=(va-vb)*(pa+pb)/2 if pa is not None and pb is not None else None;ac=(pa-pb)*(va+vb)/2 if pa is not None and pb is not None else None
     if nc is not None:nov.append(nc);acc.append(ac)
     allcells.append(dict(teacher=teacher,family=family,seed=seed,n_target=int(a['n_target']),requested_bin=a['requested_bin'],policy_yield=ya,reference_yield=yb,policy_novelty=va,reference_novelty=vb,policy_bin_accuracy=pa if pa is not None else '',reference_bin_accuracy=pb if pb is not None else '',yield_change_pp=100*(ya-yb),novelty_change_pp=100*(va-vb),bin_accuracy_change_pp=100*(pa-pb) if pa is not None and pb is not None else '',novelty_contribution_pp=100*nc if nc is not None else '',accuracy_contribution_pp=100*ac if ac is not None else '',policy_scored=int(a['scored_unique']),reference_scored=int(b['scored_unique'])))
    ya,va,pa,_,sa,ha=rates(aa);yb,vb,pb,_,sb,hb=rates(bb)
    nc=float(np.mean(nov)) if len(nov)==9 else (va-vb)*(pa+pb)/2;ac=float(np.mean(acc)) if len(acc)==9 else (pa-pb)*(va+vb)/2
    overall.append(dict(teacher=teacher,family=family,seed=seed,policy_yield=ya,reference_yield=yb,policy_novelty=va,reference_novelty=vb,policy_pooled_bin_accuracy=pa,reference_pooled_bin_accuracy=pb,yield_change_pp=100*(ya-yb),novelty_change_pp=100*(va-vb),pooled_bin_accuracy_change_pp=100*(pa-pb),novelty_contribution_pp=100*nc,accuracy_contribution_pp=100*ac,identity_residual_pp=100*((ya-yb)-(nc+ac)),decomposition='mean of nine cellwise symmetric decompositions' if len(nov)==9 else 'pooled due to an unscored cell',policy_scored_incidences=sa,reference_scored_incidences=sb,policy_inbin_hits=ha,reference_inbin_hits=hb))
 write(out/'range_R2_decomposition_by_cell.csv',allcells);write(out/'range_R2_decomposition_by_fit.csv',overall)
 summary=[]
 for teacher in ['uma','mace']:
  for family in ['ar','d2']:
   rr=[r for r in overall if r['teacher']==teacher and r['family']==family]
   for metric in ['policy_yield','reference_yield','policy_novelty','reference_novelty','policy_pooled_bin_accuracy','reference_pooled_bin_accuracy','yield_change_pp','novelty_contribution_pp','accuracy_contribution_pp']:
    mu,sd=mean_sd([r[metric] for r in rr]);summary.append(dict(teacher=teacher,family=family,metric=metric,mean=mu,sample_sd=sd,n_pairs=len(rr)))
   for count in [12,24,36]:
    for requested in ['low','middle','high']:
     rr=[r for r in allcells if r['teacher']==teacher and r['family']==family and r['n_target']==count and r['requested_bin']==requested]
     mu,sd=mean_sd([r['yield_change_pp'] for r in rr]);percell_summary.append(dict(teacher=teacher,family=family,n_target=count,requested_bin=requested,mean_yield_change_pp=mu,sample_sd_pp=sd,mean_policy_novelty=float(np.mean([r['policy_novelty'] for r in rr])),mean_reference_novelty=float(np.mean([r['reference_novelty'] for r in rr])),mean_policy_accuracy=float(np.mean([r['policy_bin_accuracy'] for r in rr])) if all(r['policy_bin_accuracy']!='' for r in rr) else '',mean_reference_accuracy=float(np.mean([r['reference_bin_accuracy'] for r in rr])) if all(r['reference_bin_accuracy']!='' for r in rr) else ''))
 write(out/'range_R2_decomposition_summary.csv',summary);write(out/'range_R2_per_cell_summary.csv',percell_summary)
 baseline=[];same_data=[]
 for roundi in [1,2]:
  agg=read(R/f'round{roundi}_evaluation/aggregate_metrics.csv');cells=read(R/f'round{roundi}_evaluation/all_cell_metrics.csv')
  for teacher in ['uma','mace']:
   for method in ['uniform','mutation']:
    rr=[r for r in agg if r['teacher']==teacher and r['family']==method];assert len(rr)==3
    for metric in ['balanced_fixed_yield','low_tail_yield','high_tail_yield']:
     mu,sd=mean_sd([100*float(r[metric]) for r in rr]);baseline.append(dict(round=roundi,teacher=teacher,method=method,metric={'low_tail_yield':'H0_fixed_low_yield','high_tail_yield':'H0_fixed_high_yield'}.get(metric,metric),mean_percent=mu,sample_sd_percent=sd,n_pairs=3))
   if roundi==1:
    for family in ['ar','d2']:
     for metric in ['balanced_fixed_yield','low_tail_yield','high_tail_yield']:
      values=[]
      for seed in [2026111301,2026111302,2026111303]:
       a=next(r for r in agg if r['teacher']==teacher and r['stage_id']==f'r1_{family}_range_{seed}');b=next(r for r in agg if r['teacher']==teacher and r['stage_id']==f'r1_{family}_fixed_{seed}');values.append(100*(float(a[metric])-float(b[metric])))
      mu,sd=mean_sd(values);same_data.append(dict(teacher=teacher,family=family,metric={'low_tail_yield':'H0_fixed_low_yield','high_tail_yield':'H0_fixed_high_yield'}.get(metric,metric),range_minus_fixed_mean_pp=mu,sample_sd_pp=sd,n_pairs=3,identical_added_rows_per_pair=2304))
 write(out/'fresh_uniform_mutation_baselines.csv',baseline);write(out/'R1_range_minus_fixed_same_data.csv',same_data)
 support=read(R/'analysis/training_bin_support.csv');dynamics=[]
 for family in ['ar','d2']:
  for seed in [2026111301,2026111302,2026111303]:
   for count in [12,24,36]:
    a=next(r for r in support if r['fit_id']==f'r1_{family}_range_{seed}' and int(r['n_target'])==count);b=next(r for r in support if r['fit_id']==f'r2_{family}_range_{seed}' and int(r['n_target'])==count)
    row=dict(family=family,seed=seed,n_target=count)
    for key in ['train_rows','train_minimum_eV_atom','train_maximum_eV_atom','low_count','middle_count','high_count','low_edge_eV_atom','high_edge_eV_atom','loss_weight_ESS']:
     row['R1_'+key]=a[key];row['R2_'+key]=b[key]
    for key in ['train_minimum_eV_atom','train_maximum_eV_atom','low_edge_eV_atom','high_edge_eV_atom']:row['change_'+key+'_meV_atom']=1000*(float(b[key])-float(a[key]))
    dynamics.append(row)
 write(out/'range_TRAIN_dynamics.csv',dynamics)
 fig,axs=plt.subplots(1,2,figsize=(8.5,3.5),sharey=True)
 for j,teacher in enumerate(['uma','mace']):
  ax=axs[j]
  for k,family in enumerate(['ar','d2']):
   for offset,metric,color,label in [(-.12,'novelty_contribution_pp','#377eb8','Novelty term'),(.12,'accuracy_contribution_pp','#ff7f00','Conditional accuracy term')]:
    rr=next(r for r in summary if r['teacher']==teacher and r['family']==family and r['metric']==metric);ax.bar(k+offset,rr['mean'],width=.22,color=color,label=label if k==0 else None);ax.errorbar(k+offset,rr['mean'],yerr=rr['sample_sd'],color='k',capsize=3,fmt='none')
  ax.axhline(0,color='k',lw=.8);ax.set_xticks([0,1],['AR','D2']);ax.set_title(teacher.upper());ax.grid(axis='y',alpha=.2)
 axs[0].set_ylabel('R2 range vs no-new yield change (pp)');axs[0].legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(out/'R2_range_yield_decomposition.pdf');fig.savefig(out/'R2_range_yield_decomposition.png',dpi=180);plt.close(fig)
 (out/'interpretation.md').write_text('All thresholds here are original H0 fixed-low/fixed-high teacher-schema thresholds, not pilot409. Novelty means selected eligible unique canonical masks divided by all128attempts in a cell. Conditional bin accuracy is in-bin hits divided by selected masks. Per-cell yield is exactly novelty times conditional accuracy. The symmetric numerical decomposition is deltaY = deltaNovelty * meanAccuracy + deltaAccuracy * meanNovelty; cellwise contributions are averaged over all nine cells where both accuracies exist, with pooled fallback explicitly flagged for unscored cells. It is an algebraic decomposition, not a causal explanation of why either rate changed. Sparse classes alone do not establish a failure mechanism; R1 range was positive. Actual TRAIN support/extrema/edge motion/loss-weight concentration and moving-bin UMA fidelity are separate observations. Existing raw tables and checks are unchanged. Sample SD is across three paired fit/draw replicates, not a confidence interval.\n')
 inputs={str(p):sha(p) for p in [R/f'round{i}_evaluation/{name}.csv' for i in [1,2] for name in ['aggregate_metrics','all_cell_metrics']]+[R/'analysis/training_bin_support.csv']}
 (out/'receipt.json').write_text(json.dumps(dict(status='EXISTING_CORE_INTERPRETATION_DIAGNOSTICS_COMPLETE',created_unix=time.time(),source_sha256=sha(Path(__file__)),inputs=inputs,outputs={p.name:sha(p) for p in out.iterdir()},no_new_fits_or_oracles=True),indent=2)+'\n')
if __name__=='__main__':main()

"""Post-hoc observed acquisition endpoints, distinct from frozen H0 conditional-yield goals."""
from pathlib import Path
import csv,hashlib,json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def write(p,rr):
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def main():
 out=R/'acquisition_coverage';out.mkdir(exist_ok=False);reference=R.parent/'causal_range/h0_range_schema/condition_schema.json';h0=json.loads(reference.read_text());data=[];inputs={str(reference):sha(reference)}
 for roundi in [1,2]:
  phase=R/f'round{roundi}_acquisition';vpath=phase/'verified_uma.json';v=json.loads(vpath.read_text());assert v['status']=='VERIFIED_FRESH_ADAPTIVE_ORACLE' and v['teacher']=='uma';p=Path(v['score']['path']);assert sha(p)==v['score']['sha256'];inputs[str(vpath)]=sha(vpath);inputs[str(p)]=sha(p)
  with np.load(p,allow_pickle=False) as z:energy={str(s):float(e)/int(a) for s,e,a in zip(z['structure_ids'],z['energy_eV'],z['n_atoms'])}
  metadata=json.loads((phase/'models.json').read_text())
  for model in metadata:
   stage=Path(model['stage']);receipt=json.loads((stage/'receipt.json').read_text());assert any(b['sha256']==sha(stage/'receipt.json') for b in v['stages']);pool=stage/'candidates.npz';assert sha(pool)==receipt['outputs']['candidates.npz'];attempts=read(stage/'attempts.csv')
   with np.load(pool,allow_pickle=False) as z:ids=z['structure_ids'].astype(str);nn=z['n_target']
   for count in [12,24,36]:
    es=np.asarray([energy[s] for s in ids[nn==count]]);entry=h0['compositions'][str(count)];lo=entry['observed_train_minimum_eV_atom'];hi=entry['observed_train_maximum_eV_atom'];n=len(es);a=sum(int(r['n_target'])==count for r in attempts);assert a==1536
    lower=max(0.,lo-float(es.min())) if n else 0.;upper=max(0.,float(es.max())-hi) if n else 0.;nlow=int(np.sum(es<lo));nhigh=int(np.sum(es>hi));policy=model['family']+'_'+model['mode'] if model['family'] in ['ar','d2'] else model['family']
    data.append(dict(round=roundi,teacher='uma',policy=policy,family=model['family'],schema_mode=model['mode'],seed=model['seed'],n_target=count,attempts=a,declared_querycap=768,selected_unique_physical_queries=n,phase_shared_unique_query_union=v['unique_structures'],H0_TRAIN_minimum_eV_atom=lo,H0_TRAIN_maximum_eV_atom=hi,new_minimum_eV_atom=float(es.min()) if n else '',new_maximum_eV_atom=float(es.max()) if n else '',below_H0_minimum=nlow,above_H0_maximum=nhigh,below_fraction_per_query=nlow/n if n else '',above_fraction_per_query=nhigh/n if n else '',outside_fraction_per_all_attempts=(nlow+nhigh)/a,lower_endpoint_gain_meV_atom=1000*lower,upper_endpoint_gain_meV_atom=1000*upper,observed_span_growth_meV_atom=1000*(lower+upper)))
 write(out/'by_fit_count.csv',data);summary=[]
 for roundi,policy,count in sorted({(r['round'],r['policy'],r['n_target']) for r in data}):
  rr=[r for r in data if (r['round'],r['policy'],r['n_target'])==(roundi,policy,count)];assert len(rr)==3
  for metric in ['selected_unique_physical_queries','below_H0_minimum','above_H0_maximum','outside_fraction_per_all_attempts','lower_endpoint_gain_meV_atom','upper_endpoint_gain_meV_atom','observed_span_growth_meV_atom']:
   values=[float(r[metric]) for r in rr];summary.append(dict(round=roundi,teacher='uma',policy=policy,n_target=count,metric=metric,mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)),n_seeds=3))
 write(out/'summary.csv',summary)
 moving=[]
 for roundi in [1,2]:
  source=R/f'round{roundi}_evaluation/aggregate_metrics.csv';inputs[str(source)]=sha(source);rows=read(source)
  for family in ['ar','d2']:
   for mode in ['fixed','adaptive','range']:
    rr=[r for r in rows if r['teacher']=='uma' and r['stage_id'].startswith(f'r{roundi}_{family}_{mode}_')];assert len(rr)==3
    moving.append(dict(round=roundi,family=family,mode=mode,moving_yield_mean_percent=float(np.mean([100*float(r['balanced_moving_yield']) for r in rr])),moving_yield_sample_sd_percent=float(np.std([100*float(r['balanced_moving_yield']) for r in rr],ddof=1)),H0_fixed_yield_mean_percent=float(np.mean([100*float(r['balanced_fixed_yield']) for r in rr])),same_property_across_rounds=mode=='fixed',description='Each checkpoint scored against its own declared TRAIN-fitted intervals; descriptive only.'))
 write(out/'UMA_own_moving_target_fidelity.csv',moving)
 policies=['ar_fixed','ar_adaptive','ar_range','d2_fixed','d2_adaptive','d2_range','uniform','mutation'];labels=['AR fixed','AR quantile','AR range','D2 fixed','D2 quantile','D2 range','Uniform','Mutation'];colors=['#377eb8','#ff7f00','#4daf4a']*2+['#777777','#a65628']
 fig,axs=plt.subplots(2,3,figsize=(12,6.5),sharey='row');plt.rcParams.update({'font.size':10})
 for col,count in enumerate([12,24,36]):
  for row,(metric,title) in enumerate([('lower_endpoint_gain_meV_atom','Lower endpoint'),('upper_endpoint_gain_meV_atom','Upper endpoint')]):
   ax=axs[row,col]
   for j,policy in enumerate(policies):
    r=next(r for r in summary if r['round']==2 and r['policy']==policy and r['n_target']==count and r['metric']==metric);ax.bar(j,r['mean'],color=colors[j],alpha=.8);ax.errorbar(j,r['mean'],yerr=r['sample_sd'],fmt='none',color='k',capsize=2)
   ax.set_title(f'nLi={count}: {title}');ax.set_xticks(range(8),labels,rotation=50,ha='right',fontsize=8);ax.grid(axis='y',alpha=.2)
   if col==0:ax.set_ylabel('Beyond H0 TRAIN (meV/atom)')
 fig.suptitle('Post-hoc observed acquisition endpoints: UMA, matched512/256 cell budgets',fontsize=13);fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(out/'R2_observed_endpoint_expansion.pdf',bbox_inches='tight',pad_inches=.2);fig.savefig(out/'R2_observed_endpoint_expansion.png',dpi=180,bbox_inches='tight',pad_inches=.2);plt.close(fig)
 (out/'interpretation.md').write_text('This post-hoc coverage diagnostic uses acquisition pools only and preserves the primary fixed-H0 yield criterion. Each method has512attempts and at most256new queries per composition/request cell; pooling three requested bins gives1536attempts and at most768physical query IDs per composition. Actual unique selected IDs are reported, so endpoint order statistics are not assumed equal-cost when novelty differs. A shared oracle evaluates overlapping method IDs only once; per-method counts must not be added as billed cost, and phase-level physical unions are shown separately. Outside-range means strictly below/above original H0 TRAIN minima/maxima, frozen by the TRAIN-only H0 range schema. No tolerance or metric was retuned. This measures observed surrogate-energy endpoints and new samples outside a finite training range, not global configurational coverage, entropy, equilibrium, DFT accuracy, or convergence. Only UMA has a frozen H0 extrema reference in this diagnostic; independent MACE fixed-bin audit remains in the core tables. Moving-target fidelity uses each checkpoint\'s own intervals and is descriptive, not a direct longitudinal same-property gain.\n')
 (out/'receipt.json').write_text(json.dumps(dict(status='POSTHOC_ACQUISITION_COVERAGE_COMPLETE',created_unix=time.time(),source_sha256=sha(Path(__file__)),inputs=inputs,outputs={p.name:sha(p) for p in out.iterdir()},primary_H0_metric_unchanged=True,no_new_fits_or_oracles=True),indent=2)+'\n')
if __name__=='__main__':main()

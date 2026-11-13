"""Allocation-only analysis: five paired fits, terminal primary, no model selection."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cathode_cvae.runtime import require_compute_allocation
from cathode_cvae.conditions import load_schema,classify
require_compute_allocation()
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['descriptors','oracle'],required=True);p.add_argument('--output-dir',type=Path);p.add_argument('--run-root',type=Path,required=True);a=p.parse_args()
r=a.run_root.resolve();b=r.parent;out=a.output_dir or r/('analysis_'+a.stage);out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
families=['ar','d2']+[f'lambda10_{m}_d2' for m in ['latent','condition','shuffled']]+[f'mean_lambda10_{m}_d2' for m in ['latent','condition','shuffled']]
pretty={f:f.replace('mean_lambda10_','Mean ').replace('lambda10_','Sampled ').replace('_d2','').replace('_',' ') for f in families}
records=[];probes=[];checkpoints=[];histories=[];bindings=[]
for family in families:
 for seed in range(2026111301,2026111306):
  d=r/'diagnostics'/f'{family}_{seed}';receipt=json.loads((d/'receipt.json').read_text());fit=(b/'initial_fits'/d.name if family in ['ar','d2'] else b/'latent_phase2/fits'/d.name if family.startswith('lambda10_') else r/'fits'/d.name)
  if receipt['status']!='PASS' or receipt['fit_seed']!=seed or receipt['TEST_processed']:raise ValueError('Diagnostic receipt mismatch')
  row=pd.read_csv(d/'generated_physical_outcomes.csv');row['family']=family;row['seed']=seed;records.append(row)
  for probe,v in receipt['metrics'].items():probes.append(dict(family=family,seed=seed,probe=probe,**v))
  h=pd.read_csv(fit/'history.csv');h['family']=family;h['seed']=seed;histories.append(h)
  for selection in ['last_update','best_validation_beta_0p1','best_validation_beta_1']:
   side=fit/(selection+'.pt.latent.json')
   if not side.exists():continue
   v=json.loads(side.read_text());q=v['pooled'];checkpoints.append(dict(family=family,seed=seed,selection=selection,optimizer_updates=v['optimizer_updates'],VAL_KL=q['total_KL_nats'],active001=q['active_components_by_cutoff']['0.001'],active01=q['active_components_by_cutoff']['0.01'],active1=q['active_components_by_cutoff']['0.1'],mean_posterior_variance=np.mean(q['posterior_variance_mean_by_dimension']) if q['latent_dim'] else np.nan))
  bindings.extend([{'path':str(x),'sha256':sha(x)} for x in [d/'receipt.json',d/'generated_physical_outcomes.csv',fit/'history.csv']])
raw=pd.concat(records,ignore_index=True);pr=pd.DataFrame(probes);hh=pd.concat(histories,ignore_index=True);cp=pd.DataFrame(checkpoints)
pr.to_csv(out/'VAL_probes_per_fit.csv',index=False);cp.to_csv(out/'checkpoint_information_per_fit.csv',index=False);hh.to_csv(out/'optimization_curves.csv',index=False)
# Descriptor effects keep ALL valid attempts, including history duplicates. These are mechanics, not novelty.
group=['family','seed','n_target','requested_bin','intervention']
desc=raw.groupby(group,sort=False).agg(attempts=('structure_id','size'),unique_orbits=('structure_id','nunique'),mean_Cs=('vacancy_pair_fraction','mean'),std_Cs=('vacancy_pair_fraction','std'),same_orbit_as_primary=('same_orbit_as_primary','mean'),mean_Cs_change=('pair_fraction_change_from_primary','mean')).reset_index()
desc.to_csv(out/'descriptor_cells.csv',index=False)
cell=['family','seed','n_target','requested_bin']
primary=raw[raw.intervention=='primary'].set_index(cell+['attempt'])
clamp=raw[raw.intervention=='latent_clamp_only'].set_index(cell+['attempt'])
effects=[]
for inter in ['guide_lower_quartile','guide_upper_quartile']:
 g=raw[raw.intervention==inter].set_index(cell+['attempt']);joined=g.join(clamp[['vacancy_pair_fraction','structure_id']],rsuffix='_clamp').join(primary[['vacancy_pair_fraction','structure_id']],rsuffix='_primary')
 joined['change_vs_clamp']=joined.vacancy_pair_fraction-joined.vacancy_pair_fraction_clamp
 joined['change_vs_primary']=joined.vacancy_pair_fraction-joined.vacancy_pair_fraction_primary
 joined['orbit_change_vs_clamp']=joined.structure_id!=joined.structure_id_clamp
 joined['orbit_change_vs_primary']=joined.structure_id!=joined.structure_id_primary
 joined['actual_target_squared_error']=(joined.vacancy_pair_fraction-joined.guide_target_raw)**2
 joined['clamp_target_squared_error']=(joined.vacancy_pair_fraction_clamp-joined.guide_target_raw)**2
 joined['predicted_initial_squared_error']=(joined.guide_prediction_initial-joined.guide_target_standardized)**2
 joined['predicted_final_squared_error']=(joined.guide_prediction_final-joined.guide_target_standardized)**2
 cols=['change_vs_clamp','change_vs_primary','orbit_change_vs_clamp','orbit_change_vs_primary','actual_target_squared_error','clamp_target_squared_error','predicted_initial_squared_error','predicted_final_squared_error']
 e=joined.groupby(cell)[cols].mean().reset_index();e['intervention']=inter;effects.append(e)
effects=pd.concat(effects,ignore_index=True);effects.to_csv(out/'guided_descriptor_effects_cells.csv',index=False)
seed_desc=desc.groupby(['family','seed','intervention'])[['mean_Cs_change','same_orbit_as_primary']].mean().reset_index();seed_desc.to_csv(out/'descriptor_seed_balanced.csv',index=False)
effects.groupby(['family','seed','intervention']).mean(numeric_only=True).to_csv(out/'guided_descriptor_effects_seed_balanced.csv')
# Five independent fits; all plotted error bars are SD across seeds unless stated otherwise.
fig,ax=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
for family in families:
 h=hh[hh.family==family];g=h.groupby('optimizer_updates')
 for axis,key in [(ax[0,0],'val_reconstruction'),(ax[0,1],'val_KL')]:
  m=g[key].mean();sd=g[key].std();axis.plot(m.index,m,label=pretty[family]);axis.fill_between(m.index,m-sd,m+sd,alpha=.10)
 if 'val_physical_posterior_MSE' in h and h.val_physical_posterior_MSE.notna().any():
  for axis,key in [(ax[1,0],'val_physical_posterior_MSE'),(ax[1,1],'val_physical_prior_MSE')]:
   m=g[key].mean();sd=g[key].std();axis.plot(m.index,m,label=pretty[family]);axis.fill_between(m.index,m-sd,m+sd,alpha=.10)
ax[0,0].set(ylabel='VAL reconstruction (nats / mask)',xlabel='Optimizer updates');ax[0,1].set(ylabel='VAL KL sum (nats / mask)',xlabel='Optimizer updates');ax[0,1].set_yscale('symlog',linthresh=.01)
ax[1,0].set(ylabel='VAL sampled-posterior C_s target MSE',xlabel='Optimizer updates');ax[1,1].set(ylabel='Independent-prior target MSE',xlabel='Optimizer updates')
ax[0,0].legend(fontsize=7,ncol=2);ax[1,0].legend(fontsize=7);fig.suptitle('Fixed 4,000-update fits: means ± one SD across five seeds; no convergence claim');fig.savefig(out/'optimization_curves.pdf');fig.savefig(out/'optimization_curves.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for k,probe in enumerate(['condition_only','posterior_mu_plus_condition','mu_fitted_probe_on_posterior_sample','mu_fitted_probe_on_independent_prior']):
 vals=[pr[(pr.family==f)&(pr.probe==probe)].MSE for f in families];means=np.array([v.mean() for v in vals]);sd=np.array([v.std() for v in vals]);x=np.arange(len(families))+(k-1.5)*.18;ax[0].errorbar(x,means,yerr=sd,fmt='o',capsize=2,label=probe.replace('_',' '))
ax[0].set(xticks=np.arange(len(families)),xticklabels=[pretty[f] for f in families],ylabel='VAL MSE (within-cell standardized C_s)',yscale='log');ax[0].tick_params(axis='x',rotation=45);ax[0].legend(fontsize=6)
for k,inter in enumerate(['guide_lower_quartile','guide_upper_quartile']):
 e=effects[effects.intervention==inter].groupby(['family','seed']).change_vs_clamp.mean();means=[];sd=[]
 for f in families:
  v=e.loc[f];means.append(v.mean());sd.append(v.std())
 ax[1].errorbar(np.arange(len(families))+(k-.5)*.18,means,yerr=sd,fmt='o',capsize=2,label=inter.replace('guide_','').replace('_',' '))
ax[1].axhline(0,c='grey',lw=.8);ax[1].set(xticks=np.arange(len(families)),xticklabels=[pretty[f] for f in families],ylabel='Actual generated C_s change vs clamp-only');ax[1].tick_params(axis='x',rotation=45);ax[1].legend(fontsize=7);fig.suptitle('Encoding a descriptor and controlling generated structures are separate tests');fig.savefig(out/'latent_physics_diagnostics.pdf');fig.savefig(out/'latent_physics_diagnostics.png',dpi=180);plt.close(fig)
if a.stage=='oracle':
 eligibility=json.loads((r/'eligibility_receipt.json').read_text());scores={};schemas={};fixed=pd.read_csv('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_angular_goal_20261110/final_project/runs/angular_20261110/iteration_main_train_thresholds.csv')
 old=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project/artifacts/overnight_20261107/data')
 pool=r/'eligible_diagnostic_union.npz'
 if sha(pool)!=eligibility['eligible_input_sha256']:raise ValueError('Eligible pool changed')
 with np.load(pool,allow_pickle=False) as z:eligible=set(z['structure_ids'].astype(str))
 table=[];energy_effects=[]
 for teacher in ['uma','mace']:
  collected=r/f'collected_{teacher}';cr=json.loads((collected/'receipt.json').read_text())
  if cr['status']!='PASS' or cr['model_counts']!={teacher:len(eligible)}:raise ValueError('Incomplete oracle')
  schemas[teacher]=load_schema(old/f'condition_schema_{teacher}.json')
  with np.load(collected/f'{teacher}.npz',allow_pickle=False) as z:
   ids=z['structure_ids'].astype(str);e=z['energy_eV']/z['n_atoms'];n=z['n_Li'];labels=classify(schemas[teacher],n,e)
   if set(ids)!=eligible:raise ValueError('Score IDs differ')
   lookup=dict(zip(ids,e));lab=dict(zip(ids,labels))
  f=raw.copy();f['eligible']=f.structure_id.isin(eligible);f['energy_eV_atom']=f.structure_id.map(lookup);f['measured_bin']=f.structure_id.map(lab);f['condition_hit']=f.eligible&(f.measured_bin==f.requested_bin)
  for key,g in f.groupby(group,sort=False):
   family,seed,n,bin_,inter=key;total=len(g);unique=g[g.eligible].drop_duplicates('structure_id');hit=unique[unique.measured_bin==bin_];q=float(fixed[(fixed.teacher==teacher)&(fixed.n_Li==n)].q20_eV_atom.iloc[0]);low=unique[unique.energy_eV_atom<=q]
   table.append(dict(teacher=teacher,family=family,seed=seed,n_target=n,requested_bin=bin_,intervention=inter,all_attempts=total,eligible_attempts=int(g.eligible.sum()),unique_eligible=len(unique),unique_condition_hit=len(hit),condition_hit_attempts=int(g.condition_hit.sum()),novel_yield=len(unique)/total,conditioned_novel_yield=len(hit)/total,bin_hit_per_all_attempts=float(g.condition_hit.sum()/total),eligible_bin_accuracy=float(g[g.eligible].condition_hit.mean()),fixed_q20_unique=len(low),fixed_q20_yield=len(low)/total,mean_eligible_energy=float(g.energy_eV_atom.mean()),minimum_eligible_energy=float(g.energy_eV_atom.min())))
  clamp_e=f[f.intervention=='latent_clamp_only'].set_index(cell+['attempt'])[['energy_eV_atom','eligible','structure_id']]
  primary_e=f[f.intervention=='primary'].set_index(cell+['attempt'])[['energy_eV_atom','eligible','structure_id']]
  for inter in ['guide_lower_quartile','guide_upper_quartile']:
   g=f[f.intervention==inter].set_index(cell+['attempt']);j=g.join(clamp_e,rsuffix='_clamp').join(primary_e,rsuffix='_primary');j['paired_eligible_clamp']=j.eligible&j.eligible_clamp;j['paired_eligible_primary']=j.eligible&j.eligible_primary
   j['energy_change_vs_clamp']=np.where(j.paired_eligible_clamp,j.energy_eV_atom-j.energy_eV_atom_clamp,np.nan);j['energy_change_vs_primary']=np.where(j.paired_eligible_primary,j.energy_eV_atom-j.energy_eV_atom_primary,np.nan)
   ee=j.groupby(cell)[['energy_change_vs_clamp','energy_change_vs_primary','paired_eligible_clamp','paired_eligible_primary']].mean().reset_index();ee['teacher']=teacher;ee['intervention']=inter;energy_effects.append(ee)
  bindings.extend([{'path':str(x),'sha256':sha(x)} for x in [collected/'receipt.json',collected/f'{teacher}.npz',old/f'condition_schema_{teacher}.json']])
 table=pd.DataFrame(table);table.to_csv(out/'oracle_yield_cells.csv',index=False)
 metric=['novel_yield','conditioned_novel_yield','bin_hit_per_all_attempts','eligible_bin_accuracy','mean_eligible_energy','minimum_eligible_energy']
 balanced=table.groupby(['teacher','family','seed','intervention'])[metric].mean().reset_index();balanced.to_csv(out/'oracle_yield_seed_balanced.csv',index=False)
 low=table[table.requested_bin=='low'].groupby(['teacher','family','seed','intervention'])[['fixed_q20_yield']].mean().reset_index();low.to_csv(out/'fixed_tail_seed_balanced.csv',index=False)
 summary=balanced.groupby(['teacher','family','intervention'])[metric].agg(['mean','std']);summary.to_csv(out/'oracle_yield_mean_SD.csv')
 pd.concat(energy_effects,ignore_index=True).to_csv(out/'guided_energy_effects_cells.csv',index=False)
 paired=[]
 primary_b=balanced[balanced.intervention=='primary']
 for teacher in ['uma','mace']:
  bb=primary_b[primary_b.teacher==teacher].set_index(['family','seed'])
  for f in families:
   for ref in ['ar','d2','lambda10_latent_d2','lambda10_condition_d2','lambda10_shuffled_d2','mean_lambda10_condition_d2','mean_lambda10_shuffled_d2']:
    for key in ['conditioned_novel_yield','novel_yield','bin_hit_per_all_attempts']:
     delta=bb.loc[f,key].sort_index()-bb.loc[ref,key].sort_index();paired.append(dict(teacher=teacher,family=f,reference=ref,metric=key,n=5,mean_difference=float(delta.mean()),SD_difference=float(delta.std()),CI95_halfwidth=float(2.776445105*delta.std()/np.sqrt(5)),seed_differences=json.dumps(delta.tolist())))
 pd.DataFrame(paired).to_csv(out/'paired_primary_effects.csv',index=False)
 fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
 for k,teacher in enumerate(['uma','mace']):
  p0=primary_b[primary_b.teacher==teacher];m=p0.groupby('family').conditioned_novel_yield.mean();sd=p0.groupby('family').conditioned_novel_yield.std();ax[k].bar(np.arange(len(families)),[100*m[f] for f in families],yerr=[100*sd[f] for f in families],capsize=3,color=['#999999']+['#4477AA']*3+['#66CCEE','#228833','#CCBB44','#EE6677']);ax[k].set(title=teacher.upper(),ylabel='Novel correctly conditioned orbits / all attempts (%)',xticks=np.arange(len(families)),xticklabels=[pretty[f] for f in families]);ax[k].tick_params(axis='x',rotation=45)
 fig.suptitle('Terminal 4,000-update generators: equal proposal budgets, mean ± SD of five fits');fig.savefig(out/'primary_oracle_yields.pdf');fig.savefig(out/'primary_oracle_yields.png',dpi=180);plt.close(fig)
receipt=dict(status='PASS',stage=a.stage,fit_replicates=5,checkpoint='last_update primary; validation-selected posterior metadata sensitivity only',assessment='New diagnostic development streams, not TEST',source_sha256=sha(__file__),bindings=bindings,raw_attempts=len(raw),interpretation='Descriptor/probe signals alone do not establish useful physical control; oracle yields and matched guide-vs-clamp effects are required. Error bars SD over five fits; paired CI uses Student-t(df4).')
(out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='bindings'},indent=2),flush=True)

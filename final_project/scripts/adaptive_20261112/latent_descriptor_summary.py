from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from cathode_cvae.runtime import require_compute_allocation
require_compute_allocation();import argparse
parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args();r=args.run_root.resolve();a=r/'analysis_descriptors_v2'
p=pd.read_csv(a/'VAL_probes_per_fit.csv');d=pd.read_csv(a/'guided_descriptor_effects_seed_balanced.csv');cp=pd.read_csv(a/'checkpoint_information_per_fit.csv');hh=pd.read_csv(a/'optimization_curves.csv')
summary={}
def paired(f,ref,probe='auxiliary_head_posterior_sample'):
 x=p[p.probe==probe].set_index(['family','seed']).MSE
 v=x.loc[f].sort_index()-x.loc[ref].sort_index()
 return dict(mean=float(v.mean()),SD=float(v.std()),CI95_halfwidth=float(2.776445105*v.std()/np.sqrt(5)),seeds=v.to_dict())
summary['VAL_genuine_minus_condition_MSE']=paired('physical_latent_d2','physical_condition_d2')
summary['VAL_genuine_minus_shuffled_MSE']=paired('physical_latent_d2','physical_shuffled_d2')
piv=d.pivot(index=['family','seed'],columns='intervention',values='change_vs_clamp')
piv['upper_minus_lower_actual_Cs']=piv.guide_upper_quartile-piv.guide_lower_quartile
summary['guided_actual_Cs_upper_minus_lower']={f:dict(mean=float(v.mean()),SD=float(v.std()),CI95_halfwidth=float(2.776445105*v.std()/np.sqrt(5))) for f,v in piv.groupby(level='family').upper_minus_lower_actual_Cs}
summary['genuine_head_predicted_loss_reduction']=float((d[d.family=='physical_latent_d2'].predicted_initial_squared_error-d[d.family=='physical_latent_d2'].predicted_final_squared_error).mean())
summary['genuine_actual_target_MSE_change']=float((d[d.family=='physical_latent_d2'].actual_target_squared_error-d[d.family=='physical_latent_d2'].clamp_target_squared_error).mean())
summary['condition_raw_exact_equality']={}
for seed in range(2026111301,2026111306):
 with np.load(r/f'diagnostics/d2_{seed}/raw_intervention_proposals.npz',allow_pickle=False) as z:x={k:z[k] for k in z.files}
 with np.load(r/f'diagnostics/physical_condition_d2_{seed}/raw_intervention_proposals.npz',allow_pickle=False) as z:y={k:z[k] for k in z.files}
 summary['condition_raw_exact_equality'][str(seed)]={k:bool(np.array_equal(x[k],y[k])) for k in x}
summary['best_checkpoint_vs_terminal']=cp.groupby(['family','selection'])[['optimizer_updates','VAL_KL','active001']].mean().reset_index().to_dict('records')
summary['terminal_validation'] = hh.sort_values('optimizer_updates').groupby(['family','seed']).tail(1).groupby('family')[['val_reconstruction','val_KL']].agg(['mean','std']).to_dict()
# JSON dict tuple keys are unsuitable; keep explicit rows instead.
summary['terminal_validation']=hh.sort_values('optimizer_updates').groupby(['family','seed']).tail(1).groupby('family')[['val_reconstruction','val_KL']].mean().reset_index().to_dict('records')
(a/'descriptor_findings.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['best_checkpoint_vs_terminal','condition_raw_exact_equality','terminal_validation']},indent=2))

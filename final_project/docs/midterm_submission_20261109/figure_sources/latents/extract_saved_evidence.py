"""Extract saved summaries only: no encoder, generator or physical model calls."""
from pathlib import Path
import csv, hashlib, json, math, shutil, statistics
HERE=Path(__file__).absolute().parent
BASE=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project')
OLD=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_next_goal_worktree/final_project')
DATA=HERE/'data'; DATA.mkdir(exist_ok=True)
sources={}
def register(path):
    path=Path(path); h=hashlib.sha256(path.read_bytes()).hexdigest();sources[str(path)]=h;return path
def read(path):
    with register(path).open(newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
    with (DATA/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fits=read(BASE/'reports/conditioning_20261107/latent_diagnostics/latent_fit_diagnostic_summary.csv')
perf=read(OLD/'runs/conditioning_20261107/evaluation/latent/seed_metrics.csv')
lookup={(r['method'],r['replicate_id']):r for r in perf if r['teacher']=='uma'}
dimension=[]
for row in fits:
    if row['method'] not in ['latent4','latent8','latent16','latent32']:continue
    point=lookup[row['method'],row['fit_seed']]
    dirname=('balance_1_d8_' if row['method']=='latent8' else 'latent_d'+row['latent_dim']+'_')+row['fit_seed']
    original_directory=OLD/'runs/conditioning_20261107/fits'/dirname
    cfg=json.loads(register(original_directory/'configuration.json').read_text());summary_original=json.loads(register(original_directory/'summary.json').read_text());latent_original=json.loads(register(original_directory/'best_validation_beta_0p1.pt.latent.json').read_text())
    assert cfg['encoder_kind']=='dense' and not cfg['graph_context'] and cfg['latent_dim']==int(row['latent_dim']) and cfg['hidden_dim']==64
    assert cfg['data_sha256']=='d2c8fffa90628660400420fbb4d81566335578bb18fab49f20b0018d4e31a465'
    selected_epoch=summary_original['selection_checkpoints']['0.1']['epoch']
    original_history=read(original_directory/'history.csv');selected_history=next(q for q in original_history if int(q['epoch'])==selected_epoch)
    assert float(selected_history['val_reconstruction'])==float(row['reconstruction_nats'])
    assert latent_original['pooled']['total_KL_nats']==float(row['total_KL_nats'])
    dimension.append(dict(study='DENSE_POSTERIOR_NO_SPATIAL_CONTEXT_DEVELOPMENT',method=row['method'],D=int(row['latent_dim']),seed=int(row['fit_seed']),selection_beta=0.1,selected_epoch=selected_epoch,parameter_count=int(row['parameter_count']),optimizer_updates=int(row['actual_updates']),useful_yield=float(point['macro_composition_bin_inbin_yield']),val_reconstruction_nats=float(row['reconstruction_nats']),val_total_KL_nats=float(row['total_KL_nats']),active_cutoff_0p01=int(row['active_at_0p01'])))
assert len(dimension)==12 and {r['D'] for r in dimension}=={4,8,16,32}
write('latent_dimension_fit_evidence.csv',dimension)
component=[];model_fit=[]
for family,label,count in [('A_radial_shells','A-shell',5),('C_dense_cvae','C-dense',3)]:
    directories=sorted((BASE/'runs/overnight_20261108/fits').glob(family+'_*'));assert len(directories)==count
    for directory in directories:
        path=register(directory/'best_validation_beta_0p1.pt.latent.json');record=json.loads(path.read_text());stats=record['pooled'];seed=int(directory.name.rsplit('_',1)[1]);assert stats['rows']==4967 and stats['latent_dim']==4
        total=stats['total_KL_nats'];model_fit.append(dict(family=label,fit=directory.name,seed=seed,selected_epoch=record['epoch'],VAL_rows=stats['rows'],KL_nats=total,conditional_information_upper_bound_bits=total/math.log(2),active_cutoff_0p001=stats['active_components_by_cutoff']['0.001'],active_cutoff_0p01=stats['active_components_by_cutoff']['0.01'],active_cutoff_0p1=stats['active_components_by_cutoff']['0.1']))
        for j in range(4):component.append(dict(family=label,fit=directory.name,seed=seed,component=j,selection_beta=0.1,posterior_mean=stats['posterior_mean_by_dimension'][j],variance_of_posterior_mean=stats['posterior_mean_variance_by_dimension'][j],mean_posterior_variance=stats['posterior_variance_mean_by_dimension'][j],component_KL_nats=stats['KL_nats_by_dimension'][j]))
write('latent_component_evidence.csv',component);write('latent_fit_evidence.csv',model_fit)
intervention=[]
for row in read(BASE/'runs/overnight_20261108/final/static_sensitivity_v2/by_fit.csv'):
    if row['teacher']!='uma' or row['method'] not in ['A_radial_shells','C_dense_cvae'] or row['contrast'] not in ['condition_high_minus_low','alternate_latent_minus_low']:continue
    assert row['complete_pairs']=='True' and row['compositions']=='3'
    intervention.append(dict(study='FROZEN_FINAL_UMA',family={'A_radial_shells':'A-shell','C_dense_cvae':'C-dense'}[row['method']],seed=int(row['replicate_id']),contrast=row['contrast'],mean_signed_delta_E_meV_atom=float(row['macro_mean_signed_delta_E_meV_atom']),mean_absolute_delta_E_meV_atom=float(row['macro_mean_absolute_delta_E_meV_atom']),RMS_delta_E_meV_atom=float(row['RMS_from_macro_mean_squared_delta_E_meV_atom']),mean_Hamming=float(row['macro_mean_raw_Hamming']),same_orbit_fraction=float(row['macro_same_orbit_fraction']),paired_draws=int(row['attempted_pairs'])))
assert len(intervention)==16;write('latent_intervention_fit_evidence.csv',intervention)
curves=[]
for name,label in [('A_radial_shells_2026110801','A-shell'),('C_dense_cvae_2026110801','C-dense')]:
    for row in read(BASE/'runs/overnight_20261108/fits'/name/'history.csv'):
        curves.append(dict(family=label,seed=2026110801,epoch=int(row['epoch']),updates=int(row['optimizer_updates']),beta=float(row['beta']),VAL_reconstruction_nats=float(row['val_reconstruction']),VAL_KL_nats=float(row['val_KL'])))
write('optimization_trace_evidence.csv',curves)
selected=[x for x in model_fit if x['family']=='A-shell'];a=[x for x in component if x['family']=='A-shell'];summary=dict(selected_shell_KL_nats=dict(mean=statistics.mean(x['KL_nats'] for x in selected),minimum=min(x['KL_nats'] for x in selected),maximum=max(x['KL_nats'] for x in selected)),selected_shell_information_upper_bound_bits=dict(mean=statistics.mean(x['conditional_information_upper_bound_bits'] for x in selected),minimum=min(x['conditional_information_upper_bound_bits'] for x in selected),maximum=max(x['conditional_information_upper_bound_bits'] for x in selected)),selected_shell_mu_variance_range=[min(x['variance_of_posterior_mean'] for x in a),max(x['variance_of_posterior_mean'] for x in a)],selected_shell_posterior_variance_range=[min(x['mean_posterior_variance'] for x in a),max(x['mean_posterior_variance'] for x in a)],information_identity='E_(m,c) KL[q(z|m,c)||p(z)] = I_q(m;z|c) + E_c KL[q(z|c)||p(z)]',interpretation='Upper bound for the specified base-weighted empirical VAL distribution; not an estimate of actual mutual information, physical entropy or semantic axes.',no_new_NN_or_MLIP_calls=True,no_training=True,scope_notes=['Dimension study is older dense/no-spatial DEVELOPMENT, not size convergence of the selected graph model.','All named selection payloads use validation_beta0.1.','Activity thresholds are descriptive; components within a fit are not independent fits.','Latent axes may change sign/permutation across fits.','Posterior collapse means input-independence, not a point distribution; broad variances near prior persist.','Optimization traces use one prespecified seed and fixed4000updates; they do not prove convergence.'])
(DATA/'latent_information_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(HERE/'source_hashes.json').write_text(json.dumps(dict(status='EXTRACTED_SAVED_EVIDENCE_ONLY',sources=sources,extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
print(json.dumps(dict(dimension_fits=len(dimension),posterior_fits=len(model_fit),posterior_component_rows=len(component),intervention_rows=len(intervention),optimization_rows=len(curves),summary=summary),indent=2))

"""Compact FINAL tables only after authoritative and independent metrics PASS."""
import csv,json,hashlib,shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
from cathode_cvae.runtime import require_compute_allocation
ROOT=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project');F=ROOT/'runs/overnight_20261108/final';B=ROOT/'reports/overnight_20261108/final_static'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def copy(source,target):
    if target.exists():assert source.read_bytes()==target.read_bytes()
    else:shutil.copy2(source,target)
def main():
    require_compute_allocation();ev=F/'static_evaluation';ind=F/'static_independent_metrics';sens=F/'static_sensitivity_v2'
    assert json.loads((ev/'receipt.json').read_text())['status']=='PASS' and json.loads((ind/'receipt.json').read_text())['status']=='PASS' and json.loads((sens/'receipt.json').read_text())['status']=='PASS'
    B.mkdir(exist_ok=True);summary=read(ev/'summary.csv');seeds=read(ind/'independent_seed_metrics.csv');metrics=read(ev/'metrics.csv')
    for src,name in [(ev/'summary.csv','summary.csv'),(ind/'contrast_summary.csv','contrast_summary.csv'),(ind/'perfit_contrasts.csv','perfit_contrasts.csv')]:copy(src,B/name)
    enriched=[]
    for r in summary:
        low=[float(q['original_q20_LOW_yield']) for q in seeds if q['teacher']==r['teacher'] and q['method']==r['method']]
        enriched.append(dict(r,descriptive_seed_SD_fixed_q20_LOW_request_yield=float(np.std(low,ddof=1))))
    write(B/'summary_with_LOW_SD.csv',enriched)
    selected=['A_radial_shells','B_radial_shells','mutation','uniform'];write(B/'selected_main_table.csv',[r for r in enriched if r['method'] in selected])
    mechanism=[]
    for teacher in ['uma','mace']:
        for method in ['A_graph_cvae','B_spatial_ar','C_dense_cvae','D_nonphysical_ar']:
            rr=[r for r in seeds if r['teacher']==teacher and r['method']==method and r['replicate_id'] in ['2026110801','2026110802','2026110803']];assert len(rr)==3
            y=[float(r['macro_yield']) for r in rr];low=[float(r['original_q20_LOW_yield']) for r in rr]
            mechanism.append(dict(teacher=teacher,method=method,common_fit_replicates=3,mean_macro_yield=float(np.mean(y)),descriptive_seed_SD=float(np.std(y,ddof=1)),mean_original_q20_LOW_yield=float(np.mean(low)),descriptive_seed_SD_original_q20_LOW_yield=float(np.std(low,ddof=1))))
    write(B/'mechanism_first3.csv',mechanism)
    write(B/'five_fit_2x2.csv',[r for r in enriched if r['method'] in ['A_graph_cvae','B_spatial_ar','A_radial_shells','B_radial_shells']])
    bybin=[]
    for teacher,method,bin in sorted({(r['teacher'],r['method'],r['requested_bin']) for r in metrics}):
        fit=[]
        for rep in sorted({r['replicate_id'] for r in metrics if r['teacher']==teacher and r['method']==method}):
            rr=[r for r in metrics if r['teacher']==teacher and r['method']==method and r['replicate_id']==rep and r['requested_bin']==bin]
            fit.append(float(np.mean([float(r['novel_inbin_yield_per_attempt']) for r in rr])))
        bybin.append(dict(teacher=teacher,method=method,requested_bin=bin,fit_or_sampler_replicates=len(fit),mean_novel_inbin_yield=float(np.mean(fit)),descriptive_seed_SD=float(np.std(fit,ddof=1))))
    write(B/'per_bin_summary.csv',bybin)
    sensitivity=read(sens/'summary.csv');latent=[];conditional=[]
    for r in sensitivity:
        row=dict(teacher=r['teacher'],method=r['method'],contrast=r['contrast'],fit_or_sampler_replicates=r['fit_or_sampler_replicates'],
            mean_signed_delta_E_meV_atom=r['mean_macro_mean_signed_delta_E_meV_atom'],mean_absolute_delta_E_meV_atom=r['mean_macro_mean_absolute_delta_E_meV_atom'],
            RMS_delta_E_meV_atom=r['mean_RMS_from_macro_mean_squared_delta_E_meV_atom'],mean_raw_Hamming=r['mean_macro_mean_raw_Hamming'],same_orbit_fraction=r['mean_macro_same_orbit_fraction'])
        (latent if r['contrast']=='alternate_latent_minus_low' else conditional).append(row)
    write(B/'latent_sensitivity.csv',latent);write(B/'conditional_response.csv',conditional)
    distance=F/'static_distances_v2';assert json.loads((distance/'receipt.json').read_text())['status']=='PASS'
    drows=read(distance/'metrics_by_seed.csv');ds=[]
    for method in sorted({r['method'] for r in drows}):
        rr=[r for r in drows if r['method']==method];v=[float(r['mean_nearest_TRAIN_reference_orbit_Hamming']) for r in rr]
        ds.append(dict(method=method,fit_or_sampler_replicates=len(rr),mean_nearest_reference_orbit_Hamming=float(np.mean(v)),descriptive_seed_SD=float(np.std(v,ddof=1)),reference_orbits_per_count=1024,scope='UPPER bound on nearest full-TRAIN distance; mobile-site Hamming units'))
    write(B/'structural_distance_summary.csv',ds)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,column,label in zip(axes,['macro_yield','original_q20_LOW_yield'],['Novel in requested bin / all attempts (%)','Novel below original q20; LOW request (%)']):
        for teacher,offset in [('uma',-.17),('mace',.17)]:
            vv=[[100*float(r[column]) for r in seeds if r['teacher']==teacher and r['method']==method] for method in selected]
            ax.bar(np.arange(4)+offset,[np.mean(v) for v in vv],.32,yerr=[np.std(v,ddof=1) for v in vv],label=teacher,capsize=3)
        ax.set(xticks=np.arange(4),xticklabels=['Shell CVAE','Shell AR','Mutation','Uniform'],ylabel=label);ax.legend()
    fig.suptitle('Fresh FINAL: five fits / sampler replicates; same UMA-conditioned proposals');fig.tight_layout();fig.savefig(B/'final_main_comparison.pdf');fig.savefig(B/'final_main_comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.5));methods=['A_graph_cvae','B_spatial_ar','C_dense_cvae','D_nonphysical_ar']
    for teacher,offset in [('uma',-.17),('mace',.17)]:
        rr={r['method']:r for r in mechanism if r['teacher']==teacher};ax.bar(np.arange(4)+offset,[100*rr[m]['mean_macro_yield'] for m in methods],.32,yerr=[100*rr[m]['descriptive_seed_SD'] for m in methods],label=teacher,capsize=3)
    ax.set(xticks=np.arange(4),xticklabels=['A: graph CVAE','B: no-latent AR','C: dense CVAE','D: scrambled AR'],ylabel='Novel in requested bin / all attempts (%)',title='Mechanism comparison: common first three fits');ax.legend();fig.tight_layout();fig.savefig(B/'final_mechanism_first3.pdf');fig.savefig(B/'final_mechanism_first3.png',dpi=150);plt.close(fig)
    receipt=dict(status='PASS_COMPACT_FINAL_TABLES',evaluation_receipt_sha256=sha(ev/'receipt.json'),independent_receipt_sha256=sha(ind/'receipt.json'),sensitivity_receipt_sha256=sha(sens/'receipt.json'),distance_receipt_sha256=sha(distance/'receipt.json'),
        source_sha256=sha(__file__),scope='Fresh FINAL outcomes only; no training, model selection, NN forwards or new scoring',
        uncertainty='Descriptive variation across independent fit/sampler replicates. Common first3 for A/B/C/D; full5 for the base/shell CVAE/AR2x2.',
        outputs={p.name:sha(p) for p in B.iterdir() if p.is_file()},new_NN_forwards=0,new_MLIP_calls=0,new_optimizer_updates=0)
    (B/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('PASS compact final tables and plots',B)
if __name__=='__main__':main()

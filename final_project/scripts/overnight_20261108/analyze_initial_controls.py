"""Descriptive, fit-grouped interpretation of complete initial DEV labels."""
import csv,json,hashlib,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from cathode_cvae.runtime import require_compute_allocation
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/overnight_20261108/initial_controls_v1'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def table(p,rr):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def mean(v):return float(np.mean(v))
def sd(v):return float(np.std(v,ddof=1)) if len(v)>1 else None
def main():
    require_compute_allocation();started=time.perf_counter();ev=OUT/'evaluation'
    receipt=json.loads((ev/'receipt.json').read_text());assert receipt['status']=='PASS'
    assert all(n==0 for n in receipt['missing_or_nonterminal_candidate_labels'].values())
    metrics=rows(ev/'metrics.csv');seeds=rows(ev/'seed_metrics.csv');paired=rows(ev/'paired_summary.csv')
    bybin=defaultdict(list)
    for r in metrics:bybin[r['teacher'],r['method'],r['requested_bin'],r['replicate_id']].append(r)
    br=[]
    for (teacher,method,b,rep),rr in sorted(bybin.items()):
        br.append(dict(teacher=teacher,method=method,requested_bin=b,replicate_id=rep,
            novel_inbin_yield=mean([float(r['novel_inbin_yield_per_attempt']) for r in rr]),
            inbin_hit_rate=mean([float(r['bin_hit_rate_per_attempt']) for r in rr]),
            novel_fixed_q20_yield=mean([float(r['novel_fixed_q20_yield_per_attempt']) for r in rr])))
    table(OUT/'per_bin_seed_summary.csv',br)
    bs=[]
    for teacher,method,b in sorted({(r['teacher'],r['method'],r['requested_bin']) for r in br}):
        rr=[r for r in br if (r['teacher'],r['method'],r['requested_bin'])==(teacher,method,b)]
        bs.append(dict(teacher=teacher,method=method,requested_bin=b,replicates=len(rr),
            mean_novel_inbin_yield=mean([r['novel_inbin_yield'] for r in rr]),
            SD_novel_inbin_yield=sd([r['novel_inbin_yield'] for r in rr]),
            mean_inbin_hit_rate=mean([r['inbin_hit_rate'] for r in rr])))
    table(OUT/'per_bin_macro_summary.csv',bs)
    diag=[]
    for teacher,method,rep,contrast in sorted({(r['teacher'],r['method'],r['replicate_id'],r['contrast']) for r in paired}):
        rr=[r for r in paired if (r['teacher'],r['method'],r['replicate_id'],r['contrast'])==(teacher,method,rep,contrast)]
        assert all(int(r['known_pairs'])==int(r['paired_attempts']) for r in rr)
        diag.append(dict(teacher=teacher,method=method,replicate_id=rep,contrast=contrast,
            mean_energy_shift_meV_atom=mean([float(r['mean_energy_shift_meV_atom']) for r in rr]),
            mean_raw_Hamming=mean([float(r['mean_raw_Hamming']) for r in rr]),
            same_orbit_fraction=mean([float(r['same_orbit_fraction']) for r in rr])))
    table(OUT/'diagnostics_by_fit.csv',diag)
    contrasts=[]
    lookup={(r['teacher'],r['method'],r['replicate_id']):r for r in seeds}
    for teacher in sorted({r['teacher'] for r in seeds}):
        for rep in sorted({r['replicate_id'] for r in seeds if r['teacher']==teacher}):
            ref=lookup[teacher,'B_spatial_ar',rep]
            for method in ['A_graph_cvae','C_dense_cvae','uniform','mutation']:
                r=lookup[teacher,method,rep]
                contrasts.append(dict(teacher=teacher,method=method,reference='B_spatial_ar',replicate_id=rep,
                    difference_macro_novel_inbin_yield=float(r['macro_composition_bin_yield'])-float(ref['macro_composition_bin_yield']),
                    difference_original_q20_LOW_yield=float(r['fixed_q20_LOW_request_yield'])-float(ref['fixed_q20_LOW_request_yield'])))
    table(OUT/'contrasts_to_spatial_ar_by_seed.csv',contrasts)
    summary=rows(ev/'summary.csv');uma=[r for r in summary if r['teacher']=='uma']
    rank=sorted(uma,key=lambda r:(float(r['mean_novel_inbin_yield']),float(r['mean_fixed_q20_LOW_request_yield'])),reverse=True)
    genuine=[r for r in rank if r['method'] in ['A_graph_cvae','C_dense_cvae']]
    result=dict(status='COMPLETE_DEV_DESCRIPTIVE_ANALYSIS',source_sha256=digest(__file__),
        evaluation_receipt_sha256=digest(ev/'receipt.json'),
        new_MLIP_calls=0,new_optimizer_updates=0,new_model_forward_calls=0,
        best_point_estimate_genuine_CVAE=genuine[0]['method'],simple_spatial_AR='B_spatial_ar',
        UMA_point_estimate_ranking=[dict(method=r['method'],macro_yield=float(r['mean_novel_inbin_yield']),
            original_q20_LOW_yield=float(r['mean_fixed_q20_LOW_request_yield'])) for r in rank],
        evidence_scope='New DEV only. Rank is a descriptive point estimate, not a final assessment or statistical superiority claim.',
        units='Energies meV/atom; Hamming counts mobile-site occupations; yields dimensionless all-attempt proportions',
        uncertainty='Means and descriptive SD across3 independent fits/sampler replicates after equal composition averaging; no proposal-row pseudoreplication',
        diagnostic_scope='Condition pairs share actual latent and AR uniforms; latent-resample probes share only AR uniforms; all are prior draws, not posterior reconstruction',
        elapsed_seconds=time.perf_counter()-started)
    (OUT/'interpretation_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=['A_graph_cvae','B_spatial_ar','C_dense_cvae','uniform','mutation'];teachers=['uma','mace']
    fig,axes=plt.subplots(1,2,figsize=(12,3.7))
    for ax,column,label in zip(axes,['macro_composition_bin_yield','fixed_q20_LOW_request_yield'],['Novel in requested bin / all attempts (%)','Novel below original q20; LOW request (%)']):
        for teacher,offset,color in [('uma',-.17,'C0'),('mace',.17,'C1')]:
            v=[[100*float(r[column]) for r in seeds if r['teacher']==teacher and r['method']==m] for m in methods]
            ax.bar(np.arange(5)+offset,[mean(q) for q in v],.32,yerr=[sd(q) for q in v],color=color,label=teacher,capsize=3)
        ax.set(xticks=np.arange(5),xticklabels=methods,ylabel=label);ax.tick_params(axis='x',rotation=20);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'prior_proposal_quality.pdf');fig.savefig(OUT/'prior_proposal_quality.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,3.7))
    for ax,contrast in zip(axes,['condition_middle_minus_low','condition_high_minus_low','alternate_latent_minus_low']):
        mm=methods if contrast!='alternate_latent_minus_low' else methods[:1]+methods[2:3]
        for teacher,offset,color in [('uma',-.17,'C0'),('mace',.17,'C1')]:
            v=[[float(r['mean_energy_shift_meV_atom']) for r in diag if r['teacher']==teacher and r['method']==m and r['contrast']==contrast] for m in mm]
            ax.bar(np.arange(len(mm))+offset,[mean(q) for q in v],.32,yerr=[sd(q) for q in v],color=color,label=teacher,capsize=3)
        ax.axhline(0,color='.3',lw=.8);ax.set(xticks=np.arange(len(mm)),xticklabels=mm,ylabel='Paired energy difference (meV/atom)',title=contrast);ax.tick_params(axis='x',rotation=20)
    axes[0].legend();fig.tight_layout();fig.savefig(OUT/'paired_condition_and_latent_response.pdf');fig.savefig(OUT/'paired_condition_and_latent_response.png',dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

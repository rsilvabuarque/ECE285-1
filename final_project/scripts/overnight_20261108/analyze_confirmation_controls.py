"""Three-fit DEV confirmation contrasts; FINAL-only fits are never read."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from cathode_cvae.runtime import require_compute_allocation
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/overnight_20261108/confirmation_v1'
REFERENCES={'A_radial_shells':'A_graph_cvae','B_radial_shells':'B_spatial_ar','B_remaining':'B_spatial_ar'}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    require_compute_allocation();ev=OUT/'evaluation';receipt=json.loads((ev/'receipt.json').read_text());assert receipt['status']=='PASS'
    rows=read(ev/'seed_metrics.csv');metrics=read(ev/'metrics.csv');contrasts=[]
    for teacher in ['uma','mace']:
        lookup={(r['method'],r['fit_seed']):r for r in rows if r['teacher']==teacher and r['fit_seed']}
        for method,reference in REFERENCES.items():
            for seed in ['2026110801','2026110802','2026110803']:
                a=lookup[method,seed];b=lookup[reference,seed]
                contrasts.append(dict(teacher=teacher,method=method,reference=reference,fit_seed=seed,
                    macro_yield=float(a['macro_composition_bin_yield']),reference_macro_yield=float(b['macro_composition_bin_yield']),
                    delta_macro_yield=float(a['macro_composition_bin_yield'])-float(b['macro_composition_bin_yield']),
                    original_q20_LOW_yield=float(a['fixed_q20_LOW_request_yield']),reference_original_q20_LOW_yield=float(b['fixed_q20_LOW_request_yield']),
                    delta_original_q20_LOW_yield=float(a['fixed_q20_LOW_request_yield'])-float(b['fixed_q20_LOW_request_yield'])))
    write(OUT/'paired_fit_contrasts.csv',contrasts)
    summary=[]
    for teacher,method in sorted({(r['teacher'],r['method']) for r in contrasts}):
        rr=[r for r in contrasts if r['teacher']==teacher and r['method']==method]
        summary.append(dict(teacher=teacher,method=method,reference=REFERENCES[method],independent_fit_pairs=len(rr),
            mean_delta_macro_yield=float(np.mean([r['delta_macro_yield'] for r in rr])),
            descriptive_SD_delta_macro_yield=float(np.std([r['delta_macro_yield'] for r in rr],ddof=1)),
            minimum_fit_delta_macro_yield=min(r['delta_macro_yield'] for r in rr),maximum_fit_delta_macro_yield=max(r['delta_macro_yield'] for r in rr),
            mean_delta_original_q20_LOW_yield=float(np.mean([r['delta_original_q20_LOW_yield'] for r in rr])),
            descriptive_SD_delta_original_q20_LOW_yield=float(np.std([r['delta_original_q20_LOW_yield'] for r in rr],ddof=1))))
    write(OUT/'paired_fit_contrast_summary.csv',summary)
    bybin=[]
    for teacher,method,bin in sorted({(r['teacher'],r['method'],r['requested_bin']) for r in metrics}):
        fit=[]
        for rep in sorted({r['replicate_id'] for r in metrics if r['teacher']==teacher and r['method']==method}):
            rr=[r for r in metrics if r['teacher']==teacher and r['method']==method and r['replicate_id']==rep and r['requested_bin']==bin]
            fit.append(float(np.mean([float(r['novel_inbin_yield_per_attempt']) for r in rr])))
        bybin.append(dict(teacher=teacher,method=method,requested_bin=bin,independent_replicates=len(fit),
            mean_novel_inbin_yield=float(np.mean(fit)),descriptive_seed_SD=float(np.std(fit,ddof=1))))
    write(OUT/'per_bin_summary.csv',bybin)
    aggregate=read(ev/'summary.csv');uma=[r for r in aggregate if r['teacher']=='uma']
    rank=lambda methods:sorted([r for r in uma if r['method'] in methods],key=lambda r:(float(r['mean_novel_inbin_yield']),float(r['mean_fixed_q20_LOW_request_yield'])),reverse=True)
    genuine=rank(['A_graph_cvae','A_radial_shells']);simple=rank(['B_spatial_ar','B_radial_shells','B_remaining'])
    result=dict(status='COMPLETE_THREE_FIT_DEV_CONFIRMATION',source_sha256=digest(__file__),evaluation_receipt_sha256=digest(ev/'receipt.json'),
        best_point_estimate_genuine_CVAE=genuine[0]['method'],best_point_estimate_simple_AR=simple[0]['method'],
        selection_rule='UMA equal-composition/bin macro all-attempt useful yield primary; original fixed-q20 LOW secondary; three fit seeds only',
        protected_assessment='FINAL-only0804/05 fits and all FINAL labels were not read. This is DEV selection evidence, not final confirmation.',
        uncertainty='Descriptive differences across three fit pairs, with common actual generation latent/uniform streams; no proposal-row p-values or equivalence claim',
        new_MLIP_calls=0,new_optimizer_updates=0,new_model_forward_calls=0,
        outputs={p.name:digest(p) for p in [OUT/'paired_fit_contrasts.csv',OUT/'paired_fit_contrast_summary.csv',OUT/'per_bin_summary.csv']})
    (OUT/'confirmation_interpretation_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=['A_graph_cvae','A_radial_shells','B_spatial_ar','B_radial_shells','B_remaining','uniform','mutation'];fig,axes=plt.subplots(1,2,figsize=(13,3.8))
    for ax,column,label in zip(axes,['macro_composition_bin_yield','fixed_q20_LOW_request_yield'],['Macro useful yield / all attempts (%)','Novel below original q20; LOW request (%)']):
        for teacher,offset in [('uma',-.17),('mace',.17)]:
            vv=[[100*float(r[column]) for r in rows if r['teacher']==teacher and r['method']==method] for method in methods]
            ax.bar(np.arange(len(methods))+offset,[np.mean(v) for v in vv],.32,yerr=[np.std(v,ddof=1) for v in vv],label=teacher,capsize=3)
        ax.set(xticks=np.arange(len(methods)),xticklabels=methods,ylabel=label);ax.tick_params(axis='x',rotation=25);ax.legend()
    fig.suptitle('Fresh DEV confirmation: three independent fits / sampler replicates');fig.tight_layout();fig.savefig(OUT/'three_fit_confirmation.pdf');fig.savefig(OUT/'three_fit_confirmation.png',dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

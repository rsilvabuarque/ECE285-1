"""First-fit controlled screen contrasts; no outcome-dependent thresholds."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from cathode_cvae.runtime import require_compute_allocation
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/overnight_20261108/screens_v1'
REFERENCES={'A_remaining':'A_graph_cvae','A_radial_shells':'A_graph_cvae','A_beta0p03':'A_graph_cvae',
            'B_remaining':'B_spatial_ar','B_radial_shells':'B_spatial_ar','D_nonphysical_ar':'B_spatial_ar'}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    require_compute_allocation();ev=OUT/'evaluation';receipt=json.loads((ev/'receipt.json').read_text());assert receipt['status']=='PASS'
    seedrows=read(ev/'seed_metrics.csv');metrics=read(ev/'metrics.csv');contrasts=[]
    for teacher in ['uma','mace']:
        lookup={(r['method'],r['fit_seed']):r for r in seedrows if r['teacher']==teacher and r['fit_seed']}
        for method,reference in REFERENCES.items():
            a=lookup[method,'2026110801'];b=lookup[reference,'2026110801']
            contrasts.append(dict(teacher=teacher,method=method,reference=reference,fit_seed=2026110801,
                macro_yield=float(a['macro_composition_bin_yield']),reference_macro_yield=float(b['macro_composition_bin_yield']),
                delta_macro_yield=float(a['macro_composition_bin_yield'])-float(b['macro_composition_bin_yield']),
                original_q20_LOW_yield=float(a['fixed_q20_LOW_request_yield']),reference_original_q20_LOW_yield=float(b['fixed_q20_LOW_request_yield']),
                delta_original_q20_LOW_yield=float(a['fixed_q20_LOW_request_yield'])-float(b['fixed_q20_LOW_request_yield']),
                inference_scope='First-fit DEV screen contrast only; no statistical superiority or three-fit feature confirmation'))
    write(OUT/'first_fit_contrasts.csv',contrasts)
    bybin=[]
    for r in contrasts:
        for bin in ['low','middle','high']:
            a=[q for q in metrics if q['teacher']==r['teacher'] and q['method']==r['method'] and q['fit_seed']=='2026110801' and q['requested_bin']==bin]
            b=[q for q in metrics if q['teacher']==r['teacher'] and q['method']==r['reference'] and q['fit_seed']=='2026110801' and q['requested_bin']==bin]
            assert len(a)==len(b)==3
            av=float(np.mean([float(q['novel_inbin_yield_per_attempt']) for q in a]));bv=float(np.mean([float(q['novel_inbin_yield_per_attempt']) for q in b]))
            bybin.append(dict(teacher=r['teacher'],method=r['method'],reference=r['reference'],fit_seed=2026110801,requested_bin=bin,
                macro_composition_novel_inbin_yield=av,reference_macro_composition_novel_inbin_yield=bv,delta_yield=av-bv))
    write(OUT/'first_fit_per_bin_contrasts.csv',bybin)
    rank=sorted([r for r in contrasts if r['teacher']=='uma' and r['method']!='D_nonphysical_ar'],
        key=lambda r:(r['delta_macro_yield'],r['delta_original_q20_LOW_yield']),reverse=True)
    result=dict(status='COMPLETE_DESCRIPTIVE_SCREEN',source_sha256=digest(__file__),evaluation_receipt_sha256=digest(ev/'receipt.json'),
        first_fit_UMA_ranking=rank,decision_rule='Primary UMA macro all-attempt useful-yield contrast; original fixed-q20 LOW secondary; root selects at most3 arms for two additional confirmation fits',
        D_limit='D has3fits while fresh A/B references have1fit in this DEV pool. First-seed D/B contrast is paired; the D three-fit mean is descriptive and cannot substitute for a matched three-fit comparison.',
        independent_units='Fit seed, not proposal row. No feature SD or p-value is estimated from this one-fit screen.',
        outputs={p.name:digest(p) for p in OUT.glob('*contrasts.csv')},new_model_forward_calls=0,new_MLIP_calls=0,new_optimizer_updates=0)
    (OUT/'screen_interpretation_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=list(REFERENCES);fig,axes=plt.subplots(1,2,figsize=(12,3.8))
    for ax,column,label in zip(axes,['delta_macro_yield','delta_original_q20_LOW_yield'],['Change in macro useful yield (percentage points)','Change in original-q20 LOW yield (percentage points)']):
        for teacher,offset in [('uma',-.17),('mace',.17)]:
            rr={r['method']:r for r in contrasts if r['teacher']==teacher}
            ax.bar(np.arange(len(methods))+offset,[100*rr[m][column] for m in methods],.32,label=teacher)
        ax.axhline(0,color='.25',lw=.8);ax.set(xticks=np.arange(len(methods)),xticklabels=methods,ylabel=label);ax.tick_params(axis='x',rotation=25);ax.legend()
    fig.suptitle('First fit only: each arm minus its matched fresh reference');fig.tight_layout();fig.savefig(OUT/'first_fit_screen_contrasts.pdf');fig.savefig(OUT/'first_fit_screen_contrasts.png',dpi=150);plt.close(fig)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

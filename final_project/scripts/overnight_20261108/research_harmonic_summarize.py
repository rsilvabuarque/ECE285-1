"""Summarize all phase-resolved harmonic null errors without dropping failures."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from cathode_cvae.data import sha256
from cathode_cvae.runtime import require_compute_allocation


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();require_compute_allocation()
    receipt=json.loads((args.analysis/'receipt.json').read_text())
    if receipt['status']!='COMPLETE_HARMONIC_NULL' or receipt['completed_analyses']!=27:raise ValueError('The full prespecified27-analysis null is required')
    args.output.mkdir(parents=True,exist_ok=False);rows=list(csv.DictReader((args.analysis/'analyses.csv').open()))
    summaries=[]
    fields=['F_error_classical_anchor_matched_eV','F_error_quantum_anchor_matched_eV','entropy_error_classical_eV_K',
        'entropy_error_quantum_eV_K','fluidicity','S_cage_quantum_eV_K','negative_dos_fraction_of_modes']
    for mode in ['1PT','2PT','3PT']:
        for length in [10.,20.,50.]:
            selected=[r for r in rows if r['mode']==mode and float(r['nominal_length_ps'])==length]
            if len(selected)!=3:raise ValueError('Missing phase realization')
            r=dict(mode=mode,length_ps=length,phase_realizations=3)
            for field in fields:
                values=np.array([float(x[field]) for x in selected]);r.update({field+'_mean':values.mean(),field+'_SD':values.std(ddof=1),field+'_min':values.min(),field+'_max':values.max()})
            summaries.append(r)
    with (args.output/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.6));colors={'1PT':'tab:blue','2PT':'tab:orange','3PT':'tab:green'}
    for mode in ['1PT','2PT','3PT']:
        subset=[r for r in summaries if r['mode']==mode];x=[r['length_ps'] for r in subset]
        for ax,field in zip(axes,['F_error_classical_anchor_matched_eV','F_error_quantum_anchor_matched_eV','fluidicity']):
            for seed in sorted({r['phase_seed'] for r in rows}):
                raw=sorted([r for r in rows if r['mode']==mode and r['phase_seed']==seed],key=lambda r:float(r['nominal_length_ps']))
                ax.plot([float(r['nominal_length_ps']) for r in raw],[float(r[field]) for r in raw],color=colors[mode],alpha=.22,lw=.8)
            ax.errorbar(x,[r[field+'_mean'] for r in subset],yerr=[r[field+'_SD'] for r in subset],label=mode,color=colors[mode],marker='o',capsize=3)
    axes[0].set(title='Classical anchor-matched error',ylabel='F estimator minus exact harmonic (eV/cell)')
    axes[1].set(title='Quantum anchor-matched error',ylabel='F estimator minus exact harmonic (eV/cell)')
    axes[2].set(title='Spurious fluidicity',ylabel='Fitted fluid-like fraction')
    for ax in axes:ax.set_xlabel('Observation length (ps)');ax.axhline(0,color='k',lw=.6);ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Actual537-mode synthetic harmonic null;3 phases; nested lengths,0.5fs samples',fontsize=10)
    fig.tight_layout();fig.savefig(args.output/'harmonic_null_bias.pdf');fig.savefig(args.output/'harmonic_null_bias.png',dpi=160);plt.close(fig)
    result=dict(status='COMPLETE_SUMMARY',analysis_receipt=dict(path=str((args.analysis/'receipt.json').resolve()),sha256=sha256(args.analysis/'receipt.json')),
        analyses_csv_sha256=sha256(args.analysis/'analyses.csv'),source_sha256=sha256(__file__),phase_realizations=3,analyses=27,
        max_absolute_energy_anchor_roundtrip_eV=max(abs(float(r['energy_anchor_roundtrip_error_eV'])) for r in rows),
        max_absolute_thermodynamic_identity_error_eV=max(abs(float(r['quantum_thermodynamic_identity_error_eV'])) for r in rows),
        max_absolute_mode_integral_error=max(abs(float(r['mode_integral_error'])) for r in rows),
        native_temperature_range_K=[min(float(r['observed_temperature_K']) for r in rows),max(float(r['observed_temperature_K']) for r in rows)],
        limitations='Fixed-energy/modal-phase null tests spectral estimators, not canonical MD or anharmonic corrections. Phase SD is descriptive; nested lengths are correlated. Nonzero fluidicity has no physical diffusion in this bounded analytic signal.',
        outputs={p.name:sha256(p) for p in args.output.iterdir() if p.is_file()})
    (args.output/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()

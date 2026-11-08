"""Fit-grouped absolute/RMS paired energy responses from saved DEV outcomes.

Changing two draws from the same prior has expected signed energy difference
zero, even when latent changes strongly alter the output. Absolute and RMS
differences quantify that response; they are not energy prediction errors.
"""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from cathode_cvae.data import sha256
from cathode_cvae.runtime import require_compute_allocation


def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    if not rows:Path(path).write_text('');return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def truth(value):return value is True or value=='True'


def summarize(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['teacher'],r['method'],r['replicate_id'],int(r['n_Li']),r['contrast']].append(r)
    per_composition=[]
    for (teacher,method,rep,n,contrast),rr in sorted(groups.items()):
        known=[r for r in rr if truth(r['both_energies_known'])]
        delta=np.asarray([float(r['energy_shift_meV_atom']) for r in known])
        if not np.isfinite(delta).all():raise ValueError('A measured paired energy difference is nonfinite')
        per_composition.append(dict(teacher=teacher,method=method,replicate_id=rep,n_Li=n,contrast=contrast,
            attempted_pairs=len(rr),known_pairs=len(known),
            mean_signed_delta_E_meV_atom=float(delta.mean()) if len(delta) else '',
            mean_absolute_delta_E_meV_atom=float(np.abs(delta).mean()) if len(delta) else '',
            mean_squared_delta_E_meV2_atom2=float(np.square(delta).mean()) if len(delta) else '',
            RMS_delta_E_meV_atom=float(np.sqrt(np.square(delta).mean())) if len(delta) else '',
            fraction_positive_delta=float((delta>0).mean()) if len(delta) else '',
            fraction_zero_delta=float((delta==0).mean()) if len(delta) else '',
            mean_raw_Hamming=float(np.mean([float(r['raw_Hamming']) for r in known])) if known else '',
            same_orbit_fraction=float(np.mean([truth(r['same_orbit']) for r in known])) if known else ''))
    by_fit=defaultdict(list)
    for r in per_composition:by_fit[r['teacher'],r['method'],r['replicate_id'],r['contrast']].append(r)
    fits=[]
    for (teacher,method,rep,contrast),rr in sorted(by_fit.items()):
        complete=all(r['known_pairs']==r['attempted_pairs'] for r in rr)
        fits.append(dict(teacher=teacher,method=method,replicate_id=rep,contrast=contrast,
            compositions=len(rr),attempted_pairs=sum(r['attempted_pairs'] for r in rr),known_pairs=sum(r['known_pairs'] for r in rr),complete_pairs=complete,
            macro_mean_signed_delta_E_meV_atom=float(np.mean([r['mean_signed_delta_E_meV_atom'] for r in rr])) if complete else '',
            macro_mean_absolute_delta_E_meV_atom=float(np.mean([r['mean_absolute_delta_E_meV_atom'] for r in rr])) if complete else '',
            RMS_from_macro_mean_squared_delta_E_meV_atom=float(np.sqrt(np.mean([r['mean_squared_delta_E_meV2_atom2'] for r in rr]))) if complete else '',
            macro_fraction_positive_delta=float(np.mean([r['fraction_positive_delta'] for r in rr])) if complete else '',
            macro_fraction_zero_delta=float(np.mean([r['fraction_zero_delta'] for r in rr])) if complete else '',
            macro_mean_raw_Hamming=float(np.mean([r['mean_raw_Hamming'] for r in rr])) if complete else '',
            macro_same_orbit_fraction=float(np.mean([r['same_orbit_fraction'] for r in rr])) if complete else ''))
    by_method=defaultdict(list)
    for r in fits:by_method[r['teacher'],r['method'],r['contrast']].append(r)
    summary=[]
    columns=['macro_mean_signed_delta_E_meV_atom','macro_mean_absolute_delta_E_meV_atom',
        'RMS_from_macro_mean_squared_delta_E_meV_atom','macro_fraction_positive_delta',
        'macro_fraction_zero_delta','macro_mean_raw_Hamming','macro_same_orbit_fraction']
    for (teacher,method,contrast),rr in sorted(by_method.items()):
        r=dict(teacher=teacher,method=method,contrast=contrast,fit_or_sampler_replicates=len(rr),all_pairs_complete=all(q['complete_pairs'] for q in rr))
        for column in columns:
            values=[q[column] for q in rr if q['complete_pairs']]
            r['mean_'+column]=float(np.mean(values)) if values else ''
            r['descriptive_seed_SD_'+column]=float(np.std(values,ddof=1)) if len(values)>1 else ''
        summary.append(r)
    return per_composition,fits,summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--paired-attempts',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv);require_compute_allocation();rows=read_csv(a.paired_attempts)
    if not rows:raise ValueError('No paired attempts were supplied')
    per_composition,fits,summary=summarize(rows);a.output_dir.mkdir(parents=True,exist_ok=False)
    for name,value in [('by_composition_fit.csv',per_composition),('by_fit.csv',fits),('summary.csv',summary)]:write_csv(a.output_dir/name,value)
    result=dict(status='PASS' if all(r['complete_pairs'] for r in fits) else 'INCOMPLETE_PAIRS',
        source_sha256=sha256(__file__),input=dict(path=str(a.paired_attempts.resolve()),sha256=sha256(a.paired_attempts)),
        units='delta E in meV/atom; MSE in (meV/atom)^2; Hamming in mobile-site occupations',
        definitions={'signed':'mean_j(delta_E_j)','absolute':'mean_j(abs(delta_E_j))','RMS':'sqrt(mean_j(delta_E_j^2))',
            'fit_macro':'Equal-weight composition mean; fit-level RMS is sqrt(equal-weight composition mean of squared differences)',
            'seed_summary':'Descriptive mean/SD across independent fit/sampler replicates, never proposal rows'},
        interpretation={'latent_resample':'Both latent draws have the same prior and share AR uniforms. Expected signed delta E is zero by identical marginals. Nonzero absolute/RMS response measures sensitivity, not conditional utility or energy accuracy.',
            'conditions':'LOW to MIDDLE/HIGH shares actual latent and AR uniforms. Positive signed delta E is the direction of the requested energy change.',
            'mutation':'Condition changes select different TRAIN-bin seed pools; common seed-choice quantiles/edit streams do not make those seed identities equal.'},
        outputs={p.name:sha256(p) for p in a.output_dir.iterdir()},new_model_forward_calls=0,new_MLIP_calls=0,new_optimizer_updates=0)
    (a.output_dir/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result


if __name__=='__main__':main()

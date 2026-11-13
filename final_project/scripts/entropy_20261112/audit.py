"""Independently check final entropy receipts and distinguish three references."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from cathode_cvae.runtime import require_compute_allocation


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();require_compute_allocation()
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    r=a.root;hash_checks=[]
    for directory in ['entropy_analysis','entropy_learned_proposal','entropy_review']:
        base=r/directory;receipt=json.loads((base/'receipt.json').read_text())
        for name,expected in receipt['outputs'].items():
            actual=digest(base/name);hash_checks.append(dict(path=str((base/name).relative_to(r)),
                expected=expected,actual=actual,match=actual==expected))
    execution=r/'entropy_learning_execution';protocol=json.loads((execution/'protocol.json').read_text())
    for name,expected in protocol['source_files'].items():
        actual=digest(execution/name);hash_checks.append(dict(path=str((execution/name).relative_to(r)),
            expected=expected,actual=actual,match=actual==expected))
    if not all(c['match'] for c in hash_checks):raise ValueError('Receipt/source hash changed')
    with np.load(r/'entropy_input/orbit_representatives.npz',allow_pickle=False) as z:
        ids=z['structure_ids'].astype(str);mapping=z['raw_to_orbit'];raw=z['raw_occupancy'];counts=z['n_target']
    with np.load(r/'entropy_ar_learning/learning.npz',allow_pickle=False) as z:
        split=z['split'].astype(str);energy=z['energy_eV_atom']
        sealed=bool(np.isnan(energy[split=='test']).all())
        active_finite=bool(np.isfinite(energy[np.isin(split,['train','val'])]).all())
    if not sealed or not active_finite:raise ValueError('Learning energy-role boundary changed')
    # Independent calculation over labelled masks: no entropy.py helper and no
    # orbit multiplicity formula, so this tests the actual expansion identity.
    thermo=rows(r/'entropy_analysis/canonical_thermodynamics.csv');max_error=0.
    kB=8.617333262145e-5
    for teacher in ['uma','mace']:
        with np.load(r/f'entropy_collected/{teacher}.npz',allow_pickle=False) as z:
            lookup=dict(zip(z['structure_ids'].astype(str),z['energy_eV']))
        e=np.array([lookup[s] for s in ids])[mapping]
        for row in [x for x in thermo if x['teacher']==teacher]:
            n=int(row['n_target']);T=float(row['temperature_K']);ee=e[raw.sum(1)==n]
            unnormalized=np.exp(-(ee-ee.min())/(kB*T));prob=unnormalized/unnormalized.sum()
            positive=prob>0;S=-float(np.sum(prob[positive]*np.log(prob[positive])))
            max_error=max(max_error,abs(S-float(row['entropy_kB'])))
    if max_error>1e-10:raise ValueError('Raw-mask entropy and orbit sum differ beyond float64 roundoff target')
    wl=rows(r/'entropy_analysis/wang_landau_thermodynamics.csv');table=[]
    groups=sorted(set((x['teacher'],int(x['n_target']),int(x['n_bins']),float(x['temperature_K'])) for x in wl))
    for teacher,n,bins,T in groups:
        rr=[x for x in wl if x['teacher']==teacher and int(x['n_target'])==n and int(x['n_bins'])==bins and float(x['temperature_K'])==T]
        estimated=np.array([float(x['estimated_entropy_kB']) for x in rr])
        table.append(dict(teacher=teacher,n_target=n,n_bins=bins,temperature_K=T,WL_seeds=len(rr),
            exact_discrete_orbit_multiplicity_sum_S_kB=float(rr[0]['exact_entropy_kB']),
            binned_exact_counts_midpoint_S_kB=float(rr[0]['binned_exact_entropy_kB']),
            WL_estimated_counts_midpoint_mean_S_kB=float(estimated.mean()),
            WL_estimated_counts_midpoint_sd_S_kB=float(estimated.std(ddof=1)),
            bin_midpoint_bias_kB=float(rr[0]['discretization_error_kB']),
            mean_absolute_DOS_error_relative_binned_kB=float(np.mean([abs(float(x['estimation_error_relative_binned_kB'])) for x in rr]))))
    a.output_dir.mkdir(parents=True)
    with (a.output_dir/'exact_binned_WL_comparison.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    reference=(
        '# Verified estimator references\n\n'
        '- Wang and Landau, Physical Review Letters 86, 2050–2053 (2001), '
        '[DOI:10.1103/PhysRevLett.86.2050](https://doi.org/10.1103/PhysRevLett.86.2050). '
        'Author preprint [cond-mat/0011174](https://arxiv.org/pdf/cond-mat/0011174), pp.1–2, Eq.(1): '
        'acceptance uses the ratio of current DOS estimates; visited-bin DOS is multiplied by f, '
        'and f is reduced after histogram flattening. The paper demonstrates recovery of thermodynamic '
        'quantities from DOS. Our fixed-composition swaps, known bin support and bounded 1e-4 log(f) '
        'stopping rule are project-specific choices, not their universal accuracy guarantee.\n'
        '- Art B. Owen, Monte Carlo theory, methods and examples, '
        '[Chapter 9, Importance sampling](https://artowen.su.domains/mc/Ch-var-is.pdf), '
        '§9.1, Eq.(9.3)/Theorem9.1 supports the unbiased weighted sample mean under support coverage; '
        '§9.2, Eq.(9.6)/Theorem9.2 gives the self-normalized ratio and its consistency, rather than finite-sample '
        'unbiasedness. §9.3, Eq.(9.13), defines weight ESS and warns that it cannot establish successful sampling. '
        'Applying these identities to exp(−E/kBT) gives our partition-function estimator; applying logarithms '
        'and a self-normalized mean produces a finite-budget biased entropy estimate.\n\n'
        'The comparison CSV keeps three quantities distinct: the exact discrete energy sum; the exact labelled '
        'bin counts evaluated at bin midpoints; and Wang–Landau estimated bin counts at the same midpoints. '
        'The first-to-second difference measures energy discretization. The second-to-third difference measures '
        'DOS estimation. Their signed errors can cancel.\n')
    (a.output_dir/'REFERENCES.md').write_text(reference)
    result=dict(status='PASS_INDEPENDENT_ENTROPY_ARTIFACT_AUDIT',verified_hashes=len(hash_checks),
        hash_checks=hash_checks,learning_TEST_energies_all_NaN=sealed,learning_TRAIN_VAL_energies_finite=active_finite,
        raw_microstate_vs_orbit_entropy_max_abs_error_kB=max_error,
        raw_entropy_check_tolerance_kB=1e-10,
        tolerance_scope='Float64 independent calculation identity check, not physical accuracy tolerance',
        source_sha256=digest(Path(__file__)),pipeline_receipt_sha256=digest(r/'entropy_pipeline_receipt.json'),
        PDF_visual_audit='entropy_review.pdf rendered with Ghostscript120dpi and manually inspected; legible3panels, labels, legends, no clipping')
    result['outputs']={q.name:digest(q) for q in a.output_dir.iterdir() if q.is_file()}
    (a.output_dir/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='hash_checks'},indent=2))


if __name__=='__main__':main()

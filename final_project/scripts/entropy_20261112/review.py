"""Summarize completed N12 entropy diagnostics without changing fitted models.

Orientation differences are scaled by kBT rather than subjected to an arbitrary
energy pass threshold. The reference remains a finite fixed-lattice MLIP model.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from cathode_cvae.data import sha256
from cathode_cvae.entropy import KB_EV_K
from cathode_cvae.runtime import require_compute_allocation


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(); require_compute_allocation()
    pipeline = json.loads((a.root/'entropy_pipeline_receipt.json').read_text())
    if pipeline['status'] != 'COMPLETE_ENTROPY_PIPELINE':
        raise ValueError('Need completed frozen pipeline')
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    analysis = a.root/'entropy_analysis'; learned = a.root/'entropy_learned_proposal'
    orientation = read_csv(analysis/'symmetry_orientation_energy_checks.csv')
    exact = read_csv(analysis/'canonical_thermodynamics.csv')
    multiplicity = read_csv(analysis/'multiplicity_effect.csv')
    wl = read_csv(analysis/'wang_landau_summary.csv')
    wl_thermo = read_csv(analysis/'wang_landau_thermodynamics.csv')
    imp = read_csv(learned/'learned_proposal_importance.csv')
    quality = read_csv(learned/'learned_proposal_quality.csv')
    normalization = read_csv(learned/'AR_normalization.csv')
    teachers = sorted(set(r['teacher'] for r in exact))
    temperatures = sorted(set(float(r['temperature_K']) for r in exact))
    terminal = sorted(set(r['model'] for r in imp if r['model'].endswith('_last')))
    a.output_dir.mkdir(parents=True)
    scaled = []; orientation_max = {}
    for teacher in teachers:
        rows = [r for r in orientation if r['teacher'] == teacher]
        difference = np.array([float(r['difference_eV']) for r in rows])
        max_delta = float(np.max(np.abs(difference))); orientation_max[teacher] = max_delta
        for T in temperatures:
            ratio = max_delta/(KB_EV_K*T)
            scaled.append(dict(teacher=teacher,temperature_K=T,orientation_pairs=len(rows),
                maximum_absolute_delta_eV=max_delta,rms_delta_eV=float(np.sqrt(np.mean(difference**2))),
                maximum_absolute_delta_over_kBT=ratio,
                largest_single_state_Boltzmann_factor_ratio=math.exp(ratio),
                scope='36 tested orientations; not a bound for every untested symmetry image'))
    write_csv(a.output_dir/'symmetry_energy_temperature_scale.csv',scaled)
    # Average draw-seed errors within each independent fitted model first. The
    # spread of three fit means is descriptive, not an independent 9-fit CI.
    comparisons = []
    for teacher in teachers:
        for n in (3,6,9):
            for T in (300.,1000.,5000.):
                for draws in (128,512,2048,8192):
                    for tag,models in [('terminal_AR_mixture',terminal),('uniform_microstate',['uniform_microstate'])]:
                        selected = [r for r in imp if r['teacher']==teacher and int(r['n_target'])==n
                            and float(r['temperature_K'])==T and int(r['draws'])==draws and r['model'] in models]
                        means=[];naive=[];ess=[]
                        for model in models:
                            rr=[r for r in selected if r['model']==model]
                            means.append(float(np.mean([abs(float(r['importance_entropy_kB'])-float(r['exact_entropy_kB'])) for r in rr])))
                            naive.append(float(np.mean([abs(float(r['naive_microstate_histogram_entropy_kB'])-float(r['exact_entropy_kB'])) for r in rr])))
                            ess.append(float(np.mean([float(r['importance_ESS'])/draws for r in rr])))
                        comparisons.append(dict(teacher=teacher,n_target=n,temperature_K=T,draws=draws,
                            method=tag,fits=len(models),draw_seeds_per_fit=len(selected)//len(models),
                            mean_absolute_importance_entropy_error_kB=float(np.mean(means)),
                            sd_fit_mean_absolute_error_kB=float(np.std(means,ddof=1)) if len(means)>1 else 0.,
                            mean_absolute_naive_entropy_error_kB=float(np.mean(naive)),
                            mean_weight_ESS_fraction=float(np.mean(ess))))
    write_csv(a.output_dir/'terminal_AR_uniform_comparison.csv',comparisons)
    # Restrict compact reporting to actual independent terminal fits; all best
    # checkpoint sensitivities remain in the full original CSVs.
    quality_summary=[]
    for teacher in teachers:
        for n in (3,6,9):
            for T in (300.,1000.,5000.):
                rr=[r for r in quality if r['teacher']==teacher and int(r['n_target'])==n
                    and float(r['temperature_K'])==T and r['model'] in terminal]
                quality_summary.append(dict(teacher=teacher,n_target=n,temperature_K=T,
                    mean_proposal_vs_physical_TV=float(np.mean([float(r['proposal_vs_physical_total_variation']) for r in rr])),
                    mean_physical_to_proposal_KL_nats=float(np.mean([float(r['physical_to_proposal_KL_nats']) for r in rr]))))
    write_csv(a.output_dir/'terminal_AR_target_mismatch.csv',quality_summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(14,4.2))
    for teacher in teachers:
        rr=[r for r in scaled if r['teacher']==teacher]
        axs[0].plot([r['temperature_K'] for r in rr],
            [r['maximum_absolute_delta_over_kBT'] for r in rr],marker='o',label=teacher)
        for method,style in [('terminal_AR_mixture','-'),('uniform_microstate','--')]:
            rr=[r for r in comparisons if r['teacher']==teacher and r['n_target']==6
                and r['temperature_K']==1000. and r['method']==method]
            axs[1].plot([r['draws'] for r in rr],
                [r['mean_absolute_importance_entropy_error_kB'] for r in rr],style,
                marker='o',label=f'{teacher}: '+('AR mixture' if method.startswith('terminal') else 'uniform'))
        for bins,style in [(16,'-'),(32,'--')]:
            rr=[r for r in wl_thermo if r['teacher']==teacher and int(r['n_target'])==6 and int(r['n_bins'])==bins]
            values=[np.mean([abs(float(r['estimated_entropy_kB'])-float(r['exact_entropy_kB']))
                for r in rr if float(r['temperature_K'])==T]) for T in temperatures]
            axs[2].plot(temperatures,values,style,marker='o',label=f'{teacher}, {bins} bins')
    for ax in axs:
        ax.set_xscale('log');ax.grid(alpha=.2);ax.legend(fontsize=7)
    axs[0].set_xlabel('Temperature (K)');axs[0].set_ylabel('Tested max |orientation ΔE| / kBT')
    axs[1].set_xlabel('Proposal draws, Li6/12 at 1000 K');axs[1].set_ylabel('Mean absolute importance S error / kB')
    axs[2].set_xlabel('Temperature (K), Li6/12');axs[2].set_ylabel('Mean absolute WL S error / kB')
    fig.suptitle('Finite fixed-lattice estimator checks: 87 orbits, 4096 labelled states')
    fig.tight_layout();fig.savefig(a.output_dir/'entropy_review.pdf');plt.close(fig)
    converged=sum(r['converged']=='True' for r in wl)
    lines=['# Small-cell entropy validation: measured results','',
        'This is an idealized O3 Li12Co12O24 fixed-cell vacancy lattice. Its 4096 labelled masks form 87 symmetry orbits. '
        'Every orbit representative was scored by both UMA and MACE, with 36 additional as-written symmetry checks per model. '
        'No DFT or relaxation labels are used; the teacher energy is the total eV per periodic cell.','',
        'The symmetry checks measure numerical consistency of the MLIP under the lattice operations. They do not validate '
        'physical energies, nor bound every untested orientation. Multiplicity-expanded thermodynamics assumes invariance '
        'to the numerical scale shown below.','']
    for teacher in teachers:
        rr=[r for r in scaled if r['teacher']==teacher and r['temperature_K']==100.][0]
        lines.append(f'- {teacher}: maximum tested |ΔE| = {rr["maximum_absolute_delta_eV"]:.8g} eV per cell; '
            f'|ΔE|/(kB × 100 K) = {rr["maximum_absolute_delta_over_kBT"]:.6g}; '
            f'single-state Boltzmann-factor ratio ≤ {rr["largest_single_state_Boltzmann_factor_ratio"]:.6g} '
            'over those tested pairs.')
    lines+=['','For fixed composition n, Z = Σo go exp(−Eo/kBT), where go is the number of labelled masks '
        'in orbit o. S/kB = ln Z + ⟨E⟩/(kBT); constant energy offsets cancel. The infinite-temperature limit is '
        'ln[C(12,n)], not ln(number of symmetry orbits). These are finite-cell configurational quantities, '
        'excluding vibrations, spin states, strain, phase transformations, and inter-composition exchange.','',
        f'The bounded Wang–Landau experiment completed {len(wl)} runs; {converged} met its prospectively specified '
        'flatness/modification-factor schedule. Exact nonempty-bin support was supplied. A successful schedule is '
        'a numerical diagnostic rather than proof of unknown-space coverage; midpoint binning and DOS errors remain distinct.','',
        'Three composition-only autoregressive models were trained on the frozen 57 TRAIN / 15 VAL / 15 TEST orbit split '
        'with multiplicity weights and symmetry augmentation. TEST energies are NaN in the learning export. The terminal '
        '4000-update checkpoints are primary; validation-selected checkpoints are retained as sensitivity results. '
        'No physical energy target was conditioned on during this tiny entropy test.','',
        'Each learned proposal is mixed with 10% uniform labelled-mask probability to guarantee support. Exact AR '
        'likelihoods were normalized by exhaustive enumeration, then independently checked against decoder samples. '
        'The proposal distribution is not a Boltzmann distribution. Importance weights divide by the actual proposal '
        'probability; naive generated-sample frequencies estimate proposal entropy instead. Partition-function sample '
        'means are unbiased under full support; self-normalized energies and the resulting entropy retain finite-budget bias.','',
        'The comparison table first averages over three draw seeds within each fit, then averages the three independent '
        'fit means. The reported fit spread is descriptive; it is not a confidence interval from nine independent fits. '
        'Weight ESS measures weight concentration and does not measure Markov-chain independence.','']
    for teacher in teachers:
        for bins in (16,32):
            rr=[r for r in wl_thermo if r['teacher']==teacher and int(r['n_target'])==6
                and float(r['temperature_K'])==300. and int(r['n_bins'])==bins]
            lines.append(f'- {teacher}, Li6/12, 300 K, {bins} WL bins: midpoint-only entropy bias '
                f'{float(rr[0]["discretization_error_kB"]):+.6g} kB per cell; mean absolute DOS-estimation '
                f'error relative to the binned reference {np.mean([abs(float(r["estimation_error_relative_binned_kB"])) for r in rr]):.6g} kB. '
                'The two contributions may cancel; small total error alone does not establish either convergence.')
    lines.append('')
    for teacher in teachers:
        for method in ['terminal_AR_mixture','uniform_microstate']:
            rr=[r for r in comparisons if r['teacher']==teacher and r['n_target']==6
                and r['temperature_K']==1000. and r['draws']==8192 and r['method']==method][0]
            lines.append(f'- {teacher}, Li6/12, 1000 K, 8192 draws, {method}: importance entropy MAE '
                f'{rr["mean_absolute_importance_entropy_error_kB"]:.5g} kB per cell; naive entropy MAE '
                f'{rr["mean_absolute_naive_entropy_error_kB"]:.5g} kB; weight ESS/draws '
                f'{rr["mean_weight_ESS_fraction"]:.5g}.')
    lines+=['','These checks establish estimator behavior for a complete small lattice and the chosen MLIP energies. '
        'They do not establish physical LCO entropy, transfer to N48, or a likelihood-corrected CVAE entropy estimator. '
        'A CVAE would require its latent-marginal proposal probability, not the decoder likelihood at one sampled latent vector.','']
    (a.output_dir/'RESULTS.md').write_text('\n'.join(lines))
    receipt=dict(status='COMPLETE_ENTROPY_REVIEW',pipeline_receipt_sha256=sha256(a.root/'entropy_pipeline_receipt.json'),
        source_sha256=sha256(Path(__file__)),orientation_max_absolute_delta_eV=orientation_max,
        normalization_max_absolute_error=max(float(r['absolute_error']) for r in normalization),
        wl_converged=converged,wl_runs=len(wl),
        input_hashes={str(q.relative_to(a.root)):sha256(q) for q in [analysis/'receipt.json',learned/'receipt.json']})
    receipt['outputs']={q.name:sha256(q) for q in a.output_dir.iterdir() if q.is_file()}
    (a.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()

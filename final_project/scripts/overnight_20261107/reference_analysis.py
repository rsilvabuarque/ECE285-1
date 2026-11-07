"""Join the complete prespecified raw-uniform reference to measured UMA energies.

No inference is performed. Prefixes are nested sensitivity checks, not independent
replicates or confidence intervals. All weights use total eV/cell and retained
raw sampling frequencies, never an additional symmetry multiplicity.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cathode_cvae.acquisition import (main as acquisition_main, static_weights,
    bin_probabilities, exploration_policy, allocate_integer)
from cathode_cvae.conditions import load_schema, classify, supported_bins
from cathode_cvae.data import sha256, target_indices
from cathode_cvae.mlip.registry import MODELS


TEMPERATURES = [393.15, 433.15, 473.15, 493.15]
PREFIXES = [512, 1024, 2048, 4096, 8192]
PROTOCOL = 'pristine-derived deletion; unrelaxed; exact canonical parent coordinates'


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    ap.add_argument('--root', type=Path, default=default_root)
    ap.add_argument('--run-dir', type=Path)
    ap.add_argument('--scores', type=Path)
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--reuse-joined', action='store_true', help='Validate and reuse an already completed exact join after a downstream analysis failure')
    a = ap.parse_args()
    root = a.root.resolve()
    run = a.run_dir or root/'runs/overnight_20261107'
    report = a.output_dir or root/'reports/overnight_20261107/reference_diagnostics'
    raw_path = run/'data/reference.npz'
    score_path = a.scores or run/'collected_reference/uma.npz'
    schema_path = run/'data/condition_schema_uma.json'
    parent_path = root/'artifacts/current/core/data/parents/O3_N48_pilot.npz'
    joined_path = run/'data/reference_scored.npz'
    allocation_dir = run/'allocation'
    if (joined_path.exists() and not a.reuse_joined) or allocation_dir.exists() or report.exists():
        raise FileExistsError('Use new reference/allocation/diagnostic output paths; existing evidence is protected')
    if not score_path.is_file():
        raise FileNotFoundError('Complete collected UMA reference scores are required before this analysis')
    with np.load(raw_path, allow_pickle=False) as z:
        raw = {k:z[k] for k in z.files}
    with np.load(score_path, allow_pickle=False) as z:
        scores = {k:z[k] for k in z.files}
    with np.load(parent_path, allow_pickle=False) as z:
        parent = {k:z[k] for k in z.files}
    schema = load_schema(schema_path)
    ids = raw['structure_ids'].astype(str)
    score_ids = scores['structure_ids'].astype(str)
    if len(ids) != 24575 or len(set(ids)) != len(ids) or len(set(score_ids)) != len(score_ids):
        raise ValueError('The declared 24,575 unique reference geometries or score identities are inconsistent')
    if str(raw['parent_sha256'].item()) != sha256(parent_path):
        raise ValueError('Reference parent hash differs from the provided parent')
    missing = sorted(set(ids)-set(score_ids))
    if missing:
        raise ValueError(f'Incomplete prespecified reference: {len(missing)} exact geometries missing; no cache-hit filtering is allowed')
    index = {s:i for i,s in enumerate(score_ids)}
    order = np.asarray([index[s] for s in ids])
    energy = np.asarray(scores['energy_eV'][order], dtype=float)
    atoms = np.asarray(scores['n_atoms'][order])
    occupancy = np.asarray(raw['occupancy'])
    n = occupancy.sum(1).astype(int)
    expected_atoms = len(parent['numbers'])-len(target_indices(parent))+n
    if not np.isfinite(energy).all() or not np.array_equal(atoms, expected_atoms):
        raise ValueError('Every reference requires one finite total energy and its actual atom count')
    if 'n_Li' in scores and not np.array_equal(scores['n_Li'][order], n):
        raise ValueError('Score and geometry compositions differ')
    if 'status' in scores and not np.all(np.isin(scores['status'][order].astype(str), ['success','complete','completed','ok','fresh','cached'])):
        raise ValueError('Failed reference scores cannot be dropped or treated as complete')
    frequency = np.asarray(raw['raw_frequency'], dtype=np.int64)
    draw_order = np.asarray(raw['raw_sample_indices'], dtype=np.int64)
    if frequency.shape != n.shape or np.any(frequency <= 0) or len(draw_order) != 24576:
        raise ValueError('Raw-frequency reference measure is incomplete')
    if not np.array_equal(np.bincount(draw_order, minlength=len(ids)), frequency):
        raise ValueError('Raw order and raw-frequency measures disagree')
    if not np.array_equal(np.asarray(raw['raw_n_Li']), n[draw_order]):
        raise ValueError('Raw sampling order and composition labels disagree')
    for count in [12,24,36]:
        if int(frequency[n==count].sum()) != 8192:
            raise ValueError('The declared raw reference budget must be complete at each composition')
    workers = []
    for path in sorted((run/'oracle/workers').glob('*.json')):
        d = json.loads(path.read_text())
        if d.get('model') != 'uma':
            continue
        expected = MODELS['uma']
        if (d.get('head') != 'omat' or d.get('package_version') != expected['version']
                or d.get('dtype') != 'float32' or d.get('tf32') is not False
                or set(d.get('checkpoint_hashes',{}).values()) != set(expected['files'].values())):
            raise ValueError('UMA worker identity or numerical precision differs from the declared teacher')
        workers.append(dict(receipt_sha256=sha256(path), worker=d.get('worker'),
            adapter_source_sha256=d.get('adapter_source_sha256'), package_version=d['package_version'],
            head=d['head'], dtype=d['dtype'], tf32=d['tf32'],
            checkpoint_hashes=d['checkpoint_hashes']))
    if not workers:
        raise ValueError('Actual UMA worker receipts are required to certify the measured reference')
    joined = dict(raw)
    joined.update(energy_eV=energy, energy_eV_atom=energy/atoms, n_atoms=atoms,
        teacher=np.asarray('uma'), head=np.asarray('omat'),
        checkpoint_name=np.asarray(MODELS['uma']['name']),
        checkpoint_sha256=np.asarray(next(iter(MODELS['uma']['files'].values()))),
        geometry_protocol=np.asarray(PROTOCOL),
        sampling_measure=np.asarray('independent uniform raw fixed-count masks; frequencies retained'),
        sampling_seed=np.asarray(20261107), dtype=np.asarray('float32'), tf32=np.asarray(False))
    if a.reuse_joined:
        with np.load(joined_path, allow_pickle=False) as z:
            if set(z.files) != set(joined) or any(not np.array_equal(z[k], np.asarray(v)) for k,v in joined.items()):
                raise ValueError('Existing joined reference differs from the complete validated source join')
    else:
        with joined_path.open('xb') as f:
            np.savez_compressed(f, **joined)
    # This runs the same public acquisition path supplied to the collaborators.
    acquisition_main(['--reference',str(joined_path),'--parent',str(parent_path),
        '--schema',str(schema_path),'--reference-kind','uniform_raw_sample',
        '--counts','12','24','36','--temperatures',*map(str,TEMPERATURES),
        '--mixture-weights','1','1','1','1','--exploration','.1','--budget','128',
        '--output-dir',str(allocation_dir)])
    report.mkdir(parents=True,exist_ok=False)
    prefix_rows, allocation_rows, population_rows = [], [], []
    for count in [12,24,36]:
        selected = n==count
        global_ids = np.flatnonzero(selected)
        local = np.full(len(ids),-1,dtype=np.int64)
        local[global_ids] = np.arange(len(global_ids))
        ordered = local[draw_order]
        ordered = ordered[ordered>=0]
        e = energy[selected]; f = frequency[selected]
        labels = classify(schema,np.full(len(e),count),e/atoms[selected])
        names = supported_bins(schema,count)
        minimum_full = float(e.min())
        population_rows.append(dict(n_Li=count,raw_draws=int(f.sum()),unique_geometries=len(e),
            minimum_total_energy_eV=minimum_full,maximum_total_energy_eV=float(e.max()),
            minimum_energy_eV_atom=float((e/atoms[selected]).min()),
            raw_low=int(f[labels=='low'].sum()),raw_middle=int(f[labels=='middle'].sum()),raw_high=int(f[labels=='high'].sum())))
        for temperature in TEMPERATURES:
            full = static_weights(e,f,temperature,measure_kind='uniform_raw_sample')
            full_bins = bin_probabilities(full['probability'],labels,names)
            for policy, proportions in [('static',full_bins),('static_exploration_0.1',exploration_policy(full_bins,.1))]:
                integers = allocate_integer(proportions,128)
                for name in names:
                    allocation_rows.append(dict(n_Li=count,temperature_K=temperature,allocation_policy=policy,
                        energy_bin=name,requested_fraction=proportions[name],attempts=integers[name],budget=128,
                        particle_ess=full['microstate_or_particle_ess'],representative_ess=full['representative_ess'],
                        maximum_representative_mass=full['maximum_representative_mass'],
                        weighted_mean_total_energy_eV=full['mean_energy_eV']))
            for size in PREFIXES:
                prefix_frequency = np.bincount(ordered[:size],minlength=len(e))
                keep = prefix_frequency>0
                weight = static_weights(e[keep],prefix_frequency[keep],temperature,measure_kind='uniform_raw_sample')
                bins = bin_probabilities(weight['probability'],labels[keep],names)
                prefix_rows.append(dict(n_Li=count,temperature_K=temperature,raw_pool_size=size,
                    unique_geometries=int(keep.sum()),minimum_total_energy_eV=float(e[keep].min()),
                    full_pool_minimum_total_energy_eV=minimum_full,
                    minimum_gap_to_full_minimum_eV=float(e[keep].min()-minimum_full),
                    weighted_mean_total_energy_eV=weight['mean_energy_eV'],
                    weighted_mean_above_full_minimum_eV=weight['mean_energy_eV']-minimum_full,
                    weighted_mean_difference_to_full_weighted_eV=weight['mean_energy_eV']-full['mean_energy_eV'],
                    particle_ess=weight['microstate_or_particle_ess'],representative_ess=weight['representative_ess'],
                    maximum_representative_mass=weight['maximum_representative_mass'],
                    maximum_single_particle_probability=weight['maximum_single_microstate_or_particle_probability'],
                    W_low=bins.get('low',0.),W_middle=bins.get('middle',0.),W_high=bins.get('high',0.),
                    l1_bin_fraction_difference_to_full=float(sum(abs(bins[b]-full_bins[b]) for b in names))))
    write_csv(report/'prefix_metrics.csv',prefix_rows)
    write_csv(report/'allocation_fractions.csv',allocation_rows)
    write_csv(report/'reference_population.csv',population_rows)
    colours = dict(low='#2166ac',middle='#f4a340',high='#b2182b')
    fig,axs=plt.subplots(3,2,figsize=(8.5,7),sharey=True)
    for ri,count in enumerate([12,24,36]):
        for ci,policy in enumerate(['static','static_exploration_0.1']):
            ax=axs[ri,ci];bottom=np.zeros(4)
            for name in ['low','middle','high']:
                y=np.array([next(r['requested_fraction'] for r in allocation_rows if r['n_Li']==count and r['temperature_K']==t and r['allocation_policy']==policy and r['energy_bin']==name) for t in TEMPERATURES])
                ax.bar(np.arange(4),y,bottom=bottom,color=colours[name],label=name);bottom+=y
            ax.set(xticks=range(4),xticklabels=[str(t) for t in TEMPERATURES],ylim=(0,1),title=f'Li{count}: {policy}',xlabel='Physical acquisition T (K)')
            if ci==0:ax.set_ylabel('Requested bin fraction')
    axs[0,1].legend(loc='lower center',fontsize=8,ncol=3)
    fig.suptitle('Complete empirical-reference acquisition; generated-bin outcomes are separate',fontsize=10)
    fig.tight_layout();fig.savefig(report/'requested_bin_fractions.pdf');fig.savefig(report/'requested_bin_fractions.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(10,6),sharex=True)
    for ci,count in enumerate([12,24,36]):
        for temperature in TEMPERATURES:
            rr=[r for r in prefix_rows if r['n_Li']==count and r['temperature_K']==temperature]
            axs[0,ci].plot(PREFIXES,[r['particle_ess'] for r in rr],'o-',label=f'{temperature} K',ms=3)
            axs[1,ci].plot(PREFIXES,[r['weighted_mean_above_full_minimum_eV'] for r in rr],'o-',ms=3)
        rr=[r for r in prefix_rows if r['n_Li']==count and r['temperature_K']==TEMPERATURES[0]]
        axs[1,ci].plot(PREFIXES,[r['minimum_gap_to_full_minimum_eV'] for r in rr],'k--',label='Prefix minimum',lw=1)
        axs[0,ci].set_title(f'Li{count}Co48O96')
        axs[1,ci].set_xlabel('Nested raw reference size')
        axs[1,ci].set_xscale('log',base=2)
    axs[0,0].set_ylabel('Raw-particle ESS')
    axs[1,0].set_ylabel('E minus full-pool minimum (eV/cell)')
    axs[0,2].legend(fontsize=7);axs[1,2].legend(fontsize=7)
    fig.suptitle('Nested-prefix sensitivity, not a confidence interval or equilibrium validation',fontsize=10)
    fig.tight_layout();fig.savefig(report/'prefix_sensitivity.pdf');fig.savefig(report/'prefix_sensitivity.png',dpi=170);plt.close(fig)
    receipt=dict(status='PASS',reference_rows=len(ids),raw_draws=len(draw_order),required_missing_ids=0,
        scored_reference_rows=len(order),all_reference_energies_finite=True,complete_reference_required=True,
        parent_sha256=sha256(parent_path),condition_schema_id=schema['identity'],
        teacher='uma',head='omat',checkpoint_name=MODELS['uma']['name'],checkpoint_hashes=MODELS['uma']['files'],
        geometry_protocol=PROTOCOL,energy_units='eV/cell',physical_temperature_units='K',
        reference_measure='Independent uniform raw fixed-count masks, collapsed with retained raw frequencies; no extra orbit multiplicity',
        prefix_sizes=PREFIXES,temperatures_K=TEMPERATURES,
        source_hashes=dict(reference=sha256(raw_path),scores=sha256(score_path),schema=sha256(schema_path),
            parent=sha256(parent_path),joined_reference=sha256(joined_path),script=sha256(__file__)),
        source_score_collector_receipt=json.loads((score_path.parent/'receipt.json').read_text()),
        uma_worker_receipts=workers,
        outputs={p.name:sha256(p) for p in sorted(report.iterdir()) if p.is_file()},
        interpretation='Measured complete-reference acquisition under a static UMA surrogate; nested prefixes are not independent replicates or confidence intervals; finite samples can miss rare low-energy states')
    (report/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS',reference_rows=len(ids),raw_draws=len(draw_order),allocation=str(allocation_dir),diagnostics=str(report),full_pool_rows=[r for r in prefix_rows if r['raw_pool_size']==8192]),indent=2))


if __name__=='__main__':
    main()

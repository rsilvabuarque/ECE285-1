"""Compare finite-budget entropy estimators with exhaustive fixed-lattice truth."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from cathode_cvae.data import sha256
from cathode_cvae.entropy import (KB_EV_K,thermodynamics,energy_bins,
                                  importance_benchmark,wang_landau_benchmark)
from cathode_cvae.runtime import require_compute_allocation


def write_csv(path,rows):
    if not rows: return
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def named_paths(items):
    result={}
    for item in items:
        name,path=item.split('=',1)
        if name in result:raise ValueError('Repeated teacher')
        result[name]=Path(path)
    return result


def load_scores(path,expected):
    with np.load(path,allow_pickle=False) as z:
        ids=z['structure_ids'].astype(str);energy=z['energy_eV'].astype(float)
        status=z['status'].astype(str) if 'status' in z else np.full(len(ids),'success')
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate score IDs')
    lookup={sid:i for i,sid in enumerate(ids)}
    if set(expected)-lookup.keys():raise ValueError('Missing exhaustive score IDs')
    index=np.array([lookup[sid] for sid in expected])
    if not np.all(np.isin(status[index],['success','new'])) or not np.isfinite(energy[index]).all():
        raise ValueError('Failed/nonfinite exhaustive score')
    return energy[index]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--score',action='append',required=True,help='TEACHER=collected scalar NPZ')
    p.add_argument('--orientation-input',type=Path)
    p.add_argument('--orientation-score',action='append',default=[])
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--temperatures-K',type=float,nargs='+',default=[100,200,300,500,800,1200,2000,5000,10000])
    p.add_argument('--wl-steps',type=int,default=200000)
    p.add_argument('--wl-bins',type=int,nargs='+',default=[16,32])
    p.add_argument('--seeds',type=int,nargs='+',default=[2026111201,2026111202,2026111203])
    a=p.parse_args();require_compute_allocation()
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    with np.load(a.catalog,allow_pickle=False) as z:
        d={k:z[k] for k in z.files}
    ids=d['structure_ids'].astype(str);counts=d['n_target'];g=d['multiplicity']
    raw=d['raw_occupancy'];mapping=d['raw_to_orbit'];n_sites=raw.shape[1]
    if len(raw)!=2**n_sites or g.sum()!=len(raw):raise ValueError('Incomplete catalog')
    inputs=named_paths(a.score);orientation_inputs=named_paths(a.orientation_score)
    if set(orientation_inputs)-set(inputs):raise ValueError('Orientation teachers lack exhaustive scores')
    energies={teacher:load_scores(path,ids) for teacher,path in inputs.items()}
    a.output_dir.mkdir(parents=True)
    canonical=[];degeneracy=[];importance=[];wl_summary=[];wl_thermo=[];wl_dos=[];orientation=[]
    for teacher,e in energies.items():
        for n in range(n_sites+1):
            ix=counts==n;ee=e[ix];gg=g[ix]
            for T in a.temperatures_K:
                exact=thermodynamics(ee,gg,T);orbit=thermodynamics(ee,np.ones(len(ee)),T)
                canonical.append(dict(teacher=teacher,n_target=n,n_sites=n_sites,
                    labelled_states=int(gg.sum()),orbits=int(ix.sum()),**{k:v for k,v in exact.items() if k!='orbit_probability'}))
                degeneracy.append(dict(teacher=teacher,n_target=n,temperature_K=T,
                    correct_entropy_kB=exact['entropy_kB'],wrong_orbit_uniform_entropy_kB=orbit['entropy_kB'],
                    difference_kB=orbit['entropy_kB']-exact['entropy_kB']))
        for n in (3,6,9):
            ix=counts==n
            for draws in (128,512,2048,8192):
                for seed in a.seeds:
                    for T in (300.,1000.,5000.):
                        importance.append(dict(teacher=teacher,n_target=n,
                            **importance_benchmark(e[ix],g[ix],T,draws=draws,seed=seed,
                                                   tilt_temperature_K=100.,exploration=.1)))
            for nbins in a.wl_bins:
                for seed in a.seeds:
                    result=wang_landau_benchmark(e[mapping],raw,n_target=n,n_bins=nbins,
                                                steps=a.wl_steps,seed=seed)
                    arrays=('estimated_probability','exact_counts','edges_eV','phases')
                    wl_summary.append(dict(teacher=teacher,**{k:v for k,v in result.items() if k not in arrays}))
                    mid=(result['edges_eV'][1:]+result['edges_eV'][:-1])/2
                    active=result['exact_counts']>0;total=int(result['exact_counts'].sum())
                    for b in range(len(mid)):
                        wl_dos.append(dict(teacher=teacher,n_target=n,n_bins=nbins,seed=seed,bin_index=b,
                            lower_eV=float(result['edges_eV'][b]),upper_eV=float(result['edges_eV'][b+1]),
                            exact_labelled_states=int(result['exact_counts'][b]),
                            estimated_labelled_states=float(result['estimated_probability'][b]*total)))
                    for T in a.temperatures_K:
                        exact=thermodynamics(e[ix],g[ix],T)
                        binned=thermodynamics(mid[active],result['exact_counts'][active],T)
                        estimate=thermodynamics(mid[active],result['estimated_probability'][active]*total,T)
                        wl_thermo.append(dict(teacher=teacher,n_target=n,n_bins=nbins,seed=seed,
                            temperature_K=T,converged=result['converged'],steps=result['steps'],
                            exact_entropy_kB=exact['entropy_kB'],binned_exact_entropy_kB=binned['entropy_kB'],
                            estimated_entropy_kB=estimate['entropy_kB'],
                            discretization_error_kB=binned['entropy_kB']-exact['entropy_kB'],
                            estimation_error_relative_binned_kB=estimate['entropy_kB']-binned['entropy_kB']))
        if teacher in orientation_inputs:
            if a.orientation_input is None:raise ValueError('Need orientation input')
            with np.load(a.orientation_input,allow_pickle=False) as z:
                oi=z['structure_ids'].astype(str);orig=z['original_structure_ids'].astype(str)
            oe=load_scores(orientation_inputs[teacher],oi);reference=dict(zip(ids,e))
            for sid,original,value in zip(oi,orig,oe):
                orientation.append(dict(teacher=teacher,structure_id=sid,original_structure_id=original,
                    reference_energy_eV=float(reference[original]),orientation_energy_eV=float(value),
                    difference_eV=float(value-reference[original])))
    for name,rows in [('canonical_thermodynamics.csv',canonical),('multiplicity_effect.csv',degeneracy),
        ('synthetic_importance_benchmark.csv',importance),('wang_landau_summary.csv',wl_summary),
        ('wang_landau_thermodynamics.csv',wl_thermo),('wang_landau_density.csv',wl_dos),
        ('symmetry_orientation_energy_checks.csv',orientation)]:write_csv(a.output_dir/name,rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    colors={3:'tab:blue',6:'tab:orange',9:'tab:green'};styles=['-','--',':','-.']
    for j,teacher in enumerate(energies):
        for n in (3,6,9):
            rows=[r for r in canonical if r['teacher']==teacher and r['n_target']==n]
            for ax,key in zip(axs,['entropy_kB','heat_capacity_kB']):
                ax.plot([r['temperature_K'] for r in rows],[r[key] for r in rows],
                    color=colors[n],linestyle=styles[j%len(styles)],label=f'{teacher}, Li{n}/12')
    for ax,label in zip(axs,['Configurational entropy / kB per cell','Configurational heat capacity / kB per cell']):
        ax.set_xscale('log');ax.set_xlabel('Temperature (K)');ax.set_ylabel(label);ax.grid(alpha=.2)
    axs[0].legend(fontsize=8);fig.suptitle('Exact idealized O3 N12 vacancy-lattice reference (MLIP energies)')
    fig.tight_layout();fig.savefig(a.output_dir/'exact_entropy_heat_capacity.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    for teacher in energies:
        for nbins in a.wl_bins:
            rows=[r for r in wl_summary if r['teacher']==teacher and r['n_target']==6 and r['n_bins']==nbins]
            x=np.arange(len(rows))+.15*(list(energies).index(teacher))
            axs[0].scatter(x,[r['density_total_variation'] for r in rows],label=f'{teacher}, {nbins} bins')
    axs[0].set_xlabel('Seed index (Li6/12)');axs[0].set_ylabel('DOS total-variation error');axs[0].legend(fontsize=8)
    for teacher in energies:
        rows=[r for r in importance if r['teacher']==teacher and r['n_target']==6 and r['temperature_K']==1000.]
        for key,style in [('naive_sample_entropy_kB',':'),('importance_entropy_kB','-')]:
            budgets=sorted(set(r['draws'] for r in rows));means=[np.mean([abs(r[key]-r['exact_entropy_kB']) for r in rows if r['draws']==b]) for b in budgets]
            axs[1].plot(budgets,means,style,marker='o',label=f'{teacher}: '+('naive' if key.startswith('naive') else 'importance'))
    axs[1].set_xscale('log');axs[1].set_xlabel('Synthetic proposal draws');axs[1].set_ylabel('Mean absolute entropy error / kB');axs[1].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.2)
    fig.suptitle('Finite-budget diagnostics; synthetic proposal is not a trained CVAE')
    fig.tight_layout();fig.savefig(a.output_dir/'entropy_estimator_diagnostics.pdf');plt.close(fig)
    receipt=dict(status='COMPLETE_EXHAUSTIVE_REFERENCE_AND_BOUNDED_ESTIMATORS',
        catalog_sha256=sha256(a.catalog),teacher_score_sha256={k:sha256(v) for k,v in inputs.items()},
        oracle_orbits=len(ids),labelled_masks=len(raw),n_sites=n_sites,
        temperatures_K=a.temperatures_K,KB_eV_K=KB_EV_K,energy_units='total eV per periodic cell',
        wang_landau_runs=len(wl_summary),wang_landau_converged=sum(r['converged'] for r in wl_summary),
        wl_budget_steps=a.wl_steps,wl_exact_support_supplied=True,
        synthetic_importance_runs=len(importance),symmetry_orientation_rows=len(orientation),
        orientation_max_absolute_delta_eV={t:max(abs(r['difference_eV']) for r in orientation if r['teacher']==t) for t in orientation_inputs},
        limitations=['Finite idealized O3 lattice; no DFT or magnetic/vibrational/free-cell validation',
        'WL flatness is a provisional numerical setting; convergence flag is not unknown-support discovery',
        'Synthetic importance sampler has known q and is not CVAE; entropy ratios finite-sample biased',
        'Small bin-width convergence is assessed separately from density estimation'],
        source_sha256=sha256(Path(__file__)),module_sha256=sha256(Path(__file__).parents[2]/'src/cathode_cvae/entropy.py'))
    receipt['outputs']={p.name:sha256(p) for p in a.output_dir.iterdir() if p.is_file()}
    (a.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (a.output_dir/'REPORT.md').write_text(
        '# Exact N12 vacancy-lattice entropy validation\n\n'
        f'Both orbit multiplicities and all {len(raw)} labelled masks are enumerated; '
        f'{len(ids)} representative energies are evaluated for each supplied teacher. '
        'Total configuration energies, rather than energy per atom, enter Boltzmann weights.\n\n'
        'The exact entropy/heat-capacity curves are properties of this idealized fixed lattice and supplied MLIP, '
        'not physical LCO thermodynamics validated against DFT or experiment. Orbit-uniform counting omits '
        'multiplicities; the companion table quantifies its error. The synthetic biased proposal shows why '
        'raw generator frequencies are not equilibrium probabilities and why known-probability correction matters.\n\n'
        f'Wang–Landau: {receipt["wang_landau_converged"]}/{len(wl_summary)} bounded runs met the stated '
        'flatness/modification-factor schedule. Exact nonempty energy-bin support was supplied, so this is '
        'an estimator validation under favourable support information. Every result, including nonconvergence, '
        'is retained. Density estimation and midpoint binning errors are reported separately.\n')
    print(json.dumps({k:receipt[k] for k in ['status','oracle_orbits','labelled_masks','wang_landau_runs','wang_landau_converged']},indent=2))


if __name__=='__main__':main()

"""Complete raw finite-Hu reference, approximate-orbit error and sampled-pool bias.

All probabilities are normalized separately by teacher and Li count. Sampling
errors use a known raw-state universe, not an entropy of a proposal histogram.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from cathode_cvae.acquisition import KB_EV_K
from cathode_cvae.data import sha256
from cathode_cvae.runtime import require_compute_allocation

spec=importlib.util.spec_from_file_location('smallcell_reference',Path(__file__).with_name('smallcell_reference.py'))
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def csv_write(path,rows):
    if not rows:raise ValueError('Cannot silently write an empty scientific table')
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def orbit_statistics(energy,keys,representative_indices,temperature):
    """Compare exact raw probabilities to two explicitly different approximations."""
    e=np.asarray(energy,float);keys=np.asarray(keys).astype(str)
    exact=reference.raw_thermodynamics(e,temperature);p=exact['probability']
    groups=sorted(set(keys));indices=[np.flatnonzero(keys==key) for key in groups]
    mass=np.array([p[ii].sum() for ii in indices]);g=np.array([len(ii) for ii in indices])
    # Keeping the *true* orbit mass but flattening its conditional distribution
    # isolates the missing entropy caused by unequal literal-coordinate energies.
    positive=mass>0
    uniform_entropy=-KB_EV_K*float(np.sum(mass[positive]*np.log(mass[positive]/g[positive])))
    rep_energy=np.array([e[int(representative_indices[key])] for key in groups])
    rep_logw=np.log(g)-(rep_energy-rep_energy.min())/(KB_EV_K*temperature)
    m=rep_logw.max();logz=m+np.log(np.exp(rep_logw-m).sum());rep_mass=np.exp(rep_logw-logz)
    rep_mean=float(np.dot(rep_mass,rep_energy));positive=rep_mass>0
    rep_entropy=-KB_EV_K*float(np.sum(rep_mass[positive]*np.log(rep_mass[positive]/g[positive])))
    rep_free=float(rep_energy.min()-KB_EV_K*temperature*logz)
    return dict(exact=exact,groups=groups,indices=indices,mass=mass,multiplicity=g,
        representative_mass=rep_mass,representative_energy=rep_energy,
        exact_mass_uniform_members_entropy_eV_K=uniform_entropy,
        representative_mean_energy_eV=rep_mean,representative_entropy_eV_K=rep_entropy,
        representative_free_energy_eV=rep_free,
        orbit_mass_total_variation=float(np.abs(rep_mass-mass).sum()/2))


def sampled_pool(energy,draw_indices,temperature):
    """Uniform-raw IID partition/energy estimates with the known universe size.

    E_ref is the exact reference minimum, used only as a numerical origin.
    With M raw states and L draws: Z_rel_hat=M/L*sum(exp[-beta(E-E_ref)]).
    Repeated draws remain separate particles; ESS is not symmetry multiplicity.
    """
    e=np.asarray(energy,float);draw=np.asarray(draw_indices,int)
    if not len(draw):raise ValueError('A sampled pool needs at least one draw')
    beta=1/(KB_EV_K*temperature);origin=float(e.min());sample=e[draw]
    logw=-beta*(sample-origin);maximum=float(logw.max())
    logsum=maximum+float(np.log(np.exp(logw-maximum).sum()))
    normalized=np.exp(logw-logsum);logz=float(np.log(len(e)/len(draw))+logsum)
    mean=float(np.dot(normalized,sample))
    entropy=KB_EV_K*(logz+beta*(mean-origin))
    free=origin-KB_EV_K*temperature*logz
    return dict(mean_energy_eV=mean,entropy_eV_K=float(entropy),free_energy_eV=float(free),
        particle_ESS=float(1/np.dot(normalized,normalized)),weights=normalized)


def collect(prepared,score_roots,model):
    states=reference.load_npz(prepared/'states.npz');protocol=json.loads((prepared/'protocol.json').read_text())
    expected=len(states['label_ids']);rows={};source=[];teachers=[];force={};stress={}
    for root in score_roots:
        for file in sorted(root.rglob('chunk_*.npz')):
            meta=json.loads(file.with_suffix('.json').read_text());binding=meta['binding']
            if binding['model']!=model:continue
            if binding['states_sha256']!=sha256(prepared/'states.npz'):raise ValueError('Different raw reference states')
            if sha256(file)!=meta['sha256']:raise ValueError('Score chunk changed')
            teacher=json.loads((file.parent/'teacher.json').read_text())
            if teacher['dtype']!='float32' or teacher['tf32'] is not False:raise ValueError('Unexpected inference precision')
            teachers.append(teacher);z=reference.load_npz(file)
            for i,label in enumerate(z['label_index']):
                label=int(label)
                if label in rows:raise ValueError(f'Duplicate physical label index {model}/{label}')
                state=int(states['label_state_rows'][label])
                if str(z['geometry_sha256'][i])!=str(states['geometry_sha256'][state]):raise ValueError('Label geometry changed')
                if str(z['label_id'][i])!=str(states['label_ids'][label]):raise ValueError('Label identity changed')
                if str(z['status'][i])!='success' or not np.isfinite(z['energy_eV'][i]):raise ValueError('Incomplete/nonfinite reference; retain failures but do not claim exact reference')
                rows[label]=float(z['energy_eV'][i]);force[label]=z['forces_parent_eV_A'][i];stress[label]=z['stress_eV_A3'][i]
            source.append(dict(path=str(file.resolve()),sha256=sha256(file)))
    if set(rows)!=set(range(expected)):raise ValueError(f'{model}: expected {expected} labels, found {len(rows)}')
    contexts={(t['checkpoint_name'],t['head'],t['package_version'],t['dtype'],t['tf32'],t['adapter_source_sha256'],
        json.dumps(t['checkpoint_hashes'],sort_keys=True)) for t in teachers}
    if len(contexts)!=1:raise ValueError('Mixed teacher contexts within the finite reference')
    e=np.array([rows[i] for i in range(expected)])
    f=np.array([force[i] for i in range(expected)]);s=np.array([stress[i] for i in range(expected)])
    return states,protocol,e,f,s,dict(model=model,labels=expected,context=list(contexts)[0],sources=source)


def analyze(args):
    require_compute_allocation();args.output.mkdir(parents=True,exist_ok=False)
    states=reference.load_npz(args.prepared/'states.npz');representatives=json.loads((args.prepared/'representatives.json').read_text())
    raw_ids=states['raw_ids'].astype(str);keys=states['canonical_orbit_keys'].astype(str);nli=states['n_Li']
    tables=defaultdict(list);bindings=[];probability_arrays={};energies={}
    for model in ['uma','mace']:
        _,protocol,e,force,stress,binding=collect(args.prepared,args.scores,model);bindings.append(binding);energies[model]=e[:1364]
        raw_lookup={sid:i for i,sid in enumerate(raw_ids)}
        for i in range(1364):
            tables['raw_energies'].append(dict(teacher=model,raw_id=raw_ids[i],state_row=i,n_Li=int(nli[i]),
                orbit_key=keys[i],geometry_sha256=str(states['geometry_sha256'][i]),energy_eV=e[i],
                energy_eV_atom=e[i]/states['n_atoms'][i],physical_Fmax_eV_A=float(np.linalg.norm(force[i],axis=1).max()),
                stress_norm_eV_A3=float(np.linalg.norm(stress[i]))))
        for state in np.unique(states['label_state_rows'][states['label_repeat_index']>0]):
            labels=np.flatnonzero(states['label_state_rows']==state);primary=int(state)
            tables['repeatability'].append(dict(teacher=model,raw_id=raw_ids[state],n_Li=int(nli[state]),evaluations=len(labels),
                energy_range_eV_cell=float(np.ptp(e[labels])),energy_range_meV_atom=float(np.ptp(e[labels])*1000/states['n_atoms'][state]),
                max_abs_force_difference_eV_A=float(np.max(np.abs(force[labels]-force[primary]))),
                max_abs_stress_difference_eV_A3=float(np.max(np.abs(stress[labels]-stress[primary])))))
        for n in protocol['counts']:
            ii=np.flatnonzero(nli==n);local_e=e[ii];local_keys=keys[ii]
            local_lookup={raw_ids[global_i]:j for j,global_i in enumerate(ii)}
            rep_indices={key:local_lookup[representatives[key]['raw_id']] for key in set(local_keys)}
            q20,q80=np.quantile(local_e,[.2,.8]);bins=np.where(local_e<=q20,0,np.where(local_e>q80,2,1))
            low_order=np.argsort(local_e,kind='stable')[:int(np.ceil(.2*len(ii)))];low_selector=np.zeros(len(ii),bool);low_selector[low_order]=True
            tables['thresholds'].append(dict(teacher=model,n_Li=n,raw_states=len(ii),orbits=len(rep_indices),
                q20_eV_cell=float(q20),q80_eV_cell=float(q80),low_tie_rule='E<=q20; high E>q80; otherwise middle',
                exact_lowest_subset_size=len(low_order)))
            for key in sorted(rep_indices):
                group=np.flatnonzero(local_keys==key);rep=rep_indices[key]
                tables['orbit_spreads'].append(dict(teacher=model,n_Li=n,orbit_key=key,multiplicity=len(group),
                    representative_raw_id=raw_ids[ii[rep]],min_energy_eV=float(local_e[group].min()),max_energy_eV=float(local_e[group].max()),
                    energy_range_eV_cell=float(np.ptp(local_e[group])),energy_range_meV_atom=float(np.ptp(local_e[group])*1000/(36+n)),
                    representative_minus_group_mean_eV=float(local_e[rep]-local_e[group].mean())))
            for T in protocol['temperatures_K']:
                result=orbit_statistics(local_e,local_keys,rep_indices,T);exact=result['exact'];p=exact['probability']
                probability_arrays[f'{model}_n{n}_T{str(T).replace(".","p")}']=p
                exact_bin=np.array([p[bins==b].sum() for b in range(3)])
                tables['thermodynamics'].append(dict(teacher=model,n_Li=n,temperature_K=T,raw_states=len(ii),orbits=len(rep_indices),
                    mean_energy_eV_cell=exact['mean_energy_eV'],free_energy_eV_cell=exact['free_energy_eV'],
                    configurational_entropy_eV_K_cell=exact['entropy_eV_K'],entropy_kB_cell=exact['entropy_eV_K']/KB_EV_K,
                    raw_state_ESS=exact['state_ESS'],mass_low=float(exact_bin[0]),mass_middle=float(exact_bin[1]),mass_high=float(exact_bin[2]),
                    mass_exact_lowest20percent=float(p[low_selector].sum()),
                    exact_mass_flat_members_entropy_excess_eV_K=result['exact_mass_uniform_members_entropy_eV_K']-exact['entropy_eV_K'],
                    representative_mean_energy_error_eV=result['representative_mean_energy_eV']-exact['mean_energy_eV'],
                    representative_entropy_error_eV_K=result['representative_entropy_eV_K']-exact['entropy_eV_K'],
                    representative_free_energy_error_eV=result['representative_free_energy_eV']-exact['free_energy_eV'],
                    representative_orbit_mass_total_variation=result['orbit_mass_total_variation']))
                for j,key in enumerate(result['groups']):
                    tables['orbit_probabilities'].append(dict(teacher=model,n_Li=n,temperature_K=T,orbit_key=key,
                        multiplicity=int(result['multiplicity'][j]),exact_probability=float(result['mass'][j]),
                        representative_probability=float(result['representative_mass'][j])))
                # Common draw identities across teachers and T. Prefix budgets
                # are correlated within each seed, not independent replicates.
                for seed in args.pool_seeds:
                    draw=np.random.default_rng(np.random.SeedSequence([seed,n])).integers(0,len(ii),max(args.pool_sizes))
                    for size in args.pool_sizes:
                        selected=draw[:size];estimate=sampled_pool(local_e,selected,T);w=estimate['weights']
                        sampled_bin=np.array([w[bins[selected]==b].sum() for b in range(3)])
                        tables['sampled_pools'].append(dict(teacher=model,n_Li=n,temperature_K=T,seed=seed,raw_draws=size,
                            distinct_raw_states=len(np.unique(selected)),reference_mass_covered=float(p[np.unique(selected)].sum()),
                            particle_ESS=estimate['particle_ESS'],ESS_fraction=estimate['particle_ESS']/size,
                            mean_energy_error_eV=estimate['mean_energy_eV']-exact['mean_energy_eV'],
                            free_energy_error_eV=estimate['free_energy_eV']-exact['free_energy_eV'],
                            entropy_error_eV_K=estimate['entropy_eV_K']-exact['entropy_eV_K'],
                            bin_mass_L1_error=float(np.abs(sampled_bin-exact_bin).sum()),
                            low_mass_error=float(sampled_bin[0]-exact_bin[0])))
    for name,rows in tables.items():csv_write(args.output/f'{name}.csv',rows)
    np.savez_compressed(args.output/'raw_probabilities.npz',**probability_arrays)
    # Teacher comparisons use the exact same physical raw structures, not
    # comparable-looking compositions from a relaxed DFT cohort.
    from scipy.stats import spearmanr,kendalltau
    cross=[]
    for n in protocol['counts']:
        ii=np.flatnonzero(nli==n);a=energies['uma'][ii];b=energies['mace'][ii]
        aa=np.argsort(a,kind='stable')[:int(np.ceil(.2*len(ii)))];bb=np.argsort(b,kind='stable')[:len(aa)]
        cross.append(dict(n_Li=n,raw_states=len(ii),spearman=float(spearmanr(a,b).statistic),kendall_tau_b=float(kendalltau(a,b).statistic),
            lowest20percent_overlap_fraction=len(set(aa)&set(bb))/len(aa),energy_offset_mean_eV_cell=float(np.mean(b-a)),
            offset_removed_RMSE_meV_atom=float(np.sqrt(np.mean(((b-a)-(b-a).mean())**2))*1000/(36+n))))
    csv_write(args.output/'teacher_comparison.csv',cross)
    if args.plots:plot(args.output)
    outputs={p.name:sha256(p) for p in args.output.iterdir() if p.is_file()}
    write_json(args.output/'receipt.json',dict(status='COMPLETE_EXACT_RAW_STATIC_REFERENCE',parent=protocol['parent'],
        prepared_protocol_sha256=sha256(args.prepared/'protocol.json'),source_sha256=sha256(__file__),
        teachers=bindings,primary_physical_EFS_labels=2728,additional_repeat_EFS_labels=24,
        pool_seeds=args.pool_seeds,pool_sizes=args.pool_sizes,temperatures_K=protocol['temperatures_K'],outputs=outputs,
        limits='Exact finite literal-parent, fixed-geometry MLIP ensemble at each count. Not real LCO entropy, a relaxed ensemble, DFT validation or 48-to12-site transfer. IID pool errors are descriptive five-seed estimator checks.'))


def plot(directory):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (directory/'thermodynamics.csv').open() as f:rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for j,n in enumerate([3,6,9]):
        ax=axes[j]
        for model in ['uma','mace']:
            r=[a for a in rows if a['teacher']==model and int(a['n_Li'])==n]
            ax.plot([float(a['temperature_K']) for a in r],[float(a['entropy_kB_cell']) for a in r],'-o',label=model.upper())
        ax.set(title=f'Li{n}Co12O24',xlabel='Temperature (K)',ylabel='Configurational entropy ($k_B$/cell)');ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(directory/'finite_entropy.pdf');plt.close(fig)
    with (directory/'sampled_pools.csv').open() as f:rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for j,n in enumerate([3,6,9]):
        for model in ['uma','mace']:
            r=[a for a in rows if a['teacher']==model and int(a['n_Li'])==n and float(a['temperature_K'])==433.15]
            budgets=sorted({int(a['raw_draws']) for a in r});means=[];spread=[]
            for budget in budgets:
                values=np.array([float(a['entropy_error_eV_K'])/KB_EV_K for a in r if int(a['raw_draws'])==budget]);means.append(values.mean());spread.append(values.std(ddof=1))
            axes[j].errorbar(budgets,means,yerr=spread,marker='o',label=model.upper(),capsize=2)
        axes[j].axhline(0,color='k',lw=.6);axes[j].set(xscale='log',title=f'Li{n},433.15K',xlabel='IID raw draws',ylabel='Entropy error ($k_B$/cell)');axes[j].grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(directory/'sampled_pool_entropy.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--scores',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pool-seeds',nargs='+',type=int,default=[20261108,20261109,20261110,20261111,20261112])
    p.add_argument('--pool-sizes',nargs='+',type=int,default=[32,64,128,256,512,1024]);p.add_argument('--plots',action='store_true')
    args=p.parse_args();analyze(args)


if __name__=='__main__':main()

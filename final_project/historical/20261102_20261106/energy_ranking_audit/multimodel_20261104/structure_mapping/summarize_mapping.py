"""Export exact joint-training candidates, observable support and mapping figures."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import csv
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import map_frames as m

HERE=Path(__file__).resolve().parent

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use compute allocation')
    m.initialize();d=m.DATA;parents=m.PARENTS
    with (HERE/'frame_mapping.csv').open() as f:rows=list(csv.DictReader(f))
    with np.load(HERE/'site_displacements.npz') as z:enc={k:z[k] for k in z.files}
    observables=[]
    for i,r in enumerate(rows):
        sl=slice(d['offsets'][i],d['offsets'][i+1]);force=d['forces'][sl]
        s=d['stress'][i]*160.21766208
        mean=s[:3].mean();dev=s.copy();dev[:3]-=mean
        observables.append(dict(index=i,geometry_key=r['geometry_key'],dataset=r['dataset'],phase=r['phase'],cohort=r['cohort'],
                                n_Li=r['n_Li'],x=r['x'],energy_eV=d['energy'][i],
                                force_component_RMS_eV_A=float(np.sqrt(np.mean(force**2))),
                                maximum_force_norm_eV_A=float(np.linalg.norm(force,axis=1).max()),
                                configurational_pressure_GPa=float(-mean),
                                von_Mises_deviatoric_stress_GPa=float(np.sqrt(1.5*(np.sum(dev[:3]**2)+2*np.sum(dev[3:]**2)))),
                                joint_template_eligible=r['joint_template_eligible']))
    m.csv_write(HERE/'conditioning_observables.csv',observables)
    summaries=[]
    for dataset,phase,cohort in sorted(set((r['dataset'],r['phase'],r['cohort']) for r in rows)):
        rr=[r for r in observables if (r['dataset'],r['phase'],r['cohort'])==(dataset,phase,cohort)]
        for quantity in ['force_component_RMS_eV_A','maximum_force_norm_eV_A','configurational_pressure_GPa','von_Mises_deviatoric_stress_GPa']:
            qq=np.quantile([r[quantity] for r in rr],[0,.05,.5,.95,1])
            summaries.append(dict(dataset=dataset,phase=phase,cohort=cohort,n_geometries=len(rr),quantity=quantity,
                                  minimum=qq[0],p05=qq[1],median=qq[2],p95=qq[3],maximum=qq[4]))
    m.csv_write(HERE/'observable_support.csv',summaries)
    selected=np.array([i for i,r in enumerate(rows) if r['joint_template_eligible']=='True'])
    atom_indices=np.concatenate([np.arange(d['offsets'][i],d['offsets'][i+1]) for i in selected])
    offsets=np.r_[0,np.cumsum(np.diff(d['offsets'])[selected])]
    parent_names=list(parents)
    masks=np.full((len(selected),max(p['n_removable_sites'] for p in parents.values())),-1,dtype=np.int8)
    lengths=[]
    for j,i in enumerate(selected):
        v=rows[i]['vacancy_mask'];masks[j,:len(v)]=list(map(int,v));lengths.append(len(v))
    np.savez_compressed(HERE/'joint_training_candidates.npz',reference_index=selected,offsets=offsets,
                        numbers=d['numbers'][atom_indices],positions_A=d['positions'][atom_indices],
                        dft_forces_eV_A=d['forces'][atom_indices],dft_energy_eV=d['energy'][selected],
                        dft_stress_eV_A3=d['stress'][selected],cell_A=d['cell'][selected],
                        displacement_A=enc['displacement_A'][atom_indices],atom_parent_slot=enc['atom_parent_slot'][atom_indices],
                        parent_names=np.array(parent_names),parent_index=np.array([parent_names.index(rows[i]['parent_id']) for i in selected]),
                        orientation=enc['orientation'][selected],origin_shift=enc['origin_shift'][selected],
                        vacancy_mask=masks,mask_length=np.array(lengths))
    # Explicit operations with site permutations and cycle decompositions.
    operation_examples=[]
    for name,p in parents.items():
        f=np.asarray(p['fractional_sites']);n=np.asarray(p['numbers']);li=f[n==3];h=np.asarray(p['cell_A'])
        categories={}
        for R,t in zip(p['rotations'],p['translations']):
            R=np.asarray(R);t=np.asarray(t)
            if np.array_equal(R,np.eye(3,dtype=int)):
                category='identity' if np.linalg.norm(t-np.rint(t))<1e-6 else 'internal_periodic_translation'
            elif np.array_equal(R,-np.eye(3,dtype=int)):category='inversion'
            elif np.array_equal(R@R@R,np.eye(3,dtype=int)) and not np.array_equal(R,np.eye(3,dtype=int)):category='threefold_rotation'
            else:continue
            if category in categories:continue
            cost=m.distance_matrix((li@R.T+t)%1,li,h);perm=cost.argmin(axis=1).tolist()
            seen=set();cycles=[]
            for a in range(len(li)):
                if a in seen:continue
                cyc=[];b=a
                while b not in seen:seen.add(b);cyc.append(b);b=perm[b]
                cycles.append(cyc)
            categories[category]=dict(parent_id=name,category=category,rotation_fractional=R.tolist(),translation_fractional=t.tolist(),
                                       Li_site_permutation_source_to_destination=perm,cycles=cycles)
        operation_examples.extend(categories.values())
    m.dump(HERE/'operation_examples.json',operation_examples)
    # Theoretical support for all small parents; original exact Burnside counts
    # for the 24-site parent are reused rather than re-enumerating 2^24 masks.
    countfile=HERE.parents[2]/'collaborator_brief/assets/symmetry_counts.json'
    old=json.loads(countfile.read_text())
    possible={c['name']:c['orbit_counts_by_k'] for c in old['cases']}
    for name,p in parents.items():
        M=p['n_removable_sites']
        if M>12:continue
        states=np.arange(1<<M,dtype=np.uint32);canonical=states.copy()
        for perm in p['li_permutations']:
            tr=np.zeros_like(states)
            for a,b in enumerate(perm):tr|=((states>>a)&1)<<b
            canonical=np.minimum(canonical,tr)
        pop=np.array([int(x).bit_count() for x in states])
        possible[name]=[len(set(canonical[pop==k])) for k in range(M+1)]
    m.dump(HERE/'possible_mask_orbits.json',possible)
    plt.rcParams.update({'font.size':10,'savefig.dpi':180})
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,(name,p) in zip(axs.flat,parents.items()):
        for migration,style,label in [(False,'-','bulk / historical'),(True,'--','migration reference only')]:
            rr=[r for r in rows if r['parent_id']==name and (r['phase']=='migration')==migration]
            if not rr:continue
            vals=np.sort([max(float(r['host_half_spacing_ratio_max']),float(r['Li_half_spacing_ratio_max'])) for r in rr])
            ax.plot(vals,np.arange(1,len(vals)+1)/len(vals),style,label=f'{label} (n={len(rr)})')
        ax.axvline(1,color='black',ls=':',label='localization boundary')
        ax.set(title=name,xlabel='max displacement / half same-species site spacing',ylabel='fraction of geometries')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    for suffix in ('pdf','png'):fig.savefig(HERE/f'mapping_diagnostics.{suffix}')
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,(name,p) in zip(axs.flat,parents.items()):
        M=p['n_removable_sites'];allowed=possible[name]
        rr=[r for r in rows if r['parent_id']==name and r['phase']!='migration']
        local=[];assigned=[]
        for nli in range(M+1):
            gg=[r for r in rr if int(r['n_Li'])==nli]
            assigned.append(len(set(r['mask_orbit_id'] for r in gg)))
            local.append(len(set(r['mask_orbit_id'] for r in gg if r['joint_template_eligible']=='True')))
        ax.plot(range(M+1),allowed[::-1],'o-',color='black',label='possible parent-mask orbits')
        ax.plot(range(M+1),assigned,'s--',color='gray',label='all assigned (includes review)')
        ax.plot(range(M+1),local,'o-',label='localized joint subset')
        ax.set(title=name,xlabel='number of retained Li atoms',ylabel='distinct mask orbits',yscale='symlog')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    for suffix in ('pdf','png'):fig.savefig(HERE/f'observed_orbit_coverage.{suffix}')
    plt.close(fig)
    m.dump(HERE/'export_receipt.json',dict(n_joint_geometries=len(selected),n_joint_atoms=len(atom_indices),
           n_all_unique_geometries=len(rows),joint_export_sha256=m.file_sha(HERE/'joint_training_candidates.npz'),
           semantics='Exact archived geometry and labels; vacancy bit 1=absent, 0=present, padded -1. No symmetry-mask deduplication; no SCF/spin recertification; no training split yet.'))
    print('EXPORT_COMPLETE',len(selected),flush=True)

if __name__=='__main__':main()

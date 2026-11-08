"""Read-only saddle, route-tangent and off-site endpoint geometry audit.

No calculator is constructed. Viewing exports retain exact NPZ provenance and
continuous parent atom identities; coordination is reported as neighbor distances
rather than asserted from an unvalidated bonding cutoff.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import ase
from ase.geometry import find_mic
from ase.io import write
from ase.mep import NEB
from ase.mep.neb import NEBState
from ase.neighborlist import neighbor_list
import numpy as np
from scipy.optimize import linear_sum_assignment
from cathode_cvae.runtime import require_compute_allocation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'conditioning_20261107'))
from neb_common import atoms_from,digest,geometry_hash,load_npz,read_json,write_json


def save_csv(path,rows):
    with Path(path).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def assignment(atom,parent,ids,migrator):
    host=np.flatnonzero(atom.numbers!=3);li=np.flatnonzero(atom.numbers==3);sites=parent['li_indices']
    difference,_=find_mic(atom.positions[host]-parent['positions_A'][ids[host]],atom.cell,atom.pbc);shift=difference.mean(0)
    vector=atom.positions[li,None,:]-shift-parent['positions_A'][sites][None,:,:]
    _,distance=find_mic(vector.reshape(-1,3),atom.cell,atom.pbc);distance=distance.reshape(len(li),len(sites))
    ii,jj=linear_sum_assignment(distance**2);nearest=distance.argmin(1);sorted_distance=np.sort(distance,axis=1)
    occupancy=np.zeros(len(sites),np.uint8);occupancy[jj]=1;vacancy=1-occupancy
    key=min(''.join(map(str,vacancy[inv].tolist())) for inv in np.argsort(parent['permutations'],axis=1))
    ordinal=int(np.flatnonzero(li==migrator)[0])
    return dict(raw_vacancy_mask=''.join(map(str,vacancy.tolist())),canonical_orbit_key=key,
        migrating_nearest_parent_site=int(nearest[ordinal]),migrating_assigned_parent_site=int(jj[ordinal]),
        migrating_nearest_distance_A=float(sorted_distance[ordinal,0]),migrating_second_distance_gap_A=float(sorted_distance[ordinal,1]-sorted_distance[ordinal,0]),
        nearest_collisions=int(len(nearest)-len(set(nearest.tolist()))),assignment_vs_nearest_disagreements=int(np.sum(jj!=nearest)),
        host_shift_A=shift.tolist())


def aligned_geometry(atom,reference):
    host=atom.numbers!=3;delta,_=find_mic(atom.positions-reference.positions,atom.cell,atom.pbc)
    shift=delta[host].mean(0);d,_=find_mic(atom.positions-reference.positions-shift,atom.cell,atom.pbc)
    return dict(RMS_A=float(np.sqrt(np.mean(np.sum(d*d,axis=1)))),maximum_atom_A=float(np.linalg.norm(d,axis=1).max()),host_shift_A=shift.tolist())


def direction_overlap(mode,tangent,masses):
    phi=np.asarray(mode,float).ravel();t=np.asarray(tangent,float).ravel().copy();root=np.repeat(np.sqrt(masses),3)
    tm=t*root;tm/=np.linalg.norm(tm);cart=phi/root;cart/=np.linalg.norm(cart);t/=np.linalg.norm(t)
    return dict(mass_metric_signed_overlap=float(phi@tm),mass_metric_absolute_overlap=float(abs(phi@tm)),
        Cartesian_signed_overlap=float(cart@t),Cartesian_absolute_overlap=float(abs(cart@t)))


def audit_hessian(folder,receipt,h):
    """Reconstruct central derivatives/eigenpairs and bind all displaced geometries."""
    delta=receipt['binding']['delta_A'];dof=3*len(h['numbers']);plus=h['forces_plus'];minus=h['forces_minus']
    if plus.shape!=(dof,len(h['numbers']),3) or not np.isfinite([plus,minus]).all():raise ValueError('Incomplete finite-difference forces')
    raw=-(plus-minus).reshape(dof,dof).T/(2*delta)
    raw_difference=float(np.max(np.abs(raw-h['hessian_raw'])))
    if raw_difference>128*np.finfo(float).eps*max(1.,np.abs(raw).max()):raise ValueError('Saved Cartesian Hessian differs from raw central forces')
    mass=np.repeat(h['masses_amu'],3);dynamic=.5*(raw+raw.T)/np.sqrt(mass[:,None]*mass[None,:])
    translation=np.zeros((dof,3))
    for axis in range(3):translation[axis::3,axis]=np.sqrt(h['masses_amu'])
    internal=np.linalg.qr(translation,mode='complete')[0][:,3:]
    values,vectors=np.linalg.eigh(internal.T@dynamic@internal);mode=internal@vectors[:,0]
    eigenvalue_difference=float(np.max(np.abs(values-h['internal_eigenvalues'])))
    if eigenvalue_difference>128*np.finfo(float).eps*max(1.,np.abs(values).max()):raise ValueError('Recomputed projected eigenvalues differ')
    calls=[json.loads(line) for line in (folder/'calculator_calls.jsonl').read_text().splitlines() if line]
    central=[r for r in calls if r['phase']=='central_displacement_Hessian']
    if len(central)!=2*dof:raise ValueError('Unexpected actual central-force call count')
    max_force_norm_difference=0.
    for column in range(dof):
        for sign,force in [(1,plus[column]),(-1,minus[column])]:
            row=central[2*column+(sign<0)];atom=atoms_from(h);atom.positions.flat[column]+=sign*delta
            if geometry_hash(atom)!=row['geometry_sha256'] or row['status']!='success':raise ValueError('Displacement geometry/force lineage differs')
            difference=abs(float(np.linalg.norm(force,axis=1).max())-row['physical_Fmax_eV_A'])
            if difference>8*np.finfo(np.float32).eps*max(1.,row['physical_Fmax_eV_A']):raise ValueError('Force array disagrees with actual call norm')
            max_force_norm_difference=max(max_force_norm_difference,difference)
    return dict(delta_A=delta,reconstructed_force_columns=dof,verified_actual_displacement_calls=len(central),
        max_Cartesian_Hessian_difference=raw_difference,max_projected_eigenvalue_difference=eigenvalue_difference,
        negative_mode_overlap_stored_vs_recomputed=float(abs(mode@h['eigenvectors_mass_weighted'][:,0])),
        max_stored_vs_call_force_norm_difference_eV_A=max_force_norm_difference,
        force_norm_check='8*float32eps*max(1,loggedFmax), accounting for float32norm versus storedfloat64 force-vector norm; arithmetic serialization check only')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--saddles',nargs=2,type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--migrating-atom',type=int,default=59)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();require_compute_allocation()
    args.output.mkdir(parents=True,exist_ok=False);parent=load_npz(args.parent)
    executed_source=Path(__file__).read_bytes();(args.output/'executed_source.py').write_bytes(executed_source)
    sources=[];loaded=[];independent=[]
    for folder in args.saddles:
        receipt=read_json(folder/'receipt.json');h=load_npz(folder/'hessian.npz')
        if receipt['status']!='COMPLETE_CURVATURE_DIAGNOSTIC':raise ValueError('Require complete actual finite-difference outputs')
        if receipt['binding']['candidate_geometry_sha256']!=geometry_hash(atoms_from(h)):
            raise ValueError('Candidate coordinate hash differs from actual Hessian provenance')
        sources.append(dict(path=str((folder/'receipt.json').resolve()),sha256=digest(folder/'receipt.json'),hessian_sha256=digest(folder/'hessian.npz')))
        independent.append(audit_hessian(folder,receipt,h))
        loaded.append((folder,receipt,h))
    a,b=loaded
    if a[1]['binding']['candidate_geometry_sha256']!=b[1]['binding']['candidate_geometry_sha256']:raise ValueError('Displacement-size comparison changed candidate')
    phi=[r[2]['eigenvectors_mass_weighted'][:,0] for r in loaded]
    delta_comparison=dict(candidate_geometry_sha256=a[1]['binding']['candidate_geometry_sha256'],
        displacement_A=[r[1]['binding']['delta_A'] for r in loaded],
        lowest_internal_eigenvalues=[float(r[2]['internal_eigenvalues'][0]) for r in loaded],
        next_internal_eigenvalues=[float(r[2]['internal_eigenvalues'][1]) for r in loaded],
        strictly_negative_counts=[int(np.sum(r[2]['internal_eigenvalues']<0)) for r in loaded],
        absolute_negative_mode_overlap=float(abs(phi[0]@phi[1])),
        relative_Hessian_difference=float(np.linalg.norm(a[2]['hessian_symmetric']-b[2]['hessian_symmetric'])/np.linalg.norm(a[2]['hessian_symmetric'])))
    folder,receipt,h=next(r for r in loaded if r[1]['binding']['delta_A']==.01)
    if digest(receipt['binding']['band']['path'])!=receipt['binding']['band']['sha256']:raise ValueError('Actual source band changed')
    data=load_npz(receipt['binding']['band']['path']);image=receipt['binding']['image'];ids=data['parent_indices']
    if not np.all(data['labels_match_geometry']) or not np.array_equal(data['positions_A'][image],h['positions_A']):
        raise ValueError('Band labels or candidate geometry differ from their exact sources')
    if not np.array_equal(h['numbers'],parent['numbers'][ids]):raise ValueError('Continuous atom identities disagree with parent chemistry')
    images=[atoms_from(data,row) for row in data['positions_A']];neb=NEB(images,method='improvedtangent',climb=True,k=.1)
    state=NEBState(neb,images,data['energy_eV']);s1,s2=state.spring(image-1),state.spring(image)
    tangent=neb.neb_method.get_tangent(state,s1,s2,image)
    overlaps=dict(improved_tangent=direction_overlap(h['eigenvectors_mass_weighted'][:,0],tangent,h['masses_amu']),
        local_chord=direction_overlap(h['eigenvectors_mass_weighted'][:,0],s1.t+s2.t,h['masses_amu']),
        convention='Mass metric uses normalized sqrt(M)*CartesianNEBtangent against unit mass-weighted eigenvector; Cartesian alternative compares normalized M^-1/2 eigenvector. Signed mode orientation is arbitrary.',
        no_new_force_calls=True,ASE_version=ase.__version__)
    modes=h['eigenvectors_mass_weighted'][:,0].reshape(-1,3);participation=np.sum(modes*modes,axis=1)
    save_csv(args.output/'negative_mode_participation.csv',[dict(local_atom=i,parent_atom=int(ids[i]),Z=int(h['numbers'][i]),mass_weighted_norm_fraction=float(participation[i])) for i in np.argsort(participation)[::-1]])
    states=[]
    for i,atom in enumerate(images):states.append((f'band_image_{i:02d}',atom,float(data['energy_eV'][i]),np.asarray(data['forces_eV_A'][i]),data['stress_eV_A3'][i]))
    for name in ['minus','plus']:
        d=load_npz(folder/f'downhill_{name}.npz')
        if not np.all(d['labels_match_geometry']):raise ValueError('Downhill endpoint labels do not match its geometry')
        states.append(('downhill_'+name,atoms_from(d,d['positions_A'][0]),float(d['energy_eV'][0]),d['forces_eV_A'][0],d['stress_eV_A3'][0]))
    rows=[];neighbors=[];exports=[];output=args.output/'geometries';output.mkdir()
    special={0:'original_start',len(images)-1:'original_end',image:'candidate_CI'}
    for name,atom,energy,force,stress in states:
        m=assignment(atom,parent,ids,args.migrating_atom);compare=[aligned_geometry(atom,end) for end in [images[0],images[-1]]]
        ni,nj,shift,distance=neighbor_list('ijSd',atom,5.,self_interaction=False)
        nearby=np.flatnonzero(ni==args.migrating_atom);oxygen=[]
        for Z,number in [(8,12),(27,8),(3,8)]:
            selected=sorted([int(i) for i in nearby if atom.numbers[nj[i]]==Z],key=lambda i:distance[i])[:number]
            for rank,j in enumerate(selected,1):
                neighbors.append(dict(state=name,neighbor_Z=Z,rank=rank,distance_A=float(distance[j]),neighbor_local_atom=int(nj[j]),neighbor_parent_atom=int(ids[nj[j]]),cell_image=' '.join(map(str,shift[j].tolist()))))
                if Z==8:oxygen.append(float(distance[j]))
        row=dict(state=name,energy_eV_cell=energy,physical_Fmax_eV_A=float(np.linalg.norm(force,axis=1).max()),
            geometry_sha256=geometry_hash(atom),**m,
            start_RMS_A=compare[0]['RMS_A'],start_maximum_atom_A=compare[0]['maximum_atom_A'],
            end_RMS_A=compare[1]['RMS_A'],end_maximum_atom_A=compare[1]['maximum_atom_A'],
            O_distance_4_A=oxygen[3],O_distance_5_A=oxygen[4],O_distance_6_A=oxygen[5],O_5_minus_4_gap_A=oxygen[4]-oxygen[3])
        rows.append(row)
        display=special.get(int(name.rsplit('_',1)[1]),name) if name.startswith('band_image_') else name
        if display:
            path=output/(display+'.exact.npz')
            np.savez_compressed(path,positions_A=atom.positions,cell_A=np.asarray(atom.cell),pbc=atom.pbc,numbers=atom.numbers,
                parent_indices=ids,energy_eV=np.array(energy),forces_eV_A=force,stress_eV_A3=stress,
                geometry_sha256=np.array(geometry_hash(atom)),labels_match_geometry=np.array(True))
            write(output/(display+'.POSCAR'),atom,format='vasp',direct=True,sort=False)
            write(output/(display+'.cif'),atom,format='cif',wrap=False)
            exports.append(dict(label=display,state=name,geometry_sha256=geometry_hash(atom),
                exact_npz_sha256=digest(path),POSCAR_sha256=digest(output/(display+'.POSCAR')),CIF_sha256=digest(output/(display+'.cif')),
                source_band=receipt['binding']['band'],
                source_downhill=None if name.startswith('band_image_') else dict(path=str((folder/(name+'.npz')).resolve()),sha256=digest(folder/(name+'.npz')))))
    save_csv(args.output/'states.csv',rows);save_csv(args.output/'neighbor_distances.csv',neighbors)
    save_csv(args.output/'geometries'/'atom_ID_mapping.csv',[dict(local_atom=i,parent_atom=int(ids[i]),Z=int(h['numbers'][i])) for i in range(len(ids))])
    write_json(output/'provenance.json',dict(teacher=receipt['teacher'],candidate_sources=sources,exports=exports,
        labels_scope='Exact-coordinateNPZcontains original same-geometry MLIP labels. POSCAR/CIF are viewing exports only; no newSCFs/DFTinputs submitted. Atom order/parentIDs retained.'))
    # Neighbor-distance plot avoids a chemically uncalibrated coordination cutoff.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4));chosen={'band_image_00':'original start',f'band_image_{len(images)-1:02d}':'original end',f'band_image_{image:02d}':'CI','downhill_minus':'downhill−','downhill_plus':'downhill+'}
    origin=images[0];host=origin.numbers!=3
    end_shift=np.mean(find_mic(images[-1].positions[host]-origin.positions[host],origin.cell,origin.pbc)[0],axis=0)
    direction=images[-1].positions[args.migrating_atom]-origin.positions[args.migrating_atom]-end_shift;direction/=np.linalg.norm(direction)
    band_xy=[]
    for name,atom,energy,force,stress in states:
        t=np.mean(find_mic(atom.positions[host]-origin.positions[host],origin.cell,origin.pbc)[0],axis=0)
        v=atom.positions[args.migrating_atom]-origin.positions[args.migrating_atom]-t;along=float(v@direction);off=float(np.linalg.norm(v-along*direction))
        if name.startswith('band_image_'):band_xy.append([along,off])
        if name in chosen:
            o=[r['distance_A'] for r in neighbors if r['state']==name and r['neighbor_Z']==8][:8]
            axes[0].plot(np.arange(1,len(o)+1),o,'-o',label=chosen[name]);axes[1].scatter(along,off,label=chosen[name],s=45)
            axes[1].annotate(chosen[name],(along,off),xytext=(4,4),textcoords='offset points',fontsize=8)
    band_xy=np.asarray(band_xy);axes[1].plot(band_xy[:,0],band_xy[:,1],'-o',color='grey',alpha=.4,zorder=0)
    axes[0].set(xlabel='O neighbor distance rank',ylabel='Li–O distance (Å)',title='Actual migrating-Li environments');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Li displacement along original endpoint vector (Å)',ylabel='Perpendicular Li displacement magnitude (Å)',title='Host-aligned continuous marked Li')
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.output/'candidate_environments.pdf');fig.savefig(args.output/'candidate_environments.png',dpi=160);plt.close(fig)
    result=dict(status='COMPLETE_READ_ONLY_SADDLE_REVIEW',source_sha256=hashlib.sha256(executed_source).hexdigest(),sources=sources,
        independent_central_force_Hessian_checks=independent,
        displacement_size_comparison=delta_comparison,mode_route_overlaps=overlaps,
        source_band=receipt['binding']['band'],candidate_image=image,migrating_atom=args.migrating_atom,
        coordination_convention='Full periodic neighbor images within5A used only as a search window; ranked O/Co/Li distances, no inferred bonding cutoff or phase assignment.',
        force_calls=0,DFT_calls=0,limits='Stable local index-one curvature does not verify that this whole marked path is one elementary hop. Same occupation orbit is not physical PES-basin identity.',
        outputs={p.name:digest(p) for p in args.output.iterdir() if p.is_file()})
    write_json(args.output/'receipt.json',result)


if __name__=='__main__':main()

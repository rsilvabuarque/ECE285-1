"""Bounded finite-displacement internal-mode diagnostic of one retained CI image.

A negative curvature is not automatically a verified transition state. Inspect
stationarity, displacement-size sensitivity, unstable-mode count and optional
downhill connections. No endpoint or saddle is symmetry canonicalized.
"""
import time
PROCESS_START=time.monotonic()
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import traceback

import numpy as np
from ase import units
from ase.geometry import find_mic
from ase.optimize import FIRE
from cathode_cvae.runtime import require_compute_allocation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'conditioning_20261107'))
from neb_common import (CallLedger,BudgetExpired,attach_calculators,atoms_from,digest,
    geometry_hash,load_npz,mapping,read_json,snapshot,write_json)

FREQUENCY_FACTOR_THz=float(np.sqrt(units._e/(1e-20*units._amu))/(2*np.pi)*1e-12)


def internal_modes(hessian,masses):
    """Remove only three mass-weighted translations of a periodic fixed cell."""
    mass=np.repeat(np.asarray(masses,float),3);sym=.5*(hessian+hessian.T)
    dynamical=sym/np.sqrt(mass[:,None]*mass[None,:]);translation=np.zeros((len(mass),3))
    for a in range(3):translation[a::3,a]=np.sqrt(masses)
    q=np.linalg.qr(translation,mode='complete')[0];internal=q[:,3:]
    projected=internal.T@dynamical@internal;value,vector=np.linalg.eigh(projected)
    full=internal@vector;residual=np.linalg.norm(projected@vector-vector*value[None,:],axis=0)
    return dict(hessian_symmetric=sym,dynamical=dynamical,translation_basis=q[:,:3],internal_basis=internal,
        internal_eigenvalues=value,eigenvectors_mass_weighted=full,
        frequencies_signed_THz=np.sign(value)*np.sqrt(np.abs(value))*FREQUENCY_FACTOR_THz,
        eigenpair_projected_residuals=residual)


def endpoint_comparison(atom,endpoint):
    host=np.flatnonzero(atom.numbers!=3)
    delta,_=find_mic(atom.positions-endpoint.positions,atom.cell,atom.pbc)
    shift=delta[host].mean(0);aligned,_=find_mic(atom.positions-endpoint.positions-shift,atom.cell,atom.pbc)
    return dict(identity_preserving_host_aligned_RMS_A=float(np.sqrt(np.mean(np.sum(aligned*aligned,axis=1)))),
        maximum_atom_difference_A=float(np.linalg.norm(aligned,axis=1).max()),host_translation_A=shift.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--band',type=Path,required=True)
    p.add_argument('--band-receipt',type=Path,required=True);p.add_argument('--image',type=int,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--root-protocol',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--delta-A',type=float,default=.01);p.add_argument('--maximum-candidate-Fmax',type=float,default=.02)
    p.add_argument('--downhill',action='store_true');p.add_argument('--downhill-amplitude-A',type=float,default=.05)
    p.add_argument('--downhill-steps',type=int,default=300);p.add_argument('--downhill-fmax',type=float,default=.02)
    p.add_argument('--max-seconds',type=float,default=2700);p.add_argument('--max-calls',type=int,default=1800)
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=4);args=p.parse_args()
    require_compute_allocation();read_json(args.root_protocol);source=read_json(args.band_receipt);data=load_npz(args.band)
    if not source['converged'] or args.image!=source['climbing_image']:raise ValueError('Require the named climbing image of a numerically converged band')
    if source['physical_image_Fmax_eV_A'][args.image]>args.maximum_candidate_Fmax:raise ValueError('Source candidate misses the declared physical-force precondition')
    if not 0<args.delta_A<=.02 or not 0<args.max_seconds<=2700 or not 0<args.downhill_steps<=300:
        raise ValueError('Outside bounded displacement/time/downhill-step settings')
    parent=load_npz(args.parent);ids=data['parent_indices'];candidate=atoms_from(data,data['positions_A'][args.image])
    endpoints=[atoms_from(data,row) for row in data['positions_A'][[0,-1]]]
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(band=dict(path=str(args.band.resolve()),sha256=digest(args.band)),
        band_receipt=dict(path=str(args.band_receipt.resolve()),sha256=digest(args.band_receipt)),image=args.image,
        candidate_geometry_sha256=geometry_hash(candidate),parent_sha256=digest(args.parent),root_protocol_sha256=digest(args.root_protocol),
        checkpoint_sha256=digest(args.checkpoint),source_sha256=digest(__file__),delta_A=args.delta_A,
        maximum_candidate_physical_Fmax_eV_A=args.maximum_candidate_Fmax,
        provisional_setting_provenance='0.01A central difference matches prior saved harmonic lane; sensitivity must be checked separately.0.02eV/A is the declared numerical candidate gate, not physical saddle accuracy.',
        Hessian='-dF/dR central finite differences, fixed cell; symmetrized then mass weighted; remove exactly3 translations',
        frequency_factor_THz=FREQUENCY_FACTOR_THz,downhill_requested=args.downhill,
        frequency_factor_provenance='sqrt(ASE units._e J/eV /(1e-20 m2/A2 *ASE units._amu kg/amu))/(2pi)*1e-12 ps/s',
        downhill_amplitude_A=args.downhill_amplitude_A,downhill_steps=args.downhill_steps,downhill_Fmax_eV_A=args.downhill_fmax,
        maximum_seconds_including_loading=args.max_seconds,maximum_EFS_calls=args.max_calls,
        scope='Potential-energy saddle diagnostic only. No DFT or finite-temperature activation free energy.')
    write_json(args.output/'protocol.json',protocol)
    ledger=CallLedger(args.output/'calculator_calls.jsonl',args.max_seconds,args.max_calls,start=PROCESS_START)
    dof=3*len(candidate);plus=np.full((dof,len(candidate),3),np.nan);minus=plus.copy()
    receipt=dict(status='PREPARED',binding=protocol);code=2
    try:
        from cathode_cvae.mlip.calculators import make_calculator
        calc,teacher=make_calculator('uma',args.checkpoint,args.device,args.threads);receipt['teacher']=teacher
        attach_calculators([candidate],calc,ledger,'candidate');ledger.phase='candidate_stationarity'
        initial_force=candidate.get_forces();initial_energy=candidate.get_potential_energy()
        actual_fmax=float(np.linalg.norm(initial_force,axis=1).max())
        receipt['candidate_physical_Fmax_eV_A']=actual_fmax
        snapshot([candidate],args.output/'candidate.npz',parent_indices=ids)
        if actual_fmax>args.maximum_candidate_Fmax:raise ValueError('Fresh candidate force misses the frozen numerical precondition')
        ledger.phase='central_displacement_Hessian';base=candidate.positions.copy()
        for column in range(dof):
            for sign,destination in [(1,plus),(-1,minus)]:
                atom=candidate.copy();atom.positions.flat[column]+=sign*args.delta_A
                attach_calculators([atom],calc,ledger,f'dof{column}_{sign:+d}')
                destination[column]=atom.get_forces()
            if column%12==0:
                np.savez_compressed(args.output/'hessian_progress.npz',forces_plus=plus,forces_minus=minus,
                    candidate_positions_A=base,parent_indices=ids,delta_A=args.delta_A)
        raw=-((plus-minus)/(2*args.delta_A)).reshape(dof,dof).T
        modes=internal_modes(raw,candidate.get_masses())
        np.savez_compressed(args.output/'hessian.npz',hessian_raw=raw,forces_plus=plus,forces_minus=minus,
            positions_A=base,cell_A=np.asarray(candidate.cell),numbers=candidate.numbers,pbc=candidate.pbc,
            parent_indices=ids,masses_amu=candidate.get_masses(),delta_A=args.delta_A,**modes)
        value=modes['internal_eigenvalues'];receipt.update(status='COMPLETE_CURVATURE_DIAGNOSTIC',
            strictly_negative_internal_eigenvalues=int(np.sum(value<0)),lowest_eigenvalues_eV_A2_amu=value[:12].tolist(),
            signed_lowest_frequencies_THz=modes['frequencies_signed_THz'][:12].tolist(),
            antisymmetric_relative_Frobenius=float(np.linalg.norm(raw-raw.T)/max(np.linalg.norm(raw),np.finfo(float).tiny)),
            maximum_projected_eigenpair_residual=float(modes['eigenpair_projected_residuals'].max()),
            candidate_energy_eV=float(initial_energy),downhill=[])
        if args.downhill and value[0]<0:
            displacement=(modes['eigenvectors_mass_weighted'][:,0]/np.repeat(np.sqrt(candidate.get_masses()),3)).reshape(-1,3)
            displacement*=args.downhill_amplitude_A/np.linalg.norm(displacement,axis=1).max()
            for sign in [-1,1]:
                atom=candidate.copy();atom.positions[:]=base+sign*displacement;name='minus' if sign<0 else 'plus'
                attach_calculators([atom],calc,ledger,'downhill_'+name);ledger.phase='downhill_'+name
                opt=FIRE(atom,logfile=str(args.output/f'downhill_{name}.log'),trajectory=str(args.output/f'downhill_{name}.traj'),dt=.1,dtmax=1.,maxstep=.05)
                converged=bool(opt.run(fmax=args.downhill_fmax,steps=args.downhill_steps))
                force=atom.get_forces();energy=atom.get_potential_energy()
                snapshot([atom],args.output/f'downhill_{name}.npz',parent_indices=ids)
                receipt['downhill'].append(dict(sign=sign,numerically_converged=converged,steps=opt.nsteps,
                    physical_Fmax_eV_A=float(np.linalg.norm(force,axis=1).max()),energy_eV=float(energy),
                    energy_minus_candidate_eV=float(energy-initial_energy),site_mapping=mapping(atom,parent,ids),
                    identity_preserving_endpoint_comparisons=[endpoint_comparison(atom,e) for e in endpoints]))
        receipt['index_one_validation']='Not established by strict sign count alone; review force, displacement-size sensitivity and both downhill connections'
        code=0
    except BudgetExpired as e:receipt.update(status='BOUNDED_PARTIAL_PRESERVED',error=str(e));code=3
    except Exception as e:receipt.update(status='FAILED_PRESERVED',error=str(e),exception_type=type(e).__name__,traceback=traceback.format_exc())
    finally:
        np.savez_compressed(args.output/'hessian_last_state.npz',forces_plus=plus,forces_minus=minus,
            candidate_positions_A=candidate.positions,parent_indices=ids,delta_A=args.delta_A)
        receipt.update(call_accounting=ledger.summary(),completed_UTC=datetime.now(timezone.utc).isoformat())
        write_json(args.output/'receipt.json',receipt)
    return code


if __name__=='__main__':raise SystemExit(main())

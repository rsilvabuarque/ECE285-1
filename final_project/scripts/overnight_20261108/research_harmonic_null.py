"""Finite-window xPT diagnostic on the actual saved positive internal spectrum.

Fixed kBT energy/mode and random phases reproduce canonical harmonic velocity
second moments. They do not constitute a canonical MD trajectory or measure any
anharmonic correction. The complete spectrum/eigenvectors remain source bound.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from cathode_cvae.data import sha256
from cathode_cvae.runtime import require_compute_allocation

OLD_ADAPTER=Path(__file__).resolve().parents[1]/'overnight_20261107'/'thermal_xpt.py'
spec=importlib.util.spec_from_file_location('thermal_xpt',OLD_ADAPTER)
xpt=importlib.util.module_from_spec(spec);spec.loader.exec_module(xpt)


def generate_velocities(frequencies,eigenvectors,masses,time_ps,phase,temperature,block=1024):
    """qdot_j=sqrt(2 kBT/C)cos(2 pi nu_j t+phi_j), v=M^-1/2 Q qdot.

    nu in cycles/ps (THz), mass amu, v Angstrom/ps. C converts
    amu Angstrom^2/ps^2 into eV. Blocks avoid a full frames×modes temporary.
    """
    nu=np.asarray(frequencies,float);q=np.asarray(eigenvectors,float);mass=np.asarray(masses,float)
    if q.shape!=(3*len(mass),len(nu)) or len(phase)!=len(nu) or np.any(nu<=0):
        raise ValueError('Expected a full positive internal eigenbasis and one phase/mode')
    if not np.allclose(q.T@q,np.eye(len(nu)),atol=1e-9,rtol=0):raise ValueError('Mass-weighted eigenvectors are not orthonormal')
    amplitude=np.sqrt(2*xpt.KB*temperature/xpt.KINETIC_EV_PER_AMU_A2_PS2)
    velocity=np.empty((len(time_ps),len(mass),3),dtype=np.float64)
    for first in range(0,len(time_ps),block):
        last=min(first+block,len(time_ps));angle=2*np.pi*np.asarray(time_ps[first:last,None])*nu+phase
        modal=amplitude*np.cos(angle)
        velocity[first:last]=(modal@q.T).reshape(-1,len(mass),3)/np.sqrt(mass)[None,:,None]
    return velocity


def matched_references(frequencies,temperature,minimum_energy,total_anchor):
    exact=xpt.harmonic(frequencies,temperature,minimum_energy)
    if exact['status']!='CALCULATED':raise ValueError('Nonpositive internal mode invalidates this harmonic null')
    classical_mean=minimum_energy+len(frequencies)*xpt.KB*temperature
    return dict(exact=exact,
        classical_anchor_matched_F_eV=total_anchor-temperature*exact['S_classical_eV_K'],
        quantum_anchor_matched_F_eV=total_anchor+exact['U_quantum_eV']-classical_mean-temperature*exact['S_quantum_eV_K'],
        anchor_minus_canonical_mean_eV=total_anchor-classical_mean)


def array_sha(array):
    h=hashlib.sha256()
    for first in range(0,len(array),1024):h.update(np.ascontiguousarray(array[first:first+1024]).tobytes())
    return h.hexdigest()


def augment_row(row,seed,length,frequencies,minimum_energy,anchor,target):
    row=dict(row);native=row['observed_temperature_K'];ref=matched_references(frequencies,native,minimum_energy,anchor)
    row.update(phase_seed=seed,nominal_length_ps=length,target_temperature_K=target,
        exact_entropy_classical_native_eV_K=ref['exact']['S_classical_eV_K'],
        exact_entropy_quantum_native_eV_K=ref['exact']['S_quantum_eV_K'],
        entropy_error_classical_eV_K=row['S_classical_eV_K']-ref['exact']['S_classical_eV_K'],
        entropy_error_quantum_eV_K=row['S_quantum_eV_K']-ref['exact']['S_quantum_eV_K'],
        F_error_classical_anchor_matched_eV=row['F_classical_MD_anchored_eV']-ref['classical_anchor_matched_F_eV'],
        F_error_quantum_anchor_matched_eV=row['F_quantum_MD_anchored_eV']-ref['quantum_anchor_matched_F_eV'],
        finite_window_anchor_minus_canonical_native_mean_eV=ref['anchor_minus_canonical_mean_eV'])
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--hessian',type=Path,required=True)
    p.add_argument('--minimum',type=Path,required=True);p.add_argument('--vendor',type=Path,required=True)
    p.add_argument('--root-protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase-seeds',nargs='+',type=int,default=[20261108,20261109,20261110])
    p.add_argument('--lengths-ps',nargs='+',type=float,default=[10.,20.,50.]);p.add_argument('--dt-fs',type=float,default=.5)
    p.add_argument('--modes',nargs='+',choices=['1PT','2PT','3PT'],default=['1PT','2PT','3PT'])
    p.add_argument('--temperature-K',type=float,default=433.15);p.add_argument('--corlen',type=float,default=.25)
    p.add_argument('--resume-from',type=Path,help='Read completed immutable result.json files from an interrupted same-protocol run; all new work goes to a fresh output')
    p.add_argument('--max-seconds',type=float,default=3600);args=p.parse_args();require_compute_allocation()
    started=time.monotonic();json.loads(args.root_protocol.read_text());h,minimum=xpt.bound_hessian(args.hessian,args.minimum)
    nu=h['frequencies_signed_THz'];q=h['eigenvectors_mass_weighted'];mass=h['masses_amu'];count=len(mass);ndof=3*count-3
    if len(nu)!=537 or ndof!=537 or q.shape!=(540,537):raise ValueError('This optional lane requires the actual537-mode180-atom spectrum')
    if np.any(nu<=0):raise ValueError('Never abs-value, floor or discard nonpositive internal modes')
    if args.dt_fs<=0 or max(args.lengths_ps)>50 or len(args.phase_seeds)>3:raise ValueError('Bounded initial null: <=3 phases and <=50ps')
    vendor=xpt.audit_vendor(args.vendor);emin=xpt.minimum_energy(minimum);anchor=emin+ndof*xpt.KB*args.temperature_K
    time_ps=np.arange(int(round(max(args.lengths_ps)*1000/args.dt_fs)))*(args.dt_fs/1000)
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(status='FROZEN_BEFORE_SYNTHETIC_ANALYSIS',source_sha256=sha256(__file__),
        adapter=dict(path=str(OLD_ADAPTER),sha256=sha256(OLD_ADAPTER)),vendor=vendor,
        hessian=dict(path=str(args.hessian.resolve()),sha256=sha256(args.hessian)),
        minimum=dict(path=str(args.minimum.resolve()),sha256=sha256(args.minimum)),
        root_protocol=dict(path=str(args.root_protocol.resolve()),sha256=sha256(args.root_protocol)),
        phase_seeds=args.phase_seeds,lengths_ps=args.lengths_ps,dt_fs=args.dt_fs,modes=args.modes,
        temperature_K=args.temperature_K,corlen_fraction=args.corlen,apodization='none',gas_gate='debye_warn',
        amplitude_convention='Every harmonic mode has fixed total energy kBT, independent uniform phase; equal canonical second moments, not canonical modal-energy sampling',
        energy_anchor_eV=anchor,energy_anchor_rule='Eminimum+537*kB*Ttarget, included once',
        positions_scope='Constant minimum coordinates broadcast only for unused atomic FrameData positions; no synthetic atomic geometry trajectory is claimed',
        exact_reference_target=xpt.harmonic(nu,args.temperature_K,emin),no_new_MLIP_DFT_or_MD=True)
    xpt.write_json(args.output/'protocol.json',protocol)
    rows=[];phases=[];failures=[];velocity_records=[];reuse={}
    if args.resume_from:
        original=json.loads((args.resume_from/'protocol.json').read_text())
        for key in ['hessian','minimum','phase_seeds','lengths_ps','dt_fs','modes','temperature_K','corlen_fraction','apodization','gas_gate','amplitude_convention','energy_anchor_eV']:
            if original[key]!=protocol[key]:raise ValueError(f'Interrupted scientific protocol differs in {key}')
        for seed in args.phase_seeds:
            for length in args.lengths_ps:
                for mode in args.modes:
                    saved=args.resume_from/f'seed{seed}'/f'{length:g}ps_{mode}'/'result.json'
                    if saved.exists():reuse[(seed,length,mode)]=saved
        protocol['immutable_completed_results_reused']=[dict(path=str(file.resolve()),sha256=sha256(file)) for file in reuse.values()]
        protocol['interrupted_source_protocol']=dict(path=str((args.resume_from/'protocol.json').resolve()),sha256=sha256(args.resume_from/'protocol.json'))
        xpt.write_json(args.output/'protocol.json',protocol)
    # Store all phases before analysis; interruption cannot lose their identity.
    all_phases=np.asarray([np.random.default_rng(seed).uniform(0,2*np.pi,ndof) for seed in args.phase_seeds])
    np.savez_compressed(args.output/'phases.npz',phase_seeds=np.asarray(args.phase_seeds),phases=all_phases,
        frequencies_THz=nu,temperature_K=args.temperature_K,dt_fs=args.dt_fs)
    for seed in args.phase_seeds:
        if time.monotonic()-started>=args.max_seconds:break
        phase=all_phases[args.phase_seeds.index(seed)];phases.append(phase)
        velocity=generate_velocities(nu,q,mass,time_ps,phase,args.temperature_K)
        com=(velocity*mass[None,:,None]).sum(1)/mass.sum()
        kinetic=.5*xpt.KINETIC_EV_PER_AMU_A2_PS2*np.sum(velocity**2*mass[None,:,None],axis=(1,2))
        d=dict(time_fs=time_ps*1000,velocities_A_ps=velocity,masses_amu=mass,cell_A=h['cell_A'],numbers=h['numbers'],
            positions_A=np.broadcast_to(h['positions_A'],velocity.shape),energy_total_eV=np.full(len(time_ps),anchor),
            kinetic_eV=kinetic)
        xpt.validate_trajectory(d)
        velocity_records.append(dict(seed=seed,velocity_sha256=array_sha(velocity),frames=len(time_ps),
            maximum_COM_speed_A_ps=float(np.linalg.norm(com,axis=1).max()),
            mean_kinetic_eV=float(kinetic.mean()),phase_sha256=array_sha(phase)))
        for length in args.lengths_ps:
            indices=np.arange(int(round(length*1000/args.dt_fs)))
            for mode in args.modes:
                if time.monotonic()-started>=args.max_seconds:break
                output=args.output/f'seed{seed}'/f'{length:g}ps_{mode}'
                try:
                    if (seed,length,mode) in reuse:row=json.loads(reuse[(seed,length,mode)].read_text())
                    else:row=xpt.run_package(d,indices,output,mode=mode,corlen=args.corlen,apodization='none',gas_gate='debye_warn',target_K=args.temperature_K)
                    row=augment_row(row,seed,length,nu,emin,anchor,args.temperature_K)
                    row['raw_result_path']=str(reuse.get((seed,length,mode),output/'result.json').resolve())
                    row['raw_result_sha256']=sha256(row['raw_result_path'])
                    row['reused_completed_analysis']=(seed,length,mode) in reuse
                    rows.append(row)
                    xpt.save_rows(args.output/'analyses_partial.csv',rows)
                    xpt.write_json(args.output/'progress.json',dict(completed=len(rows),failed=len(failures),last_case=[seed,length,mode]))
                except Exception as e:
                    failures.append(dict(phase_seed=seed,length_ps=length,mode=mode,exception_type=type(e).__name__,error=str(e)))
                    xpt.write_json(args.output/'failures_partial.json',failures)
        del velocity,d,kinetic,com
    if rows:xpt.save_rows(args.output/'analyses.csv',rows)
    xpt.write_json(args.output/'failures.json',failures)
    expected=len(args.phase_seeds)*len(args.lengths_ps)*len(args.modes)
    xpt.write_json(args.output/'receipt.json',dict(status='COMPLETE_HARMONIC_NULL' if len(rows)==expected else 'BOUNDED_OR_FAILED_PARTIAL',
        completed_analyses=len(rows),expected_analyses=expected,failures=len(failures),elapsed_seconds=time.monotonic()-started,
        velocity_records=velocity_records,protocol_sha256=sha256(args.output/'protocol.json'),
        source_sha256=sha256(__file__),limitations='Synthetic harmonic estimator diagnostic, not physical MD, an anharmonic correction or validation of a real-LCO free energy. Nested lengths share one realization; phase spread is descriptive.',
        outputs={file.name:sha256(file) for file in args.output.iterdir() if file.is_file()}))


if __name__=='__main__':main()

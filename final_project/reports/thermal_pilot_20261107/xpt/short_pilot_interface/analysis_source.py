#!/usr/bin/env python3
"""Atomic crystal analysis with pinned, unmodified ATLAS py-xPT 2.1.0.

This is an array-I/O and accounting adapter to the actual ATLAS engine, not a
replacement 2PT/3PT implementation. Native thermodynamics use its observed VAC
temperature. Classical Helmholtz energy is separately anchored to the MD total
energy because upstream anchors only quantum fields. Common-temperature values
are explicitly first-order transports, not canonical free-energy calculations.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import warnings

import numpy as np

PIN = '197bdcf5f57c9d04b5527f9e7d6f4fdb8bf3b82d'
KB = 8.617333262145e-5  # eV/K
H_THZ = 4.135667696e-3  # h nu in eV for nu in THz
CM_TO_THZ = 0.0299792458
# Match the pinned package's own output-unit conversion for an exact round trip
# of the supplied eV MD anchor; do not use its rounded eV->kcal factor 23.0605.
EV_PER_KJ_MOL = 1000.0 * 1.03642688e-5
EV_PER_KCAL_MOL = 4.184 * EV_PER_KJ_MOL
KINETIC_EV_PER_AMU_A2_PS2 = 1.0364269656262175e-4
SOURCE_BYTES=Path(__file__).read_bytes()
SOURCE_SHA256=hashlib.sha256(SOURCE_BYTES).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + '\n')


def named_paths(values):
    result = {}
    for value in values:
        key, path = value.split('=', 1)
        if key in result: raise ValueError(f'Repeated identity: {key}')
        result[key] = Path(path)
    return result


def audit_vendor(vendor):
    vendor = Path(vendor).resolve()
    actual = subprocess.check_output(['git', '-C', str(vendor), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != PIN: raise ValueError(f'ATLAS source {actual} differs from pinned {PIN}')
    dirty = subprocess.check_output(['git', '-C', str(vendor), 'status', '--porcelain', '--', 'py-xPT/src'], text=True)
    if dirty: raise ValueError('ATLAS implementation has local source changes')
    import pyxpt
    package_path = Path(pyxpt.__file__).resolve()
    if not package_path.is_relative_to(vendor / 'py-xPT' / 'src'):
        raise ValueError(f'Imported another py-xPT: {package_path}')
    return dict(repository='https://github.com/atlas-nano/codes', commit=actual,
                package_version='2.1.0', imported_path=str(package_path),
                source_hashes={str(p.relative_to(vendor)): sha(p)
                               for p in sorted((vendor / 'py-xPT' / 'src').rglob('*.py'))})


def validate_trajectory(d):
    required = ['velocities_A_ps', 'time_fs', 'masses_amu', 'cell_A',
                'positions_A', 'energy_total_eV']
    for key in required:
        if key not in d: raise ValueError(f'Missing trajectory array {key}')
        if not np.isfinite(d[key]).all(): raise ValueError(f'Non-finite trajectory {key}')
    v = d['velocities_A_ps']; times = d['time_fs']; masses = d['masses_amu']
    if v.ndim != 3 or v.shape[-1] != 3 or times.shape != (len(v),):
        raise ValueError('Velocities must be frames x atoms x 3, with one timestamp per frame')
    if len(v) < 32 or masses.shape != (v.shape[1],) or np.any(masses <= 0):
        raise ValueError('Need positive per-atom masses and at least 32 velocity frames')
    if d['positions_A'].shape != v.shape or d['energy_total_eV'].shape != times.shape:
        raise ValueError('Positions, energies and velocities must refer to the same frames')
    if d['cell_A'].shape != (3, 3) or np.linalg.det(d['cell_A']) <= 0:
        raise ValueError('Expected one right-handed common fixed cell')
    spacing = np.diff(times)
    if np.any(spacing <= 0) or not np.allclose(spacing, spacing[0], rtol=1e-8, atol=1e-8):
        raise ValueError('Saved velocity frames must be uniformly spaced')
    if 'kinetic_eV' in d:
        kinetic = .5 * KINETIC_EV_PER_AMU_A2_PS2 * np.sum(v*v*masses[None,:,None], axis=(1,2))
        if not np.allclose(kinetic, d['kinetic_eV'], rtol=2e-6, atol=1e-7):
            raise ValueError('Velocity units/masses disagree with saved kinetic energies')
    return float(spacing[0])


def velocity_axis(d):
    """Use full-rate velocity/energy frames; atomic xPT does not use positions.

    Detailed geometric diagnostics remain on the producer's separate time_fs
    axis. A constant first saved geometry fills FrameData's unused position
    member; it is never represented as a measured full-rate trajectory.
    """
    if 'velocity_time_fs' not in d: return d
    required=['velocity_energy_total_eV','velocity_kinetic_eV','velocity_temperature_K']
    for key in required:
        if key not in d: raise ValueError(f'Full-rate velocity axis missing {key}')
    frames=len(d['velocity_time_fs']); adapted=dict(d)
    adapted['time_fs']=d['velocity_time_fs']
    adapted['energy_total_eV']=d['velocity_energy_total_eV']
    adapted['kinetic_eV']=d['velocity_kinetic_eV']
    adapted['temperature_K']=d['velocity_temperature_K']
    adapted['positions_A']=np.broadcast_to(d['positions_A'][0],(frames,len(d['masses_amu']),3))
    return adapted


def harmonic(frequencies_THz, temperature_K, minimum_energy_eV):
    """Finite-cell harmonic partition over exactly the non-translational modes.

    Classical phase-space convention includes h, matching xPT's high-T QHO
    weights. Imaginary or zero internal modes invalidate this reference; they
    are never abs-valued, floored, or discarded here.
    """
    nu = np.asarray(frequencies_THz)
    if not np.isfinite(nu).all() or np.any(nu <= 0):
        return dict(status='UNSTABLE_OR_UNRESOLVED_INTERNAL_MODES',
                    nonpositive_modes=int(np.sum(nu <= 0)))
    hv = H_THZ * nu; u = hv/(KB*temperature_K)
    log_one_minus = np.log(-np.expm1(-u))
    occupation = 1/np.expm1(u)
    sq = KB*np.sum(u*occupation-log_one_minus)
    sc = KB*np.sum(1-np.log(u))
    return dict(status='CALCULATED', modes=len(nu), temperature_K=temperature_K,
                F_classical_eV=minimum_energy_eV+KB*temperature_K*np.log(u).sum(),
                F_quantum_eV=minimum_energy_eV+np.sum(.5*hv+KB*temperature_K*log_one_minus),
                S_classical_eV_K=sc, S_quantum_eV_K=sq,
                U_quantum_eV=minimum_energy_eV+np.sum(.5*hv+hv*occupation),
                Cv_classical_eV_K=len(nu)*KB,
                Cv_quantum_eV_K=KB*np.sum((.5*u/np.sinh(.5*u))**2),
                zero_point_eV=float(.5*hv.sum()))


def run_package(d, indices, output, *, mode='3PT', corlen=.25,
                apodization='none', gas_gate='debye_warn', target_K=433.15):
    """Supply real ATLAS System/FrameData objects, then call its complete engine."""
    from pyxpt.config import Config
    from pyxpt.config.base import validate
    from pyxpt.constants import ModexPT, R as ATLAS_R
    from pyxpt.io.trajectory import System, AtomInfo, MolInfo, GroupInfo, FrameData
    from pyxpt.thermo.engine import xPTEngine

    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    masses = d['masses_amu']; cell = d['cell_A']; count = len(masses)
    mean_energy = float(d['energy_total_eV'][indices].mean())
    cfg = Config(trajectory=['NPZ_ARRAY_API'], trajectory_format='NPZ_ARRAY_API',
                 topology='NPZ masses_amu + explicit atomic topology adapter',
                 molecular=False, dimension=3, constraints=0,
                 mode=ModexPT(mode), refinement='none' if mode=='3PT' else 'rigorous',
                 cage_entropy=mode=='3PT', cage_entropy_rot=False,
                 cage_ws='consistent', cage_quadrature='trapezoid',
                 cage_nf_run=3, gas_gate=gas_gate, gas_gate_tol=.01,
                 corlen=corlen, apodization=apodization,
                 subtract_com_velocity=True, finite_size_correction='off',
                 normalize=False, show_classical_thermo=True, show_xpt_split=True,
                 prefix=str((output/'atlas').resolve()), out_units='ev',
                 temperature=target_K, volume=float(np.linalg.det(cell)),
                 energy_avg=mean_energy/EV_PER_KCAL_MOL, energy_std=0.,
                 energy_units='kcal/mol', vel_scale=1., use_gpu=False)
    cfg._temperature_provided=True; cfg._volume_provided=True
    cfg._energy_avg_provided=True; cfg._energy_units_provided=True
    cfg._vel_scale_provided=True; cfg._thermo_section_seen=True
    validate(cfg)
    system = System(); system.natom=count; system.nmol=count; system.ngrp=1
    system.periodic=True; system.total_mass=float(masses.sum()); system._cfg=cfg
    system.atoms=[AtomInfo(i,float(m),i,0,type_label=str(int(d['numbers'][i])) if 'numbers' in d else '') for i,m in enumerate(masses)]
    system.mols=[MolInfo(i,[i],float(m)) for i,m in enumerate(masses)]
    system.groups=[GroupInfo(0,list(range(count)),list(range(count)),float(masses.sum()),constraint=0)]
    def frames():
        for i in indices:
            yield FrameData(float(d['time_fs'][i])/1000.,d['positions_A'][i],
                            d['velocities_A_ps'][i],np.zeros((count,3)),
                            np.empty((0,6)),cell,temperature=float(d.get('temperature_K',np.full(len(d['time_fs']),target_K))[i]),
                            pressure=0.,volume=float(np.linalg.det(cell)),
                            total_energy=float(d['energy_total_eV'][i])/EV_PER_KCAL_MOL)
    log_handler=logging.FileHandler(output/'package.log'); log_handler.setFormatter(logging.Formatter('%(levelname)s %(name)s: %(message)s'))
    logger=logging.getLogger('pyxpt'); logger.addHandler(log_handler); old_level=logger.level; logger.setLevel(logging.INFO)
    started=time.monotonic()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            engine=xPTEngine(cfg,system); engine._setup(len(indices)); engine.accumulate(frames())
            result=engine.compute(); engine.write(result)
        numpy_warnings=[str(w.message) for w in caught]
    finally:
        logger.removeHandler(log_handler); log_handler.close(); logger.setLevel(old_level)
    write_json(output/'configuration.json',asdict(cfg))
    temperature=float(result.temperature[0,0]); s_q=float(result.S_quantum[0,0]); s_c=float(result.S_classical[0,0])
    if not np.isfinite([temperature,s_q,s_c,result.A_quantum[0,0]]).all() or temperature<=0:
        raise ValueError('Non-finite native thermodynamics')
    trace=np.trace(result.vac[0,0],axis1=-2,axis2=-1); normalized=trace/trace[0]
    frequency=result.frequencies; dos=result.dos[0,0]; spacing=float(frequency[1]-frequency[0])
    expected_modes=3*count-3
    direct_dof=float(engine._compute_dof()[0,0]-engine._translation_rot_dof())
    if direct_dof != expected_modes: raise ValueError('COM mode accounting differs from finite harmonic convention')
    integrated_modes=float(np.trapezoid(dos,frequency)); negative_area=float(np.trapezoid(np.maximum(-dos,0),frequency))
    dt_ps=float(engine._vacdtime)
    omega2=float(np.trapezoid(dos*(2*np.pi*frequency*CM_TO_THZ)**2,frequency)/integrated_modes)
    omega=float(np.sqrt(max(omega2,0)))
    com=(d['velocities_A_ps'][indices]*masses[None,:,None]).sum(1)/masses.sum()
    centered=d['velocities_A_ps'][indices]-com[:,None,:]
    kinetic_mean=float(.5*KINETIC_EV_PER_AMU_A2_PS2*np.sum(centered*centered*masses[None,:,None],axis=(1,2)).mean())
    temperature_modern=2*kinetic_mean/(KB*expected_modes)
    producer_temperature=float(d['temperature_K'][indices].mean()) if 'temperature_K' in d else None
    native_temperature_identity=float(trace[0]/(.1*ATLAS_R*expected_modes))
    native=dict(mode=mode,frames=len(indices),duration_ps=float((d['time_fs'][indices[-1]]-d['time_fs'][indices[0]])/1000.),
                first_time_fs=float(d['time_fs'][indices[0]]),last_time_fs=float(d['time_fs'][indices[-1]]),
                saved_interval_fs=dt_ps*1000.,corlen_fraction=corlen,maximum_lag_ps=(engine._vacmaxf-1)*dt_ps,
                apodization=apodization,gas_gate=gas_gate,observed_temperature_K=temperature,target_temperature_K=target_K,
                physical_removed_translation_dof=3,package_additional_constraints=0,expected_modes=expected_modes,
                integrated_modes=integrated_modes,mode_integral_error=integrated_modes-expected_modes,
                negative_dos_area_modes=negative_area,negative_dos_fraction_of_modes=negative_area/expected_modes,
                com_speed_rms_A_ps=float(np.sqrt(np.mean(np.sum(com*com,axis=1)))),
                fluidicity=float(result.fluidicity[0,0]),packing_fraction=float(result.packing_fraction[0]),
                S_cage_quantum_eV_K=float(result.S_cage[0,0]),S_quantum_eV_K=s_q,S_classical_eV_K=s_c,
                native_E_md_eV=float(result.E_md[0,0]),input_MD_total_mean_eV=mean_energy,
                native_E_quantum_eV=float(result.E_quantum[0,0]),native_E_classical_DOS_eV=float(result.E_classical[0,0]),
                native_A_quantum_eV=float(result.A_quantum[0,0]),native_A_classical_DOS_eV=float(result.A_classical[0,0]),
                F_classical_MD_anchored_eV=mean_energy-temperature*s_c,
                F_quantum_MD_anchored_eV=float(result.A_quantum[0,0]),
                quantum_thermodynamic_identity_error_eV=float(result.A_quantum[0,0]-result.E_quantum[0,0]+temperature*s_q),
                energy_anchor_roundtrip_error_eV=float(result.E_md[0,0])-mean_energy,
                Einstein_Omega0_per_ps=omega,saved_dt_times_Omega0=dt_ps*omega,
                package_cage_resolution_recommendation=.066,
                cage_resolution_condition_met=dt_ps*omega<=.066,
                VAC_tail_last_quarter_rms=float(np.sqrt(np.mean(normalized[-max(1,len(normalized)//4):]**2))),
                numpy_warnings=numpy_warnings,elapsed_seconds=time.monotonic()-started,
                package_warnings=list(getattr(engine,'_captured_warnings',[])),
                native_prefix=str(output/'atlas'))
    native.update(producer_mean_temperature_K=producer_temperature,
                  independent_COM_kinetic_temperature_K=temperature_modern,
                  native_over_independent_temperature_ratio=temperature/temperature_modern,
                  native_minus_independent_temperature_K=temperature-temperature_modern,
                  native_over_producer_temperature_ratio=temperature/producer_temperature if producer_temperature else None,
                  native_minus_producer_temperature_K=temperature-producer_temperature if producer_temperature else None,
                  native_temperature_from_VAC0_internal_DOF_K=native_temperature_identity,
                  native_VAC_temperature_identity_error_K=temperature-native_temperature_identity,
                  independent_temperature_if_all_3N_DOF_K=temperature_modern*expected_modes/(3*count),
                  ATLAS_R_J_mol_K=ATLAS_R)
    native['F_classical_first_order_target_T_eV']=native['F_classical_MD_anchored_eV']-s_c*(target_K-temperature)
    native['F_quantum_first_order_target_T_eV']=native['F_quantum_MD_anchored_eV']-s_q*(target_K-temperature)
    native['temperature_transport_delta_K']=target_K-temperature
    gate_ratio=float(result.gas_gate_plateau_ratio[0,0])
    native['gas_gate_plateau_ratio']=gate_ratio if np.isfinite(gate_ratio) else None
    native['gas_gate_would_fire']=bool(result.gas_gate_fired[0,0])
    native['gas_entropy_component_eV_K']=float(result.gas_entropy_component[0,0])
    np.savez_compressed(output/'spectrum.npz',frequency_cm=frequency,frequency_THz=frequency*CM_TO_THZ,
                        dos_per_cm_cell=dos,dos_gas_per_cm_cell=result.dos_gas[0,0],
                        dos_cage_per_cm_cell=result.dos_cage[0,0],dos_solid_per_cm_cell=result.dos_solid[0,0],
                        lag_ps=np.arange(len(trace))*dt_ps,mass_weighted_VACF=trace,normalized_VACF=normalized)
    write_json(output/'result.json',native)
    return native


def minimum_energy(d):
    for key in ['energy_eV','relaxed_energy_eV','minimum_energy_eV','potential_eV']:
        if key in d and np.asarray(d[key]).size==1: return float(np.asarray(d[key]).item())
    raise ValueError('Minimum NPZ needs scalar energy_eV (or relaxed_energy_eV)')


def bound_hessian(hessian_path,minimum_path):
    h=load_npz(hessian_path); minimum=load_npz(minimum_path)
    if 'input_sha256' in h and str(h['input_sha256'])!=sha(minimum_path):
        raise ValueError('Hessian input hash does not identify the supplied minimum')
    for key in ['cell_A','positions_A','numbers','masses_amu','parent_indices']:
        if key in h and key in minimum and not np.allclose(h[key],minimum[key],rtol=0,atol=1e-10):
            raise ValueError(f'Hessian and minimum differ in {key}')
    return h,minimum


def producer_receipt(path):
    p=Path(path).with_name('receipt.json')
    if not p.exists(): return None
    value=json.loads(p.read_text())
    teacher=value.get('teacher')
    identity=None if not teacher else {k:teacher.get(k) for k in ['model','checkpoint_name','head','checkpoint_hashes','dtype','tf32','nitro','compile','dispersion']}
    return dict(path=str(p),sha256=sha(p),status=value.get('status'),teacher_identity=identity)


def harmonic_reference(args):
    evidence=audit_vendor(args.vendor)
    hessians=named_paths(args.hessian); minima=named_paths(args.minimum)
    args.output_dir.mkdir(parents=True,exist_ok=False); (args.output_dir/'analysis_source.py').write_bytes(SOURCE_BYTES); rows=[]; sources={}; cells=[]
    for name,path in hessians.items():
        h,minimum=bound_hessian(path,minima[name]); frequencies=h['frequencies_signed_THz']
        n=len(minimum['masses_amu']); expected=3*n-3
        if len(frequencies)!=expected: raise ValueError('Expected3N−3 finite-cell internal modes')
        cells.append(minimum['cell_A'])
        if not np.allclose(cells[-1],cells[0],rtol=0,atol=1e-10): raise ValueError('Different reference cells')
        energy=minimum_energy(minimum); row=harmonic(frequencies,args.target_temperature_K,energy)
        row.update(basin_id=name,minimum_energy_eV=energy,minimum_frequency_THz=float(frequencies.min()),maximum_frequency_THz=float(frequencies.max()),expected_modes=expected,delta_A=float(h['delta_A']) if 'delta_A' in h else None)
        rows.append(row); sources[name]=dict(hessian=str(path),hessian_sha256=sha(path),minimum=str(minima[name]),minimum_sha256=sha(minima[name]),minimum_producer=producer_receipt(minima[name]))
    save_rows(args.output_dir/'harmonic_reference.csv',rows)
    write_json(args.output_dir/'receipt.json',dict(status='CALCULATED_FINITE_CELL_REFERENCE',temperature_K=args.target_temperature_K,modes='Exactly 3N−3 projected internal modes',script_sha256=SOURCE_SHA256,source=evidence,sources=sources,results=rows,limitations=['Single common cell and finite Gamma sampling','Harmonic force/mode convergence is independently audited; no negative frequency is abs-valued','Same MLIP reference, not DFT or thermodynamic integration']))
    print(json.dumps(dict(status='CALCULATED_FINITE_CELL_REFERENCE',rows=len(rows),output=str(args.output_dir))))


def attach_harmonic(row, frequencies, energy, target):
    row['minimum_energy_eV']=energy
    observed=harmonic(frequencies,row['observed_temperature_K'],energy)
    common=harmonic(frequencies,target,energy)
    if observed['status']!='CALCULATED':
        row['harmonic_status']=observed['status']; return
    row['harmonic_status']='CALCULATED'
    for treatment in ['classical','quantum']:
        row[f'harmonic_F_{treatment}_matched_T_eV']=observed[f'F_{treatment}_eV']
        row[f'harmonic_F_{treatment}_target_T_eV']=common[f'F_{treatment}_eV']
        row[f'xPT_minus_harmonic_{treatment}_matched_T_eV']=row[f'F_{treatment}_MD_anchored_eV']-observed[f'F_{treatment}_eV']
        row[f'temperature_transport_harmonic_second_order_scale_{treatment}_eV']=.5*observed[f'Cv_{treatment}_eV_K']/row['observed_temperature_K']*row['temperature_transport_delta_K']**2


def save_rows(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def summarize(main, sensitivity, differences, output):
    """Descriptive repeat/block spreads only; no invented confidence intervals."""
    groups={}
    for row in main: groups.setdefault((row['basin_id'],row['mode']),[]).append(row)
    rows=[]
    for (basin,mode),members in groups.items():
        summary=dict(basin_id=basin,mode=mode,repeats=len(members))
        fields=['observed_temperature_K','F_classical_first_order_target_T_eV',
                'F_quantum_first_order_target_T_eV','temperature_transport_delta_K',
                'fluidicity','S_cage_quantum_eV_K','integrated_modes',
                'negative_dos_fraction_of_modes','saved_dt_times_Omega0']
        for key in fields:
            values=np.array([r[key] for r in members]); summary[key+'_mean']=float(values.mean())
            summary[key+'_sample_sd']=float(values.std(ddof=1)) if len(values)>1 else ''
            summary[key+'_min']=float(values.min()); summary[key+'_max']=float(values.max())
        rows.append(summary)
    save_rows(output/'repeat_summary.csv',rows)
    difference_groups={}
    for row in differences: difference_groups.setdefault((row['basin_id'],row['mode']),[]).append(row)
    rows=[]
    for (basin,mode),members in difference_groups.items():
        summary=dict(basin_id=basin,reference_basin_id=members[0]['reference_basin_id'],mode=mode,repeats=len(members),delta_static_eV=members[0]['delta_static_eV'])
        for treatment in ['classical','quantum']:
            key=f'delta_xPT_first_order_target_{treatment}_eV'; values=np.array([r[key] for r in members])
            summary[key+'_mean']=float(values.mean()); summary[key+'_sample_sd']=float(values.std(ddof=1)) if len(values)>1 else ''
            summary[key+'_min']=float(values.min()); summary[key+'_max']=float(values.max())
            summary[key+'_changes_sign_across_repeats']=bool(values.min()<0<values.max())
            harmonic_key=f'delta_harmonic_target_{treatment}_eV'
            if harmonic_key in members[0]: summary[harmonic_key]=members[0][harmonic_key]
        rows.append(summary)
    if rows: save_rows(output/'difference_summary.csv',rows)
    comparison=[]
    for row in sensitivity:
        baseline=next((r for r in main if r['basin_id']==row['basin_id'] and r['repeat_id']==row['repeat_id'] and r['mode']==row['mode']),None)
        if baseline is None: continue
        entry=dict(basin_id=row['basin_id'],repeat_id=row['repeat_id'],mode=row['mode'],scope=row['scope'],frames=row['frames'],observed_temperature_K=row['observed_temperature_K'],saved_interval_fs=row['saved_interval_fs'],maximum_lag_ps=row['maximum_lag_ps'],fluidicity=row['fluidicity'],negative_dos_fraction_of_modes=row['negative_dos_fraction_of_modes'])
        for treatment in ['classical','quantum']:
            key=f'F_{treatment}_first_order_target_T_eV'; entry[f'F_shift_from_full_same_mode_{treatment}_eV']=row[key]-baseline[key]
            matched=next((r for r in sensitivity if r['basin_id']==row['basin_id'] and r['repeat_id']==row['repeat_id'] and r['scope']=='full_matched_block_lag' and r['mode']==row['mode']),None)
            if 'blocks_' in row['scope'] and matched is not None:
                entry[f'F_shift_from_matched_lag_full_same_mode_{treatment}_eV']=row[key]-matched[key]
        comparison.append(entry)
    if comparison: save_rows(output/'sensitivity_shifts.csv',comparison)


def plot_results(main,sensitivity,differences,output,hessians):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    basins=sorted({r['basin_id'] for r in main}); modes=['1PT','2PT','3PT']
    if differences:
        figure,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
        other=[b for b in basins if b!=basins[0]]
        for ax,treatment in zip(axes,['classical','quantum']):
            for j,mode in enumerate(modes):
                for i,basin in enumerate(other):
                    selected=[r for r in differences if r['basin_id']==basin and r['mode']==mode]
                    if not selected: continue
                    values=[r[f'delta_xPT_first_order_target_{treatment}_eV'] for r in selected]
                    x=i+(j-1)*.18; ax.scatter(np.full(len(values),x),values,color=f'C{j}',s=28,label=mode if i==0 else None)
                    ax.plot([x-.045,x+.045],[np.mean(values)]*2,color=f'C{j}',linewidth=2)
            for i,basin in enumerate(other):
                selected=next(r for r in differences if r['basin_id']==basin)
                ax.scatter(i,selected['delta_static_eV'],marker='x',s=50,color='black',label='Static minimum' if i==0 else None)
                harmonic_key=f'delta_harmonic_target_{treatment}_eV'
                if harmonic_key in selected: ax.scatter(i+.33,selected[harmonic_key],marker='D',s=32,color='gray',label='Harmonic at target T' if i==0 else None)
            ax.axhline(0,color='gray',linewidth=.7); ax.set_xticks(range(len(other)),other)
            ax.set_title(treatment.capitalize()); ax.grid(axis='y',alpha=.2)
        axes[0].set_ylabel(f'Free-energy difference from {basins[0]} (eV/cell)')
        axes[0].legend(fontsize=8); figure.suptitle('Same-MLIP diagnostic; xPT transported to target T at first order\nDots are declared trajectory repeats; no thermodynamic-integration reference',fontsize=10)
        figure.tight_layout(rect=[0,0,1,.88]); figure.savefig(output/'free_energy_differences.pdf'); figure.savefig(output/'free_energy_differences.png',dpi=180); plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(10,4.2))
    for basin in basins:
        selected=next((r for r in main if r['basin_id']==basin and r['mode']=='3PT'),next(r for r in main if r['basin_id']==basin))
        spectrum=load_npz(Path(selected['native_prefix']).parent/'spectrum.npz')
        axes[0].plot(spectrum['frequency_THz'],spectrum['dos_per_cm_cell']/CM_TO_THZ,label=basin,linewidth=.7)
        axes[1].plot(spectrum['lag_ps'],spectrum['normalized_VACF'],label=basin,linewidth=.7)
    max_nu=max(float(np.max(load_npz(path)['frequencies_signed_THz'])) for path in hessians.values())
    axes[0].set_xlim(0,max_nu*1.2); axes[0].set_xlabel('Frequency (THz)'); axes[0].set_ylabel('Modes per THz per cell')
    axes[1].set_xlim(0,min(2.,max(float(r['maximum_lag_ps']) for r in main))); axes[1].set_xlabel('Lag time (ps)'); axes[1].set_ylabel('Mass-weighted VACF / VACF(0)')
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2)
    figure.suptitle('One declared NVE repeat per basin; native ATLAS spectra, unmodified',fontsize=10); figure.tight_layout(rect=[0,0,1,.94]); figure.savefig(output/'spectral_diagnostics.pdf'); figure.savefig(output/'spectral_diagnostics.png',dpi=180); plt.close(figure)
    if sensitivity:
        figure,ax=plt.subplots(figsize=(11,4.5)); tags=list(dict.fromkeys(r['scope'] for r in sensitivity if r['mode']=='3PT'))
        for i,basin in enumerate(basins):
            base=next((r for r in main if r['basin_id']==basin and r['mode']=='3PT'),None)
            if base is None: continue
            rows=[r for r in sensitivity if r['basin_id']==basin and r['mode']=='3PT']
            ax.scatter([tags.index(r['scope'])+(i-(len(basins)-1)/2)*.15 for r in rows],
                       [r['F_quantum_first_order_target_T_eV']-base['F_quantum_first_order_target_T_eV'] for r in rows],label=basin,s=25)
        ax.axhline(0,color='gray',linewidth=.8); ax.set_xticks(range(len(tags)),tags,rotation=45,ha='right',fontsize=8)
        ax.set_ylabel('Quantum F shift from full-repeat 3PT (eV/cell)'); ax.legend(fontsize=8); ax.grid(axis='y',alpha=.2)
        ax.set_title('Correlated block / frame / lag / window diagnostics; first-order target-T transport',fontsize=10)
        figure.tight_layout(); figure.savefig(output/'sensitivity_diagnostics.pdf'); figure.savefig(output/'sensitivity_diagnostics.png',dpi=180); plt.close(figure)


def analyze(args):
    evidence=audit_vendor(args.vendor); trajectories=named_paths(args.trajectory)
    hessians=named_paths(args.hessian); minima=named_paths(args.minimum)
    output=args.output_dir; output.mkdir(parents=True,exist_ok=False)
    (output/'analysis_source.py').write_bytes(SOURCE_BYTES)
    raw_output=args.raw_output_dir or output/'native'
    main=[]; sensitivity=[]; sources={}; cells=[]
    sensitivity_basins=set()
    for name,path in trajectories.items():
        basin,repeat=name.split(':',1); d=velocity_axis(load_npz(path)); validate_trajectory(d)
        h,minimum=bound_hessian(hessians[basin],minima[basin]); frequencies=h['frequencies_signed_THz']
        energy=minimum_energy(minimum)
        if len(frequencies)!=3*len(d['masses_amu'])-3: raise ValueError('Harmonic and MD mode counts differ')
        if not np.allclose(d['cell_A'],minimum['cell_A'],rtol=0,atol=1e-10): raise ValueError('MD/minimum cells differ')
        for key in ['numbers','masses_amu','parent_indices']:
            if key in d and key in minimum and not np.array_equal(d[key],minimum[key]): raise ValueError(f'MD/minimum differ in{key}')
        cells.append(d['cell_A'])
        if any(not np.allclose(cell,cells[0],rtol=0,atol=1e-10) for cell in cells): raise ValueError('Competing basins do not share one cell')
        md_producer=producer_receipt(path); minimum_producer=producer_receipt(minima[basin])
        if md_producer and minimum_producer and md_producer['teacher_identity']!=minimum_producer['teacher_identity']:
            raise ValueError('MD/minimum producer teachers differ')
        sources[name]=dict(trajectory=str(path),trajectory_sha256=sha(path),hessian=str(hessians[basin]),hessian_sha256=sha(hessians[basin]),minimum=str(minima[basin]),minimum_sha256=sha(minima[basin]),MD_producer=md_producer,minimum_producer=minimum_producer,teacher_identity_verified=bool(md_producer and minimum_producer and md_producer['teacher_identity']))
        index=np.arange(len(d['time_fs'])); safe_name=name.replace(':','_')
        if args.max_duration_ps is not None:
            if args.max_duration_ps<=0: raise ValueError('Prefix duration must be positive')
            index=index[d['time_fs']<=d['time_fs'][0]+1000*args.max_duration_ps]
        if len(index)<32: raise ValueError('Analysis window needs at least32 frames')
        for mode in args.modes:
            row=run_package(d,index,raw_output/safe_name/mode,mode=mode,corlen=args.corlen,target_K=args.target_temperature_K)
            row.update(basin_id=basin,repeat_id=repeat,scope=args.scope); attach_harmonic(row,frequencies,energy,args.target_temperature_K); main.append(row)
        if args.sensitivity and basin not in sensitivity_basins:
            sensitivity_basins.add(basin); specs=[]
            duration_ps=float((d['time_fs'][index[-1]]-d['time_fs'][index[0]])/1000.)
            block_lag_ps=min(args.corlen*duration_ps,duration_ps/16.)
            for block_mode in ['1PT','3PT']:
                specs.append(('full_matched_block_lag',index,block_lag_ps/duration_ps,'none','debye_warn',block_mode))
            for blocks in [2,4]:
                for j,indices in enumerate(np.array_split(index,blocks)):
                    block_duration=float((d['time_fs'][indices[-1]]-d['time_fs'][indices[0]])/1000.)
                    for block_mode in ['1PT','3PT']:
                        specs.append((f'{blocks}blocks_{j}',indices,block_lag_ps/block_duration,'none','debye_warn',block_mode))
            for stride in [2,4]: specs.append((f'stride_{stride}',index[::stride],args.corlen,'none','debye_warn','3PT'))
            for corlen in [.1,.5]: specs.append((f'corlen_{corlen}',index,corlen,'none','debye_warn','3PT'))
            specs.append(('hann',index,args.corlen,'hann','debye_warn','3PT'))
            specs.append(('debye_gate',index,args.corlen,'none','debye','3PT'))
            for tag,indices,corlen,apodization,gate,mode in specs:
                row=run_package(d,indices,raw_output/safe_name/(tag+'_'+mode),mode=mode,corlen=corlen,apodization=apodization,gas_gate=gate,target_K=args.target_temperature_K)
                row.update(basin_id=basin,repeat_id=repeat,scope=tag); attach_harmonic(row,frequencies,energy,args.target_temperature_K); sensitivity.append(row)
    save_rows(output/'native_results.csv',main)
    if sensitivity: save_rows(output/'sensitivity_results.csv',sensitivity)
    # Pair differences remain repeat-resolved. Cross-repeat spread is descriptive;
    # a handful of independent starts does not provide a calibrated CI.
    differences=[]; basins=sorted({r['basin_id'] for r in main}); reference=basins[0]
    lookup={(r['basin_id'],r['repeat_id'],r['mode']):r for r in main}
    for row in main:
        key=(reference,row['repeat_id'],row['mode'])
        if row['basin_id']==reference or key not in lookup: continue
        ref=lookup[key]; difference=dict(basin_id=row['basin_id'],reference_basin_id=reference,repeat_id=row['repeat_id'],mode=row['mode'],delta_static_eV=row.get('minimum_energy_eV',np.nan)-ref.get('minimum_energy_eV',np.nan),observed_T_basin_K=row['observed_temperature_K'],observed_T_reference_K=ref['observed_temperature_K'])
        for treatment in ['classical','quantum']:
            difference[f'delta_xPT_first_order_target_{treatment}_eV']=row[f'F_{treatment}_first_order_target_T_eV']-ref[f'F_{treatment}_first_order_target_T_eV']
            if row['harmonic_status']==ref['harmonic_status']=='CALCULATED': difference[f'delta_harmonic_target_{treatment}_eV']=row[f'harmonic_F_{treatment}_target_T_eV']-ref[f'harmonic_F_{treatment}_target_T_eV']
        differences.append(difference)
    if differences: save_rows(output/'basin_differences.csv',differences)
    summarize(main,sensitivity,differences,output)
    plot_results(main,sensitivity,differences,output,hessians)
    write_json(output/'receipt.json',dict(status='CALCULATED_DIAGNOSTIC',source=evidence,input_hashes=sources,script_sha256=SOURCE_SHA256,python=platform.python_version(),package_requirements=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines(),main_analyses=len(main),sensitivity_analyses=len(sensitivity),target_temperature_K=args.target_temperature_K,max_duration_ps=args.max_duration_ps,analysis_scope=args.scope,conventions=dict(atomic=True,whole_system=True,per_cell=True,COM_removed_physical_dof=3,additional_package_constraints=0,native_temperature='VAC-derived kinetic temperature',classical_anchor='mean total MD energy - T * native classical entropy',quantum_anchor='native MD energy + quantum-classical DOS correction - T * quantum entropy',common_temperature='explicit first-order -S deltaT transport, not exact canonical reweighting',full_rate_velocity_axis='velocity_time_fs when available; energy anchor from exactly matching velocity_energy_total_eV',unused_FrameData_positions='first detailed saved positions; positions are unused in atomic thermodynamics; no interpolated geometry is claimed'),limitations=['No same-potential thermodynamic integration validates anharmonic free-energy accuracy.','Same-MLIP agreement is not physical DFT validation.','Native package classical A is unanchored and retained separately.','Native NVE total-energy fluctuations are not a canonical heat-capacity estimator.','Negative DOS mass, mode-integral error, warnings and cage resolution are retained.','Native source replaces some non-finite thermodynamic intermediates with zero; raw diagnostics and warnings must be inspected.','Basin labels require the separate occupation/host stability checks; xPT does not establish non-hopping.']))
    print(json.dumps(dict(status='CALCULATED_DIAGNOSTIC',main=len(main),sensitivity=len(sensitivity),output=str(output))))


def selftest(args):
    """Synthetic harmonic velocities, not measurements of LCO or a new MD run."""
    evidence=audit_vendor(args.vendor); rng=np.random.default_rng(731)
    count=8; masses=np.array([6.94,58.933,15.999,15.999]*2); ndof=3*count-3; target=433.15
    translation=np.zeros((3*count,3))
    for axis in range(3): translation[axis::3,axis]=np.sqrt(masses)
    q=np.linalg.qr(translation,mode='complete')[0][:,3:]
    time_ps=np.arange(40000)*.0005; frequencies=np.linspace(2,12,ndof)
    phase=rng.uniform(0,2*np.pi,ndof)
    modal=np.sqrt(2*KB*target/KINETIC_EV_PER_AMU_A2_PS2)*np.cos(2*np.pi*time_ps[:,None]*frequencies+phase)
    velocities=(modal@q.T).reshape(-1,count,3)/np.sqrt(masses)[None,:,None]
    kinetic=.5*KINETIC_EV_PER_AMU_A2_PS2*np.sum(velocities**2*masses[None,:,None],axis=(1,2))
    d=dict(time_fs=time_ps*1000,velocities_A_ps=velocities,positions_A=np.zeros_like(velocities),masses_amu=masses,cell_A=np.eye(3)*10,energy_total_eV=np.full(len(time_ps),-100+ndof*KB*target),kinetic_eV=kinetic,numbers=np.array([3,27,8,8]*2))
    validate_trajectory(d); args.output_dir.mkdir(parents=True,exist_ok=False); (args.output_dir/'analysis_source.py').write_bytes(SOURCE_BYTES)
    split=dict(d,time_fs=d['time_fs'][::4],positions_A=d['positions_A'][::4],energy_total_eV=d['energy_total_eV'][::4],kinetic_eV=kinetic[::4],velocity_time_fs=d['time_fs'],velocity_energy_total_eV=d['energy_total_eV'],velocity_kinetic_eV=kinetic,velocity_temperature_K=np.full(len(kinetic),target))
    validate_trajectory(velocity_axis(split))
    rows=[]
    for mode in ['1PT','2PT','3PT']:
        row=run_package(d,np.arange(len(time_ps)),args.output_dir/mode,mode=mode,target_K=target)
        assert row['expected_modes']==ndof
        assert abs(row['energy_anchor_roundtrip_error_eV'])<1e-10
        assert abs(row['quantum_thermodynamic_identity_error_eV'])<1e-10
        assert abs(row['observed_temperature_K']-target)<.01
        attach_harmonic(row,frequencies,-100,target); rows.append(row)
    if not np.isclose(rows[1]['F_quantum_MD_anchored_eV'],rows[2]['F_quantum_MD_anchored_eV'],atol=1e-8):
        raise AssertionError('Synthetic nondiffusive 2PT and 3PT differ unexpectedly')
    assert harmonic(np.array([1.,-1.]),target,-100)['status']=='UNSTABLE_OR_UNRESOLVED_INTERNAL_MODES'
    save_rows(args.output_dir/'synthetic_harmonic_check.csv',rows)
    write_json(args.output_dir/'receipt.json',dict(status='PASS',synthetic=True,source=evidence,tests=['actual package modes1PT/2PT/3PT execute','3N-3 mode convention','velocity/kinetic unit consistency','MD energy anchor exact round trip','native quantum A=E-TS identity','nondiffusive cage vanishes','imaginary internal modes never abs-valued'],results=rows))
    print(json.dumps(dict(status='PASS',synthetic=True,observed_T=rows[0]['observed_temperature_K'],modes=ndof,integrated_modes=rows[0]['integrated_modes'])))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    for name in ['analyze','selftest','harmonic']:
        p=sub.add_parser(name); p.add_argument('--vendor',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True)
        if name=='harmonic':
            p.add_argument('--hessian',nargs='+',required=True); p.add_argument('--minimum',nargs='+',required=True)
            p.add_argument('--target-temperature-K',dest='target_temperature_K',type=float,default=433.15)
        if name=='analyze':
            p.add_argument('--trajectory',nargs='+',required=True,help='BASIN:REPEAT=production.npz')
            p.add_argument('--hessian',nargs='+',required=True,help='BASIN=hessian.npz')
            p.add_argument('--minimum',nargs='+',required=True,help='BASIN=relaxed.npz')
            p.add_argument('--target-temperature-K',dest='target_temperature_K',type=float,default=433.15)
            p.add_argument('--modes',nargs='+',choices=['1PT','2PT','3PT'],default=['1PT','2PT','3PT'])
            p.add_argument('--corlen',type=float,default=.25)
            p.add_argument('--sensitivity',action='store_true',help='First declared repeat per basin only: blocks,strides,lag,taper,crystal gate')
            p.add_argument('--raw-output-dir',type=Path,help='Native spectra/logs on scratch; small tables/figures remain in output-dir')
            p.add_argument('--max-duration-ps',type=float,help='Analyze this recorded prefix of each completed trajectory')
            p.add_argument('--scope',default='full_trajectory',help='Explicit experiment/window label, e.g.matched5ps_NVT')
    args=parser.parse_args(argv)
    if os.environ.get('NERSC_HOST') and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('On NERSC, run this analysis inside an authorized compute allocation')
    if args.command=='analyze': analyze(args)
    elif args.command=='harmonic': harmonic_reference(args)
    else: selftest(args)


if __name__=='__main__': main()

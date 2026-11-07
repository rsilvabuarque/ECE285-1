#!/usr/bin/env python3
"""Compare two declared temperature transports; perform no MLIP or MD calls.

Direct transport: F_x(T0) ~= F_x(T) - S_x(T) (T0-T).
Harmonic-reference transport: F_h(T0) + R(T) - [S_x(T)-S_h(T)] (T0-T),
where R(T)=F_x(T)-F_h(T). The latter keeps known harmonic curvature exactly;
unknown curvature of the residual is neither measured nor bounded here.
All F are eV/cell, S eV/(cell K), and temperatures K. The original xPT wrapper
is hash-checked and remains unchanged. These transports are sensitivity choices,
not competing estimates from which a favorable basin ordering may be selected.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

WRAPPER_SHA='1019fb739af11cf107dae2967fd39fd1ce3dbb00d7789661d4539a220707a894'
SOURCE=Path(__file__).read_bytes()


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open() as f: return list(csv.DictReader(f))


def write_csv(path,rows):
    if not rows: return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def transport(F_x,S_x,F_h_obs,S_h_obs,F_h_target,T_obs,T_target):
    values=np.array([F_x,S_x,F_h_obs,S_h_obs,F_h_target,T_obs,T_target],float)
    if not np.isfinite(values).all() or min(T_obs,T_target)<=0:
        raise ValueError('Need finite energies/entropies and positive temperatures')
    delta_T=T_target-T_obs
    direct=F_x-S_x*delta_T
    residual=F_x-F_h_obs
    reference=F_h_target+residual-(S_x-S_h_obs)*delta_T
    known_harmonic_change=F_h_target-F_h_obs+S_h_obs*delta_T
    if not np.isclose(reference-direct,known_harmonic_change,rtol=0,atol=1e-10):
        raise AssertionError('Temperature-transport difference identity failed')
    return dict(observed_temperature_K=T_obs,target_temperature_K=T_target,
                delta_temperature_K=delta_T,F_x_observed_eV=F_x,S_x_observed_eV_K=S_x,
                F_harmonic_observed_eV=F_h_obs,S_harmonic_observed_eV_K=S_h_obs,
                F_harmonic_target_eV=F_h_target,residual_observed_eV=residual,
                residual_entropy_observed_eV_K=S_x-S_h_obs,
                F_direct_first_order_eV=direct,F_harmonic_reference_transport_eV=reference,
                reference_minus_direct_eV=reference-direct,
                exact_known_harmonic_curvature_change_eV=known_harmonic_change)


def load_references(args):
    if sha(args.wrapper)!=WRAPPER_SHA: raise ValueError('Frozen xPT wrapper hash changed')
    spec=importlib.util.spec_from_file_location('frozen_thermal_xpt',args.wrapper)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    hessians=module.named_paths(args.hessian); minima=module.named_paths(args.minimum)
    references={}; hashes={'wrapper':WRAPPER_SHA}
    for basin,path in hessians.items():
        h,minimum=module.bound_hessian(path,minima[basin])
        frequencies=h['frequencies_signed_THz']; energy=module.minimum_energy(minimum)
        if len(frequencies)!=3*len(minimum['masses_amu'])-3 or np.any(frequencies<=0):
            raise ValueError('Reference needs all3N−3 positive internal modes')
        references[basin]=(frequencies,energy)
        hashes[basin]=dict(hessian=str(path),hessian_sha256=sha(path),minimum=str(minima[basin]),minimum_sha256=sha(minima[basin]))
    return module,references,hashes


def selftest(args,module,references):
    rows=[]
    for basin,(frequencies,energy) in references.items():
        target=module.harmonic(frequencies,args.target_temperature_K,energy)
        for T in [390.,470.]:
            observed=module.harmonic(frequencies,T,energy)
            for treatment in ['classical','quantum']:
                Fh=observed[f'F_{treatment}_eV']; Sh=observed[f'S_{treatment}_eV_K']; F0=target[f'F_{treatment}_eV']
                r=transport(Fh,Sh,Fh,Sh,F0,T,args.target_temperature_K)
                error=r['F_harmonic_reference_transport_eV']-F0
                assert abs(error)<1e-10
                r.update(basin_id=basin,treatment=treatment,test='zero_residual_actual_harmonic_reference',reference_target_error_eV=error,synthetic_residual=True)
                rows.append(r)
                # Synthetic linear residual tests both the entropy sign and
                # reference transport beyond the zero-residual limit.
                a=.025; slope=1e-4  # eV/cell and eV/(cell K), synthetic only
                R=a+slope*(T-args.target_temperature_K)
                linear=transport(Fh+R,Sh-slope,Fh,Sh,F0,T,args.target_temperature_K)
                error=linear['F_harmonic_reference_transport_eV']-(F0+a)
                assert abs(error)<1e-10
                linear.update(basin_id=basin,treatment=treatment,test='synthetic_linear_residual',reference_target_error_eV=error,synthetic_residual=True)
                rows.append(linear)
    write_csv(args.output_dir/'harmonic_limit_tests.csv',rows)
    return rows


def apply(args,module,references):
    rows=[]; source_hashes={}
    for path in args.native_results:
        source_hashes[str(path)]=sha(path)
        for original in read_csv(path):
            basin=original['basin_id']; frequencies,energy=references[basin]
            T=float(original['observed_temperature_K']); observed=module.harmonic(frequencies,T,energy)
            target=module.harmonic(frequencies,args.target_temperature_K,energy)
            for treatment in ['classical','quantum']:
                result=transport(float(original[f'F_{treatment}_MD_anchored_eV']),
                                 float(original[f'S_{treatment}_eV_K']),
                                 observed[f'F_{treatment}_eV'],observed[f'S_{treatment}_eV_K'],
                                 target[f'F_{treatment}_eV'],T,args.target_temperature_K)
                result.update(native_file=str(path),basin_id=basin,repeat_id=original['repeat_id'],
                              mode=original['mode'],scope=original['scope'],treatment=treatment,
                              first_time_fs=original['first_time_fs'],last_time_fs=original['last_time_fs'])
                rows.append(result)
    write_csv(args.output_dir/'transports_per_repeat.csv',rows)
    lookup={(r['native_file'],r['basin_id'],r['repeat_id'],r['mode'],r['scope'],r['treatment']):r for r in rows}
    if len(lookup)!=len(rows): raise ValueError('Ambiguous repeated native result rows')
    differences=[]; reference_basin=sorted(references)[0]
    for r in rows:
        if r['basin_id']==reference_basin: continue
        key=(r['native_file'],reference_basin,r['repeat_id'],r['mode'],r['scope'],r['treatment'])
        if key not in lookup: continue
        ref=lookup[key]
        differences.append(dict(native_file=r['native_file'],basin_id=r['basin_id'],reference_basin_id=reference_basin,repeat_id=r['repeat_id'],mode=r['mode'],scope=r['scope'],treatment=r['treatment'],delta_direct_first_order_eV=r['F_direct_first_order_eV']-ref['F_direct_first_order_eV'],delta_harmonic_reference_transport_eV=r['F_harmonic_reference_transport_eV']-ref['F_harmonic_reference_transport_eV'],transport_change_in_basin_difference_eV=r['reference_minus_direct_eV']-ref['reference_minus_direct_eV']))
    write_csv(args.output_dir/'basin_difference_transport_sensitivity.csv',differences)
    return rows,source_hashes


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=['selftest','apply'])
    p.add_argument('--wrapper',type=Path,default=Path('scripts/overnight_20261107/thermal_xpt.py'))
    p.add_argument('--hessian',nargs='+',required=True); p.add_argument('--minimum',nargs='+',required=True)
    p.add_argument('--target-temperature-K',dest='target_temperature_K',type=float,default=433.15)
    p.add_argument('--native-results',type=Path,nargs='+'); p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(argv)
    if os.environ.get('NERSC_HOST') and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run this analysis inside the authorized compute allocation')
    module,references,hashes=load_references(args)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'analysis_source.py').write_bytes(SOURCE)
    if args.command=='selftest': rows=selftest(args,module,references); inputs={}
    else:
        if not args.native_results: raise ValueError('apply needs complete --native-results CSV files')
        rows,inputs=apply(args,module,references)
    receipt=dict(status='PASS' if args.command=='selftest' else 'CALCULATED_TRANSPORT_SENSITIVITY',command=args.command,rows=len(rows),target_temperature_K=args.target_temperature_K,reference_hashes=hashes,native_csv_hashes=inputs,source_sha256=hashlib.sha256(SOURCE).hexdigest(),units=dict(F='eV/cell',S='eV/(cell K)',T='K'),unknown_residual_curvature='Not measured or bounded; both transports retained, no ordering selected',numerical_test_tolerance_eV=1e-10,scope='Known harmonic curvature retained exactly; only xPT-minus-harmonic residual transported to first order')
    (args.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],rows=len(rows),output=str(args.output_dir))))


if __name__=='__main__': main()

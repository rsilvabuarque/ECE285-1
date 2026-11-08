"""Bounded fixed-endpoint comparison of saved continuous NEB route families.

The same old endpoints remain byte-identical in every image-count comparison.
No endpoint quench, symmetry canonicalization, new event search, or DFT occurs.
"""
import time
PROCESS_START = time.monotonic()  # Includes Python imports and foundation-model loading.

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import sys
import traceback

import ase
from ase.mep import NEB
from ase.optimize import FIRE
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'conditioning_20261107'))
from neb_common import (BudgetExpired, CallLedger, attach_calculators, atoms_from,
    digest, discrete_barriers, export_images, geometry_hash, load_npz, mapping,
    read_json, snapshot, write_json)
from neb_run import resample_band
from cathode_cvae.runtime import require_compute_allocation


def fmax(force):
    return float(np.linalg.norm(np.asarray(force).reshape(-1, 3), axis=1).max())


def prepare_band(data, interior):
    """Resample saved unwrapped Cartesian 3N arclength without changing endpoints."""
    if interior < 1: raise ValueError('At least one interior image is required')
    xyz = np.asarray(data['positions_A'])
    if xyz.ndim != 3 or xyz.shape[1:] != (len(data['numbers']), 3):
        raise ValueError('Malformed saved band')
    images = [atoms_from(data, row) for row in xyz]
    old_hashes = [geometry_hash(images[0]), geometry_hash(images[-1])]
    result = resample_band(images, interior)
    if [geometry_hash(result[0]), geometry_hash(result[-1])] != old_hashes:
        raise ValueError('Route resampling changed an endpoint')
    return result, old_hashes


def request_binding(args, images, endpoint_hashes):
    return dict(root_protocol=dict(path=str(args.root_protocol.resolve()), sha256=digest(args.root_protocol)),
        source_band=dict(path=str(args.band.resolve()), sha256=digest(args.band)),
        parent=dict(path=str(args.parent.resolve()), sha256=digest(args.parent)),
        checkpoint=dict(path=str(args.checkpoint.resolve()), sha256=digest(args.checkpoint)),
        source_sha256=digest(__file__), helpers={name:digest(Path(__file__).resolve().parents[1]/'conditioning_20261107'/name)
            for name in ['neb_common.py', 'neb_run.py']},
        event=args.event, route=args.route, model='uma', interior_images=args.interior,
        endpoint_geometry_sha256=endpoint_hashes,
        target_projected_Fmax_eV_A=args.fmax, maximum_total_optimizer_steps=args.steps,
        maximum_seconds_including_imports_loading=args.max_seconds, maximum_EFS_calls=args.max_calls,
        practical_numerical_span_stability_eV_hop=0.02,
        numerical_target_scope='Provisional numerical stability criterion only; not model or DFT accuracy',
        ASE_version=ase.__version__, ASE_implementation={o.__name__:dict(path=inspect.getsourcefile(o),
            sha256=digest(inspect.getsourcefile(o))) for o in [NEB,FIRE]},
        settings=dict(method='improvedtangent', climb=True, spring_eV_A2=0.1,
            optimizer='FIRE',dt_ASE=0.1,dtmax_ASE=1.,maxstep_A=0.05,
            parallel=False,remove_rotation_translation=False, shared_calculator=False),
        initialization='Piecewise linear full-3N Cartesian arclength resampling of saved continuous route; no new IDPP',
        periodicity='Saved continuously unwrapped positions retained; no independent endpoint MIC or atom reorder',
        no_new_DFT=True, fixed_endpoints=True, fixed_cell=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root-protocol',required=True,type=Path);p.add_argument('--band',required=True,type=Path)
    p.add_argument('--parent',required=True,type=Path);p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--event',required=True);p.add_argument('--route',required=True);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--interior',type=int,default=7);p.add_argument('--fmax',type=float,default=0.02)
    p.add_argument('--steps',type=int,default=500);p.add_argument('--max-seconds',type=float,default=2700)
    p.add_argument('--max-calls',type=int,default=5000);p.add_argument('--device',default='cuda')
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args(argv);require_compute_allocation()
    if not (0<args.fmax<=0.02 and 1<=args.steps<=500 and 0<args.max_seconds<=2700):
        raise ValueError('This bounded lane allows <=500 total steps, <=45 min and <=0.02 eV/A target')
    # Reading the central protocol is deliberate: a missing not-yet-frozen protocol aborts.
    read_json(args.root_protocol)
    data=load_npz(args.band);parent=load_npz(args.parent)
    images, endpoint_hashes=prepare_band(data,args.interior)
    parent_indices=np.asarray(data['parent_indices'],int)
    if not np.array_equal(images[0].numbers,parent['numbers'][parent_indices]):
        raise ValueError('Saved continuous atom identities disagree with parent chemistry')
    args.output.mkdir(parents=True,exist_ok=False)
    binding=request_binding(args,images,endpoint_hashes);write_json(args.output/'protocol.json',binding)
    ledger=CallLedger(args.output/'calculator_calls.jsonl',args.max_seconds,args.max_calls,start=PROCESS_START)
    result=dict(status='PREPARED',binding=binding,started_UTC=datetime.now(timezone.utc).isoformat(),
        energy_units='eV per simulation cell / per marked hop; potential energy, not activation free energy',
        converged=False, scientific_validation='No Hessian, independent saddle verification or DFT validation')
    current_projected=None;exitcode=2
    try:
        ledger.check()
        from cathode_cvae.mlip.calculators import make_calculator
        print('MODEL_LOADING',flush=True)
        inner,teacher=make_calculator('uma',args.checkpoint,args.device,args.threads)
        result['teacher']=teacher;write_json(args.output/'teacher.json',teacher)
        print('MODEL_READY',flush=True);ledger.check()
        attach_calculators(images,inner,ledger,'fixed_route')
        ledger.phase='initial_full_path'
        for im in images: im.get_forces()
        snapshot(images,args.output/'initial_band.npz',parent_indices=parent_indices)
        result['endpoint_site_maps']=[mapping(im,parent,parent_indices) for im in [images[0],images[-1]]]
        neb=NEB(images,k=0.1,method='improvedtangent',climb=True,parallel=False,
            remove_rotation_and_translation=False,allow_shared_calculator=False)
        opt=FIRE(neb,logfile=str(args.output/'optimizer.log'),trajectory=str(args.output/'band.traj'),
            dt=0.1,dtmax=1.0,maxstep=0.05)
        ledger.phase='climbing_band'
        def observe():
            nonlocal current_projected
            current_projected=neb.get_forces().copy()
            if [geometry_hash(images[0]),geometry_hash(images[-1])]!=endpoint_hashes:
                raise ValueError('Fixed endpoints changed during NEB')
            energy=[float(im.calc.results['energy']) for im in images]
            row=dict(step=opt.nsteps,elapsed_seconds=time.monotonic()-PROCESS_START,
                projected_Fmax_eV_A=fmax(current_projected),energy_eV=energy,
                peak_image=int(np.argmax(energy)),climbing_image=int(neb.imax),
                physical_image_Fmax_eV_A=[fmax(im.calc.results['forces']) for im in images],
                actual_EFS_attempts=ledger.attempts)
            with (args.output/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if opt.nsteps%10==0:snapshot(images,args.output/'latest_band.npz',
                projected_forces=current_projected,parent_indices=parent_indices)
        opt.attach(observe,interval=1)
        result['converged']=bool(opt.run(fmax=args.fmax,steps=args.steps))
        current_projected=neb.get_forces().copy()
        final=snapshot(images,args.output/'final_band.npz',projected_forces=current_projected,parent_indices=parent_indices)
        result.update(optimizer_steps=opt.nsteps,projected_Fmax_eV_A=fmax(current_projected),
            physical_image_Fmax_eV_A=[fmax(f) for f in final['forces_eV_A']],
            climbing_image=int(neb.imax), **discrete_barriers(final['energy_eV']))
        result['image_site_maps']=[mapping(im,parent,parent_indices) for im in images]
        result['endpoint_geometry_sha256_after']=[geometry_hash(images[0]),geometry_hash(images[-1])]
        if result['endpoint_geometry_sha256_after']!=endpoint_hashes:raise ValueError('Endpoint hashes changed')
        result['status']='NUMERICALLY_CONVERGED_UNVALIDATED' if result['converged'] else 'STEP_CAP_TARGET_NOT_MET'
        export_images(images,args.output/'geometry_exports',parent_indices)
        exitcode=0 if result['converged'] else 2
    except BudgetExpired as error:
        result.update(status='BOUNDED_BUDGET_TERMINATION',error=str(error));exitcode=3
    except Exception as error:
        result.update(status='FAILED_PRESERVED',error=str(error),exception_type=type(error).__name__,traceback=traceback.format_exc())
    finally:
        # Serialization never requests new labels, including at unknown last-state positions.
        snapshot(images,args.output/'last_state.npz',projected_forces=current_projected,parent_indices=parent_indices)
        result['call_accounting']=ledger.summary();result['completed_UTC']=datetime.now(timezone.utc).isoformat()
        write_json(args.output/'receipt.json',result)
        print(json.dumps(dict(status=result['status'],converged=result['converged'],**ledger.summary())),flush=True)
    return exitcode


if __name__=='__main__':raise SystemExit(main())

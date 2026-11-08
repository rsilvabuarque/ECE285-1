"""Independently tighten the SAME original endpoints under UMA or MACE.

This is a new endpoint experiment, separate from the fixed-original-endpoint
image/tolerance study. It prepares a displacement-tapered band but never launches
a NEB calculation. Site persistence is a prerequisite, not proof of PES identity.
"""
import time
PROCESS_START=time.monotonic()
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import traceback

import numpy as np
from ase.geometry import find_mic
from ase.optimize import FIRE
from cathode_cvae.runtime import require_compute_allocation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'conditioning_20261107'))
from neb_common import (BudgetExpired,CallLedger,attach_calculators,atoms_from,digest,
    export_images,geometry_hash,load_npz,mapping,read_json,snapshot,
    validate_endpoint_event,write_json)


def tapered_band(data,start,end):
    """Keep every old interior image, adding a linear endpoint-displacement taper.

    The taper coordinate is the original full3N Cartesian arclength. This does
    not resample the old path, independently unwrap atoms or apply IDPP.
    """
    xyz=np.asarray(data['positions_A'],float)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0).reshape(len(xyz)-1,-1),axis=1))]
    if arc[-1]<=0:raise ValueError('A collapsed route cannot define the displacement taper')
    s=arc/arc[-1];d0=start.positions-xyz[0];d1=end.positions-xyz[-1]
    moved=xyz+(1-s[:,None,None])*d0+s[:,None,None]*d1
    moved[0]=start.positions;moved[-1]=end.positions
    images=[atoms_from(data,row) for row in moved]
    return images,s


def geometry_change(atom,original):
    host=atom.numbers!=3;difference,_=find_mic(atom.positions-original.positions,atom.cell,atom.pbc)
    shift=difference[host].mean(0);aligned,_=find_mic(atom.positions-original.positions-shift,atom.cell,atom.pbc)
    distance=np.linalg.norm(aligned,axis=1)
    return dict(host_translation_A=shift.tolist(),identity_preserving_host_aligned_RMS_A=float(np.sqrt(np.mean(distance**2))),
        host_RMS_A=float(np.sqrt(np.mean(distance[host]**2))),Li_RMS_A=float(np.sqrt(np.mean(distance[~host]**2))),
        maximum_atom_difference_A=float(distance.max()),scope='Continuous geometry-change measurements, no arbitrary basin-equivalence cutoff')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-band',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--root-protocol',type=Path,required=True)
    p.add_argument('--model',choices=['uma','mace'],required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--migrating-atom',type=int,required=True)
    p.add_argument('--source-site',type=int,required=True);p.add_argument('--destination-site',type=int,required=True)
    p.add_argument('--fmax',type=float,default=.01);p.add_argument('--steps',type=int,default=500)
    p.add_argument('--max-seconds',type=float,default=2700);p.add_argument('--max-calls',type=int,default=5000)
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=4);args=p.parse_args()
    require_compute_allocation();read_json(args.root_protocol)
    if not (0<args.fmax<=.01 and 0<args.steps<=500 and 0<args.max_seconds<=2700 and 0<args.max_calls<=5000):
        raise ValueError('Endpoint study permits at most500steps/endpoint,45min/process,5000actualcalls and.01eV/A force target')
    data=load_npz(args.input_band);parent=load_npz(args.parent);ids=np.asarray(data['parent_indices'],int)
    original=[atoms_from(data,row) for row in data['positions_A'][[0,-1]]]
    original_maps=[mapping(im,parent,ids) for im in original]
    expected=[m['assigned_parent_Li_sites'] for m in original_maps]
    valid,reason=validate_endpoint_event(*original_maps,*expected,args.migrating_atom,args.source_site,args.destination_site)
    if not valid:raise ValueError('Original endpoint marked event is not the declared event: '+reason)
    endpoints=[im.copy() for im in original];args.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(status='FROZEN_BEFORE_ENDPOINT_LABELS',source_sha256=digest(__file__),model=args.model,
        root_protocol=dict(path=str(args.root_protocol.resolve()),sha256=digest(args.root_protocol)),
        input_band=dict(path=str(args.input_band.resolve()),sha256=digest(args.input_band)),
        parent=dict(path=str(args.parent.resolve()),sha256=digest(args.parent)),checkpoint_sha256=digest(args.checkpoint),
        original_endpoint_geometry_sha256=[geometry_hash(im) for im in original],parent_indices=ids.tolist(),
        migrating_atom=args.migrating_atom,source_parent_Li_site=args.source_site,destination_parent_Li_site=args.destination_site,
        fixed_cell=True,target_physical_Fmax_eV_A=args.fmax,maximum_steps_per_endpoint=args.steps,
        maximum_seconds_including_loading=args.max_seconds,maximum_EFS_calls=args.max_calls,
        optimizer=dict(name='FIRE',dt=.1,dtmax=1.,maxstep_A=.05),
        new_experiment='New independently teacher-tightened endpoint identities; not part of fixed-original-endpoint image/tolerance comparison',
        prepared_band_policy='Same image count and continuous atom order; linear endpoint-displacement taper in original full3N arclength; no resampling or IDPP')
    write_json(args.output/'protocol.json',protocol)
    ledger=CallLedger(args.output/'calculator_calls.jsonl',args.max_seconds,args.max_calls,start=PROCESS_START)
    receipt=dict(status='PREPARED',binding=protocol,endpoints=[],band_gate='NOT_ASSESSED');code=2
    try:
        from cathode_cvae.mlip.calculators import make_calculator
        calc,teacher=make_calculator(args.model,args.checkpoint,args.device,args.threads);receipt['teacher']=teacher
        for i,im in enumerate(endpoints):
            ledger.phase=f'endpoint_{i}';attach_calculators([im],calc,ledger,f'endpoint_{i}')
            initial_force=im.get_forces();initial_energy=im.get_potential_energy()
            opt=FIRE(im,logfile=str(args.output/f'endpoint_{i}.log'),trajectory=str(args.output/f'endpoint_{i}.traj'),dt=.1,dtmax=1.,maxstep=.05)
            def observe(index=i,atom=im,optimizer=opt):
                force=atom.get_forces();row=dict(endpoint=index,step=optimizer.nsteps,energy_eV=atom.get_potential_energy(),
                    physical_Fmax_eV_A=float(np.linalg.norm(force,axis=1).max()),EFS_attempts=ledger.attempts)
                with (args.output/'progress.jsonl').open('a') as f:f.write(__import__('json').dumps(row)+'\n')
                if optimizer.nsteps%10==0:snapshot([atom],args.output/f'endpoint_{index}_latest.npz',parent_indices=ids)
            opt.attach(observe,interval=1);converged=bool(opt.run(fmax=args.fmax,steps=args.steps))
            f=im.get_forces();snapshot([im],args.output/f'endpoint_{i}_final.npz',parent_indices=ids)
            receipt['endpoints'].append(dict(index=i,numerically_converged=converged,steps=opt.nsteps,
                physical_Fmax_eV_A=float(np.linalg.norm(f,axis=1).max()),energy_eV=float(im.get_potential_energy()),
                initial_teacher_energy_eV=float(initial_energy),energy_change_eV=float(im.get_potential_energy()-initial_energy),
                initial_teacher_physical_Fmax_eV_A=float(np.linalg.norm(initial_force,axis=1).max()),
                geometry_sha256=geometry_hash(im),site_mapping=mapping(im,parent,ids),
                geometry_change_from_original=geometry_change(im,original[i])))
        new_maps=[r['site_mapping'] for r in receipt['endpoints']]
        retained,reason=validate_endpoint_event(*new_maps,*expected,args.migrating_atom,args.source_site,args.destination_site)
        no_ambiguity=all(m['nearest_collision_count']==0 and m['hungarian_differs_from_nearest']==0 for m in new_maps)
        stationary=all(r['numerically_converged'] and r['physical_Fmax_eV_A']<=args.fmax for r in receipt['endpoints'])
        receipt.update(status='COMPLETE_ENDPOINT_STUDY',marked_event_retained=retained,marked_event_reason=reason,
            unambiguous_nearest_and_assignment=no_ambiguity,both_stationary=stationary,
            new_endpoint_geometry_sha256=[geometry_hash(im) for im in endpoints],
            band_gate='ELIGIBLE_FOR_PHYSICAL_REVIEW' if stationary and retained and no_ambiguity else 'DO_NOT_RUN_BAND')
        if stationary and retained and no_ambiguity:
            images,s=tapered_band(data,*endpoints)
            snapshot(images,args.output/'prepared_tapered_band.npz',parent_indices=ids)
            np.savez_compressed(args.output/'endpoint_taper.npz',original_arclength_fraction=s,
                start_displacement_A=endpoints[0].positions-original[0].positions,end_displacement_A=endpoints[1].positions-original[1].positions)
            export_images(images,args.output/'prepared_geometry_exports',ids)
            receipt['prepared_tapered_band_sha256']=digest(args.output/'prepared_tapered_band.npz')
        receipt['PES_basin_identity_limit']='Same marked/otherLi assignments and fixedhostidentities do not prove the same continuous PES basins. Inspect geometries/forces before approving the separate new-endpoint band.'
        code=0 if stationary else 2
    except BudgetExpired as e:receipt.update(status='BOUNDED_PARTIAL_PRESERVED',error=str(e));code=3
    except Exception as e:receipt.update(status='FAILED_PRESERVED',exception_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
    finally:
        snapshot(endpoints,args.output/'last_endpoints.npz',parent_indices=ids)
        receipt.update(call_accounting=ledger.summary(),completed_UTC=datetime.now(timezone.utc).isoformat())
        write_json(args.output/'receipt.json',receipt)
    return code


if __name__=='__main__':raise SystemExit(main())

"""Fresh alternate-teacher E/F on unchanged saved UMA path coordinates.

This is a fixed-coordinate profile, never an alternate-teacher optimized barrier.
"""
import time
PROCESS_START=time.monotonic()
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import traceback
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'conditioning_20261107'))
from neb_common import (CallLedger,attach_calculators,atoms_from,digest,discrete_barriers,
    load_npz,read_json,snapshot,write_json)
from cathode_cvae.runtime import require_compute_allocation


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--band',required=True,type=Path)
    p.add_argument('--root-protocol',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--model',choices=['uma','mace'],default='mace')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--device',default='cuda');p.add_argument('--max-seconds',type=float,default=1200)
    args=p.parse_args();require_compute_allocation();read_json(args.root_protocol)
    data=load_npz(args.band);images=[atoms_from(data,row) for row in data['positions_A']]
    args.output.mkdir(parents=True,exist_ok=False)
    binding=dict(band=dict(path=str(args.band.resolve()),sha256=digest(args.band)),
        root_protocol=dict(path=str(args.root_protocol.resolve()),sha256=digest(args.root_protocol)),
        checkpoint_sha256=digest(args.checkpoint),source_sha256=digest(__file__),model=args.model,
        coordinate_scope='Exactly the supplied UMA band; no relaxation, remapping, canonicalization or interpolation')
    write_json(args.output/'protocol.json',binding)
    ledger=CallLedger(args.output/'calculator_calls.jsonl',args.max_seconds,len(images),start=PROCESS_START)
    receipt=dict(status='PREPARED',binding=binding,limitations='Alternate-teacher energies/forces on UMA coordinates, not its optimized migration barrier')
    code=2
    try:
        from cathode_cvae.mlip.calculators import make_calculator
        calc,teacher=make_calculator(args.model,args.checkpoint,args.device,args.threads)
        receipt['teacher']=teacher;ledger.phase='unchanged_path_profile'
        attach_calculators(images,calc,ledger,args.model)
        for im in images:im.get_forces()
        result=snapshot(images,args.output/'profile.npz',parent_indices=data['parent_indices'])
        if not np.array_equal(result['positions_A'],data['positions_A']):raise ValueError('Profile coordinates changed')
        receipt.update(status='COMPLETE_FIXED_COORDINATE_PROFILE',**discrete_barriers(result['energy_eV']),
            physical_image_Fmax_eV_A=np.linalg.norm(result['forces_eV_A'],axis=2).max(1).tolist(),
            physical_endpoint_Fmax_eV_A=np.linalg.norm(result['forces_eV_A'][[0,-1]],axis=2).max(1).tolist())
        code=0
    except Exception as e:
        receipt.update(status='FAILED_OR_BOUNDED_PRESERVED',exception_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
    finally:
        snapshot(images,args.output/'last_state.npz',parent_indices=data['parent_indices'])
        receipt.update(call_accounting=ledger.summary(),completed_UTC=datetime.now(timezone.utc).isoformat())
        write_json(args.output/'receipt.json',receipt)
    return code


if __name__=='__main__':raise SystemExit(main())

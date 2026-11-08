"""Timed single-GPU inference pilot on three existing protected training masks."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from cathode_cvae.data import deletion_atoms, sha256
from cathode_cvae.mlip.calculators import make_calculator, evaluate
from cathode_cvae.runtime import require_compute_allocation


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--checkpoint',required=True)
    a=p.parse_args();require_compute_allocation()
    out=a.root/'runs/conditioning_20261107/pilot'/a.model
    out.mkdir(parents=True,exist_ok=False)
    parent_path=a.root/'artifacts/current/core/data/parents/O3_N48_pilot.npz'
    with np.load(parent_path) as z:parent={k:z[k] for k in z.files}
    data=a.root/'artifacts/current/improvement_v1/training_round0.npz'
    with np.load(data) as z:d={k:z[k] for k in z.files}
    begin=time.perf_counter()
    calc,receipt=make_calculator(a.model,a.checkpoint,'cuda',4)
    rows=[]
    for n in [12,24,36]:
        i=np.flatnonzero((d['split']=='train')&(d['n_Li']==n))[0]
        atoms,_=deletion_atoms(parent,d['occupancy'][i])
        t=time.perf_counter();E,F,S=evaluate(calc,atoms)
        rows.append(dict(structure_id=str(d['structure_ids'][i]),n_Li=n,
            energy_eV=E,force_shape=list(F.shape),stress_shape=list(S.shape),
            seconds=time.perf_counter()-t,finite=bool(np.isfinite(E) and np.isfinite(F).all() and np.isfinite(S).all())))
    result=dict(status='PASS' if all(r['finite'] for r in rows) else 'FAIL',
        scope='Existing training geometries; engineering and time profile, not a new assessment',
        rows=rows,wall_seconds=time.perf_counter()-begin,model_receipt=receipt,
        parent_sha256=sha256(parent_path),source_sha256=sha256(__file__))
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

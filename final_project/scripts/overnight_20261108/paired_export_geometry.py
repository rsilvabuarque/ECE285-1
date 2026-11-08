"""Lossless compact coordinate view of the original 6000 acquisition lineages.

Original roles, including interrupted cases, remain unchanged. TEST is written
separately. Missing endpoints use empty ragged segments; missing scalar/tensor
labels use NaN plus explicit known flags, never interpolated values.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed,TimeoutError
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from paired_common import atomic_json,check_prepared,geometry_hash,identity,sha256


def read_one(item):
    task,queue,parent_cell=item
    directory=Path(queue)/'results'/task['task_id'];record_path=directory/'result.json';record=json.loads(record_path.read_text())
    if record['physical_key']!=task['physical_key'] or record['context_hash']!=task['context_hash']:
        raise ValueError('Original task/result binding changed')
    array_path=directory/'arrays.npz'
    if sha256(array_path)!=record['arrays_sha256']:raise ValueError('Original raw array bytes changed')
    with np.load(array_path,allow_pickle=False) as z:
        out={k:z[k] for k in ['numbers','parent_indices','initial_positions_A','cell_A','occupancy']}
        if not np.array_equal(out['cell_A'],parent_cell):raise ValueError('Acquisition fixed cell changed')
        for key in ['endpoint_positions_A','endpoint_forces_eV_A','endpoint_stress_eV_A3','initial_energy_eV','endpoint_energy_eV']:
            out[key]=z[key] if key in z else None
    if geometry_hash(out['numbers'],out['initial_positions_A'],out['cell_A'])!=record['initial_geometry_hash']:
        raise ValueError('Initial coordinate binding differs')
    if out['endpoint_positions_A'] is not None:
        if geometry_hash(out['numbers'],out['endpoint_positions_A'],out['cell_A'])!=record['endpoint_literal_geometry_hash']:
            raise ValueError('Endpoint coordinate binding differs')
    if out['endpoint_forces_eV_A'] is not None:
        if float(np.linalg.norm(out['endpoint_forces_eV_A'],axis=1).max())!=record['endpoint_force_max_eV_A']:
            raise ValueError('Endpoint force/header mismatch')
    for key in ['initial_energy_eV','endpoint_energy_eV']:
        if out[key] is not None and float(out[key])!=record[key]:raise ValueError('Energy/header mismatch')
    out.update(task=task,record=record,result_sha256=sha256(record_path),task_record_sha256=identity(task))
    return out


def write_view(path,items,cohort,rows):
    offsets=np.r_[0,np.cumsum([len(x['numbers']) for x in items])]
    endpoint_offsets=np.r_[0,np.cumsum([len(x['numbers']) if x['endpoint_positions_A'] is not None else 0 for x in items])]
    force_offsets=np.r_[0,np.cumsum([len(x['numbers']) if x['endpoint_forces_eV_A'] is not None else 0 for x in items])]
    def cat(key,width=None,dtype=float):
        values=[x[key] for x in items if x[key] is not None]
        return np.concatenate(values,axis=0) if values else np.empty((0,width),dtype=dtype) if width else np.empty(0,dtype=dtype)
    def scalar(x,key):return float(x[key]) if x[key] is not None else np.nan
    data=dict(cohort_rows=np.array(rows),structure_ids=cohort['structure_ids'][rows],split=cohort['split'][rows],
        n_target=cohort['n_target'][rows],occupancy=cohort['occupancy'][rows],canonical_vacancy_keys=cohort['canonical_masks'][rows],
        task_ids=np.array([x['task']['task_id'] for x in items]),status=np.array([x['record']['status'] for x in items]),
        endpoint_labels_usable=np.array([x['record']['endpoint_labels_usable'] for x in items],bool),
        input_offsets=offsets,numbers=cat('numbers',dtype=int),parent_indices=cat('parent_indices',dtype=int),
        initial_positions_A=cat('initial_positions_A',3),cell_A=np.stack([x['cell_A'] for x in items]),
        endpoint_offsets=endpoint_offsets,endpoint_positions_A=cat('endpoint_positions_A',3),
        endpoint_geometry_known=np.array([x['endpoint_positions_A'] is not None for x in items],bool),
        force_offsets=force_offsets,endpoint_forces_eV_A=cat('endpoint_forces_eV_A',3),
        endpoint_force_known=np.array([x['endpoint_forces_eV_A'] is not None for x in items],bool),
        endpoint_stress_eV_A3=np.stack([x['endpoint_stress_eV_A3'] if x['endpoint_stress_eV_A3'] is not None else np.full(6,np.nan) for x in items]),
        endpoint_stress_known=np.array([x['endpoint_stress_eV_A3'] is not None for x in items],bool),
        initial_energy_eV=np.array([scalar(x,'initial_energy_eV') for x in items]),
        endpoint_energy_eV=np.array([scalar(x,'endpoint_energy_eV') for x in items]),
        initial_energy_known=np.array([x['initial_energy_eV'] is not None for x in items],bool),
        endpoint_energy_known=np.array([x['endpoint_energy_eV'] is not None for x in items],bool),
        endpoint_fmax_eV_A=np.array([x['record']['endpoint_force_max_eV_A'] if x['record']['endpoint_force_max_eV_A'] is not None else np.nan for x in items]),
        initial_geometry_hash=np.array([x['record']['initial_geometry_hash'] for x in items]),
        endpoint_geometry_hash=np.array([x['record']['endpoint_literal_geometry_hash'] or '' for x in items]),
        input_record_sha256=np.array([x['task_record_sha256'] for x in items]),
        result_sha256=np.array([x['result_sha256'] for x in items]),arrays_sha256=np.array([x['record']['arrays_sha256'] for x in items]))
    if any(v.dtype.hasobject for v in data.values()):raise ValueError('Compact views must contain no object arrays')
    np.savez_compressed(path,**data)
    return dict(path=path.name,sha256=sha256(path),rows=len(items),roles=dict(Counter(str(x) for x in data['split'])),
        statuses=dict(Counter(str(x) for x in data['status'])),endpoint_geometry_known=int(data['endpoint_geometry_known'].sum()),
        usable_endpoints=int(data['endpoint_labels_usable'].sum()),bytes=path.stat().st_size)


def main(argv=None):
    from cathode_cvae.runtime import require_compute_allocation
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','queue','output-dir']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--read-budget-seconds',type=float,default=120)
    a=p.parse_args(argv);require_compute_allocation();started=time.monotonic()
    preparation,context=check_prepared(a.prepared)
    tasks_path=a.prepared/'tasks_all.json';tasks=json.loads(tasks_path.read_text())['tasks']
    if len(tasks)!=6000:raise ValueError('Only the original acquisition6000 is allowed')
    with np.load(a.prepared/'cohort.npz',allow_pickle=False) as z:cohort={k:z[k] for k in ['structure_ids','split','n_target','occupancy','canonical_masks']}
    with np.load(a.prepared/'parent.npz',allow_pickle=False) as z:parent_cell=z['cell_A']
    executor=ProcessPoolExecutor(max_workers=a.workers);futures={executor.submit(read_one,(task,str(a.queue),parent_cell)):i for i,task in enumerate(tasks)}
    data={}
    try:
        for future in as_completed(futures,timeout=a.read_budget_seconds):data[futures[future]]=future.result()
    except BaseException:
        for f in futures:f.cancel()
        executor.shutdown(wait=False,cancel_futures=True)
        raise
    executor.shutdown()
    a.output_dir.mkdir(parents=True,exist_ok=False)
    views=[]
    for name,rows in [('TRAINVAL',np.flatnonzero(np.isin(cohort['split'],['train','val']))),('TEST',np.flatnonzero(cohort['split']=='test'))]:
        views.append(write_view(a.output_dir/(name+'.npz'),[data[int(i)] for i in rows],cohort,rows))
    shutil.copyfile(a.prepared/'parent.npz',a.output_dir/'parent.npz')
    (a.output_dir/'README.md').write_text('''# Original acquisition geometry view

TRAINVAL.npz preserves all4800 original TRAIN/VAL rows, including interrupted rows. TEST.npz preserves all1200 original TEST rows and is assessment only. No FINAL7616 proposals occur here. Every array is numeric or fixed-width Unicode; use numpy.load(..., allow_pickle=False).

For row i, input_offsets[i:i+2] selects numbers, parent_indices, and initial_positions_A. endpoint_offsets selects observed endpoint_positions_A; force_offsets selects endpoint_forces_eV_A. Present endpoint/force segments have the same atom order as the input segment. An empty segment means unavailable data. NaN labels/stresses explicitly mark missing values; known flags must be checked. Interrupted last observed poses are retained and never promoted to converged endpoints: check status and endpoint_labels_usable.

Positions use Angstrom, energy eV/cell, forces eV/Angstrom, and ASE tensile-positive stress eV/Angstrom^3 in xx,yy,zz,yz,xz,xy order. Geometry is fixed-cell UMA-specific relaxation data, not DFT or a certified global minimum. Original occupancy is1=present, canonical vacancy keys use1=vacant. Parent indices preserve original atom identities without independent endpoint canonicalization.
''')
    receipt=dict(status='PASS_COMPLETE_ACQUISITION_GEOMETRY_VIEW',completed_UTC=datetime.now(timezone.utc).isoformat(),
        views=views,original_rows=6000,source_cohort_sha256=sha256(a.prepared/'cohort.npz'),source_tasks_sha256=sha256(tasks_path),
        parent_sha256=sha256(a.prepared/'parent.npz'),context_hash=context['context_hash'],source_sha256=sha256(__file__),
        all_raw_arrays_rehashed=True,original_splits_preserved=True,assessment7616_included=False,missing_values_imputed=False,
        no_NEW_masks=True,new_model_calls=0,new_training_updates=0,elapsed_seconds=time.monotonic()-started)
    atomic_json(a.output_dir/'receipt.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

"""Freeze coordinates-only history from explicit unsealed research trajectories.

No paired-cohort directory is accepted. ASE may deserialize existing unsealed
research labels while opening its trajectory format, but only geometry and
continuous original atom identities are exported; no TEST file is opened.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

from ase.io import iread
import numpy as np
from cathode_cvae.data import sha256
from cathode_cvae.runtime import require_compute_allocation


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def geometry_hash(numbers,positions,cell,pbc):
    h=hashlib.sha256()
    for a in [np.asarray(numbers,dtype='<i8'),np.asarray(positions,dtype='<f8'),np.asarray(cell,dtype='<f8'),np.asarray(pbc,np.uint8)]:h.update(a.tobytes())
    return h.hexdigest()


def companion(path):
    candidates=['initial_band.npz','candidate.npz',path.stem+'_final.npz',path.stem+'.npz',
        'hessian.npz','final_band.npz','last_state.npz','last_endpoints.npz','endpoint_0_final.npz','endpoint_0_latest.npz','latest_band.npz']
    for name in candidates:
        file=path.parent/name
        if not file.exists():continue
        before=sha256(file)
        with np.load(file,allow_pickle=False) as z:
            if 'parent_indices' not in z:continue
            result={k:z[k] for k in ['parent_indices','numbers','cell_A']}
        if sha256(file)!=before:continue
        return result,dict(path=str(file.resolve()),sha256=before)
    raise ValueError('No stable original-ID companion for '+str(path))


def stable_copy(source,destination,attempts):
    observations=[]
    for attempt in range(attempts):
        before=sha256(source);shutil.copyfile(source,destination);after=sha256(source);copy=sha256(destination)
        observations.append(dict(attempt=attempt+1,source_before_sha256=before,source_after_sha256=after,copy_sha256=copy))
        if before==after==copy:return observations
    raise ValueError('Source changed during every bounded snapshot copy: '+str(source))


def export(args):
    require_compute_allocation();started=time.monotonic();args.output.mkdir(parents=True,exist_ok=False)
    for root in args.roots:
        if 'paired' in root.parts:raise ValueError('Only explicit unsealed research roots are accepted; paired cohorts are forbidden')
    with np.load(args.parent,allow_pickle=False) as z:parent={k:z[k] for k in ['numbers','cell_A']}
    identity=args.output/'identity';raw=args.output/'stable_raw';identity.mkdir();raw.mkdir()
    sources=sorted({p.resolve() for root in args.roots for p in root.rglob('*.traj')})
    listed_UTC=datetime.now(timezone.utc).isoformat();coordinate_files=[];rows=[];failures=[]
    for source in sources:
        stem=source.parent.name+'_'+source.stem+'_'+hashlib.sha256(str(source).encode()).hexdigest()[:10]
        destination=raw/(stem+'.traj')
        try:
            observation=stable_copy(source,destination,args.copy_attempts);binding,binding_source=companion(source)
            slots=np.asarray(binding['parent_indices'],int);numbers=np.asarray(binding['numbers']);cell=np.asarray(binding['cell_A'])
            if len(set(slots.tolist()))!=len(slots) or not np.array_equal(numbers,parent['numbers'][slots]):raise ValueError('Original IDs do not match parent chemistry')
            if not np.array_equal(cell,parent['cell_A']):raise ValueError('Research trajectory belongs to another cell namespace')
            frames=[]
            for atom in iread(destination,index=':'):
                if not np.array_equal(atom.numbers,numbers) or not np.array_equal(np.asarray(atom.cell),cell):
                    raise ValueError('Trajectory changed atom order/species/cell')
                frames.append(atom.positions.copy())
            if not frames:raise ValueError('No fully committed trajectory frames')
            output=identity/(stem+'.npz')
            np.savez_compressed(output,positions_A=np.asarray(frames),numbers=numbers,cell_A=cell,parent_indices=slots,
                source_frame_index=np.arange(len(frames)),source_trajectory_sha256=np.array(observation[-1]['copy_sha256']),
                parent_sha256=np.array(sha256(args.parent)))
            coordinate_files.append(dict(path=str(output.resolve()),sha256=sha256(output),positions_key='positions_A'))
            rows.append(dict(kind='accepted_saved_trajectory_frames',source=str(source),source_snapshot_checks=observation,
                original_ID_companion=binding_source,frames=len(frames),atoms=len(numbers),terminal_receipt_present=(source.parent/'receipt.json').exists(),
                cutoff_stable_copy=str(destination.resolve()),output=coordinate_files[-1]))
        except Exception as e:
            failures.append(dict(source=str(source),exception_type=type(e).__name__,error=str(e)))
        dump(args.output/'progress.json',dict(completed=len(rows),failures=failures,source_inventory=len(sources)))
    for folder in args.hessians:
        source=folder/'hessian.npz';receipt=json.loads((folder/'receipt.json').read_text())
        if receipt['status']!='COMPLETE_CURVATURE_DIAGNOSTIC':raise ValueError('Only complete actual FD experiments are included')
        before=sha256(source)
        with np.load(source,allow_pickle=False) as z:
            base=z['positions_A'];numbers=z['numbers'];cell=z['cell_A'];pbc=z['pbc'];slots=z['parent_indices'];delta=float(z['delta_A'])
        if sha256(source)!=before:raise ValueError('FD source changed while read')
        if not np.array_equal(numbers,parent['numbers'][slots]) or not np.array_equal(cell,parent['cell_A']):raise ValueError('FD parent identity differs')
        calls=[json.loads(line) for line in (folder/'calculator_calls.jsonl').read_text().splitlines() if line]
        central=[r for r in calls if r['phase']=='central_displacement_Hessian'];dof=base.size
        if len(central)!=2*dof:raise ValueError('Unexpected actual FD coordinate-call count')
        frames=np.repeat(base[None],2*dof,axis=0)
        for column in range(dof):
            for j,sign in enumerate([1,-1]):
                index=2*column+j;frames[index].flat[column]+=sign*delta
                if central[index]['status']!='success' or geometry_hash(numbers,frames[index],cell,pbc)!=central[index]['geometry_sha256']:
                    raise ValueError('Reconstructed FD input differs from actual successful call')
        output=identity/(folder.name+'_actual_FD_inputs.npz')
        np.savez_compressed(output,positions_A=frames,numbers=numbers,cell_A=cell,parent_indices=slots,
            displacement_A=np.array(delta),source_hessian_sha256=np.array(before),parent_sha256=np.array(sha256(args.parent)))
        coordinate_files.append(dict(path=str(output.resolve()),sha256=sha256(output),positions_key='positions_A'))
        rows.append(dict(kind='actual_complete_FD_inputs',source=str(source.resolve()),source_sha256_before=before,
            source_sha256_after=sha256(source),actual_input_geometries=len(frames),atoms=len(numbers),
            receipt_sha256=sha256(folder/'receipt.json'),call_log_sha256=sha256(folder/'calculator_calls.jsonl'),output=coordinate_files[-1]))
    after=sorted({p.resolve() for root in args.roots for p in root.rglob('*.traj')})
    changed=[str(p) for p in sources if p.exists() and sha256(p)!=next((r['source_snapshot_checks'][-1]['source_after_sha256'] for r in rows if r.get('source')==str(p)),sha256(p))]
    result=dict(status='COMPLETE_COORDINATES_ONLY_CUTOFF_SNAPSHOTS' if not failures and sources==after else 'PARTIAL_EXPOSURE_SNAPSHOT',
        listed_UTC=listed_UTC,completed_UTC=datetime.now(timezone.utc).isoformat(),source_sha256=sha256(__file__),
        source_roots=[str(x.resolve()) for x in args.roots],trajectory_sources=len(sources),trajectories_exported=sum(r['kind']=='accepted_saved_trajectory_frames' for r in rows),
        accepted_saved_frames=sum(r.get('frames',0) for r in rows),actual_FD_inputs=sum(r.get('actual_input_geometries',0) for r in rows),
        failures=failures,source_inventory_unchanged=sources==after,sources_changed_after_individual_cutoff=changed,
        coordinate_files=coordinate_files,records=rows,elapsed_seconds=time.monotonic()-started,
        label_fields_exported=0,new_MLIP_calls=0,paired_TEST_files_opened=0,
        scope='Every fully committed frame in stable trajectory byte copies plus every actual complete FD input; coordinates/IDs only. Later appended frames are outside the declared per-file cutoff and must be considered before a later FINAL freeze.')
    dump(args.output/'receipt.json',result);dump(args.output/'exposure_manifest.json',dict(coordinate_files=coordinate_files))
    print(json.dumps({k:result[k] for k in ['status','trajectory_sources','trajectories_exported','accepted_saved_frames','actual_FD_inputs','failures','sources_changed_after_individual_cutoff']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',nargs='+',type=Path,required=True)
    p.add_argument('--hessians',nargs='+',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--copy-attempts',type=int,default=3)
    export(p.parse_args())


if __name__=='__main__':main()

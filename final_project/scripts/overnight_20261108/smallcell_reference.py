"""Literal-parent finite-cell enumeration, fresh labels and thermodynamics.

Raw occupation states are never symmetry-aliased for physical scoring. The Hu
parent has only approximate geometric symmetry; orbit keys describe its retained
site-permutation group, not exact equality of the stored coordinate energies.
"""
import time
PROCESS_START=time.monotonic()  # A worker's physical timer includes imports/model loading.
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from cathode_cvae.acquisition import KB_EV_K,canonical_keys,static_weights
from cathode_cvae.data import deletion_atoms,sha256,target_indices
from cathode_cvae.runtime import require_compute_allocation

COUNTS=(3,6,9)
TEMPERATURES=(393.15,433.15,473.15,493.15)


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def geometry_hash(atoms):
    h=hashlib.sha256()
    for value in [atoms.numbers.astype('<i8'),atoms.positions.astype('<f8'),np.asarray(atoms.cell).astype('<f8'),atoms.pbc.astype(np.uint8)]:h.update(value.tobytes())
    return h.hexdigest()


def enumerate_occupancies(n_sites,counts):
    rows=[]
    for n in counts:
        for occupied in itertools.combinations(range(n_sites),n):
            x=np.zeros(n_sites,np.uint8);x[list(occupied)]=1;rows.append(x)
    return np.asarray(rows,np.uint8)


def prepare(args):
    require_compute_allocation();parent=load_npz(args.parent);indices=target_indices(parent)
    if len(indices)!=12 or len(parent['numbers'])!=48:raise ValueError('This campaign requires the literal 12-site/48-atom Hu parent')
    provenance=json.loads(args.provenance.read_text());source=next(r for r in provenance['parents'] if r['parent']=='Hu_O3_12')
    if source['sha256']!=sha256(args.parent):raise ValueError('Parent bytes differ from the traced Hu geometry')
    args.output.mkdir(parents=True,exist_ok=False)
    x=enumerate_occupancies(12,COUNTS);vacancy=1-x;keys=canonical_keys(x,parent['permutations'])
    if len(x)!=1364 or len({r.tobytes() for r in x})!=1364:raise ValueError('Raw-state enumeration is incomplete')
    raw_codes=[''.join(map(str,row)) for row in vacancy]
    ids=np.array(['hu12raw_'+format(int(code,2),'03x') for code in raw_codes])
    geometry=[]
    for row in x:
        atoms,_=deletion_atoms(parent,row);geometry.append(geometry_hash(atoms))
    representatives={}
    for sid,mask in zip(parent['structure_ids'].astype(str),parent['masks']):
        if int((1-mask).sum()) not in COUNTS:continue
        key=str(canonical_keys((1-mask)[None],parent['permutations'])[0])
        raw=''.join(map(str,mask.astype(int)))
        representatives[key]=dict(legacy_id=sid,raw_id='hu12raw_'+format(int(raw,2),'03x'),raw_vacancy_mask=raw)
    if len(representatives)!=29:raise ValueError('Expected6/17/6 approximate orbit representatives')
    # Two fixed raw geometries/count receive two additional exact repeats each.
    # Their primary evaluation plus repeats diagnose numerical repeatability.
    repeat_rows=[]
    for n in COUNTS:
        ii=np.flatnonzero(x.sum(1)==n)
        repeat_rows.extend(int(i) for i in [ii[0],ii[-1]])
    label_rows=np.r_[np.arange(len(x)),np.repeat(repeat_rows,2)]
    repeat_index=np.r_[np.zeros(len(x),int),np.tile([1,2],len(repeat_rows))]
    label_ids=np.array([str(ids[i])+f'_r{r}' for i,r in zip(label_rows,repeat_index)])
    np.savez_compressed(args.output/'states.npz',occupancy=x,masks=vacancy,raw_ids=ids,
        raw_vacancy_keys=np.asarray(raw_codes),canonical_orbit_keys=keys,n_Li=x.sum(1),n_atoms=36+x.sum(1),
        geometry_sha256=np.asarray(geometry),parent_sha256=np.asarray(sha256(args.parent)),
        label_state_rows=label_rows,label_repeat_index=repeat_index,label_ids=label_ids)
    write_json(args.output/'representatives.json',representatives)
    protocol=dict(status='FROZEN_BEFORE_NEW_LABELS',created_UTC=datetime.now(timezone.utc).isoformat(),
        parent=dict(path=str(args.parent.resolve()),sha256=sha256(args.parent)),
        parent_provenance=dict(path=str(args.provenance.resolve()),sha256=sha256(args.provenance),record=source),
        root_protocol=dict(path=str(args.root_protocol.resolve()),sha256=sha256(args.root_protocol)),
        counts=list(COUNTS),temperatures_K=list(TEMPERATURES),raw_states=1364,labels_per_teacher=len(label_rows),
        additional_repeats_per_teacher=len(label_rows)-len(x),models=['uma','mace'],no_relaxation=True,
        geometry='Literal saved parent coordinates/species/cell with chosen Li deleted; no canonical coordinate permutation or symmetrization',
        conventions='occupancy1=present Li; mask1=vacancy; raw IDs identify actual stored masks; orbit key is separate',
        primary_measure='Every raw occupation state once, normalized separately by composition and teacher',
        representative_join='Use the stored parent mask, not a geometry inferred from its canonical-code legacy ID',
        symmetry_scope='Retained approximate site-permutation group; physical within-orbit energy equality must be measured',
        outputs={name:sha256(args.output/name) for name in ['states.npz','representatives.json']},source_sha256=sha256(__file__))
    write_json(args.output/'protocol.json',protocol)
    print(json.dumps({k:protocol[k] for k in ['status','raw_states','labels_per_teacher','additional_repeats_per_teacher']},indent=2))


def score(args):
    require_compute_allocation();started=PROCESS_START
    protocol=json.loads((args.prepared/'protocol.json').read_text())
    for record in [protocol['parent'],protocol['root_protocol']]:
        if sha256(record['path'])!=record['sha256']:raise ValueError('Frozen source changed')
    path=args.prepared/'states.npz'
    if sha256(path)!=protocol['outputs']['states.npz']:raise ValueError('Frozen states changed')
    states=load_npz(path);parent=load_npz(Path(protocol['parent']['path']))
    if not 0<=args.shard<args.shards or args.shards<1:raise ValueError('Invalid worker shard')
    expected=np.arange(len(states['label_ids']))[args.shard::args.shards]
    args.output.mkdir(parents=True,exist_ok=args.resume)
    binding=dict(model=args.model,parent_sha256=protocol['parent']['sha256'],states_sha256=sha256(path),
        checkpoint_sha256=sha256(args.checkpoint),shard=args.shard,shards=args.shards,source_sha256=sha256(__file__))
    binding_path=args.output/'binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text())!=binding:raise ValueError('Resume context differs')
    if not binding_path.exists():write_json(binding_path,binding)
    completed=set()
    for saved in args.output.glob('chunk_*.npz'):
        side=json.loads(saved.with_suffix('.json').read_text())
        if side['sha256']!=sha256(saved) or side['binding']!=binding:raise ValueError('Saved score chunk changed')
        completed.update(side['label_indices'])
    if not completed<=set(expected.tolist()):raise ValueError('Shard contains unexpected labels')
    pending=[int(i) for i in expected if i not in completed]
    if args.limit_labels is not None:pending=pending[:args.limit_labels]
    from cathode_cvae.mlip.calculators import make_calculator,evaluate
    print('MODEL_LOADING',args.model,flush=True)
    calc,teacher=make_calculator(args.model,args.checkpoint,args.device,args.threads)
    write_json(args.output/'teacher.json',teacher);print('MODEL_READY',args.model,flush=True)
    rows=[];forces=[];stresses=[];calls=0;failures=0
    def flush():
        if not rows:return
        output=args.output/f"chunk_{rows[0]['label_index']:05d}.npz"
        if output.exists():raise FileExistsError('Do not overwrite completed chunks')
        np.savez_compressed(output,**{key:np.asarray([r[key] for r in rows]) for key in rows[0]},
            forces_parent_eV_A=np.asarray(forces),stress_eV_A3=np.asarray(stresses))
        write_json(output.with_suffix('.json'),dict(binding=binding,label_indices=[r['label_index'] for r in rows],sha256=sha256(output)))
        rows.clear();forces.clear();stresses.clear()
    for label in pending:
        if time.monotonic()-started>=args.max_seconds or calls>=args.max_calls:break
        state=int(states['label_state_rows'][label]);atoms,slots=deletion_atoms(parent,states['occupancy'][state])
        if geometry_hash(atoms)!=str(states['geometry_sha256'][state]):raise ValueError('Actual inference geometry differs from prepared hash')
        begin=time.perf_counter();calls+=1
        try:
            calc.reset();energy,force,stress=evaluate(calc,atoms)
            if not np.isfinite(energy) or not np.isfinite(force).all() or not np.isfinite(stress).all():raise ValueError('Nonfinite label')
            padded=np.zeros((len(parent['numbers']),3));padded[slots]=force
            status='success';error='';stress=np.asarray(stress)
        except Exception as failure:
            failures+=1;energy=np.nan;padded=np.full((len(parent['numbers']),3),np.nan);stress=np.full(6,np.nan)
            status='failed';error=f'{type(failure).__name__}: {failure}'
        rows.append(dict(label_index=label,label_id=str(states['label_ids'][label]),state_row=state,
            raw_id=str(states['raw_ids'][state]),repeat_index=int(states['label_repeat_index'][label]),
            n_Li=int(states['n_Li'][state]),n_atoms=len(atoms),geometry_sha256=geometry_hash(atoms),
            energy_eV=float(energy),status=status,error=error,seconds=time.perf_counter()-begin))
        forces.append(padded);stresses.append(stress)
        if len(rows)>=args.chunk_size:flush();print('SCORED',args.model,calls,flush=True)
    flush()
    done=[];total_failures=0
    for file in args.output.glob('chunk_*.json'):
        done.extend(json.loads(file.read_text())['label_indices'])
        with np.load(file.with_suffix('.npz'),allow_pickle=False) as z:total_failures+=int(np.sum(z['status']!='success'))
    complete=set(done)==set(expected.tolist())
    record=dict(status=('COMPLETE' if not total_failures else 'COMPLETE_WITH_FAILURES') if complete else 'BOUNDED_PARTIAL',binding=binding,expected_labels=len(expected),
        completed_label_rows=len(done),new_EFS_attempts=calls,failed_EFS_attempts=failures,
        all_saved_failed_label_rows=total_failures,
        elapsed_seconds_including_model_loading=time.monotonic()-started,teacher=teacher,
        limitations='Fresh literal raw-state static model only; no DFT, relaxations or physical entropy validation')
    write_json(args.output/'receipt.json',record);print(json.dumps({k:v for k,v in record.items() if k not in ['teacher','binding']},indent=2))


def raw_thermodynamics(energy,temperature):
    """Exact finite raw-state model, stable even when the energies are negative."""
    e=np.asarray(energy,float)
    if e.ndim!=1 or not len(e) or not np.isfinite(e).all():raise ValueError('Every enumerated raw state needs a finite energy')
    logw=-(e-e.min())/(KB_EV_K*temperature);m=float(logw.max());logz=m+float(np.log(np.exp(logw-m).sum()))
    logp=logw-logz;p=np.exp(logp)
    entropy=-KB_EV_K*float(np.sum(p*logp))
    mean=float(np.dot(p,e));free=float(e.min()-KB_EV_K*temperature*logz)
    return dict(probability=p,mean_energy_eV=mean,entropy_eV_K=entropy,free_energy_eV=free,
        state_ESS=float(1/np.dot(p,p)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--parent',type=Path,required=True);p.add_argument('--provenance',type=Path,required=True)
    p.add_argument('--root-protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.set_defaults(function=prepare)
    p=sub.add_parser('score');p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=['uma','mace'],required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=4);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--chunk-size',type=int,default=64);p.add_argument('--resume',action='store_true')
    p.add_argument('--max-seconds',type=float,default=1800);p.add_argument('--max-calls',type=int,default=2000);p.add_argument('--limit-labels',type=int)
    p.set_defaults(function=score);args=parser.parse_args();args.function(args)


if __name__=='__main__':main()

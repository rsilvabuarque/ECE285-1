"""Small synthetic regressions for adaptive binning and attempt accounting.

These tests do not train a model or load an interatomic potential. The fake
score verifier below tests merge/summary consumption only; raw oracle receipt
validation is independently covered by the existing iterative-comparison tests.
"""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cathode_cvae import adaptive_iteration as adaptive
from cathode_cvae.conditions import classify, fit_schema, load_schema, save_schema
from cathode_cvae.data import sha256


def fixture(tmp_path, energies=None):
    parent=tmp_path/'parent.npz'
    np.savez(parent,numbers=np.full(48,3),li_indices=np.arange(48),
             permutations=np.arange(48)[None,:],positions_A=np.zeros((48,3)),cell_A=np.eye(3)*10)
    energies=np.asarray(np.arange(1,101) if energies is None else energies,dtype=float)
    size=len(energies);rng=np.random.default_rng(94);keys=set();x=[]
    while len(x)<size+2:
        row=np.zeros(48,np.uint8);row[rng.choice(48,12,replace=False)]=1
        key=''.join(map(str,1-row))
        if key not in keys:keys.add(key);x.append(row)
    x=np.asarray(x);roles=np.asarray(['train']*size+['val','test'])
    energy=np.r_[energies,50.,np.nan]
    data=tmp_path/'data.npz'
    np.savez_compressed(data,occupancy=x,n_Li=np.full(size+2,12),energy_eV_atom=energy,
                        split=roles,weight=np.ones(size+2,np.float32),
                        structure_ids=np.asarray([f'original_{i}' for i in range(size+2)]),
                        canonical_masks=np.asarray([''.join(map(str,1-row)) for row in x]),
                        source=np.full(size+2,'synthetic'))
    reference=fit_schema(np.full(size,12),energies,np.full(size,'train'),n_sites=48,
                         provenance={'parent_sha256':sha256(parent)})
    schema=tmp_path/'reference.json';save_schema(reference,schema)
    boundary=make_boundary(tmp_path/'boundary',parent,keys,
                           {''.join(map(str,1-row)) for row in x[-2:]})
    return parent,data,schema,boundary


def make_boundary(path,parent,known,protected):
    path.mkdir();adaptive.write_json(path/'known_history.json',sorted(known))
    adaptive.write_json(path/'protected.json',sorted(protected))
    adaptive.write_json(path/'receipt.json',dict(status='FROZEN_ADAPTIVE_CANONICAL_BOUNDARY',
        parent=adaptive.binding(parent),history=adaptive.binding(path/'known_history.json'),
        protected=adaptive.binding(path/'protected.json')))
    return path


def proposal(tmp_path,parts,name,**kwargs):
    parent,_,schema,boundary=parts
    return adaptive.propose(parent,schema,boundary,tmp_path/name,namespace='synthetic_round1',
                seed=5,method='uniform',sampler='uniform',counts=[12],bins=['low','middle','high'],
                attempts_per_cell=10,queries_per_cell=4,device='cpu',**kwargs)


def fake_verified_scores(stage,out,energies):
    with np.load(stage/'candidates.npz',allow_pickle=False) as z:
        ids=z['structure_ids'];n=z['n_target']
    path=out/'scores.npz';np.savez(path,structure_ids=ids,energy_eV=np.asarray(energies)*n,n_atoms=n,n_Li=n)
    verifier=out/'verifier.json'
    adaptive.write_json(verifier,dict(status='VERIFIED_FRESH_ADAPTIVE_ORACLE',teacher='uma',
        stages=[adaptive.binding(stage/'receipt.json')],score=adaptive.binding(path)))
    return verifier


def test_schema_modes_are_train_only_and_have_distinct_meanings(tmp_path):
    parent,data,reference,boundary=fixture(tmp_path)
    # Extend TRAIN only while retaining a deliberately nonfinite TEST energy.
    with np.load(data,allow_pickle=False) as z:base={k:z[k] for k in z.files}
    added=np.asarray([-20.,-10.,50.]);n=len(base['split'])
    base['n_Li']=np.r_[base['n_Li'],[12]*3]
    base['energy_eV_atom']=np.r_[base['energy_eV_atom'],added]
    base['split']=np.r_[base['split'],['train']*3]
    base['structure_ids']=np.r_[base['structure_ids'],['new1','new2','new3']]
    # Schema fitting accesses these fields only; no geometry is scored.
    changed=tmp_path/'changed.npz';np.savez(changed,**base)
    got={}
    for mode in ('fixed','adaptive','range'):
        adaptive.fit_round_schema(changed,parent,reference,mode,tmp_path/mode)
        got[mode]=load_schema(tmp_path/mode/'condition_schema.json')
    assert got['fixed']['compositions']['12']['edges_eV_atom']==[20.,80.]
    assert got['adaptive']['compositions']['12']['edges_eV_atom']==[19.,80.]
    assert got['range']['compositions']['12']['edges_eV_atom']==[4.,76.]
    assert got['range']['compositions']['12']['q20_eV_atom']==19.
    assert got['range']['compositions']['12']['lower_quantile_eV_atom']==19.
    assert got['range']['compositions']['12']['lower_condition_edge_eV_atom']==4.
    assert 'condition edges' in got['range']['quantile_estimator']
    assert got['fixed']['identity']==load_schema(reference)['identity']
    receipt=json.loads((tmp_path/'adaptive/receipt.json').read_text())
    assert receipt['training_rows']==103


def test_proposals_are_paired_count_constrained_and_cap_without_energy(tmp_path):
    parts=fixture(tmp_path);proposal(tmp_path,parts,'first');proposal(tmp_path,parts,'repeat')
    with np.load(tmp_path/'first/raw_attempts.npz') as a,np.load(tmp_path/'repeat/raw_attempts.npz') as b:
        for field in ('occupancy','latents','uniforms'):assert np.array_equal(a[field],b[field])
        assert np.all(a['occupancy'].sum(1)==12)
    rows=adaptive.read_csv(tmp_path/'first/attempts.csv')
    assert len(rows)==30 and len({r['attempt_id'] for r in rows})==30
    for requested in ('low','middle','high'):
        cell=[r for r in rows if r['requested_bin']==requested]
        assert sum(r['status']=='selected' for r in cell)==4
        assert sum(r['status']=='query_cap' for r in cell)==6
    parent,data,schema,boundary=parts
    keys={r['canonical_mask'] for r in rows}
    known={rows[0]['canonical_mask']};protected={rows[10]['canonical_mask']}
    second_boundary=make_boundary(tmp_path/'second_boundary',parent,known|protected,protected)
    adaptive.propose(parent,schema,second_boundary,tmp_path/'filtered',namespace='synthetic_round1',
         seed=5,method='uniform',sampler='uniform',counts=[12],attempts_per_cell=10,queries_per_cell=4,device='cpu')
    filtered=adaptive.read_csv(tmp_path/'filtered/attempts.csv')
    assert filtered[0]['status']=='known_boundary' and filtered[10]['status']=='protected_boundary'
    with np.load(tmp_path/'filtered/raw_attempts.npz') as a,np.load(tmp_path/'first/raw_attempts.npz') as b:
        assert np.array_equal(a['occupancy'],b['occupancy'])


def test_merge_preserves_roles_and_test_values_and_uses_measured_labels(tmp_path):
    parts=fixture(tmp_path);parent,data,schema,boundary=parts;proposal(tmp_path,parts,'stage')
    stage=tmp_path/'stage'
    with np.load(stage/'candidates.npz') as z:count=len(z['structure_ids'])
    energies=np.resize(np.asarray([-5.,50.,150.]),count)
    verifier=fake_verified_scores(stage,tmp_path,energies)
    receipt=adaptive.merge_training(data,parent,schema,stage,verifier,tmp_path/'merged',1)
    assert receipt['added_train_rows']==12 and receipt['original_rows_value_identical']
    with np.load(data) as a,np.load(tmp_path/'merged/training.npz') as b:
        for name in a.files:
            x,y=a[name],b[name][:len(a[name])]
            assert np.array_equal(x,y) if x.dtype.kind in 'US' else np.array_equal(x,y,equal_nan=True)
        assert np.all(b['split'][-12:]=='train') and np.isnan(b['energy_eV_atom'][101])
    labels=adaptive.read_csv(tmp_path/'merged/acquired_labels.csv')
    assert set(r['measured_bin'] for r in labels)=={'low','middle','high'}
    # Requested bins did not filter energies: even the high-energy miss is added.
    assert any(float(r['energy_eV_atom'])==150. for r in labels)
    adaptive.summarize(stage,verifier,schema,tmp_path/'summary')
    metrics=adaptive.read_csv(tmp_path/'summary/metrics.csv')
    assert all(float(r['fixed_inbin_yield'])<=.4 for r in metrics)


def test_extending_boundary_protects_assessment_only_and_detects_mutation(tmp_path):
    parts=fixture(tmp_path);parent,data,schema,boundary=parts;proposal(tmp_path,parts,'acq')
    adaptive.extend_boundary(boundary,[tmp_path/'acq'],[],tmp_path/'after_acq')
    _,known,protected=adaptive.load_boundary(tmp_path/'after_acq')
    original=json.loads((tmp_path/'acq/all_generated_keys.json').read_text())
    assert set(original)<=known
    assert len(protected)==2
    adaptive.extend_boundary(boundary,[],[tmp_path/'acq'],tmp_path/'after_assessment')
    _,_,protected=adaptive.load_boundary(tmp_path/'after_assessment')
    with np.load(tmp_path/'acq/candidates.npz') as z:expected={''.join(map(str,m)) for m in z['masks']}
    assert expected<=protected
    (tmp_path/'acq/attempts.csv').write_text('tampered\n')
    with pytest.raises(ValueError,match='hash mismatch'):
        adaptive.validate_stage(tmp_path/'acq')


def test_seed_bank_and_mutation_preserve_fixed_count(tmp_path):
    parts=fixture(tmp_path);parent,data,schema,boundary=parts
    adaptive.freeze_seed_bank(data,parent,schema,boundary,tmp_path/'bank')
    adaptive.propose(parent,schema,boundary,tmp_path/'mutation',namespace='mutation1',seed=9,method='mutation',
        sampler='mutation',seed_bank=tmp_path/'bank',counts=[12],attempts_per_cell=8,queries_per_cell=4,device='cpu')
    rows=adaptive.read_csv(tmp_path/'mutation/attempts.csv')
    assert all(1<=int(r['mutation_swaps'])<=3 and int(r['mutation_seed_row'])>=0 for r in rows)
    with np.load(tmp_path/'mutation/raw_attempts.npz') as z:assert np.all(z['occupancy'].sum(1)==12)


def test_raw_oracle_verifier_rejects_pre_freeze_scores_and_tampering(tmp_path):
    from cathode_cvae import pool_scoring
    from cathode_cvae.mlip.registry import MODELS
    parts=fixture(tmp_path);parent,_,_,_=parts;proposal(tmp_path,parts,'stage')
    stage=tmp_path/'stage';root=tmp_path/'oracle'
    pool_scoring.register_pool(root,stage/'candidates.npz',parent,models=('uma',),batch_size=64)
    registry=json.loads((root/'registry.json').read_text());task=registry['tasks'][0];tid=task['task_id']
    for child in ('scores/uma','workers',f'claims/{tid}'):(root/child).mkdir(parents=True,exist_ok=True)
    freeze=json.loads((stage/'receipt.json').read_text())['created_unix']
    adaptive.write_json(root/'workers/synthetic.json',dict(model='uma',head=MODELS['uma']['head'],
        package_version=MODELS['uma']['version'],dtype='float32',tf32=False,
        checkpoint_hashes={Path(k).name:v for k,v in MODELS['uma']['files'].items()}))
    adaptive.write_json(root/f'claims/{tid}/owner.json',dict(worker='synthetic',started_unix=freeze+1))
    with np.load(stage/'candidates.npz') as z:ids=z['structure_ids'];n=z['n_target']
    energy=np.arange(len(ids),dtype=float);score=root/f'scores/uma/{tid}.npz'
    np.savez_compressed(score,structure_ids=ids,status=np.full(len(ids),'success'),energy_eV=energy,
        n_atoms=n,n_Li=n,force_component_rms_eV_A=np.zeros(len(ids)),force_max_eV_A=np.zeros(len(ids)),
        stress_eV_A3=np.zeros((len(ids),6)),seconds=np.zeros(len(ids)))
    adaptive.write_json(score.with_suffix('.json'),dict(task=task,worker='synthetic',status='complete',
        output_sha256=sha256(score),rows=[dict(structure_id=str(sid),status='success',energy_eV=float(e),
            n_atoms=int(nn),n_Li=int(nn)) for sid,e,nn in zip(ids,energy,n)]))
    assert pool_scoring.collect(root,root/'collected')['status']=='PASS'
    result=adaptive.verify_oracle([stage],root,root/'collected/receipt.json',tmp_path/'verified.json')
    assert result['status']=='VERIFIED_FRESH_ADAPTIVE_ORACLE' and result['unique_structures']==12
    owner=root/f'claims/{tid}/owner.json';owner.write_text(json.dumps(dict(worker='synthetic',started_unix=freeze-1)))
    with pytest.raises(ValueError,match='Task did not start'):
        adaptive.verify_oracle([stage],root,root/'collected/receipt.json',tmp_path/'invalid.json')

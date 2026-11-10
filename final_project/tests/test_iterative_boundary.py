"""Small synthetic lineage/boundary checks; no scoring or model execution."""
import json
import shutil
import numpy as np
import pytest
from cathode_cvae.data import sha256
from cathode_cvae.iterative_boundary import binding, freeze_history, freeze_manifest
from test_iterative_acquisition import fixture_data, generation, write_json


def setup_boundary(d):
    out=d['root']/'boundary';out.mkdir()
    write_json(out/'receipt.json', {'status':'FROZEN_DEVELOPMENT_KEY_ONLY_BOUNDARY',
        'parent':binding(d['parent']), 'history':binding(d['known']), 'protected':binding(d['protected'])})
    return out


def setup_generation(d):
    generated=generation(d,d['base'])
    shutil.copyfile(generated/'primary_candidates.npz',generated/'candidates.npz')
    rp=generated/'receipt.json';r=json.loads(rp.read_text())
    r['outputs']['candidates.npz']=sha256(generated/'candidates.npz')
    r['no_energy_filtering']=True
    rp.write_text(json.dumps(r))
    return generated


def test_history_protects_all_generated_and_preserves_inherited_train(fixture_data):
    d=fixture_data;b=setup_boundary(d);g=setup_generation(d);out=d['root']/'expanded'
    r=freeze_history(b,d['base'],[g],[],out)
    protected=set(json.loads((out/'protected.json').read_text()))
    assert r['inherited_train_protected_overlap']>=1
    with np.load(g/'candidates.npz') as z: assert len(protected)>=len(z['masks'])
    assert r['no_protected_energy_members_read']
    with pytest.raises(FileExistsError): freeze_history(b,d['base'],[g],[],out)
    (g/'candidates.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'): freeze_history(b,d['base'],[g],[],d['root']/'tampered')


def test_round2_manifest_binds_merged_data_and_fresh_namespace(fixture_data):
    d=fixture_data;b=setup_boundary(d)
    merge=d['root']/'merge.json'
    write_json(merge, {'status':'MERGED_ITERATIVE_ROUND','round':1,
        'outputs':{'training.npz':sha256(d['base'])},'origin_data_sha256':sha256(d['base']),
        'inputs':{'parent':binding(d['parent']),'schema':binding(d['schema'])}})
    old=d['root']/'prior.json';write_json(old,{'namespace':'old'})
    plan=d['root']/'plan.json'; write_json(plan,{'status':'DECLARED_ITERATIVE_DEVELOPMENT',
        'purpose':'assessment','namespace':'new','previous_manifests':[str(old)],'counts':[24],
        'requested_bins':['low'],'count_per_group':3,'models':[{'method':'mutation','replicate_id':7,'sampler':'mutation'}]})
    out=d['root']/'manifest.json'
    m=freeze_manifest(plan,b,d['parent'],d['base'],d['schema'],2,merge,out)
    assert m['inputs']['data']['sha256']==sha256(d['base'])
    assert m['previous_merge_receipt']['sha256']==sha256(merge)
    assert m['expected_primary_attempts']==3
    prior=json.loads(old.read_text());prior['namespace']='new';old.write_text(json.dumps(prior))
    with pytest.raises(ValueError,match='duplicates'): freeze_manifest(plan,b,d['parent'],d['base'],d['schema'],2,merge,d['root']/'bad.json')
    merge.write_text(json.dumps({'status':'MERGED_ITERATIVE_ROUND','round':1,'outputs':{'training.npz':'bad'},
        'inputs':{'parent':binding(d['parent']),'schema':binding(d['schema'])}}))
    with pytest.raises(ValueError,match='data bytes'): freeze_manifest(plan,b,d['parent'],d['base'],d['schema'],2,merge,d['root']/'bad2.json')

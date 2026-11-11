"""Synthetic fixed-bank mutation pairing; no MLIP, training, or scheduler jobs."""
import json
import numpy as np
import pytest
from cathode_cvae.data import sha256
from cathode_cvae.iterative_boundary import binding
from cathode_cvae.fixed_seed_mutation import freeze_bank,generate
from cathode_cvae.iterative_acquisition import stage
from test_iterative_acquisition import fixture_data,write_json


def manifest(d,data,path,namespace='round1'):
    schema=json.loads(d['schema'].read_text())
    boundary=d['root']/'manifest_boundary.json'
    if not boundary.exists(): write_json(boundary,{'history':binding(d['known']),'protected':binding(d['protected'])})
    write_json(path,{'status':'FROZEN_TEST','namespace':namespace,'diagnostics':False,
      'inputs':{n:binding(p) for n,p in [('parent',d['parent']),('data',data),('schema',d['schema']),('history',d['known'])]},
      'boundary_receipt':binding(boundary),'expected_condition_schema_id':schema['identity'],'counts':[24],'requested_bins':['low'],
      'count_per_group':8,'expected_primary_attempts':8,
      'models':[{'method':'mutation','replicate_id':7,'sampler':'mutation'}]})


def test_same_bank_produces_literal_matched_edits_and_stages(fixture_data):
    d=fixture_data; bank=d['root']/'bank'
    freeze_bank(d['parent'],d['base'],d['schema'],d['protected'],[24],['low'],bank)
    with np.load(d['base']) as z: arrays={k:z[k] for k in z.files}
    arrays['weight']=arrays['weight']*2
    second=d['root']/'other_data.npz';np.savez_compressed(second,**arrays)
    m1=d['root']/'m1.json';m2=d['root']/'m2.json';manifest(d,d['base'],m1);manifest(d,second,m2)
    one=d['root']/'g1';two=d['root']/'g2'
    generate(m1,d['parent'],d['base'],d['schema'],d['known'],d['protected'],bank,one)
    generate(m2,d['parent'],second,d['schema'],d['known'],d['protected'],bank,two)
    with np.load(one/'raw_attempts.npz') as a, np.load(two/'raw_attempts.npz') as b:
        for name in ('occupancy','seed_occupancy','remove_uniforms','add_uniforms'):
            assert np.array_equal(a[name],b[name])
    result=stage(parent_path=d['parent'],base_data_path=d['base'],schema_path=d['schema'],generation_dir=one,
      known_histories=[d['known']],protected_histories=[d['protected']],method='mutation',replicate='7',
      round_index=1,role_seed=9,counts=[24],bins=['low'],attempts_per_cell=8,queries_per_cell=4,
      val_fraction=0,output_dir=d['root']/'stage')
    assert result['attempts']==8
    changed=d['root']/'changed.json';manifest(d,d['base'],changed,'round2')
    g3=d['root']/'g3';generate(changed,d['parent'],d['base'],d['schema'],d['known'],d['protected'],bank,g3)
    with np.load(one/'raw_attempts.npz') as a, np.load(g3/'raw_attempts.npz') as b:
        assert not np.array_equal(a['remove_uniforms'],b['remove_uniforms'])


def test_rejects_missing_train_seed_and_tampered_bank(fixture_data):
    d=fixture_data;bank=d['root']/'bank';freeze_bank(d['parent'],d['base'],d['schema'],d['protected'],[24],['low'],bank)
    with np.load(d['base']) as z: arrays={k:z[k] for k in z.files}
    arrays['split']=arrays['split'].copy();arrays['split'][:10]='val'
    modified=d['root']/'modified.npz';np.savez_compressed(modified,**arrays)
    m=d['root']/'m.json';manifest(d,modified,m)
    with pytest.raises(ValueError,match='missing/changed'):generate(m,d['parent'],modified,d['schema'],d['known'],d['protected'],bank,d['root']/'g')
    (bank/'bank.npz').write_bytes(b'tamper')
    with pytest.raises(ValueError,match='hash mismatch'):generate(m,d['parent'],modified,d['schema'],d['known'],d['protected'],bank,d['root']/'h')


def test_bank_seed_newly_protected_blocks_generation(fixture_data):
    d=fixture_data;bank=d['root']/'bank';freeze_bank(d['parent'],d['base'],d['schema'],d['protected'],[24],['low'],bank)
    with np.load(bank/'bank.npz') as z: seed=z['occupancy'][0]
    key=''.join(map(str,1-seed)); protected=json.loads(d['protected'].read_text());protected.append(key)
    d['protected']=d['root']/'new_protected.json'; d['protected'].write_text(json.dumps(protected))
    m=d['root']/'m.json';manifest(d,d['base'],m)
    with pytest.raises(ValueError,match='newly protected'):
        generate(m,d['parent'],d['base'],d['schema'],d['known'],d['protected'],bank,d['root']/'g')

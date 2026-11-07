"""Energy conditions are train-fitted, identity-checked, and explicit about support."""
import copy
import json

import numpy as np
import pytest
import torch

from cathode_cvae.conditions import (fit_schema, classify, encode_requests,
    encode_observations, supported_bins, schema_identity, save_schema, load_schema)
from cathode_cvae.learning.generate_mask import load_model, resolve_request, attempt_records
from cathode_cvae.learning.models import make_model


def example():
    n = np.full(12, 2)
    energy = np.r_[np.arange(10, dtype=float), -100., 100.]
    split = np.array(['train']*10 + ['val', 'test'])
    return fit_schema(n, energy, split, n_sites=4, provenance={'teacher':'synthetic'})


def test_edges_and_identity_do_not_use_heldout_energies():
    schema = example()
    n = np.full(12, 2)
    split = ['train']*10 + ['val', 'test']
    other = fit_schema(n, np.r_[np.arange(10), -1e9, 1e9], split, n_sites=4,
                       provenance={'teacher':'synthetic'})
    assert schema_identity(other) == schema_identity(schema)
    assert schema['compositions']['2']['edges_eV_atom'] == [1., 7.]
    assert classify(schema, [2]*5, [-1., 1., 1.01, 7., 7.01]).tolist() == ['low','low','middle','middle','high']
    assert schema['compositions']['2']['bin_counts'] == dict(low=2,middle=6,high=2)


def test_ties_and_sparse_compositions_are_not_forced_into_three_bins():
    schema = fit_schema([0,0,0,1,1], [4.,4.,4.,2.,3.], ['train']*5, n_sites=2)
    assert supported_bins(schema, 0) == ['low']
    assert schema['compositions']['0']['coincident_edges']
    assert schema['compositions']['0']['empty_bins'] == ['middle','high']
    assert schema['compositions']['0']['merged_bins'] == {}
    with pytest.raises(ValueError, match='no training support'):
        encode_requests(schema, 0, 'high')
    with pytest.raises(ValueError, match='absent'):
        encode_requests(schema, 2, 'low')
    # A held-out observation can genuinely fall outside supported training bins;
    # evaluating its conditional likelihood is distinct from supporting requests.
    assert encode_observations(schema, 0, 5., mode='multibin').tolist() == [0.,0.,0.,1.]


def test_encodings_keep_binary_and_composition_controls_functional():
    schema = example()
    assert encode_requests(schema, 2, 'middle').tolist() == [.5,0.,1.,0.]
    assert encode_requests(schema, 2, 'low', mode='energy').tolist() == [.5,1.]
    assert encode_requests(schema, 2, 'high', mode='energy').tolist() == [.5,0.]
    assert encode_requests(schema, 2, 'middle', mode='composition').tolist() == [.5]
    with pytest.raises(ValueError, match='Unknown requested'):
        encode_requests(schema, 2, 'lowest-ever')
    with pytest.raises(ValueError, match='finite integers'):
        encode_requests(schema, 2.3, 'low')


def test_schema_roundtrip_and_tamper_detection(tmp_path):
    schema = example();path = tmp_path/'schema.json';save_schema(schema,path)
    assert load_schema(path) == schema
    modified = json.loads(path.read_text());modified['compositions']['2']['edges_eV_atom'][0] = -2.
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match='identity'):
        load_schema(path)


def test_requested_class_and_effective_binary_control_are_distinguished():
    schema = example()
    cfg = dict(n_sites=4,condition_dim=2,condition_stats={'2':{'q20_eV_atom':1.}})
    c, requested, effective, identity, _ = resolve_request(cfg,2,'middle',request_schema=schema)
    assert c.tolist() == [.5,0.] and requested == 'middle' and effective == 'rest'
    assert identity == schema_identity(schema)
    with pytest.raises(ValueError, match='different low thresholds'):
        resolve_request({**cfg,'condition_stats':{'2':{'q20_eV_atom':0.}}},2,'low',request_schema=schema)
    with pytest.raises(ValueError, match='conflicts'):
        resolve_request(cfg,2,'middle',binary_condition=1,request_schema=schema)


def test_old_and_new_checkpoint_loading_is_explicit(tmp_path):
    for mode, dimension in [('energy',2),('composition',1),('multibin',4)]:
        model = make_model('ar_cvae',n_sites=4,condition_dim=dimension)
        cfg = dict(kind='ar_cvae',n_sites=4,condition_dim=dimension,latent_dim=8)
        if mode == 'multibin':
            cfg.update(condition_mode=mode,condition_schema=example())
        path=tmp_path/(mode+'.pt');torch.save(dict(state_dict=model.state_dict(),configuration=cfg),path)
        loaded, actual = load_model(path,'cpu')
        assert loaded.condition_dim == dimension
        if mode == 'multibin':
            corrupt=copy.deepcopy(cfg);corrupt['condition_dim']=2
            bad=tmp_path/'bad.pt';torch.save(dict(state_dict=model.state_dict(),configuration=corrupt),bad)
            with pytest.raises(ValueError,match='dimension'):
                load_model(bad,'cpu')


def test_every_attempt_retains_request_and_acquisition_provenance():
    kwargs=dict(method='toy',fit_seed=3,sampling_seed=11,n_target=2,requested_bin='middle',
        effective_model_class='middle',schema_id='frozen-schema',checkpoint_sha256='checkpoint',
        parent_sha256='parent',geometry_protocol='deletion',temperature_K=433.15,
        allocation_policy='static-energy-exploration-0.1',allocation_weight=.3)
    rows=attempt_records(3,['same','same','other'],['0101','0101','0110'],**kwargs)
    assert len(rows)==3 and len({r['scoring_structure_id'] for r in rows})==2
    assert all(r['requested_bin']=='middle' and r['fit_seed']==3 and r['sampling_seed']==11
               and r['temperature_K']==433.15 and r['condition_schema_id']=='frozen-schema' for r in rows)
    failed=attempt_records(3,[],[],**kwargs,error='unsupported bin')
    assert len(failed)==3 and all(r['status']=='failed_request' and not r['scoring_structure_id'] for r in failed)


def test_custom_quantiles_are_named_separately_from_literal_q20_q80():
    schema=fit_schema([2]*10,np.arange(10),['train']*10,n_sites=4,quantiles=(.1,.9))
    entry=schema['compositions']['2']
    assert entry['edges_eV_atom']==[0.,8.]
    assert entry['q20_eV_atom']==1. and entry['q80_eV_atom']==7.

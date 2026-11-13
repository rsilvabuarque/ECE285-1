"""Synthetic next-round boundaries; no energies or MLIPs are used."""
import json
import numpy as np
import pytest
from cathode_cvae.data import sha256
from cathode_cvae.iterative_assessment import filter_assessment
from cathode_cvae.iterative_next_boundary import build
from test_iterative_acquisition import fixture_data
from test_iterative_assessment import make_bundle


def prepare(d):
    protected=json.loads(d['protected'].read_text())
    protected.append(''.join(map(str,1-d['masks'][10]))) # H0 VAL
    d['protected'].write_text(json.dumps(protected))
    boundary,_,raw,generator=make_bundle(d)
    with np.load(raw/'raw_attempts.npz') as z:occ=z['occupancy']
    masks=1-occ;ids=[f'mask48_{int("".join(map(str,x)),2):012x}' for x in masks]
    for name in ('candidates.npz','primary_candidates.npz'):
        np.savez_compressed(raw/name,masks=masks,structure_ids=np.asarray(ids),parent_sha256=np.asarray(sha256(d['parent'])))
    receipt_path=raw/'receipt.json';receipt=json.loads(receipt_path.read_text())
    for name in ('candidates.npz','primary_candidates.npz'):receipt['outputs'][name]=sha256(raw/name)
    receipt_path.write_text(json.dumps(receipt))
    safe=d['root']/'safe';filter_assessment(raw,boundary,generator,safe)
    return boundary,raw,safe


def test_only_newly_assessable_orbits_are_protected(fixture_data):
    d=fixture_data;b,raw,safe=prepare(d)
    with np.load(d['base']) as z:original={k:z[k].copy() for k in ('occupancy','split')}
    out=d['root']/'next';r=build(b,d['base'],[(raw,safe)],out)
    known=set(json.loads((out/'known_history.json').read_text()));protected=set(json.loads((out/'protected.json').read_text()))
    old_protected=set(json.loads(d['protected'].read_text()))
    assert old_protected<=protected and r['newly_protected_assessable_orbits']==2
    assert r['newly_protected_h0_train_orbits']==0
    h0_first=''.join(map(str,1-d['masks'][0]));assert h0_first in known and h0_first not in protected
    with np.load(d['base']) as z:
        assert np.array_equal(z['occupancy'],original['occupancy']) and np.array_equal(z['split'],original['split'])
    assert r['no_protected_energy_members_read']


def test_rejects_tampered_safe_pool_and_inconsistent_attempts(fixture_data):
    d=fixture_data;b,raw,safe=prepare(d)
    path=safe/'candidates.npz'
    with np.load(path) as z: arrays={k:z[k] for k in z.files}
    arrays['masks']=arrays['masks'][:1];arrays['structure_ids']=arrays['structure_ids'][:1]
    np.savez_compressed(path,**arrays)
    receipt=safe/'receipt.json';data=json.loads(receipt.read_text());data['outputs']['candidates.npz']=sha256(path)
    receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='differs from eligible'):
        build(b,d['base'],[(raw,safe)],d['root']/'next')


def test_acquisition_raw_added_as_known_only(fixture_data):
    import shutil
    d=fixture_data;b,raw,safe=prepare(d)
    acquisition=d['root']/'acquisition';shutil.copytree(raw,acquisition)
    result=build(b,d['base'],[(raw,safe)],d['root']/'next',[acquisition])
    assert result['proposal_sources'][-1]['scope']=='known_only'
    assert result['newly_protected_assessable_orbits']==2
    with pytest.raises(ValueError,match='already paired'):
        build(b,d['base'],[(raw,safe)],d['root']/'bad',[raw])

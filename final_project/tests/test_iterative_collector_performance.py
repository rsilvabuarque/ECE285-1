"""Synthetic regression for collected-NPZ access and raw-score provenance."""
import json
from collections import Counter
import numpy as np
import pytest
from cathode_cvae.iterative_comparison import collect_oracle
from test_iterative_acquisition import fixture_data,generation,run_stage,oracle


def setup_oracle(d):
    stage=d['root']/'stage';run_stage(d,generation(d,d['base']),stage)
    root,collection,_=oracle(d,stage)
    reg=json.loads((root/'registry.json').read_text())
    ids={item['structure_id'] for item in reg['physical_keys'].values()}
    after=json.loads((stage/'stage_receipt.json').read_text())['created_unix']
    return root,collection,ids,reg['parent_sha256'],after


def test_collected_members_loaded_once_with_equivalent_receipt(fixture_data,monkeypatch):
    root,collection,ids,parent,after=setup_oracle(fixture_data)
    expected=collect_oracle(root,collection,ids,parent,'uma',after)
    collected=str(collection.parent/'uma.npz');seen=Counter()
    original=np.lib.npyio.NpzFile.__getitem__
    def track(archive,key):
        if str(archive.zip.filename)==collected:seen[key]+=1
        return original(archive,key)
    monkeypatch.setattr(np.lib.npyio.NpzFile,'__getitem__',track)
    actual=collect_oracle(root,collection,ids,parent,'uma',after)
    assert actual==expected
    assert seen==Counter({k:1 for k in ('structure_ids','status','provenance','energy_eV','n_atoms','n_Li')})


@pytest.mark.parametrize('field,value',[
    ('status','failed'),('provenance','cached'),('energy_eV',123.0),('n_atoms',5),('n_Li',1),
    ('structure_ids','mask48_ffffffffffff')])
def test_collected_tampering_still_rejected(fixture_data,field,value):
    root,collection,ids,parent,after=setup_oracle(fixture_data)
    path=collection.parent/'uma.npz'
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    arrays[field][0]=value
    np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError,match='Collector'):
        collect_oracle(root,collection,ids,parent,'uma',after)

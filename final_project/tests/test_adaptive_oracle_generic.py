import json
from pathlib import Path
import numpy as np
import pytest
from cathode_cvae.adaptive_oracle import register_pool


def inputs(tmp_path):
 p=tmp_path/'parent.npz'; np.savez(p,numbers=np.array([3,3,27]),li_indices=np.array([0,1]),permutations=np.array([[0,1],[1,0]]))
 return p

def test_generic_canonical_queue_and_as_written_are_separate(tmp_path):
 p=inputs(tmp_path);x=tmp_path/'pool.npz';np.savez(x,structure_ids=np.array(['mask2_1']),masks=np.array([[0,1]],np.uint8))
 r=register_pool(tmp_path/'canonical',x,p,models=['uma']);assert r['added_geometry_model_pairs']==1
 bad=tmp_path/'bad.npz';np.savez(bad,structure_ids=np.array(['mask2_2']),masks=np.array([[1,0]],np.uint8))
 with pytest.raises(ValueError,match='canonical'):register_pool(tmp_path/'badroot',bad,p,models=['uma'])
 raw=tmp_path/'orient.npz';np.savez(raw,structure_ids=np.array(['orientation2_0']),masks=np.array([[1,0]],np.uint8))
 r=register_pool(tmp_path/'written',raw,p,models=['mace'],as_written=True)
 assert r['added_geometry_model_pairs']==1
 reg=json.loads((tmp_path/'written/registry.json').read_text());assert reg['as_written']
 assert reg['tasks'][0]['expected_structure_ids']==['orientation2_0']
 with pytest.raises(ValueError,match='mix'):register_pool(tmp_path/'written',x,p,models=['mace'])

"""Two prespecified SP scopes share exact work without dropping source failures."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/overnight_20261108"))
import paired_common as common
import paired_mace_union as union


def test_union_keeps_scope_denominators_and_deduplicates_exact_coordinates(tmp_path):
    numbers=np.array([8,3]);xyz=np.array([[0.,0.,0.],[1.23456789123,2.,3.]]);cell=np.eye(3)*10
    geometry=common.geometry_hash(numbers,xyz,cell)
    inputs=[];before={}
    for scope in ["hash96","pilot24"]:
        directory=tmp_path/scope;directory.mkdir();inputs.append(directory)
        common.atomic_npz(directory/"endpoints.npz",structure_ids=["same"],offsets=[0,2],numbers=numbers,
            positions=xyz,cell=[cell],parent_indices=[0,1],parent_sha256=np.array("a"*64))
        common.atomic_json(directory/"manifest.json",dict(parent_sha256="a"*64,primary_context_hash="same_protocol",
            prespecified=2,exported=1,inference_npz_sha256=common.sha256(directory/"endpoints.npz"),
            records=[dict(structure_id="same",exported=True,literal_geometry_hash=geometry),
                     dict(structure_id="failed_"+scope,exported=False,status="interrupted")]))
        before[scope]=common.sha256(directory/"manifest.json")
    result=union.main(["--exports",*map(str,inputs),"--output-dir",str(tmp_path/"union")])
    assert result["unique_physical_coordinates"]==1
    assert all(v["prespecified"]==2 and v["usable"]==1 for v in result["scopes"].values())
    for directory in inputs:assert common.sha256(directory/"manifest.json")==before[directory.name]
    with np.load(tmp_path/"union/endpoints.npz") as z:assert np.array_equal(z["positions"],xyz)
    for scope in ["hash96","pilot24"]:
        rows=json.loads((tmp_path/"union"/("scope_"+scope+".json")).read_text())["records"]
        assert len(rows)==2 and rows[0]["scoring_structure_id"]=="same" and rows[1]["scoring_structure_id"] is None

"""The final-unsealing authorization is bound to actual frozen artifacts."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def test_changed_artifact_blocks_test_unsealing(tmp_path):
    scripts=Path(__file__).resolve().parents[1]/"scripts"/"overnight_20261108"
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location("paired_test_audit_test",scripts/"paired_test_audit.py")
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        values={"joint_protocol":{"no_more_model_selection":True},"independent_raw_audit":{"status":"PASS"},
                "static_receipt":{"status":"FROZEN_BEFORE_SCORING"},"paired_receipt":{"status":"FROZEN_PROPOSALS_AND_ENDPOINT_TASKS_NO_SCFS"}}
        paths={}
        gate={"status":"APPROVED_FINAL_SCORING_AFTER_ALL_PROPOSALS_AND_INDEPENDENT_RAW_AUDIT","created_UTC":"synthetic-test"}
        for name,value in values.items():
            path=tmp_path/(name+".json");path.write_text(json.dumps(value));paths[name]=path;gate[name+"_sha256"]=m.sha256(path)
        g=tmp_path/"gate.json";g.write_text(json.dumps(gate))
        order=[paths[k] for k in values]
        assert m.check_gate(g,*order)["sha256"]==m.sha256(g)
        paths["independent_raw_audit"].write_text('{"status":"FAIL"}')
        with pytest.raises(ValueError,match="changed"):
            m.check_gate(g,*order)
        assert m.initial_energy_or_unknown({}, {"initial_energy_eV":None,"status":"interrupted"}) is None
        with pytest.raises(ValueError,match="no matching raw array"):
            m.initial_energy_or_unknown({}, {"initial_energy_eV":-1})
    finally:
        sys.path.remove(str(scripts))

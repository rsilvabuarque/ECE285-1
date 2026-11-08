import importlib.util
from pathlib import Path
import sys

import pytest


def test_pairing_is_by_replicate_count_request_and_keeps_cell_denominator():
    scripts=Path(__file__).resolve().parents[1]/"scripts"/"overnight_20261108"
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location("paired_contrasts_test",scripts/"paired_contrasts.py")
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        a=[dict(replicate=str(rep),n_target=n,requested_bin="low",attempts=64,y=33/64) for rep in range(3) for n in [12,24,36]]
        b=[dict(r,y=32/64) for r in reversed(a)]
        out=m.paired_difference(a,b,"y","LOW")
        assert all(r["delta_percentage_points"]==1.5625 and r["attempts_per_method"]==192 for r in out)
        with pytest.raises(ValueError,match="Unmatched"):
            m.paired_difference(a,b[:-1],"y","LOW")
        b[0]["attempts"]=63
        with pytest.raises(ValueError,match="denominator"):
            m.paired_difference(a,b,"y","LOW")
    finally:sys.path.remove(str(scripts))

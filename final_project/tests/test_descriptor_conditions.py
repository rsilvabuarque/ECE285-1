"""Structural-condition support and held-out leakage tests, synthetic values."""

import copy

import numpy as np
import pytest

from cathode_cvae.descriptor import (classify, encode_observations, encode_requests,
    fit_descriptor_schema, load_schema, save_schema, validate_schema)


def test_joint_schema_uses_train_only_and_keeps_boundary_ties(tmp_path):
    n = np.full(8, 12)
    d = np.array([0., .1, .1, .2, .3, .4, .5, .6])
    energy = np.array(["low", "low", "middle", "middle", "high", "high", "low", "high"])
    role = np.array(["train"] * 6 + ["val", "test"])
    a = fit_descriptor_schema(d, n, energy, role, parent_hash="parent", graph_hash="graph")
    d2 = d.copy(); d2[-2:] = [0., 1.]
    e2 = energy.copy(); e2[-2:] = ["high", "low"]
    assert a == fit_descriptor_schema(d2, n, e2, role, parent_hash="parent", graph_hash="graph")
    assert a["compositions"]["12"]["boundary"] == .1
    assert a["compositions"]["12"]["joint_support"]["low"] == {"low": 2, "high": 0}
    assert classify(a, n, d).tolist() == ["low", "low", "low", "high", "high", "high", "high", "high"]
    out = tmp_path / "descriptor.json"; save_schema(a, out)
    assert load_schema(out) == a
    with pytest.raises(FileExistsError):
        save_schema(a, out)
    b = copy.deepcopy(a); b["compositions"]["12"]["boundary"] += .01
    with pytest.raises(ValueError, match="identity"):
        validate_schema(b)


def test_unsupported_generation_fails_but_heldout_outcome_is_retained():
    n = np.full(4, 12)
    d = np.array([0., .1, .2, .3])
    energy = np.array(["low", "low", "high", "high"])
    s = fit_descriptor_schema(d, n, energy, np.array(["train"] * 4))
    assert encode_requests(s, 12, "low", "low").tolist() == [1., 0.]
    with pytest.raises(ValueError, match="unsupported joint"):
        encode_requests(s, 12, "low", "high")
    # An adverse validation/test combination remains an encoded observation.
    assert encode_observations(s, 12, "low", .3).tolist() == [0., 1.]
    with pytest.raises(ValueError, match="composition"):
        encode_requests(s, 13, "low", "low")


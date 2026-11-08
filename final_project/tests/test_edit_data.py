"""Protect paired-orientation lineage and nested editor role boundaries."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "scripts/conditioning_20261107/prepare_edit_data.py"
spec = importlib.util.spec_from_file_location("prepare_edit_data", path)
prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)


def test_exact_edit_reconstruction_and_no_count_repair():
    seed = np.asarray([1, 1, 0, 0, 1, 0], np.uint8)
    child = np.asarray([0, 1, 1, 0, 1, 0], np.uint8)
    prep.validate_edit(seed, child, [0], [2], 1)
    with pytest.raises(ValueError, match="does not reproduce"):
        prep.validate_edit(seed, np.asarray([0, 1, 0, 1, 1, 0]), [0], [2], 1)
    with pytest.raises(ValueError, match="disagree"):
        prep.validate_edit(seed, child, [2], [0], 1)
    with pytest.raises(ValueError, match="Invalid distinct"):
        prep.validate_edit(seed, child, [0, 0], [2, 3], 2)


def test_joint_symmetry_dedup_preserves_relative_seed_child_orientation():
    permutations = np.asarray([np.roll(np.arange(6), k) for k in range(6)])
    seed = np.asarray([1, 1, 0, 0, 0, 0], np.uint8)
    child = np.asarray([1, 0, 1, 0, 0, 0], np.uint8)
    key, witness = prep.joint_key(seed, child, permutations)
    for inverse in np.argsort(permutations, axis=1):
        assert prep.joint_key(seed[inverse], child[inverse], permutations)[0] == key
    # Rotating only the child retains each individual symmetry orbit but changes
    # the edit event relative to the seed; it must not be deduplicated away.
    independent = np.roll(child, 2)
    assert prep.joint_key(seed, independent, permutations)[0] != key
    assert 0 <= witness < len(permutations)


def test_components_join_seed_and_child_roles_transitively():
    seeds = ["A", "B", "D", "F", "H"]
    children = ["B", "C", "E", "G", "I"]
    component, split = prep.assign_components(seeds, children, 20261107)
    assert component[0] == component[1]
    assert split[0] == split[1]
    roles = {}
    for seed, child, role in zip(seeds, children, split):
        for key in [seed, child]:
            assert roles.setdefault(key, role) == role
    again = prep.assign_components(seeds[::-1], children[::-1], 20261107)
    assert list(again[0][::-1]) == list(component)
    assert list(again[1][::-1]) == list(split)


def test_scalar_oracle_identity_duplicates_are_rejected():
    scores = dict(structure_ids=np.asarray(["same", "same"]), energy_eV=np.asarray([-1., -2.]),
        n_atoms=np.asarray([2, 2]))
    with pytest.raises(ValueError, match="duplicate"): prep.score_lookup(scores)

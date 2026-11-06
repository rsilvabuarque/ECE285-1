import numpy as np
from cathode_cvae.data import canonical_mask, deletion_atoms


def test_canonical_witness_and_species_independent_deletion():
    # Synthetic Na host: the removable atoms deliberately are not the first rows.
    parent = dict(numbers=np.array([8, 11, 8, 11]),
                  positions_A=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0.]]),
                  cell_A=np.eye(3) * 5, target_indices=np.array([1, 3]))
    perms = np.array([[0, 1], [1, 0]])
    key, witness = canonical_mask(np.array([0, 1]), perms)
    assert key == "01"
    assert np.array_equal(np.array([0, 1])[witness], [1, 0])
    assert canonical_mask(np.array([1, 0]), perms)[0] == key
    atoms, slots = deletion_atoms(parent, np.array([0, 1]))
    assert atoms.get_chemical_formula() == "NaO2"
    assert slots.tolist() == [0, 2, 3]


def test_long_mask_does_not_overflow():
    o = np.array([0] + [1] * 143)
    p = np.stack([np.arange(144), np.roll(np.arange(144), 1)])
    key, _ = canonical_mask(o, p)
    assert len(key) == 144 and key.count("1") == 1


def test_incomplete_symmetry_action_is_rejected():
    import pytest
    from cathode_cvae.data import validate_permutation_group
    assert validate_permutation_group([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    with pytest.raises(ValueError, match="not closed"):
        validate_permutation_group([[0, 1, 2], [1, 2, 0]])


def test_existing_labels_export_preserves_protected_splits(tmp_path):
    import csv
    from cathode_cvae.data import prepare_learning, sha256
    parent = tmp_path / "parent.npz"
    np.savez(parent, numbers=[8, 11, 11, 11], positions_A=np.zeros((4, 3)), cell_A=np.eye(3),
             target_indices=[1, 2, 3], permutations=[[0, 1, 2]])
    labels = tmp_path / "labels.npz"
    np.savez(labels, occupancy=np.eye(3, dtype=np.uint8), structure_ids=["a", "b", "c"],
             n_Li=[1, 1, 1], n_atoms=[2, 2, 2], energy_eV=[-2., -1., 0.], parent_sha256=sha256(parent))
    splits = tmp_path / "splits.csv"
    with splits.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["structure_id", "parent", "canonical_mask", "split"])
        w.writeheader()
        for sid, key, split in zip(["a", "b", "c"], ["011", "101", "110"], ["train", "val", "test"]):
            w.writerow(dict(structure_id=sid, parent="synthetic", canonical_mask=key, split=split))
    output = tmp_path / "learning.npz"
    r = prepare_learning(labels, parent, splits, output)
    assert r["split_counts"] == dict(train=1, val=1, test=1)
    with np.load(output) as z:
        assert z["split"].tolist() == ["train", "val", "test"]
        assert np.array_equal(z["energy_eV_atom"], [-1., -.5, 0.])


def test_prepare_parent_handles_a_non_li_target(tmp_path):
    # Synthetic periodic binary cell tests mapping, not material stability.
    from ase import Atoms
    from ase.io import write
    from cathode_cvae.data import prepare_parent, validate_permutation_group
    atoms = Atoms("NaCl", scaled_positions=[[0, 0, 0], [.5, .5, .5]], cell=np.eye(3)*4, pbc=True)
    source = tmp_path / "POSCAR"
    write(source, atoms, format="vasp")
    output = tmp_path / "parent.npz"
    result = prepare_parent(source, "Na", output, 1e-5)
    assert result["target_sites"] == 1 and result["permutation_group_verified"]
    with np.load(output) as z:
        assert z["target_symbol"].item() == "Na"
        assert z["target_indices"].tolist() == [0]
        assert validate_permutation_group(z["permutations"])
        assert len(z["permutation_operation_indices"]) == len(z["permutations"])

"""Check data/units/dispatch contracts without downloading MLIP checkpoints."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pytest

from cathode_cvae.mlip import cli
from cathode_cvae.mlip.calculators import evaluate
from cathode_cvae.mlip.checkpoints import verify


def test_uma_device_index_is_selected_before_sdk_load(monkeypatch):
    import torch
    from cathode_cvae.mlip.calculators import uma_predict_device
    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    assert uma_predict_device(torch.device("cuda:3")) == "cuda"
    assert selected == [3]
    assert uma_predict_device(torch.device("cpu")) == "cpu"


def parent_fixture(path):
    # Deliberately interleave target atoms: the target sites are not the first rows.
    np.savez(path, numbers=[27, 3, 8, 3], positions_A=np.arange(12).reshape(4, 3) / 10,
             cell_A=np.eye(3) * 5, target_indices=[1, 3])


def test_mask_site_mapping_and_displacement_geometry(tmp_path):
    parent = tmp_path / "parent.npz"
    parent_fixture(parent)
    masks = tmp_path / "masks.npz"
    np.savez(masks, occupancy=[[0, 1]], structure_ids=["one"],
             displacement_A=np.ones((1, 4, 3)) * .2, cell_A=np.eye(3)[None] * 6)
    identity, atoms, source = next(cli.read_inputs([masks], parent))
    assert identity == "one"
    assert atoms.numbers.tolist() == [27, 8, 3]
    expected = np.arange(12).reshape(4, 3) / 10 * 6 / 5 + .2
    np.testing.assert_allclose(atoms.positions, expected[[0, 2, 3]])
    np.testing.assert_allclose(atoms.cell.array, np.eye(3) * 6)
    assert source["sha256"]


def test_mask_geometry_parent_slot_alias(tmp_path):
    parent = tmp_path / "parent.npz"
    parent_fixture(parent)
    data = tmp_path / "predicted.npz"
    coordinates = np.arange(12).reshape(1, 4, 3) / 5
    np.savez(data, masks=[[1, 0]], positions_parent_A=coordinates, structure_ids=["predicted"])
    _, atoms, _ = next(cli.read_inputs([data], parent))
    np.testing.assert_allclose(atoms.positions, coordinates[0, [0, 2, 3]])


def test_parent_hash_mismatch_even_when_dimensions_match(tmp_path):
    from cathode_cvae.mlip.checkpoints import sha256
    correct = tmp_path / "correct.npz"
    parent_fixture(correct)
    wrong = tmp_path / "wrong.npz"
    # Same geometry/site count, a different host chemical species.
    with np.load(correct) as archive:
        modified = {key: archive[key] for key in archive.files}
    modified["numbers"] = np.array([28, 3, 8, 3])
    np.savez(wrong, **modified)
    data = tmp_path / "predicted.npz"
    np.savez(data, occupancy=[[0, 1]], structure_ids=["predicted"], parent_sha256=np.asarray(sha256(correct)))
    assert len(list(cli.read_inputs([data], correct))) == 1
    with pytest.raises(ValueError, match="Parent SHA256 mismatch"):
        list(cli.read_inputs([data], wrong))


@pytest.mark.parametrize("field", ["presence", "atom_presence"])
def test_presence_must_agree_with_occupancy(tmp_path, field):
    parent = tmp_path / "parent.npz"
    parent_fixture(parent)
    data = tmp_path / "bad_presence.npz"
    np.savez(data, occupancy=[[0, 1]], structure_ids=["predicted"], **{field: [[1, 1, 1, 1]]})
    with pytest.raises(ValueError, match="disagrees with occupancy"):
        list(cli.read_inputs([data], parent))


def test_nonfinite_cell_is_rejected(tmp_path):
    parent = tmp_path / "parent.npz"
    parent_fixture(parent)
    data = tmp_path / "nonfinite.npz"
    cell = np.eye(3)
    cell[0, 0] = np.nan
    np.savez(data, occupancy=[[0, 1]], cell_A=cell, structure_ids=["predicted"])
    with pytest.raises(ValueError, match="finite, nonsingular"):
        list(cli.read_inputs([data], parent))


def test_duplicate_ids_are_not_silently_merged(tmp_path):
    parent = tmp_path / "parent.npz"
    parent_fixture(parent)
    data = tmp_path / "duplicates.npz"
    np.savez(data, occupancy=[[0, 1], [1, 0]], structure_ids=["same", "same"])
    with pytest.raises(ValueError, match="Repeated structure ID"):
        list(cli.read_inputs([data], parent))


def test_units_and_ragged_scores_roundtrip(tmp_path, monkeypatch):
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes

    class AnalyticCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": float((atoms.positions**2).sum() / 2),
                            "forces": -atoms.positions.copy(), "stress": np.array([1., 2., 3., 0, 0, 0]) / 160.21766208}

    calc = AnalyticCalculator()
    atoms = Atoms("LiCoO2", positions=np.arange(12).reshape(4, 3) / 10, cell=np.eye(3) * 5, pbc=True)
    energy, force, stress = evaluate(calc, atoms)
    metrics = cli.summarize(energy, force, stress, atoms)
    assert metrics["max_abs_principal_stress_GPa"] == pytest.approx(3)
    assert metrics["pressure_GPa"] == pytest.approx(-2)
    assert metrics["force_component_rms_eV_A"] == pytest.approx(np.sqrt(np.mean(atoms.positions**2)))
    assert metrics["n_target"] == 1
    from ase.io import write
    source = tmp_path / "input.extxyz"
    write(source, atoms)
    monkeypatch.setattr(cli, "make_calculator", lambda *args: (calc, {"model": "uma", "test_only": True}))
    output = tmp_path / "output"
    args = argparse.Namespace(output=output, model="uma", checkpoint=None, device="cpu", threads=1,
                              model_dir=None, command="score", input=[source], parent=None, target="Li", fail_fast=False)
    assert cli.run_calculations(args) == 0
    with np.load(output / "predictions.npz", allow_pickle=False) as data:
        np.testing.assert_allclose(data["forces_eV_A"], force)
        np.testing.assert_allclose(data["stress_eV_A3"][0], stress)
        assert data["energy_eV"][0] == pytest.approx(energy)
        assert data["offsets"].tolist() == [0, 4]
        assert data["n_Li"].tolist() == [1]
    assert json.loads((output / "run_receipt.json").read_text())["successes"] == 1


def test_checkpoint_corruption_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "weights.pt"
    source.write_bytes(b"not the pinned weights")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify("uma", source)


def test_target_species_is_inferred_from_new_parent(tmp_path, monkeypatch):
    from ase.calculators.calculator import Calculator, all_changes

    class ZeroCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]
        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = dict(energy=0., forces=np.zeros((len(atoms), 3)), stress=np.zeros(6))

    parent = tmp_path / "na_parent.npz"
    np.savez(parent, numbers=[11, 8, 8], positions_A=np.eye(3), cell_A=np.eye(3)*5,
             target_indices=[0], target_symbol="Na")
    source = tmp_path / "one.npz"
    np.savez(source, occupancy=[[1]], structure_ids=["one"])
    monkeypatch.setattr(cli, "make_calculator", lambda *args: (ZeroCalculator(), dict(test_only=True)))
    args = argparse.Namespace(output=tmp_path / "result", model="uma", checkpoint=None, device="cpu", threads=1,
        model_dir=None, command="score", input=[source], parent=parent, target=None, fail_fast=False)
    assert cli.run_calculations(args) == 0
    with np.load(args.output / "predictions.npz") as z:
        assert z["n_target"].tolist() == [1]
        assert "n_Li" not in z


def test_dispatch_uses_venv_path_not_symlink_target(tmp_path, monkeypatch):
    # Distinct venv Python symlinks often resolve to the same base executable.
    target = tmp_path / "backend-python"
    target.symlink_to(Path(cli.sys.executable).resolve())
    config = tmp_path / "environments.json"
    config.write_text(json.dumps({"interpreters": {"uma": str(target)}, "model_dir": str(tmp_path)}))
    monkeypatch.delenv("CATHODE_CVAE_IN_BACKEND", raising=False)
    calls = []
    monkeypatch.setattr(cli.subprocess, "call", lambda command, **kwargs: calls.append((command, kwargs)) or 0)
    args = argparse.Namespace(command="score", model="uma", environment_map=config, model_dir=None)
    assert cli.dispatch_if_needed(args, ["score", "--model", "uma"]) == 0
    assert calls[0][0][0] == str(target)
    assert calls[0][1]["env"]["CATHODE_CVAE_IN_BACKEND"] == "1"

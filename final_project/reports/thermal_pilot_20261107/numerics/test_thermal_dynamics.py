"""Tiny analytic/mocked numerical tests; these do not validate the UMA potential."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("thermal_dynamics", ROOT / "scripts/overnight_20261107/thermal_dynamics.py")
td = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(td)


def test_start_record_serializes_repeated_shard_paths(tmp_path):
    parent = tmp_path / "parent.npz"; parent.write_bytes(b"parent identity")
    structure = tmp_path / "final.npz"; structure.write_bytes(b"input identity")
    shards = [tmp_path / "shard0", tmp_path / "shard1"]
    args = SimpleNamespace(command="assemble-hessian", parent=parent, input=structure,
                           output=tmp_path / "assembled", basin_id="toy", shard=shards)
    td.start_record(args)
    recorded = json.loads((args.output / "receipt.json").read_text())
    assert recorded["arguments"]["shard"] == [str(p) for p in shards]
    assert recorded["parent_sha256"] == td.digest(parent)


def test_mass_weighted_translations_and_signed_instability():
    # Three identical Cartesian springs connect two masses. Relative frequencies
    # squared are k(1/m1+1/m2); common translations have zero eigenvalues.
    identity = np.eye(3)
    k = 2.
    raw = k * np.block([[identity, -identity], [-identity, identity]])
    arrays, stats = td.analyze_hessian(raw, np.array([2., 4.]))
    np.testing.assert_allclose(arrays["eigenvalues_internal_eV_A2_amu"], 1.5, atol=1e-14)
    np.testing.assert_allclose(arrays["translation_basis"].T @ arrays["internal_basis"], 0., atol=1e-14)
    assert stats["retained_internal_modes"] == 3
    assert stats["translation_residual_norm_eV_A2_amu"] < 1e-14
    bad, stats_bad = td.analyze_hessian(-raw, np.array([2., 4.]))
    assert stats_bad["strict_negative_internal_modes"] == 3
    assert np.all(bad["frequencies_signed_THz"] < 0)


def test_raw_antisymmetry_is_retained():
    raw = np.eye(6)
    raw[0, 1] = .1
    arrays, stats = td.analyze_hessian(raw, np.ones(2))
    assert arrays["hessian_raw_eV_A2"][0, 1] == .1
    assert arrays["hessian_sym_eV_A2"][0, 1] == .05
    assert stats["raw_antisymmetric_max_eV_A2"] == .1


def toy_parent(path):
    data = dict(numbers=np.array([3, 3, 27, 8]),
                positions_A=np.array([[1., 1., 1.], [3., 1., 1.], [1., 4., 1.], [1., 1., 5.]]),
                cell_A=np.eye(3) * 10, li_indices=np.array([0, 1]))
    np.savez(path, **data)
    return data


def test_periodic_hungarian_and_host_translation(tmp_path):
    from ase import Atoms
    parent = toy_parent(tmp_path / "parent.npz")
    indices = np.array([0, 2, 3])
    atoms = Atoms(numbers=parent["numbers"][indices], positions=parent["positions_A"][indices] + [10.2, .1, 0.], cell=parent["cell_A"], pbc=True)
    result = td.mapping_diagnostics(atoms, parent, indices)
    assert result["occupancy_hungarian"] == [1, 0]
    assert result["changed_occupancy_sites"] == 0
    assert result["host_displacement_rms_A"] < 1e-14
    np.testing.assert_allclose(result["assignment_distances_A"], 0., atol=1e-14)
    atoms.positions[0] += [2., 0., 0.]
    moved = td.mapping_diagnostics(atoms, parent, indices)
    assert moved["occupancy_hungarian"] == [0, 1]
    assert moved["changed_occupancy_sites"] == 2


def test_short_mock_md_contract(tmp_path, monkeypatch):
    from ase.calculators.calculator import Calculator, all_changes
    from ase import units
    class Harmonic(Calculator):
        implemented_properties = ["energy", "forces", "stress"]
        def __init__(self, reference):
            super().__init__(); self.reference = reference
        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            displacement = self.atoms.positions - self.reference
            self.results = dict(energy=.5 * np.sum(displacement ** 2), forces=-displacement, stress=np.zeros(6))
    parent_path = tmp_path / "parent.npz"; parent = toy_parent(parent_path)
    indices = np.array([0, 2, 3]); structure = tmp_path / "minimum.npz"
    np.savez(structure, numbers=parent["numbers"][indices], positions_A=parent["positions_A"][indices], cell_A=parent["cell_A"], parent_indices=indices, energy_eV=0., parent_sha256=np.array(td.digest(parent_path)))
    out = tmp_path / "md"; out.mkdir()
    args = SimpleNamespace(parent=parent_path, input=structure, output=out, seed=123, repeat_id="mock", basin_id="toy", timestep_fs=.5, save_interval_fs=2., temperature_K=433.15, taut_fs=100., equilibration_ps=.004, production_ps=.02, production_ensemble="nve", abort_temperature_K=5000., abort_force_eV_A=100.)
    monkeypatch.setattr(td, "calculator", lambda _: (Harmonic(parent["positions_A"][indices]), {"mock": True}))
    record = {}; td.md(args, record)
    data = td.load_npz(out / "production.npz")
    assert record["status"] == "COMPLETE"
    assert record["active_degrees_of_freedom"] == 6
    assert len(data["time_fs"]) == 11
    assert len(data["velocity_time_fs"]) == 41
    np.testing.assert_allclose(np.diff(data["time_fs"]), 2.)
    np.testing.assert_allclose(np.diff(data["velocity_time_fs"]), .5)
    np.testing.assert_allclose(data["velocities_A_ps"], data["velocities_ASE"] * units.fs * 1000)
    np.testing.assert_allclose(data["temperature_K"], 2 * data["kinetic_eV"] / (6 * units.kB))
    np.testing.assert_allclose(data["energy_total_eV"], data["potential_eV"] + data["kinetic_eV"])
    assert record["phases"]["production"]["mean_total_energy_eV"] == pytest.approx(data["energy_total_eV"].mean())
    assert record["phases"]["production"]["full_rate_mean_total_energy_eV"] == pytest.approx(data["velocity_energy_total_eV"].mean())
    np.testing.assert_allclose(data["velocity_energy_total_eV"], data["velocity_potential_eV"] + data["velocity_kinetic_eV"])
    np.testing.assert_allclose(data["detailed_velocities_ASE"], data["velocities_ASE"][::4])
    assert np.linalg.norm(data["COM_velocity_ASE"], axis=1).max() < 1e-12


def test_central_force_hessian_and_fresh_precision(tmp_path, monkeypatch):
    from ase.calculators.calculator import Calculator, all_changes
    class Harmonic(Calculator):
        implemented_properties = ["energy", "forces", "stress"]
        def __init__(self, reference):
            super().__init__(); self.reference = reference; self.calls = 0
        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes); self.calls += 1
            x = self.atoms.positions - self.reference
            self.results = dict(energy=.5 * np.sum(x ** 2), forces=-x, stress=np.zeros(6))
    parent_path = tmp_path / "parent.npz"; parent = toy_parent(parent_path)
    indices = np.array([0, 2, 3]); structure = tmp_path / "minimum.npz"
    np.savez(structure, numbers=parent["numbers"][indices], positions_A=parent["positions_A"][indices], cell_A=parent["cell_A"], parent_indices=indices, energy_eV=0., parent_sha256=np.array(td.digest(parent_path)))
    calc = Harmonic(parent["positions_A"][indices])
    teacher = dict(checkpoint_name="analytic mock", head="unit spring", checkpoint_hashes={}, dtype="float64", tf32=False)
    monkeypatch.setattr(td, "calculator", lambda _: (calc, teacher))
    shard = tmp_path / "shard"; shard.mkdir()
    args = SimpleNamespace(parent=parent_path, input=structure, output=shard, delta_A=.01, shards=1, shard_index=0)
    record = {}; td.hessian(args, record); td.write_json(shard / "receipt.json", record)
    # Three fresh baseline evaluations plus two independent displacements per DOF.
    assert calc.calls == 3 + 2 * 9
    assert record["force_precision_repeats"]["fresh_predictions"] == 3
    assert record["force_precision_repeats"]["force_component_max_span_eV_A"] == 0.
    np.testing.assert_allclose(td.load_npz(shard / "hessian_columns.npz")["hessian_columns_eV_A2"], np.eye(9), atol=3e-14)
    out = tmp_path / "assembled"; out.mkdir()
    args.output = out; args.shard = [shard]
    assembled = {}; td.assemble(args, assembled)
    assert assembled["retained_internal_modes"] == 6
    np.testing.assert_allclose(td.load_npz(out / "hessian.npz")["hessian_raw_eV_A2"], np.eye(9), atol=3e-14)

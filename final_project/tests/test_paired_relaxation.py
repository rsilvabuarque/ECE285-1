"""Synthetic numerical tests: no MLIP import, fitting, or physical claim."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/overnight_20261108"
sys.path.insert(0, str(SCRIPTS))
import paired_common as common
import paired_prepare as prepare
import paired_worker as worker
import paired_collect as collect
import paired_register as register


def parent():
    return dict(numbers=np.array([8, 8, 3, 3, 3, 3]),
                positions_A=np.array([[0, 0, 0], [5, 5, 5], [1, 2, 2], [3, 2, 2], [5, 2, 2], [7, 2, 2]], float),
                cell_A=np.eye(3) * 10, li_indices=np.arange(2, 6),
                permutations=np.array([[0, 1, 2, 3]]))


class Harmonic(Calculator):
    implemented_properties = ["energy", "forces", "stress"]
    def __init__(self, target, fail_after=None):
        super().__init__()
        self.target = np.asarray(target)
        self.calls = 0
        self.fail_after = fail_after
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("Synthetic deliberate failure")
        delta = atoms.positions - self.target
        self.results = dict(energy=float(.5 * np.sum(delta ** 2)), forces=-delta, stress=np.zeros(6))


def setup_attempt(tmp_path, **override):
    p = parent()
    key = "0011"
    context = dict(context_hash="synthetic-only", optimizer=dict(wall_cap_seconds=30.,
        snapshot_stride=5, maxstep_A=.15, max_steps=500, fmax_eV_A=.05))
    context["optimizer"].update(override)
    task = dict(canonical_vacancy_key=key, physical_key="synthetic", split="train")
    target = p["positions_A"][[0, 1, 2, 3]].copy()
    target[2, 0] += .2
    return p, context, task, target


def test_canonical_witness_and_species_mapping_do_not_move_saved_geometry():
    p = parent(); p["permutations"] = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    raw = np.array([1, 0, 0, 0], np.uint8)
    key, gather = common.canonical(raw, p["permutations"])
    assert key == "0111" and np.array_equal(raw[gather], 1 - np.array(list(key), int))
    slots = np.array([0, 1, 2, 3])
    xyz = p["positions_A"][slots].copy()
    before = xyz.copy()
    mapping = common.mapping_diagnostics(p["numbers"][slots], xyz, p, slots)
    assert mapping["mobile_assignment_changes"] == mapping["host_assignment_changes"] == 0
    assert mapping["mobile_nearest_collisions"] == mapping["host_nearest_collisions"] == 0
    assert np.array_equal(xyz, before)
    xyz[3] = xyz[2] + [.01, 0, 0]
    mapping = common.mapping_diagnostics(p["numbers"][slots], xyz, p, slots)
    assert mapping["mobile_nearest_collisions"] == 1
    assert sum(mapping["occupancy_hungarian"]) == 2
    with pytest.raises(ValueError, match="binary"):
        common.canonical([1, .5, 0, 0], p["permutations"])


def test_converged_attempt_pairs_initial_endpoint_and_counts_cached_calls(tmp_path):
    p, c, t, target = setup_attempt(tmp_path)
    base = Harmonic(target)
    result, stopped = worker.run_attempt(base, p, t, c, tmp_path / "attempt")
    arrays = common.load_npz(tmp_path / "attempt/arrays.npz")
    assert result["status"] == "converged" and not stopped
    assert result["endpoint_force_max_eV_A"] < .05
    assert result["endpoint_energy_eV"] < result["initial_energy_eV"]
    assert result["calculator_EFS_attempts"] == base.calls == result["calculator_EFS_successes"]
    assert result["calculator_EFS_failures"] == 0
    assert np.array_equal(arrays["cell_A"], p["cell_A"])
    assert np.array_equal(arrays["parent_indices"], [0, 1, 2, 3])
    assert np.array_equal(arrays["initial_positions_A"], p["positions_A"][[0, 1, 2, 3]])
    assert np.array_equal(arrays["endpoint_positions_A"], arrays["trajectory_positions_A"][-1])
    assert result["arrays_sha256"] == common.sha256(tmp_path / "attempt/arrays.npz")


def test_failed_force_call_preserves_last_valid_labels_without_cleanup_evaluation(tmp_path):
    p, c, t, target = setup_attempt(tmp_path)
    base = Harmonic(target, fail_after=1)
    result, stopped = worker.run_attempt(base, p, t, c, tmp_path / "failed")
    assert result["status"] == "error" and not stopped
    assert result["calculator_EFS_attempts"] == base.calls == 2
    assert result["calculator_EFS_successes"] == result["calculator_EFS_failures"] == 1
    assert not result["endpoint_labels_usable"]
    assert result["endpoint_energy_eV"] == result["initial_energy_eV"]
    assert (tmp_path / "failed/failure_traceback.txt").exists()


@pytest.mark.parametrize("settings,status,calls", [({"max_steps": 0}, "step_limit", 1),
    ({"wall_cap_seconds": -1}, "wall_limit", 0)])
def test_numerical_caps_remain_visible(tmp_path, settings, status, calls):
    p, c, t, target = setup_attempt(tmp_path, **settings)
    base = Harmonic(target)
    result, _ = worker.run_attempt(base, p, t, c, tmp_path / status)
    assert result["status"] == status and not result["converged"]
    assert base.calls == result["calculator_EFS_attempts"] == calls
    if calls == 0:
        assert result["endpoint_energy_eV"] is None


def test_prepare_excludes_intake_preserves_draws_and_freezes_train_only_pilot(tmp_path):
    parent_path = tmp_path / "parent.npz"
    np.savez(parent_path, **parent())
    history = tmp_path / "history.json"; history.write_text(json.dumps(["0011"]))
    protected = tmp_path / "protected.npz"
    np.savez(protected, canonical_masks=["0011"], split=["test"])
    protocol = tmp_path / "protocol.json"; protocol.write_text('{"status":"frozen synthetic protocol"}')
    args = SimpleNamespace(output_dir=tmp_path / "prepared", parent=parent_path,
        history=history, protected_data=protected, goal_protocol=protocol, counts=[2],
        train_per_count=2, val_per_count=1, test_per_count=1, pilot_per_count=2,
        max_draws_per_count=1000, seed=314159)
    receipt = prepare.prepare(args)
    common.check_prepared(args.output_dir)
    cohort = common.load_npz(args.output_dir / "cohort.npz")
    draws = common.load_npz(args.output_dir / "raw_draws.npz")
    assert receipt["independent_masks"] == 4 and not receipt["labels_read"]
    assert "0011" not in cohort["canonical_masks"] and len(set(cohort["canonical_masks"])) == 4
    assert np.all(cohort["split"][cohort["pilot_indices"]] == "train")
    for row, draw in enumerate(cohort["raw_draw_index"]):
        assert np.array_equal(cohort["occupancy"][row], draws["occupancy"][draw][cohort["canonical_gather"][row]])
    assert (draws["rejection_reason"] == "intake_known").any()
    (args.output_dir / "context.json").write_text("{}")
    with pytest.raises(ValueError, match="Frozen preparation changed"):
        common.check_prepared(args.output_dir)


def test_development_collector_hashes_but_never_reads_sealed_test_labels(tmp_path):
    pp = tmp_path / "parent.npz"; np.savez(pp, **parent())
    history = tmp_path / "history.json"; history.write_text('["0011"]')
    protected = tmp_path / "protected.npz"; np.savez(protected, canonical_masks=["0011"], split=["test"])
    protocol = tmp_path / "protocol.json"; protocol.write_text('{"status":"synthetic"}')
    prepared = tmp_path / "prepared"
    prepare.prepare(SimpleNamespace(output_dir=prepared, parent=pp, history=history,
        protected_data=protected, goal_protocol=protocol, counts=[2], train_per_count=2,
        val_per_count=1, test_per_count=1, pilot_per_count=2, max_draws_per_count=1000, seed=314159))
    context = json.loads((prepared / "context.json").read_text())
    data = common.load_npz(prepared / "cohort.npz")
    tasks = json.loads((prepared / "tasks_all.json").read_text())["tasks"]
    queue = tmp_path / "queue"
    for name in ["claims", "workers", "results"]: (queue / name).mkdir(parents=True)
    fake_worker = "synthetic_only"
    model = dict(context["teacher"], ase="3.26.0")
    common.atomic_json(queue / "workers" / (fake_worker + ".json"), dict(model_receipt=model))
    for row, task in enumerate(tasks):
        occ = common.occupancy_from_key(task["canonical_vacancy_key"])
        from cathode_cvae.data import deletion_atoms
        atoms, _ = deletion_atoms(parent(), occ)
        target = atoms.positions.copy(); target[-1, 0] += .1 + row * .01
        result_dir = queue / "results" / task["task_id"]
        record, _ = worker.run_attempt(Harmonic(target), parent(), task, context, result_dir)
        record.update(worker=fake_worker, model_receipt_sha256=common.identity(model))
        if data["split"][row] == "test":
            # Invalid scalar contents prove the default collector cannot have
            # loaded or cast the TEST energy arrays/record values.
            arrays = common.load_npz(result_dir / "arrays.npz")
            arrays["initial_energy_eV"] = np.array(np.nan)
            arrays["endpoint_energy_eV"] = np.array(np.nan)
            common.atomic_npz(result_dir / "arrays.npz", **arrays)
            record.update(initial_energy_eV="SEALED_TEST", endpoint_energy_eV="SEALED_TEST",
                          arrays_sha256=common.sha256(result_dir / "arrays.npz"))
        common.publish_result(result_dir, record)
        common.atomic_json(queue / "claims" / (task["task_id"] + ".json"), dict(worker=fake_worker, task=task))
    out = tmp_path / "collection"
    receipt = collect.collect(SimpleNamespace(prepared=prepared, queue=queue, output_dir=out,
        selection="all", unseal_test=False, allow_incomplete=False, minimum_low_support=30))
    assert receipt["status"] == "PASS_COMPLETE" and receipt["test_energies_sealed"]
    deletion = common.load_npz(out / "learning_deletion.npz")
    endpoint = common.load_npz(out / "learning_endpoint.npz")
    for key in ["occupancy", "split", "weight", "structure_ids", "n_target"]:
        assert np.array_equal(deletion[key], endpoint[key])
    assert len(endpoint["split"]) == 3 and set(endpoint["split"]) == {"train", "val"}
    assert np.isfinite(endpoint["energy_eV_atom"]).all()
    assert not json.loads((out / "sealed_test_manifest.json").read_text())["energy_values_read"]


def test_registration_deduplicates_physics_without_changing_candidate_bindings(tmp_path):
    pp = tmp_path / "parent.npz"; np.savez(pp, **parent())
    history = tmp_path / "history.json"; history.write_text('["0011"]')
    protected = tmp_path / "protected.npz"; np.savez(protected, canonical_masks=["0011"], split=["test"])
    protocol = tmp_path / "protocol.json"; protocol.write_text('{"status":"synthetic"}')
    prepared = tmp_path / "prepared"
    prepare.prepare(SimpleNamespace(output_dir=prepared, parent=pp, history=history,
        protected_data=protected, goal_protocol=protocol, counts=[2], train_per_count=2,
        val_per_count=1, test_per_count=1, pilot_per_count=2, max_draws_per_count=1000, seed=314159))
    pool = tmp_path / "candidates.npz"
    np.savez(pool, occupancy=[[1, 1, 0, 0], [1, 1, 0, 0]], structure_ids=["candidate_A", "candidate_B"])
    args = SimpleNamespace(prepared=prepared, pool=pool, attempts=None, label="synthetic",
        output_dir=tmp_path / "registration", prior_task_files=[], reserved_other_protocol_attempts=0,
        maximum_primary_attempts=10)
    receipt = register.register(args)
    bindings = json.loads((args.output_dir / "candidate_bindings.json").read_text())
    assert receipt["input_candidate_rows"] == 2 and receipt["unique_physical_tasks"] == 1
    assert len(bindings) == 2 and bindings[0]["task_id"] == bindings[1]["task_id"]
    assert [r["source_structure_id"] for r in bindings] == ["candidate_A", "candidate_B"]
    assert not receipt["outcomes_read"] and receipt["no_energy_filtering"]
    args.output_dir = tmp_path / "overbudget"; args.maximum_primary_attempts = 4
    with pytest.raises(ValueError, match="cap"):
        register.register(args)

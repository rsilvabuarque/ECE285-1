"""Independent mutation locality, orbit distance, and missing-label accounting."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

from cathode_cvae.conditions import fit_schema, save_schema
from cathode_cvae.data import sha256
from cathode_cvae.descriptor import fit_descriptor_schema, save_schema as save_descriptor
from cathode_cvae.graph import build_periodic_graph
from cathode_cvae.parent_spec import ParentSpec


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/conditioning_20261107/descriptor_benchmark.py"
spec = importlib.util.spec_from_file_location("descriptor_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(spec); spec.loader.exec_module(benchmark)


def test_frozen_mutation_has_declared_exact_edit_count_and_no_energy_acceptance():
    rng = np.random.default_rng(1172)
    seed = np.array([1, 0, 1, 0, 1, 0, 1, 0], np.uint8)
    observed = set()
    for _ in range(30):
        child, k, removed, added = benchmark.frozen_swaps(seed, rng)
        observed.add(k)
        assert set(removed) <= {0, 2, 4, 6} and set(added) <= {1, 3, 5, 7}
        assert len(set(removed)) == len(set(added)) == k
        assert child.sum() == 4 and np.sum(child != seed) == 2 * k
    assert observed == {1, 2, 3}
    # No energy is accepted by this API; these are frozen proposals, not a search.


def test_reference_distance_minimizes_symmetry_rather_than_canonical_frame_only():
    reference = np.array([[1, 1, 0, 0]], np.uint8)
    rotated_and_different = np.array([[0, 1, 1, 0], [1, 0, 1, 0]], np.uint8)
    permutations = np.array([[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]])
    assert benchmark._orbit_distances(rotated_and_different, reference, permutations).tolist() == [0, 2]


def test_evaluation_keeps_duplicates_and_missing_labels_in_attempt_denominator(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "require_compute_allocation", lambda: None)
    parent_path, data_path, history_path = tmp_path / "parent.npz", tmp_path / "data.npz", tmp_path / "history.json"
    parent = dict(numbers=np.full(4, 11), positions_A=np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], float),
                  cell_A=np.diag([4., 4., 8.]), target_indices=np.arange(4),
                  permutations=np.array([[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]))
    np.savez(parent_path, **parent)
    np.savez(data_path, occupancy=np.array([[1, 1, 0, 0]], np.uint8), split=np.array(["train"]),
             structure_ids=np.array(["synthetic_seed"]), canonical_masks=np.array(["0011"]))
    history_path.write_text("[]\n")
    graph = build_periodic_graph(ParentSpec.from_legacy(parent, sha256(parent_path)), 5.)
    structural = fit_descriptor_schema(np.array([.25]), np.array([2]), np.array(["middle"]), np.array(["train"]),
        parent_hash=graph.parent.parent_hash, graph_hash=graph.to_dict()["graph_hash"])
    descriptor_path = tmp_path / "descriptor.json"; save_descriptor(structural, descriptor_path)
    energy = fit_schema(np.array([2, 2, 2]), np.array([-1., 0., 1.]), np.array(["train"] * 3), n_sites=4)
    energy_path = tmp_path / "energy.json"; save_schema(energy, energy_path)
    generated = tmp_path / "generated"; generated.mkdir()
    # Three identical novel successful proposals and one unscored known mask.
    masks = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], np.uint8)
    ids = np.array(["mask4_3", "mask4_5"])
    np.savez(generated / "candidates.npz", masks=masks, occupancy=1-masks, structure_ids=ids,
             n_Li=np.array([2, 2]), n_atoms=np.array([2, 2]))
    rows = []
    for i, (sid, key) in enumerate([(ids[1], "0101")] * 3 + [(ids[0], "0011")]):
        rows.append(dict(attempt_id=f"synthetic_{i}", method="descriptor_cvae", n_Li=2, requested_bin="middle",
            requested_descriptor_range="low", scoring_structure_id=sid, canonical_mask=key, replicate_id="1"))
    benchmark._write_csv(generated / "attempts.csv", rows)
    benchmark._write_csv(generated / "distance_reference.csv", [dict(n_target=2, structure_id="synthetic_seed", raw_occupancy="1100")])
    receipt = dict(data_sha256=sha256(data_path), parent_sha256=sha256(parent_path), history_sha256=sha256(history_path),
        descriptor_schema_sha256=sha256(descriptor_path), counts=[2],
        outputs={f: sha256(generated / f) for f in ("candidates.npz", "attempts.csv", "distance_reference.csv")})
    (generated / "generation_receipt.json").write_text(json.dumps(receipt))
    score_path = tmp_path / "scores.npz"
    np.savez(score_path, structure_ids=ids[1:], energy_eV=np.array([0.]), n_atoms=np.array([2]))
    out = tmp_path / "evaluated"
    benchmark.evaluate(argparse.Namespace(generated=generated, data=data_path, parent=parent_path, history=history_path,
        descriptor_schema=descriptor_path, schemas={"synthetic": energy_path}, scores={"synthetic": score_path}, output_dir=out))
    metric = next(csv.DictReader((out / "metrics.csv").open()))
    assert int(metric["attempts"]) == 4
    assert float(metric["score_coverage"]) == .75
    assert int(metric["distinct_novel_joint_orbits"]) == 1
    assert float(metric["novel_joint_yield_per_all_attempts"]) == .25
    final = json.loads((out / "receipt.json").read_text())
    assert final["status"] == "INCOMPLETE_UNKNOWN_ENERGIES"
    assert final["unknown_by_teacher"] == {"synthetic": ["mask4_3"]}


"""Tiny synthetic tests for orbit roles, score provenance and iterative replay."""

import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from cathode_cvae.conditions import fit_schema, save_schema
from cathode_cvae.data import sha256
from cathode_cvae.iterative_acquisition import merge, stage
from cathode_cvae.mlip.registry import MODELS
from cathode_cvae import pool_scoring
from cathode_cvae.pool_scoring import PROTOCOL


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def vacancy_key(occupancy):
    return "".join(map(str, 1 - occupancy))


def candidate_id(key):
    return f"mask48_{int(key, 2):012x}"


@pytest.fixture
def fixture_data(tmp_path):
    """All coordinates/scores here are synthetic; no model or oracle is run."""
    parent = tmp_path / "parent.npz"
    np.savez_compressed(parent, numbers=np.ones(192, dtype=int), li_indices=np.arange(48),
                        permutations=np.arange(48, dtype=int)[None, :])
    masks = []
    for occupied in itertools.islice(itertools.combinations(range(48), 24), 22):
        value = np.zeros(48, dtype=np.uint8)
        value[list(occupied)] = 1
        masks.append(value)
    base_x = np.asarray(masks[:12])
    roles = np.asarray(["train"] * 10 + ["val", "test"])
    energies = np.arange(-10, 2, dtype=np.float64)
    base = tmp_path / "base.npz"
    np.savez_compressed(base, occupancy=base_x, n_Li=np.full(12, 24),
        energy_eV_atom=energies, split=roles, weight=np.ones(12, dtype=np.float32),
        structure_ids=np.asarray([f"original-{i}" for i in range(12)]),
        canonical_masks=np.asarray([vacancy_key(x) for x in base_x]),
        source=np.asarray(["original"] * 12))
    schema = tmp_path / "schema.json"
    save_schema(fit_schema(np.full(12, 24), energies, roles, n_sites=48,
        provenance={"parent_sha256": sha256(parent), "data_sha256": sha256(base),
                    "teacher": "uma", "teacher_head": MODELS["uma"]["head"],
                    "teacher_checkpoint_hashes": MODELS["uma"]["files"],
                    "geometry_protocol": PROTOCOL}), schema)
    known = tmp_path / "known.json"
    protected = tmp_path / "protected.json"
    write_json(known, [vacancy_key(x) for x in base_x])
    # A historically protected identity can already belong to H0 TRAIN; that
    # inherited role must remain unchanged and the overlap must be reported.
    write_json(protected, [vacancy_key(base_x[-1]), vacancy_key(base_x[1])])
    return dict(root=tmp_path, parent=parent, base=base, schema=schema, known=known,
                protected=protected, masks=masks)


def generation(data, base, name="generated", sample_indices=(11, 0, 12, 12, 13, 14, None),
               method="generator", seed_index=None):
    out = data["root"] / name
    out.mkdir()
    rows = []
    raw = []
    raw_seeds = []
    if seed_index is not None:
        with np.load(base, allow_pickle=False) as archive:
            seed = archive["occupancy"][seed_index].copy()
            seed_id = str(archive["structure_ids"][seed_index])
    unique = {}
    for i, idx in enumerate(sample_indices):
        key = "" if idx is None else vacancy_key(data["masks"][idx])
        sid = "" if idx is None else candidate_id(key)
        if idx is not None:
            unique[sid] = np.asarray(list(key), dtype=np.uint8)
        raw.append(np.full(48, np.nan) if idx is None else data["masks"][idx])
        raw_seeds.append(np.full(48, np.nan) if seed_index is None or idx is None else seed)
        row = dict(attempt_id=f"example-{i}", method=method, replicate_id="7",
                   n_target="24", requested_bin="low", canonical_mask=key,
                   structure_id=sid, raw_row=str(i),
                   status="failed_request" if idx is None else "generated")
        if seed_index is not None:
            removed = [] if idx is None else np.flatnonzero((seed == 1) & (data["masks"][idx] == 0)).tolist()
            added = [] if idx is None else np.flatnonzero((seed == 0) & (data["masks"][idx] == 1)).tolist()
            row.update(seed_source_row=str(seed_index), seed_source_sha256=sha256(base),
                       seed_structure_id=seed_id, swap_count=str(len(removed)),
                       removed_sites="|".join(map(str, removed)), added_sites="|".join(map(str, added)))
        rows.append(row)
    write_csv(out / "attempts.csv", rows)
    np.savez_compressed(out / "primary_candidates.npz", masks=np.asarray(list(unique.values())),
                        structure_ids=np.asarray(list(unique)), parent_sha256=np.asarray(sha256(data["parent"])))
    np.savez_compressed(out / "raw_attempts.npz", occupancy=np.asarray(raw),
                        seed_occupancy=np.asarray(raw_seeds))
    manifest = out / "manifest.json"
    model = {"method": method, "replicate_id": "7"}
    checkpoints = []
    if seed_index is None:
        checkpoint = out / "synthetic.pt"
        checkpoint.write_bytes(b"synthetic checkpoint identity, never loaded")
        model.update(checkpoint=str(checkpoint), expected_checkpoint_sha256=sha256(checkpoint))
        checkpoints.append({"method": method, "replicate_id": "7", "actual_checkpoint_sha256": sha256(checkpoint)})
    else:
        model["sampler"] = "mutation"
    write_json(manifest, {"status": "FROZEN_SYNTHETIC", "models": [model]})
    schema = json.loads(data["schema"].read_text())
    write_json(out / "protocol.json", {"status": "FROZEN_BEFORE_GENERATION_AND_SCORING",
        "no_energy_filtering": True, "condition_schema_id": schema["identity"], "inputs": {
        "parent": {"sha256": sha256(data["parent"])}, "data": {"sha256": sha256(base)},
        "schema": {"sha256": sha256(data["schema"])},
        "history": {"path": str(data["known"]), "sha256": sha256(data["known"])}},
        "manifest": {"path": str(manifest), "sha256": sha256(manifest)}})
    write_json(out / "receipt.json", {"status": "FROZEN_BEFORE_SCORING", "no_energy_filtering": True,
        "protocol_sha256": sha256(out / "protocol.json"), "checkpoints": checkpoints, "outputs": {
            "attempts.csv": sha256(out / "attempts.csv"),
            "primary_candidates.npz": sha256(out / "primary_candidates.npz"),
            "raw_attempts.npz": sha256(out / "raw_attempts.npz")}})
    return out


def run_stage(data, generated, output, base=None, history=None, previous=None, query_budget=2,
              method="generator"):
    return stage(parent_path=data["parent"], base_data_path=base or data["base"],
        schema_path=data["schema"], generation_dir=generated,
        known_histories=[history or data["known"]], protected_histories=[data["protected"]],
        method=method, replicate="7", round_index=1 if previous is None else 2,
        role_seed=901, counts=[24], bins=["low"], attempts_per_cell=7,
        queries_per_cell=query_budget, val_fraction=0.0, output_dir=output,
        previous_merge_receipt=previous)


def oracle(data, stage_dir, name="oracle"):
    root = data["root"] / name
    pool_scoring.register_pool(root, stage_dir / "candidates.npz", data["parent"],
                               models=("uma",), batch_size=64)
    registry = json.loads((root / "registry.json").read_text())
    task = registry["tasks"][0]
    tid = task["task_id"]
    for child in ("scores/uma", "workers", f"claims/{tid}"):
        (root / child).mkdir(parents=True, exist_ok=True)
    receipt = json.loads((stage_dir / "stage_receipt.json").read_text())
    with np.load(stage_dir / "candidates.npz", allow_pickle=False) as archive:
        ids = archive["structure_ids"].astype(str)
    write_json(root / "workers" / "synthetic.json", dict(model="uma", head=MODELS["uma"]["head"],
        package_version=MODELS["uma"]["version"], dtype="float32", tf32=False,
        checkpoint_hashes={Path(k).name: v for k, v in MODELS["uma"]["files"].items()}))
    write_json(root / "claims" / tid / "owner.json",
               dict(worker="synthetic", started_unix=receipt["created_unix"] + 1))
    scores = root / "scores" / "uma" / f"{tid}.npz"
    # Two intentionally low-bin, synthetic eV/atom energies; 144 hosts + 24 Li.
    np.savez_compressed(scores, structure_ids=ids, status=np.asarray(["success"] * len(ids)),
        energy_eV=np.asarray([-10.5, -9.5][:len(ids)]) * 168,
        n_atoms=np.full(len(ids), 168), n_Li=np.full(len(ids), 24),
        force_component_rms_eV_A=np.zeros(len(ids)), force_max_eV_A=np.zeros(len(ids)),
        stress_eV_A3=np.zeros((len(ids), 6)), seconds=np.zeros(len(ids)))
    write_json(scores.with_suffix(".json"), dict(task=task, worker="synthetic",
        output_sha256=sha256(scores), status="complete", rows=[
            dict(structure_id=sid, status="success", energy_eV=e * 168, n_atoms=168, n_Li=24)
            for sid, e in zip(ids.tolist(), [-10.5, -9.5])]))
    assert pool_scoring.collect(root, root / "collected")["status"] == "PASS"
    collection = root / "collected" / "receipt.json"
    return root, collection, scores


def run_merge(data, stage_dir, root, collection, output, **options):
    return merge(stage_dir=stage_dir, parent_path=data["parent"], base_data_path=data["base"],
                 schema_path=data["schema"], oracle_root=root, collection_receipt=collection,
                 output_dir=output, **options)


def test_stage_accounts_every_attempt_and_keeps_roles_before_scoring(fixture_data):
    d = fixture_data
    out = d["root"] / "staged"
    receipt = run_stage(d, generation(d, d["base"]), out)
    assert receipt["attempts"] == 7
    assert receipt["selected_unique_orbits"] == 2
    assert receipt["base_role_external_protected_overlap_counts"] == {"train": 1, "val": 0, "test": 1}
    assert receipt["stage_status_counts"] == {
        "protected": 1, "known": 1, "selected_for_query": 2,
        "duplicate_in_round": 1, "cell_query_budget_exhausted": 1,
        "generation_failed_or_invalid": 1}
    roles = json.loads((out / "assigned_roles.json").read_text())
    assert {entry["role"] for entry in roles.values()} == {"train"}
    assert all(entry["n_target"] == 24 for entry in roles.values())


def test_merge_preserves_all_original_rows_and_reweights_new_train_only(fixture_data):
    d = fixture_data
    staged = d["root"] / "staged"
    run_stage(d, generation(d, d["base"]), staged)
    root, collection, _ = oracle(d, staged)
    output = d["root"] / "merged"
    receipt = run_merge(d, staged, root, collection, output, new_target_fraction=0.2)
    assert receipt["attempts"] == 7
    assert receipt["terminal_successful_orbits"] == 2
    assert receipt["new_train_rows"] == 2
    assert receipt["new_val_rows"] == receipt["new_test_rows"] == 0
    assert receipt["base_role_external_protected_overlap_counts"]["train"] == 1
    tid = next(iter(receipt["task_evidence"]))
    assert receipt["task_evidence"][tid]["score"]["sha256"] == sha256(root / "scores" / "uma" / f"{tid}.npz")
    with np.load(d["base"], allow_pickle=False) as original, np.load(output / "training.npz", allow_pickle=False) as result:
        for name in original.files:
            assert np.array_equal(original[name], result[name][:len(original[name])])
        assert np.allclose(result["energy_eV_atom"][-2:], [-10.5, -9.5])
        assert np.allclose(result["weight"][-2:], [0.25, 0.25])
        assert sum(result["weight"][-2:]) / (2 + sum(result["weight"][-2:])) == pytest.approx(0.2)
    with (output / "attempts_scored.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7
    assert sum(row["oracle_status"] == "success" for row in rows) == 3  # One repeated attempt shares a query.


def test_merge_rejects_score_tampering_and_prescore_role_violation(fixture_data):
    d = fixture_data
    staged = d["root"] / "staged"
    run_stage(d, generation(d, d["base"]), staged)
    root, collection, score = oracle(d, staged)
    owner_path = root / "claims" / score.stem / "owner.json"
    owner = json.loads(owner_path.read_text())
    owner["started_unix"] = 0
    write_json(owner_path, owner)
    with pytest.raises(ValueError, match="before role assignment"):
        run_merge(d, staged, root, collection, d["root"] / "early")
    owner["started_unix"] = json.loads((staged / "stage_receipt.json").read_text())["created_unix"] + 1
    write_json(owner_path, owner)
    with score.open("ab") as stream:
        stream.write(b"tampering")
    with pytest.raises(ValueError, match="score bytes differ"):
        run_merge(d, staged, root, collection, d["root"] / "tampered")


def test_target_weights_use_measured_bins_including_missed_requests(fixture_data):
    d = fixture_data
    staged = d["root"] / "staged"
    run_stage(d, generation(d, d["base"]), staged)
    root, collection, score_path = oracle(d, staged)
    with np.load(score_path, allow_pickle=False) as archive:
        scores = {key: archive[key] for key in archive.files}
    # Both requests were LOW, but the second scored structure belongs to MIDDLE.
    scores["energy_eV"][1] = -5.5 * 168
    np.savez_compressed(score_path, **scores)
    metadata = json.loads(score_path.with_suffix(".json").read_text())
    metadata["output_sha256"] = sha256(score_path)
    metadata["rows"][1]["energy_eV"] = -5.5 * 168
    write_json(score_path.with_suffix(".json"), metadata)
    pool_scoring.collect(root, root / "collected-middle")
    collection = root / "collected-middle" / "receipt.json"
    output = d["root"] / "merged"
    result = run_merge(d, staged, root, collection, output, new_target_fraction=0.2)
    assert result["weight_by_training_cell"]["24:low"]["new_train"] == 1
    assert result["weight_by_training_cell"]["24:middle"]["new_train"] == 1
    with np.load(output / "training.npz", allow_pickle=False) as archive:
        assert np.allclose(archive["weight"][-2:], [0.5, 1.5])


def test_second_round_requires_history_of_all_prior_acquisitions(fixture_data):
    d = fixture_data
    staged = d["root"] / "staged"
    run_stage(d, generation(d, d["base"]), staged)
    root, collection, score_path = oracle(d, staged)
    with np.load(score_path, allow_pickle=False) as archive:
        scores = {key: archive[key] for key in archive.files}
    scores["status"][1] = "failed"
    scores["energy_eV"][1] = np.nan
    scores["n_atoms"][1] = 0
    scores["n_Li"][1] = -1
    np.savez_compressed(score_path, **scores)
    metadata = json.loads(score_path.with_suffix(".json").read_text())
    metadata["status"] = "complete_with_failures"
    metadata["output_sha256"] = sha256(score_path)
    metadata["rows"][1] = dict(structure_id=str(scores["structure_ids"][1]), status="failed",
                               error_type="SyntheticFailure", error="test-only failure")
    write_json(score_path.with_suffix(".json"), metadata)
    assert pool_scoring.collect(root, root / "collected-failed")["status"] == "INCOMPLETE"
    collection = root / "collected-failed" / "receipt.json"
    output = d["root"] / "merged"
    first = run_merge(d, staged, root, collection, output)
    assert first["terminal_failed_orbits"] == 1
    assert first["new_train_rows"] == 1
    with (output / "labels.csv").open(newline="") as stream:
        labels = list(csv.DictReader(stream))
    assert next(row for row in labels if row["status"] == "failed")["error_type"] == "SyntheticFailure"
    base2 = output / "training.npz"
    generated2 = generation(d, base2, name="generated2", sample_indices=(11, 0, 14, 14, 15, 16, None))
    previous = output / "merge_receipt.json"
    with pytest.raises(ValueError, match="omits prior acquired or failed"):
        run_stage(d, generated2, d["root"] / "missing-history", base=base2, previous=previous)
    history = d["root"] / "extended-known.json"
    previous_keys = list(json.loads((staged / "assigned_roles.json").read_text()))
    write_json(history, json.loads(d["known"].read_text()) + previous_keys)
    receipt = run_stage(d, generated2, d["root"] / "staged2", base=base2,
                        previous=previous, history=history)
    assert receipt["round"] == 2
    assert receipt["origin_data_sha256"] == sha256(d["base"])
    assert receipt["selected_unique_orbits"] == 2


def test_generation_and_protected_history_bytes_are_frozen(fixture_data):
    d = fixture_data
    generated = generation(d, d["base"])
    with (generated / "attempts.csv").open("a") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="bind attempts.csv"):
        run_stage(d, generated, d["root"] / "bad-stage")
    generated = generation(d, d["base"], name="generated-clean")
    staged = d["root"] / "staged"
    run_stage(d, generated, staged)
    root, collection, _ = oracle(d, staged)
    with d["protected"].open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="protected_histories bytes changed"):
        run_merge(d, staged, root, collection, d["root"] / "bad-merge")


def test_mutation_seeds_and_swap_budgets_are_bound_to_train_rows(fixture_data):
    d = fixture_data
    generated = generation(d, d["base"], name="mutation",
        sample_indices=(12, 13, 14, 15, 16, 17, None), method="mutation", seed_index=0)
    receipt = run_stage(d, generated, d["root"] / "staged-mutation", method="mutation")
    assert receipt["selected_unique_orbits"] == 2
    assert receipt["mutation_swap_counts"] == {1: 6}
    with (generated / "attempts.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["seed_source_row"] = "11"  # This original row is TEST, not TRAIN.
    write_csv(generated / "attempts.csv", rows)
    gen_receipt = json.loads((generated / "receipt.json").read_text())
    gen_receipt["outputs"]["attempts.csv"] = sha256(generated / "attempts.csv")
    write_json(generated / "receipt.json", gen_receipt)
    with pytest.raises(ValueError, match="does not belong to the bound TRAIN"):
        run_stage(d, generated, d["root"] / "protected-seed", method="mutation")


def test_noncanonical_protected_representatives_exclude_their_whole_orbit(fixture_data):
    d = fixture_data
    np.savez_compressed(d["parent"], numbers=np.ones(192, dtype=int), li_indices=np.arange(48),
                        permutations=np.stack((np.arange(48), np.arange(47, -1, -1))))
    with np.load(d["base"], allow_pickle=False) as archive:
        schema = fit_schema(archive["n_Li"], archive["energy_eV_atom"], archive["split"], n_sites=48,
            provenance={"parent_sha256": sha256(d["parent"]), "data_sha256": sha256(d["base"]),
                        "teacher": "uma", "teacher_head": MODELS["uma"]["head"],
                        "teacher_checkpoint_hashes": MODELS["uma"]["files"],
                        "geometry_protocol": PROTOCOL})
    write_json(d["schema"], schema)
    protected = json.loads(d["protected"].read_text())
    protected.append(vacancy_key(d["masks"][12])[::-1])
    # Also protect an H0 TRAIN orbit only through its reversed representation;
    # this must be reported but must never remove or relabel the original row.
    protected.append(vacancy_key(d["masks"][2])[::-1])
    write_json(d["protected"], protected)
    generated = generation(d, d["base"], sample_indices=(11, 2, 12, 12, 13, 14, None))
    staged = d["root"] / "staged"
    receipt = run_stage(d, generated, staged)
    assert receipt["base_role_external_protected_overlap_counts"]["train"] == 2
    assert receipt["stage_status_counts"]["protected"] == 4
    assert vacancy_key(d["masks"][12]) not in json.loads((staged / "assigned_roles.json").read_text())

"""Independent synthetic checks for pairing, complete labels, and metric scopes."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from cathode_cvae.conditions import fit_schema, save_schema


SOURCE = Path(__file__).resolve().parents[1] / "scripts/conditioning_20261107/analyze_diagnostics.py"
spec = importlib.util.spec_from_file_location("conditioning_diagnostics", SOURCE)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def fixture(tmp_path):
    raw = np.asarray([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1],
                      [1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0]], np.uint8)
    keys = ["".join(map(str, 1-r)) for r in raw]
    ids = ["geom_" + k for k in keys]
    labels = ["low", "middle", "high", "low", "low", "low", "low"]
    kinds = ["primary", "primary", "primary", "latent_B", "posterior_0", "prior_matched_0", "primary"]
    rows = []
    for i, (key, sid, requested, kind) in enumerate(zip(keys, ids, labels, kinds)):
        r = dict(method="synthetic", replicate_id="seed7", fit_seed="7", n_Li=2,
            requested_bin=requested, diagnostic=kind, raw_row=i, canonical_mask=key,
            scoring_structure_id=sid, status="generated", attempt_id=f"attempt{i}",
            paired_id="p1" if i == 6 else "p0", checkpoint_sha256="synthetic-checkpoint",
            uniforms_sha256="common-u", latents_sha256="different-z" if i == 3 else "common-z")
        if kind.startswith(("posterior", "prior")):
            r.update(validation_target_id="validation_A", posterior_target_hamming=0,
                     prior_target_hamming=4, validation_target_energy_eV_atom=-1.)
        rows.append(r)
    schema = fit_schema([2]*10, np.linspace(-1., 0., 10), ["train"]*10, n_sites=4)
    schema_path = tmp_path / "schema.json"; save_schema(schema, schema_path)
    scores = tmp_path / "scores.npz"
    unique = list(dict.fromkeys(ids))
    energy = {ids[0]: -1., ids[1]: -.5, ids[2]: 0., ids[3]: -.75}
    np.savez(scores, structure_ids=np.asarray(unique), energy_eV=np.asarray([2*energy[s] for s in unique]),
             n_atoms=np.full(len(unique), 2))
    return rows, raw, scores, schema, schema_path


def test_common_stream_pairs_preserve_direction_and_raw_distance(tmp_path):
    rows, raw, scores, schema, _ = fixture(tmp_path)
    outcomes, missing = diagnostics.join_scores(rows, raw, scores, schema, set(), set())
    assert not missing
    conditions, latents, posterior = diagnostics.paired_diagnostics(outcomes, raw)
    assert len(conditions) == 2 and len(latents) == 1 and len(posterior) == 1
    assert conditions[0]["delta_E_eV_atom"] == .5
    assert conditions[0]["raw_Hamming"] == 2
    assert conditions[1]["delta_E_eV_cell"] == 2.
    assert latents[0]["uniform_fingerprints_equal"]
    assert not latents[0]["latent_fingerprints_equal"]
    assert latents[0]["random_stream_scope"].startswith("Shared AR uniforms; latent is resampled")
    assert conditions[0]["random_stream_scope"].startswith("Shared latent and AR uniforms")
    assert posterior[0]["random_stream_scope"].startswith("Shared condition and AR uniforms")
    assert posterior[0]["posterior_target_Hamming"] == 0
    assert posterior[0]["prior_target_Hamming"] == 4
    assert posterior[0]["posterior_minus_target_eV_atom"] == 0
    assert posterior[0]["prior_minus_target_eV_atom"] == 1


def test_missing_labels_cannot_look_like_a_complete_yield(tmp_path):
    rows, raw, scores, schema, _ = fixture(tmp_path)
    with np.load(scores) as z:
        d = {k: z[k][:-1] for k in z.files}
    np.savez(scores, **d)
    with pytest.raises(ValueError, match="Incomplete exact oracle"):
        diagnostics.join_scores(rows, raw, scores, schema, set(), set())
    outcomes, missing = diagnostics.join_scores(rows, raw, scores, schema, set(), set(), allow_incomplete=True)
    assert len(outcomes) == len(rows) and len(missing) == 1
    assert not outcomes[3]["energy_known"] and outcomes[3]["energy_eV_atom"] == ""


def test_balanced_macro_and_low_original_threshold_are_separate(tmp_path):
    rows, raw, scores, schema, _ = fixture(tmp_path)
    outcomes, _ = diagnostics.join_scores(rows, raw, scores, schema, set(), set())
    cells, scopes = diagnostics.outcome_summaries(outcomes, {2: -.95})
    low = next(r for r in cells if r["requested_bin"] == "low")
    assert low["attempts"] == 2 and low["distinct_novel_inbin_yield"] == .5
    balanced = next(r for r in scopes if r["scope"].startswith("balanced"))
    original = next(r for r in scopes if r["scope"].startswith("requested_low"))
    assert balanced["macro_useful_yield"] == pytest.approx(5/6)
    assert original["macro_useful_yield"] == .5
    # An omitted history must remain unknown, never silently count as empty.
    unknown, _ = diagnostics.join_scores(rows, raw, scores, schema, set(), None)
    _, scopes = diagnostics.outcome_summaries(unknown, {2: -.95})
    assert all(r["macro_useful_yield"] == "" for r in scopes)
    for row in outcomes:
        if row["requested_bin"] == "low":
            row["energy_known"] = False; row["requested_bin_hit"] = False
    cells, scopes = diagnostics.outcome_summaries(outcomes, {2: -.95})
    assert not next(r for r in cells if r["requested_bin"] == "low")["complete_energy_coverage"]
    assert all(r["macro_useful_yield"] == "" for r in scopes)


def test_end_to_end_diagnostic_receipt_and_val_target_guard(tmp_path):
    rows, raw, scores, schema, schema_path = fixture(tmp_path)
    attempts = tmp_path / "attempts.csv"; diagnostics.write_csv(attempts, rows)
    raw_targets = np.zeros_like(raw); raw_targets[[4, 5]] = raw[0]
    raw_path = tmp_path / "raw.npz"; np.savez(raw_path, occupancy=raw, seed_occupancy=raw_targets)
    training = tmp_path / "training.npz"
    np.savez(training, occupancy=raw[[3, 0]], canonical_masks=np.asarray(["0110", "0011"]),
        structure_ids=np.asarray(["training_A", "validation_A"]), split=np.asarray(["train", "val"]),
        n_Li=np.asarray([2, 2]), energy_eV_atom=np.asarray([-.75, -1.]))
    history = tmp_path / "history.json"; history.write_text("[]")
    output = tmp_path / "analysis"
    receipt = diagnostics.analyze(attempts, raw_path, scores, output, schema_path=schema_path,
        training_data=training, history_path=history, plots=True)
    assert receipt["status"] == "PASS_COMPLETE_ORACLE_COVERAGE"
    assert receipt["posterior_prior_pairs"] == 1 and receipt["invalid_mask_attempts"] == 0
    assert receipt["posterior_or_prior_raw_target_Hamming_verified"] == 2
    assert (output / "condition_response.pdf").exists()
    assert (output / "latent_response.pdf").exists()
    assert (output / "posterior_vs_prior.pdf").exists()
    with np.load(training) as z:
        d = {k: z[k] for k in z.files}
    d["split"] = np.asarray(["train", "test"])
    np.savez(training, **d)
    with pytest.raises(ValueError, match="protected validation"):
        diagnostics.analyze(attempts, raw_path, scores, tmp_path/"bad_analysis", schema_path=schema_path,
            training_data=training, history_path=history, plots=False)


def test_selected_fit_metadata_matches_the_actual_checkpoint_hash(tmp_path):
    root = tmp_path / "fits"; fit = root / "synthetic"; fit.mkdir(parents=True)
    (fit / "configuration.json").write_text(json.dumps(dict(seed=7, latent_dim=1, parameter_count=12)))
    history = [dict(epoch=i, optimizer_updates=i, val_reconstruction=3-i,
                    train_reconstruction=2-i/2, val_KL=1.) for i in [1, 2]]
    diagnostics.write_csv(fit/"history.csv", history)
    stat = dict(total_KL_nats=1., mean_KL_nats_per_dimension=1., KL_nats_by_dimension=[1.],
        posterior_mean_by_dimension=[0.], posterior_mean_variance_by_dimension=[.02],
        posterior_variance_mean_by_dimension=[1.], active_components_by_cutoff={"0.001": 1, "0.01": 1, "0.1": 0})
    (fit / "latent_history.jsonl").write_text("\n".join(json.dumps(dict(epoch=i, pooled=stat, by_composition={"2": stat})) for i in [1, 2]))
    selected = {}
    for beta, epoch in [(1., 1), (.1, 2)]:
        name = f"beta{beta}.pt"; (fit/name).write_bytes(f"synthetic {beta}".encode())
        selected[str(beta)] = dict(epoch=epoch, filename=name, sha256=diagnostics.sha256(fit/name))
    (fit / "summary.json").write_text(json.dumps(dict(optimizer_updates=2, training_seconds=.1, selection_checkpoints=selected)))
    histories, selections, components = diagnostics.aggregate_fits(root)
    assert len(histories) == 2 and len(selections) == 2 and len(components) == 4
    assert {r["selected_epoch"] for r in selections} == {1, 2}
    assert all(r["active_at_0.01"] == 1 for r in selections)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(models=[dict(method="specified", checkpoint=str(fit/"beta1.0.pt"))])))
    other = root/"concurrent"; other.mkdir()
    (other/"configuration.json").write_text("{}")
    # An unrelated fit can be incomplete while this manifest's stage is complete.
    assert len(diagnostics.aggregate_fits(root, manifest)[1]) == 2
    diagnostics.plot_fit_diagnostics(tmp_path, histories, selections, components)
    assert (tmp_path / "fit_learning_curves.pdf").exists()
    assert (tmp_path / "selected_latent_components.pdf").exists()
    (fit / "beta1.0.pt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checkpoint differs"):
        diagnostics.aggregate_fits(root)

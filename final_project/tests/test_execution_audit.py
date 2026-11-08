"""Independent resource and raw-evidence fixtures; no scheduler or MLIP runs."""

import importlib.util
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


FILE = Path(__file__).resolve().parents[1] / "reports/conditioning_20261107/audit_execution.py"
spec = importlib.util.spec_from_file_location("execution_audit", FILE)
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def test_gpu_subtypes_and_child_steps_are_not_double_counted(tmp_path):
    p = tmp_path / "accounting.tsv"
    p.write_text("JobIDRaw\tAccount\tAllocNodes\tAllocTRES\tState\tStart\tEnd\tElapsedRaw\n"
        "100\tm5397_g\t4\tgres/gpu=16,gres/gpu:a100=16\tCOMPLETED\t2026-11-08T08:00:00\t2026-11-08T09:00:00\t3600\n"
        "100.batch\tm5397_g\t4\tgres/gpu=16\tCOMPLETED\t2026-11-08T08:00:00\t2026-11-08T09:00:00\t3600\n"
        "101\tm5397_g\t1\tgres/gpu=2,gres/gpu:a100=2\tCOMPLETED\t2026-11-08T08:00:00\t2026-11-08T09:00:00\t3600\n")
    totals, rows = audit.allocation_audit(p, owned_jobs=["100", "101"])
    assert len(rows) == 2 and totals["ignored_child_or_out_of_scope_rows"] == ["100.batch"]
    assert totals["GPU_device_hours"] == 18.
    assert totals["physical_node_hours"] == 5.
    assert totals["four_GPU_equivalent_node_hours"] == 4.5
    assert totals["maximum_concurrently_allocated_GPUs"] == 18
    assert totals["listed_all_allocations_terminal"] and totals["findings"] == []
    with pytest.raises(ValueError, match="disagree"):
        audit.parse_gpu_count("gres/gpu=2,gres/gpu:a100=4")


def test_running_shared_allocation_is_preliminary_and_goal_interval_is_explicit(tmp_path):
    p = tmp_path / "live.tsv"
    p.write_text("JobIDRaw|Account|AllocNodes|AllocTRES|State|Start|End|ElapsedRaw\n"
        "102|m5397_g|1|gres/gpu:a100=2|RUNNING|2026-11-08T08:00:00|Unknown|3600\n")
    total, rows = audit.allocation_audit(p, goal_start=audit.parse_time("2026-11-08T08:30:00Z"),
        snapshot_utc=audit.parse_time("2026-11-08T09:00:00Z"))
    assert not total["listed_all_allocations_terminal"]
    assert rows[0]["charged_goal_seconds"] == 1800.
    assert total["GPU_device_hours"] == 1. and total["physical_node_hours"] == .5
    assert total["four_GPU_equivalent_node_hours"] == .25
    assert total["findings"] == []
    stale, _ = audit.allocation_audit(p, snapshot_utc=audit.parse_time("2026-11-08T17:00:00Z"))
    assert "UTC" in stale["findings"][0]["finding"]


def test_raw_oracle_fixture_separates_fresh_from_cache_and_detects_mutation(tmp_path):
    root = tmp_path; run = root / "runs"; oracle = run / "oracle"
    for p in (oracle / "scores/uma", oracle / "workers", oracle / "claims"):
        p.mkdir(parents=True, exist_ok=True)
    parent = root / "parent.npz"
    np.savez(parent, numbers=np.array([11, 11, 27]), li_indices=np.array([0, 1]), permutations=np.array([[0, 1]]))
    parent_sha = audit.digest(parent)
    models = {"uma": dict(head="test", version="0.0.test", files={"checkpoints/weights.pt": "synthetic-checksum"})}
    inp = root / "candidates.npz"
    np.savez(inp, structure_ids=np.array(["mask2_1", "mask2_2"]), masks=np.array([[0, 1], [1, 0]], np.uint8), parent_sha256=np.asarray(parent_sha))
    task = dict(task_id="fixture", model="uma", input_npz=str(inp), input_sha256=audit.digest(inp),
                parent_npz=str(parent), parent_sha256=parent_sha, indices=[0], expected_structure_ids=["mask2_1"])
    registry = dict(parent_sha256=parent_sha, tasks=[task], physical_keys={})
    for sid, provenance in [("mask2_1", "new_scoring"), ("mask2_2", "exact_geometry_cache")]:
        key = audit.expected_physical_key("uma", sid, parent_sha, models, 2)
        registry["physical_keys"][key] = dict(model="uma", structure_id=sid, input_sha256=audit.digest(inp), provenance=provenance)
    registry_file = oracle / "registry.json"; registry_file.write_text(json.dumps(registry))
    (oracle / "tasks.jsonl").write_text(json.dumps(task) + "\n")
    worker = dict(worker="fixtureworker", model="uma", head="test", package_version="0.0.test", dtype="float32",
                  tf32=False, nitro=False, compile=False, dispersion=False,
                  checkpoint_hashes={"weights.pt": "synthetic-checksum"}, job_id="synthetic")
    (oracle / "workers/fixtureworker.json").write_text(json.dumps(worker))
    cache = oracle / "cache_uma.npz"
    np.savez(cache, structure_ids=np.array(["mask2_2"]), energy_eV=np.array([-2.]), n_atoms=np.array([2]), n_Li=np.array([1]))
    (oracle / "cache_uma.json").write_text(json.dumps(dict(context=audit.expected_context("uma", parent_sha, models), npz_sha256=audit.digest(cache))))
    output = oracle / "scores/uma/fixture.npz"
    np.savez(output, structure_ids=np.array(["mask2_1"]), status=np.array(["success"]), energy_eV=np.array([-1.]),
             n_atoms=np.array([2]), n_Li=np.array([1]), forces_parent_eV_A=np.zeros((1, 3, 3)), stress_eV_A3=np.zeros((1, 6)))
    output.with_suffix(".json").write_text(json.dumps(dict(task=task, worker="fixtureworker",
        output_sha256=audit.digest(output), rows=[dict(structure_id="mask2_1", status="success", energy_eV=-1.)])))
    evidence = audit.Evidence()
    result, tasks, workers = audit.oracle_audit(root, run, evidence, models)
    assert result["by_teacher"]["uma"] == dict(registered_fresh_pairs=1, registered_cache_reuses=1,
        recorded_new_logical_static_attempts=1, successful_new_EFS=1, failed_new_logical_attempts=0, fresh_pairs_without_terminal_rows=0)
    assert tasks[0]["status"] == "COMPLETE" and evidence.findings == []
    assert evidence.mutation_check() == []
    registry_file.write_text(json.dumps(dict(registry, new_stage="appended while audit was reading")))
    assert str(registry_file.resolve()) in evidence.mutation_check()


def test_fit_manifests_deduplicate_identical_plans_and_choose_actual_worker_claim(tmp_path):
    root, run = tmp_path, tmp_path / "run"
    directory = run / "fits/scientific_fit"; directory.mkdir(parents=True)
    command = ["--output-dir", str(directory), "--max-updates", "4000"]
    task = dict(id="same_fit", argv=command)
    individual = run / "single_fit_tasks.json"
    combined = run / "z_combined_tasks.json"
    for path in (individual, combined):
        path.write_text(json.dumps(dict(stage="synthetic full fit", tasks=[task])))
    claim = combined.with_suffix(".claims") / "same_fit"; claim.mkdir(parents=True)
    (claim / "owner.json").write_text(json.dumps(dict(tasks_sha256=audit.digest(combined), job="synthetic")))
    (claim / "receipt.json").write_text(json.dumps(dict(status="COMPLETE")))
    cfg = dict(data_sha256="synthetic-data", maximum_updates=4000, train_rows=40157, val_rows=4967, test_rows=5123)
    (directory / "configuration.json").write_text(json.dumps(cfg))
    summary = dict(status="FIT_COMPLETE", optimizer_updates=4000, stop_reason="optimizer_update_cap",
                   train_rows=40157, val_rows=4967, test_rows=5123, heldout_test_evaluated=False)
    (directory / "summary.json").write_text(json.dumps(summary))
    protocol = dict(data=dict(round0_sha256="synthetic-data"), nonresource_science_bounds=dict(maximum_full_training_trajectories=30))
    result, rows = audit.fit_audit(root, run, audit.Evidence(), protocol)
    assert result["started_full_trajectories"] == result["completed_full_trajectories"] == 1
    assert result["eligible_matched_round0_4000_update_trajectories"] == 1
    assert rows[0]["identical_planned_manifest_count"] == 2
    assert rows[0]["worker_job"] == "synthetic"
    assert rows[0]["direct_fit_has_no_task_receipt"] is False
    other = dict(task, argv=["--output-dir", str(directory), "--max-updates", "4001"])
    individual.write_text(json.dumps(dict(tasks=[other])))
    with pytest.raises(ValueError, match="DIFFERENT commands"):
        audit.fit_audit(root, run, audit.Evidence(), protocol)


def test_finalizer_is_stdlib_only_and_refuses_payload_changes(tmp_path):
    root = tmp_path; run = root / "run"; run.mkdir()
    protocol_dir = root / "artifacts/conditioning_20261107"; protocol_dir.mkdir(parents=True)
    protocol_file = protocol_dir / "protocol.json"
    protocol_file.write_text(json.dumps(dict(start_UTC="2026-11-08T08:00:00Z", hard_stop_UTC="2026-11-08T12:00:00Z",
        user_resource_override=dict(maximum_concurrent_GPUs=36, maximum_new_GPU_node_hours=250))))
    payload = run / "raw_payload.npz"; payload.write_bytes(b"synthetic raw payload; no array is loaded")
    previous_alloc = root / "heavy_allocations.csv"
    previous_alloc.write_text("job_id,start_UTC\n100,2026-11-08T08:00:00+00:00\n")
    heavy_file = root / "heavy.json"
    bound = {str(p.resolve()): audit.digest(p) for p in (protocol_file, payload)}
    stats = {str(p.resolve()): dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in (protocol_file, payload)}
    heavy_file.write_text(json.dumps(dict(evidence_stable_during_audit=True, findings=[], evidence_file_hashes=bound,
        evidence_file_stats=stats, sidecars=dict(allocations=dict(path=str(previous_alloc), sha256=audit.digest(previous_alloc))),
        training=dict(task_definition_files_at_start=[], full_trajectories_without_terminal_summary=0,
                      defined_full_trajectories_not_started=0), oracle=dict(noncomplete_tasks=0),
        engineering_inference={}, allocations=dict(accounting_snapshot_UTC="2026-11-08T08:30:00Z"),
        physical_scope="synthetic execution fixture", exclusions="no physical claim")))
    final_accounting = root / "final.tsv"
    final_accounting.write_text("JobIDRaw|Account|AllocNodes|AllocTRES|State|Start|End|ElapsedRaw\n"
        "100|m5397_g|1|gres/gpu=2,gres/gpu:a100=2|COMPLETED|2026-11-08T08:00:00|2026-11-08T09:00:00|3600\n")
    args = argparse.Namespace(finalize_from=heavy_file, accounting=final_accounting, owned_jobs=["100"],
        account="m5397_g", accounting_snapshot_UTC=None, output=root / "closed.json")
    assert audit.finalize_from(args, root, run, b"synthetic finalizer source") == 0
    receipt = json.loads(args.output.read_text())
    assert receipt["status"] == "PASS_EXECUTION_ACCOUNTING_FINAL"
    assert receipt["allocations"]["GPU_device_hours"] == 2.
    assert receipt["finalizer_checks"]["arrays_or_models_loaded"] is False
    assert receipt["finalizer_checks"]["raw_payload_content_rehashed"] is False
    assert receipt["finalizer_checks"]["new_runtime_model_calls"] == 0
    # Changed payloads cannot be closed merely because allocations are terminal.
    payload.write_bytes(b"changed payload")
    args.output = root / "refused.json"
    assert audit.finalize_from(args, root, run, b"synthetic source") == 2
    assert str(payload.resolve()) in json.loads(args.output.read_text())["changed_files"]
    # -S disables site packages: importing the finalizer still needs no numpy,
    # torch, ASE, model access, or allocation context.
    code = ("import importlib.util,sys; s=importlib.util.spec_from_file_location('audit',sys.argv[1]); "
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
            "assert 'numpy' not in sys.modules and 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-S", "-c", code, str(FILE)], check=True)


def test_core30_amendment_allows28_plus3_but_not_a_renamed_mask_fit(tmp_path):
    contract = dict(mode="verified_core_scope_amendment", maximum_core_full_trajectories=30,
                    maximum_optional_editor_full_trajectories=3, maximum_all_full_trajectories=31)
    rows = [dict(category="full_scientific_trajectory", started=True, experiment_phase="core") for _ in range(28)]
    rows += [dict(category="full_scientific_trajectory", started=True, experiment_phase="optional_editor") for _ in range(3)]
    evidence = audit.Evidence()
    counts = audit.check_phase_caps(rows, contract, evidence)
    assert counts == dict(started_core_full_trajectories=28, started_optional_editor_full_trajectories=3,
                          started_all_full_trajectories=31)
    assert evidence.findings == []
    audit.check_phase_caps(rows + [rows[-1]], contract, evidence)
    assert {f["finding"] for f in evidence.findings} == {"optional_editor full training trajectory bound exceeded", "all full training trajectory bound exceeded"}
    # The exemption requires the editor trainer, its variational kind, and a
    # completed packet gate, not an output directory or phase label alone.
    root, run = tmp_path, tmp_path / "run"; run.mkdir()
    source_dir = root / "src/cathode_cvae/learning"; source_dir.mkdir(parents=True)
    mask_source, editor_source = source_dir / "train_mask.py", source_dir / "train_edit.py"
    mask_source.write_text("# synthetic mask trainer fixture\n")
    editor_source.write_text("# synthetic variational editor trainer fixture\n")
    cfg = dict(kind="ar_cvae", experiment_phase="optional_editor", trainer_module="cathode_cvae.learning.train_edit",
               trainer_sha256=audit.digest(mask_source), maximum_updates=4000)
    evidence = audit.Evidence()
    phase = audit.classify_fit_phase(root, run, evidence, cfg, {}, {}, "renamed editor", [], {}, contract)
    assert phase["phase"] == "core" and any("incorrectly labelled" in f["finding"] for f in evidence.findings)
    gate_file = root / "gate.json"
    gate_file.write_text(json.dumps(dict(status="COMPLETE", completed_UTC="2026-11-08T08:00:00Z")))
    cfg.update(kind="seed_edit_cvae", trainer_sha256=audit.digest(editor_source),
               required_report_gate=dict(path=str(gate_file), sha256=audit.digest(gate_file)))
    owner = dict(started_unix=audit.parse_time("2026-11-08T08:00:01Z").timestamp())
    phase = audit.classify_fit_phase(root, run, audit.Evidence(), cfg, owner, {}, "optional editor", [], {}, contract)
    assert phase["phase"] == "optional_editor" and phase["verified"]
    owner["started_unix"] -= 2
    phase = audit.classify_fit_phase(root, run, audit.Evidence(), cfg, owner, {}, "optional editor", [], {}, contract)
    assert phase["phase"] == "core"  # launching before the gate cannot be relabelled later


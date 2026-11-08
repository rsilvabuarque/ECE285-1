"""Collect paired labels with an energy-free TEST manifest by default.

The paired model files contain TRAIN/VAL only and exactly the same converged
starting masks. TEST labels remain in their raw queue files until --unseal-test
is explicitly used for final assessment. Failed attempts are never discarded
from completion/convergence denominators.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from paired_common import (atomic_json, atomic_npz, check_prepared, geometry_hash, identity,
    load_npz, mobile_indices, sha256, verify_teacher_receipt)

DELETION_PROTOCOL = "pristine-derived deletion; unrelaxed; exact canonical parent coordinates"
ENDPOINT_PROTOCOL = "paired fixed-cell UMA FIRE endpoint; fmax<0.05 eV/A; maxstep0.15A; max500steps; wall120s; v1"


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def collect(args):
    from cathode_cvae.runtime import require_compute_allocation
    from cathode_cvae.conditions import classify, fit_schema, save_schema
    require_compute_allocation()
    preparation, context = check_prepared(args.prepared)
    data = load_npz(args.prepared / "cohort.npz")
    parent = load_npz(args.prepared / "parent.npz")
    tasks = json.loads((args.prepared / "tasks_all.json").read_text())["tasks"]
    selected = data["pilot_indices"] if args.selection == "pilot" else np.arange(len(tasks))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    statuses, development, test_labels, missing, checked_sources = [], [], [], [], []
    worker_receipts = {}
    rows_by_endpoint = defaultdict(list)
    for row in selected:
        row = int(row); task = tasks[row]
        result_dir = args.queue / "results" / task["task_id"]
        sealed = str(data["split"][row]) == "test" and not args.unseal_test
        source = result_dir / ("status.json" if sealed else "result.json")
        if not source.exists():
            missing.append(task["task_id"])
            statuses.append(dict(row=row, task_id=task["task_id"], n_target=int(data["n_target"][row]),
                split=str(data["split"][row]), status="missing", converged=False, steps=0,
                EF_attempts=0, elapsed_seconds=0., endpoint_force_max_eV_A=None,
                mobile_assignment_changes=None, host_assignment_changes=None, energy_sealed=sealed))
            continue
        record = json.loads(source.read_text())
        if record["physical_key"] != task["physical_key"] or record["context_hash"] != context["context_hash"]:
            raise ValueError("Result belongs to a different geometry/protocol")
        if sealed and (not record.get("energy_fields_omitted") or
                       any(k in record for k in ["initial_energy_eV", "endpoint_energy_eV"])):
            raise ValueError("TEST header contains forbidden energy fields")
        if sealed and sha256(result_dir / "result.json") != record["result_sha256"]:
            raise ValueError("Sealed result bytes changed")
        if sha256(result_dir / "arrays.npz") != record["arrays_sha256"]:
            raise ValueError("Saved coordinate/force arrays changed")
        claim_path = args.queue / "claims" / (task["task_id"] + ".json")
        claim = json.loads(claim_path.read_text())
        worker_id = record.get("worker", claim["worker"])
        if worker_id != claim["worker"] or claim["task"]["physical_key"] != task["physical_key"]:
            raise ValueError("Result and original exclusive claim disagree")
        if worker_id not in worker_receipts:
            path = args.queue / "workers" / (worker_id + ".json")
            worker = json.loads(path.read_text())
            model = worker["model_receipt"]
            verify_teacher_receipt(model, context)
            worker_receipts[worker_id] = dict(path=str(path.absolute()), sha256=sha256(path), receipt=worker)
        model = worker_receipts[worker_id]["receipt"]["model_receipt"]
        if record.get("model_receipt_sha256", identity(model)) != identity(model):
            raise ValueError("Result's worker inference binding differs")
        if record["converged"] and (record["endpoint_force_max_eV_A"] is None or
                not record["endpoint_force_max_eV_A"] < context["optimizer"]["fmax_eV_A"]):
            raise ValueError("Converged endpoint fails the physical force criterion")
        mapping = record.get("mapping") or {}
        statuses.append(dict(row=row, task_id=task["task_id"], n_target=int(data["n_target"][row]),
            split=str(data["split"][row]), status=record["status"], converged=bool(record["converged"]),
            steps=record["n_steps"], EF_attempts=record["calculator_EFS_attempts"],
            elapsed_seconds=record["elapsed_seconds_including_mapping"],
            endpoint_force_max_eV_A=record["endpoint_force_max_eV_A"],
            mobile_assignment_changes=mapping.get("mobile_assignment_changes"),
            host_assignment_changes=mapping.get("host_assignment_changes"), energy_sealed=sealed))
        checked_sources.append(dict(task_id=task["task_id"], split=str(data["split"][row]),
            result_sha256=record.get("result_sha256", sha256(source)), arrays_sha256=record["arrays_sha256"],
            claim_sha256=sha256(claim_path), worker=worker_id))
        if mapping:
            rows_by_endpoint[mapping["endpoint_occupation_orbit"]].append(row)
        if sealed:
            continue
        if not record["endpoint_labels_usable"]:
            continue
        with np.load(result_dir / "arrays.npz", allow_pickle=False) as z:
            if not np.array_equal(z["cell_A"], parent["cell_A"]):
                raise ValueError("Fixed-cell endpoint has different cell")
            if not np.array_equal(z["occupancy"], data["occupancy"][row]):
                raise ValueError("Saved initial occupancy differs from frozen cohort")
            slots = z["parent_indices"].astype(int)
            if (not np.array_equal(z["numbers"], parent["numbers"][slots]) or
                    not np.array_equal(z["initial_positions_A"], parent["positions_A"][slots])):
                raise ValueError("Initial atomic species/coordinates differ from the parent deletion")
            if geometry_hash(z["numbers"], z["initial_positions_A"], z["cell_A"]) != record["initial_geometry_hash"]:
                raise ValueError("Initial literal geometry binding differs")
            if geometry_hash(z["numbers"], z["endpoint_positions_A"], z["cell_A"]) != record["endpoint_literal_geometry_hash"]:
                raise ValueError("Endpoint literal geometry binding differs")
            measured_fmax = float(np.linalg.norm(z["endpoint_forces_eV_A"], axis=1).max())
            if measured_fmax != record["endpoint_force_max_eV_A"]:
                raise ValueError("Physical endpoint force does not match its scalar receipt")
            e0, e1 = float(z["initial_energy_eV"]), float(z["endpoint_energy_eV"])
            if not np.isfinite([e0, e1]).all() or e0 != record["initial_energy_eV"] or e1 != record["endpoint_energy_eV"]:
                raise ValueError("Paired scalar receipt disagrees with coordinate arrays")
            natoms = len(z["numbers"])
        item = dict(row=row, deletion_energy_eV=e0, endpoint_energy_eV=e1, n_atoms=natoms)
        (test_labels if str(data["split"][row]) == "test" else development).append(item)
    write_csv(args.output_dir / "attempt_status.csv", statuses)
    atomic_json(args.output_dir / "sealed_test_manifest.json", dict(
        status="UNSEALED_FOR_FINAL_ASSESSMENT" if args.unseal_test else "TEST_ENERGIES_SEALED",
        original_cohort_roles_sha256=sha256(args.prepared / "cohort.npz"),
        records=[r for r in checked_sources if r["split"] == "test"],
        energy_values_read=bool(args.unseal_test),
        scope="Raw coordinate/force and sealed energy bytes hash-bound. No TEST values enter model selection."))
    if missing and not args.allow_incomplete:
        atomic_json(args.output_dir / "receipt.json", dict(status="INCOMPLETE", missing=missing,
            expected=len(selected), present=len(selected)-len(missing), test_energy_unsealed=args.unseal_test))
        raise RuntimeError(f"Missing {len(missing)} frozen endpoint attempts; statuses retained")
    if worker_receipts:
        representative = next(iter(worker_receipts.values()))["receipt"]["model_receipt"]
        inference = dict(status="VERIFIED_EXECUTED_PAIRED_INFERENCE", teacher=representative,
            context=context, context_hash=context["context_hash"],
            workers=[{k:v for k,v in w.items() if k != "receipt"} for w in worker_receipts.values()],
            precision=dict(dtype="float32", tf32=False),
            scope="Executed teacher and numerical endpoint context, not an accuracy certificate")
        atomic_json(args.output_dir / "inference_receipt.json", inference)
    support, schemas = [], {}
    if development and all(any(data["split"][r["row"]] == "train" and data["n_target"][r["row"]] == n for r in development)
                           for n in preparation["counts"]):
        selected_rows = np.array([r["row"] for r in development], int)
        common = dict(occupancy=data["occupancy"][selected_rows], n_target=data["n_target"][selected_rows],
            n_Li=data["n_target"][selected_rows], split=data["split"][selected_rows],
            canonical_masks=data["canonical_masks"][selected_rows], structure_ids=data["structure_ids"][selected_rows],
            weight=np.ones(len(development), np.float32), parent_sha256=np.array(context["parent_sha256"]),
            paired_cohort_row=selected_rows, authoritative_roles_sha256=np.array(sha256(args.prepared / "cohort.npz")))
        for target, protocol in [("deletion", DELETION_PROTOCOL), ("endpoint", ENDPOINT_PROTOCOL)]:
            total = np.array([r[target + "_energy_eV"] for r in development])
            energy = total / np.array([r["n_atoms"] for r in development])
            filename = f"learning_{target}.npz"
            atomic_npz(args.output_dir / filename, **common, energy_eV=total, energy_eV_atom=energy)
            schema = fit_schema(common["n_target"], energy, common["split"], n_sites=len(mobile_indices(parent)),
                provenance=dict(geometry_protocol=protocol, energy_target=target,
                    data_sha256=sha256(args.output_dir / filename), parent_sha256=context["parent_sha256"],
                    authoritative_full_roles_sha256=sha256(args.prepared / "cohort.npz"),
                    inference_receipt=dict(path=str((args.output_dir / "inference_receipt.json").absolute()),
                                           sha256=sha256(args.output_dir / "inference_receipt.json")),
                    teacher_precision=context["teacher"], endpoint_context_hash=context["context_hash"]))
            save_schema(schema, args.output_dir / f"condition_schema_{target}.json")
            schemas[target] = schema
            bins = classify(schema, common["n_target"], energy)
            for n in preparation["counts"]:
                for split in ["train", "val"]:
                    mask = (common["n_target"] == n) & (common["split"] == split)
                    support.append(dict(target=target, n_target=n, split=split, independent_masks=int(mask.sum()),
                        **{name:int(np.sum(mask & (bins == name))) for name in ["low", "middle", "high"]}))
    write_csv(args.output_dir / "development_class_support.csv", support)
    gate = bool(support) and all(r["low"] >= args.minimum_low_support for r in support)
    gate = gate and all(r[name] > 0 for r in support if r["split"] == "train" for name in ["low", "middle", "high"])
    if args.unseal_test and test_labels:
        atomic_json(args.output_dir / "final_test_labels.json", test_labels)
        if schemas:
            test_support = []
            for target, schema in schemas.items():
                n = np.array([data["n_target"][r["row"]] for r in test_labels])
                e = np.array([r[target + "_energy_eV"] / r["n_atoms"] for r in test_labels])
                bins = classify(schema, n, e)
                for count in preparation["counts"]:
                    test_support.append(dict(target=target, n_target=count, split="test",
                        independent_masks=int(np.sum(n == count)),
                        **{name:int(np.sum((n == count) & (bins == name))) for name in ["low", "middle", "high"]}))
            write_csv(args.output_dir / "final_test_class_support.csv", test_support)
    elapsed = np.array([r["elapsed_seconds"] for r in statuses if r["status"] != "missing"])
    status_counts = Counter(r["status"] for r in statuses)
    report = dict(status="PASS_COMPLETE" if not missing else "PRELIMINARY_INCOMPLETE",
        collected_UTC=datetime.now(timezone.utc).isoformat(), selection=args.selection,
        original_roles_sha256=sha256(args.prepared / "cohort.npz"), context_hash=context["context_hash"],
        expected_attempts=len(selected), terminal_attempts=len(selected)-len(missing), missing=missing,
        statuses=dict(status_counts), development_converged_pairs=len(development),
        test_energies_sealed=not args.unseal_test, test_labels_used_for_thresholds=False,
        paired_target_fit_gate="READY_FOR_MATCHED_PAIRED_TARGET_FITS" if gate else "DEFERRED_INSUFFICIENT_TRAIN_VAL_SUPPORT",
        support_gate=dict(minimum_low_masks_per_count_per_train_val_role=args.minimum_low_support,
                          provenance="Operational support gate from current goal; not a formal power calculation"),
        EF_attempts=sum(r["EF_attempts"] for r in statuses),
        worker_startup_and_idle_seconds={k:w["receipt"].get("elapsed_seconds_including_loading_idle") for k,w in worker_receipts.items()},
        per_attempt_seconds=dict(median=float(np.median(elapsed)), p95=float(np.quantile(elapsed,.95)),
            mean=float(np.mean(elapsed)), maximum=float(np.max(elapsed))) if len(elapsed) else {},
        occupation_merge_groups=[dict(endpoint_occupation_orbit=k, cohort_rows=rr,
            roles=sorted(set(data["split"][rr].tolist()))) for k,rr in rows_by_endpoint.items() if len(rr)>1],
        merge_scope="Occupation-key coincidences only, not certified common PES basins; no roles reassigned", sources=checked_sources,
        source_sha256=sha256(__file__), no_new_DFT=True)
    atomic_json(args.output_dir / "receipt.json", report)
    print(json.dumps({k:report[k] for k in ["status", "terminal_attempts", "statuses", "test_energies_sealed", "paired_target_fit_gate", "per_attempt_seconds", "EF_attempts"]}, indent=2))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--selection", choices=["all", "pilot"], default="all")
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--unseal-test", action="store_true", help="Final assessment only; never model selection")
    p.add_argument("--minimum-low-support", type=int, default=30)
    return collect(p.parse_args(argv))


if __name__ == "__main__":
    main()

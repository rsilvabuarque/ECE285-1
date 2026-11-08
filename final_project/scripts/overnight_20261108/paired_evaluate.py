"""All-attempt proposal assessment under both deletion and endpoint targets.

Only numerically converged, compatible endpoints receive endpoint classes.
Failed/limited optimizations retain their attempt denominators and unknown
endpoint classes. Orbit novelty refers to initial masks, not certified PES basins.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from paired_common import atomic_json, check_prepared, identity, load_npz, sha256


def rows_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="") as stream:
            w = csv.DictWriter(stream, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def evaluate(args):
    from cathode_cvae.runtime import require_compute_allocation
    from cathode_cvae.conditions import classify, load_schema
    require_compute_allocation()
    prep, context = check_prepared(args.prepared)
    registration = json.loads((args.registration / "registration.json").read_text())
    for name, expected in registration["outputs"].items():
        if sha256(args.registration / name) != expected:
            raise ValueError("Frozen endpoint proposal registration changed")
    binding = registration.get("source_attempts")
    if not binding or binding["sha256"] != sha256(args.attempts):
        raise ValueError("All-attempt assessment requires its exact prescoring attempt CSV binding")
    if registration["context_hash"] != context["context_hash"]:
        raise ValueError("Endpoint proposal registration has a different physical context")
    schemas = {"deletion": load_schema(args.deletion_schema), "endpoint": load_schema(args.endpoint_schema)}
    frozen_inputs = {r["sha256"] for r in registration.get("frozen_analysis_inputs", [])}
    for path in [args.deletion_schema, args.endpoint_schema, args.history]:
        if sha256(path) not in frozen_inputs:
            raise ValueError("Threshold/history input was not frozen before endpoint scoring: " + str(path))
    original_thresholds = {}
    original_path = getattr(args, "original_thresholds", None)
    if original_path is not None:
        if sha256(original_path) not in frozen_inputs:
            raise ValueError("Original fixed-q20 thresholds were not frozen before scoring")
        with original_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["teacher"] == "uma" and row["source"] == "main_train":
                    n = int(row["n_Li"])
                    if n in original_thresholds:
                        raise ValueError("Ambiguous original fixed-q20 composition threshold")
                    original_thresholds[n] = float(row["q20_eV_atom"])
    for target, schema in schemas.items():
        provenance = schema["provenance"]
        if provenance.get("endpoint_context_hash") != context["context_hash"]:
            raise ValueError("Separate TRAIN-fitted target schema has a different endpoint context")
        if provenance.get("energy_target") != target:
            raise ValueError("Deletion and endpoint condition schemas were interchanged")
    history_payload = json.loads(args.history.read_text())
    history = set(history_payload if isinstance(history_payload, list) else next(history_payload[k] for k in
        ["keys","canonical_masks","known_keys","canonical_keys"] if k in history_payload))
    facts_path=getattr(args,"facts",None);facts=None
    if facts_path is not None:
        facts=json.loads(facts_path.read_text())
        if (facts.get("status")!="FROZEN_COMPLETE_ENDPOINT_FACTS" or facts.get("schema")!="paired-endpoint-facts-v1" or
                facts.get("context_hash")!=context["context_hash"] or facts.get("context")!=context or
                identity({k:v for k,v in facts.items() if k!="identity"})!=facts["identity"]):
            raise ValueError("Compact endpoint facts are incomplete, corrupted or context incompatible")
        expected=getattr(args,"facts_sha256",None)
        if expected is not None and sha256(facts_path)!=expected:raise ValueError("Compact facts outer SHA256 differs from the release manifest")
    bindings = json.loads((args.registration / "candidate_bindings.json").read_text())
    by_id = {r["source_structure_id"]:r for r in bindings}
    by_key = {r["canonical_vacancy_key"]:r for r in bindings}
    cohort = load_npz(args.prepared / "cohort.npz")
    protected_test = set(cohort["canonical_masks"][cohort["split"] == "test"].astype(str))
    with args.attempts.open(newline="") as stream:
        attempts = list(csv.DictReader(stream))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    source_records = []
    missing = set()
    for key, b in by_key.items():
        if key in protected_test and not args.unseal_test:
            results[key] = dict(terminal=False, status="protected_test_energy_sealed")
            missing.add(b["task_id"])
            continue
        if facts is not None:
            item=facts["records"].get(b["physical_key"])
            if item is None:raise ValueError("Compact facts omit a prespecified physical query")
            if item["canonical_vacancy_key"]!=key:raise ValueError("Compact facts initial mask binding differs")
            record=item["record"];claim=dict(claimed_unix=item["claimed_unix"]);source_record=dict(item["source"])
        else:
            directory = args.queue / "results" / b["task_id"]
            path = directory / "result.json"
            if not path.exists():
                results[key] = dict(terminal=False, status="missing")
                missing.add(b["task_id"])
                continue
            record = json.loads(path.read_text())
            if sha256(directory / "arrays.npz") != record["arrays_sha256"]:
                raise ValueError("Endpoint raw data bytes changed")
            claim_path=args.queue / "claims" / (b["task_id"] + ".json")
            claim = json.loads(claim_path.read_text())
            source_record=dict(task_id=b["task_id"],result_sha256=sha256(path),arrays_sha256=record["arrays_sha256"],claim_sha256=sha256(claim_path))
        if record["physical_key"] != b["physical_key"] or record["context_hash"] != context["context_hash"]:
            raise ValueError("Candidate endpoint uses a different initial geometry or numerical protocol")
        info = dict(terminal=True, status=record["status"], endpoint_known=bool(record["endpoint_labels_usable"]),
            deletion_known=record["initial_energy_eV"] is not None,
            deletion_energy_eV_atom=record["initial_energy_eV"] / record["n_atoms"] if record["initial_energy_eV"] is not None else None,
            endpoint_energy_eV_atom=record["endpoint_energy_eV"] / record["n_atoms"] if record["endpoint_labels_usable"] else None,
            endpoint_occupation_key=(record.get("mapping") or {}).get("endpoint_occupation_orbit", ""),
            EF_attempts=record["calculator_EFS_attempts"],
            new_endpoint_task=claim["claimed_unix"] >= registration["frozen_unix"],
            preexisting_compatible_endpoint=claim["claimed_unix"] < registration["frozen_unix"])
        for target, schema in schemas.items():
            energy = info[target + "_energy_eV_atom"]
            info[target + "_bin"] = str(classify(schema, np.array([record["n_target"]]), np.array([energy]))[0]) if energy is not None else "unknown"
        results[key] = info
        source_records.append(source_record)
    outcomes = []
    for i, raw in enumerate(attempts):
        r = dict(raw)
        sid = r.get("scoring_structure_id", r.get("structure_id", ""))
        b = by_id.get(sid)
        key = r.get("canonical_mask", "")
        if b is not None and key and key != b["canonical_vacancy_key"]:
            raise ValueError("Attempt mask disagrees with its registered candidate identity")
        if b is None and key:
            b = by_key.get(key)
        valid = b is not None and r.get("status", "") not in {"failed", "error", "invalid", "generation_failed"}
        if valid:
            key = b["canonical_vacancy_key"]
        info = results.get(key, {}) if valid else {}
        n = int(r.get("n_Li") or r.get("n_target") or (len(key) - key.count("1") if key else -1))
        requested = r.get("requested_bin", "")
        if requested not in {"low", "middle", "high"}:
            raise ValueError("Every comparison attempt must retain its prespecified requested bin")
        r.update(paired_attempt_row=i, paired_n_target=n, paired_initial_key=key,
            paired_valid=valid, paired_novel=bool(valid and key not in history),
            paired_endpoint_status=info.get("status", "invalid_or_unregistered"),
            paired_endpoint_task=b["task_id"] if b else "",
            paired_endpoint_occupation_key=info.get("endpoint_occupation_key", ""),
            paired_endpoint_new_task=info.get("new_endpoint_task", False))
        for target in schemas:
            r["paired_" + target + "_bin"] = info.get(target + "_bin", "unknown")
            r["paired_" + target + "_energy_eV_atom"] = info.get(target + "_energy_eV_atom")
        e0 = info.get("deletion_energy_eV_atom")
        r["paired_original_q20_deletion_low"] = bool(e0 is not None and n in original_thresholds and e0 <= original_thresholds[n])
        outcomes.append(r)
    rows_csv(args.output_dir / "attempt_outcomes.csv", outcomes)
    grouped = defaultdict(list)
    for r in outcomes:
        replicate = r.get("replicate_id") or r.get("fit_seed") or r.get("seed", "")
        grouped[(r["method"], replicate, r["paired_n_target"], r["requested_bin"])].append(r)
    cells, confusion = [], []
    for (method, replicate, n, requested), rr in sorted(grouped.items()):
        denominator = len(rr)
        row = dict(method=method, replicate=replicate, n_target=n, requested_bin=requested,
            attempts=denominator, distinct_initial_orbits=len({r["paired_initial_key"] for r in rr if r["paired_valid"]}),
            novel_initial_orbits=len({r["paired_initial_key"] for r in rr if r["paired_novel"]}))
        if original_path is not None and requested == "low":
            if n not in original_thresholds:
                raise ValueError("Original fixed-q20 threshold unavailable for requested composition")
            old_low = {r["paired_initial_key"] for r in rr if r["paired_novel"] and r["paired_original_q20_deletion_low"]}
            row["original_q20_deletion_useful_orbits"] = len(old_low)
            row["original_q20_deletion_useful_yield"] = len(old_low) / denominator
        else:
            row["original_q20_deletion_useful_orbits"] = None
            row["original_q20_deletion_useful_yield"] = None
        for target in schemas:
            useful = {r["paired_initial_key"] for r in rr if r["paired_novel"] and r["paired_" + target + "_bin"] == requested}
            energies = [r["paired_" + target + "_energy_eV_atom"] for r in rr if r["paired_" + target + "_energy_eV_atom"] is not None]
            row[target + "_useful_novel_orbits"] = len(useful)
            row[target + "_useful_yield"] = len(useful) / denominator
            row[target + "_known_attempts"] = len(energies)
            row[target + "_mean_energy_eV_atom"] = float(np.mean(energies)) if energies else None
            if target == "endpoint":
                row["useful_endpoint_occupation_classes"] = len({r["paired_endpoint_occupation_key"] for r in rr
                    if r["paired_initial_key"] in useful and r["paired_endpoint_occupation_key"]})
            for achieved in ["low", "middle", "high", "unknown"]:
                confusion.append(dict(method=method, replicate=replicate, n_target=n,
                    requested_bin=requested, target=target, achieved_bin=achieved,
                    attempts=sum(r["paired_" + target + "_bin"] == achieved for r in rr)))
        cells.append(row)
    rows_csv(args.output_dir / "cells.csv", cells)
    rows_csv(args.output_dir / "confusion.csv", confusion)
    summaries, fit_summaries = [], []
    for method in sorted({r["method"] for r in cells}):
        rr = [r for r in cells if r["method"] == method]
        for target in schemas:
            fit_means = []
            for replicate in sorted({r["replicate"] for r in rr}):
                ff = [r for r in rr if r["replicate"] == replicate]
                value = float(np.mean([r[target + "_useful_yield"] for r in ff]))
                fit_means.append(value)
                fit_summaries.append(dict(method=method, replicate=replicate, target=target,
                    equal_weight_count_request_yield=value, cells=len(ff), attempts=sum(r["attempts"] for r in ff)))
            summaries.append(dict(method=method, target=target, equal_weight_cell_mean_yield=float(np.mean([r[target + "_useful_yield"] for r in rr])),
                cells=len(rr), attempts=sum(r["attempts"] for r in rr),
                replicates=len(fit_means), mean_replicate_macro_yield=float(np.mean(fit_means)),
                std_replicate_macro_yield=float(np.std(fit_means, ddof=1)) if len(fit_means)>1 else None,
                useful_distinct_per_cell_sum=sum(r[target + "_useful_novel_orbits"] for r in rr),
                scope="Descriptive equal-weight count/request/fit cells; draw-level repeats do not increase independent fit count"))
    rows_csv(args.output_dir / "summary.csv", summaries)
    rows_csv(args.output_dir / "fit_summary.csv", fit_summaries)
    report = dict(status="PASS_COMPLETE" if not missing else "INCOMPLETE_LABELS_OR_SEALED_TEST",
        evaluated_UTC=datetime.now(timezone.utc).isoformat(), attempts=len(attempts),
        registered_unique_physical_candidates=len(by_key), missing_or_sealed_tasks=sorted(missing),
        endpoint_statuses=dict(Counter(v["status"] for v in results.values())),
        unknown_endpoint_attempts=sum(r["paired_endpoint_bin"] == "unknown" for r in outcomes),
        all_attempts_retained=True, thresholds="Separate frozen TRAIN-only deletion and converged-endpoint schemas",
        novelty_history_sha256=sha256(args.history), test_energy_unsealed=args.unseal_test,
        source_inputs={str(path):sha256(path) for path in [args.attempts, args.deletion_schema, args.endpoint_schema,
            args.history, args.registration / "registration.json"] + ([original_path] if original_path is not None else [])}, source_results=source_records,
        EF_attempts_unique_physical_tasks=sum(v.get("EF_attempts", 0) for v in results.values()),
        newly_claimed_endpoint_tasks=sum(v.get("new_endpoint_task", False) for v in results.values()),
        cached_preexisting_compatible_tasks=sum(v.get("preexisting_compatible_endpoint", False) for v in results.values()),
        source_sha256=sha256(__file__), no_new_DFT=True,
        mean_energy_scope="Available deletion labels and converged compatible endpoints only; unknown outcomes remain in every yield denominator",
        original_fixed_q20_scope="Initial deletion energies only; never applied to shifted endpoint energies" if original_path is not None else "OMITTED; supply --original-thresholds for the required legacy LOW metric",
        compact_facts_sha256=sha256(facts_path) if facts_path is not None else None,
        raw_arrays_rehashed_in_this_replay=facts is None,
        request_vs_assessment_schema="Generator requests retain their own recorded schema. Paired-calibration deletion/endpoint schemas define separate assessment classes; the original fixed-q20 metric is separately named.",
        limitation="Distinct initial or endpoint occupation orbits are not certified distinct PES minima; failures remain in attempt denominators")
    atomic_json(args.output_dir / "receipt.json", report)
    if missing and not args.allow_incomplete:
        raise RuntimeError("Endpoint pool incomplete; all outcomes retained but no complete-comparison claim")
    print(json.dumps({k:report[k] for k in ["status", "attempts", "endpoint_statuses", "unknown_endpoint_attempts", "newly_claimed_endpoint_tasks"]}, indent=2))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--registration", type=Path, required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument("--queue", type=Path)
    source.add_argument("--facts",type=Path,help="Portable scalar facts; makes no access to the original HPC queue")
    p.add_argument("--facts-sha256",help="Expected compact facts SHA256 from the authenticated download manifest")
    p.add_argument("--attempts", type=Path, required=True)
    p.add_argument("--deletion-schema", type=Path, required=True)
    p.add_argument("--endpoint-schema", type=Path, required=True)
    p.add_argument("--original-thresholds", type=Path, help="Legacy generation_thresholds.csv; UMA main_train q20 applied only to deletion energies")
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--unseal-test", action="store_true", help="Final assessment only if a candidate overlaps the acquisition TEST set")
    return evaluate(p.parse_args(argv))


if __name__ == "__main__":
    main()

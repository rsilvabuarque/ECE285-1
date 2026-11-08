"""Freeze endpoint work for a complete proposal pool without outcome filtering.

Physical work is deduplicated under exact initial parent geometry plus endpoint
context; the original proposal attempt CSV is retained by its owner and its
denominator is never replaced with the number of physical tasks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from paired_common import (atomic_json, atomic_npz, canonical, check_prepared,
    load_npz, sha256, structure_id, task_for_row)


def register(args):
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    preparation, context = check_prepared(args.prepared)
    parent = load_npz(args.prepared / "parent.npz")
    with np.load(args.pool, allow_pickle=False) as z:
        occupancy = z["occupancy"] if "occupancy" in z else 1 - z["masks"]
        ids = z["structure_ids"].astype(str)
        if "parent_sha256" in z and str(z["parent_sha256"].item()) != context["parent_sha256"]:
            raise ValueError("Proposal pool belongs to a different physical parent")
    if len(ids) != len(occupancy) or len(set(ids)) != len(ids):
        raise ValueError("Candidate pool must have one unique record ID per geometry row")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    unique = {}
    bindings = []
    for row, (source_id, raw) in enumerate(zip(ids, occupancy)):
        key, gather = canonical(raw, parent["permutations"])
        if key not in unique:
            unique[key] = len(unique)
        bindings.append(dict(source_row=row, source_structure_id=source_id,
            canonical_vacancy_key=key, canonical_structure_id=structure_id(key),
            canonical_pool_row=unique[key], gather=gather.tolist()))
    keys = list(unique)
    canonical_pool = args.output_dir / "canonical_pool.npz"
    atomic_npz(canonical_pool, occupancy=np.asarray([1 - np.array(list(k), np.uint8) for k in keys]),
        structure_ids=np.asarray([structure_id(k) for k in keys]), canonical_masks=np.asarray(keys),
        parent_sha256=np.array(context["parent_sha256"]))
    pool_sha = sha256(canonical_pool)
    tasks = [task_for_row(args.prepared, i, key, "assessment", canonical_pool, pool_sha) for i, key in enumerate(keys)]
    known_tasks = json.loads((args.prepared / "tasks_all.json").read_text())["tasks"]
    prior_sources = []
    for path in args.prior_task_files:
        payload = json.loads(path.read_text())
        if payload["context_hash"] != context["context_hash"]:
            raise ValueError("Prior queue has a different endpoint context")
        known_tasks.extend(payload["tasks"])
        prior_sources.append(dict(path=str(path.absolute()), sha256=sha256(path)))
    union = {r["physical_key"] for r in known_tasks + tasks}
    if len(union) + args.reserved_other_protocol_attempts > args.maximum_primary_attempts:
        raise ValueError("Complete frozen endpoint queue exceeds the declared primary-attempt cap")
    for b in bindings:
        b["task_id"] = tasks[b["canonical_pool_row"]]["task_id"]
        b["physical_key"] = tasks[b["canonical_pool_row"]]["physical_key"]
    atomic_json(args.output_dir / "candidate_bindings.json", bindings)
    atomic_json(args.output_dir / "tasks.json", dict(schema="paired-relaxation-task-list-v1",
        context_hash=context["context_hash"], tasks=tasks))
    record = dict(status="FROZEN_BEFORE_NEW_ENDPOINT_SCORING", frozen_unix=time.time(),
        frozen_UTC=datetime.now(timezone.utc).isoformat(), label=args.label,
        source_pool=dict(path=str(args.pool.absolute()), sha256=sha256(args.pool)),
        source_attempts=dict(path=str(args.attempts.absolute()), sha256=sha256(args.attempts)) if args.attempts else None,
        context_hash=context["context_hash"], input_candidate_rows=len(ids), unique_physical_tasks=len(tasks),
        initial_acquisition_tasks=preparation["independent_masks"],
        complete_known_queue_unique_tasks=len(union), reserved_other_protocol_attempts=args.reserved_other_protocol_attempts,
        primary_attempt_cap=args.maximum_primary_attempts, prior_task_sources=prior_sources,
        outcomes_read=False, no_energy_filtering=True, attempts_denominator="Original attempt CSV, including duplicates, failures and repeats",
        source_sha256=sha256(__file__), outputs={name:sha256(args.output_dir / name) for name in
            ["canonical_pool.npz", "candidate_bindings.json", "tasks.json"]})
    atomic_json(args.output_dir / "registration.json", record)
    print(json.dumps({k:record[k] for k in ["status", "input_candidate_rows", "unique_physical_tasks", "complete_known_queue_unique_tasks"]}, indent=2))
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--attempts", type=Path)
    p.add_argument("--label", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prior-task-files", nargs="*", type=Path, default=[])
    p.add_argument("--maximum-primary-attempts", type=int, default=18000)
    p.add_argument("--reserved-other-protocol-attempts", type=int, default=96)
    return register(p.parse_args(argv))


if __name__ == "__main__":
    main()

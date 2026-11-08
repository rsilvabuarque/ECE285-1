"""Audit all saved accepted-step geometries without reading energy labels.

A triangle-inequality certificate avoids a full same-species assignment when
every atom remains inside half its original site's nearest-neighbor distance.
Outside that certificate the exact periodic assignment diagnostics are used.
Neither occupation keys nor this geometric check certify physical PES basins.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import time

import numpy as np

from paired_common import (atomic_json, check_prepared, load_npz, mapping_diagnostics,
                           mobile_indices, sha256)


def nearest_parent_separations(parent):
    from ase.geometry import find_mic
    xyz, numbers, cell = parent["positions_A"], parent["numbers"], parent["cell_A"]
    minimum = np.full(len(numbers), np.inf)
    for z in np.unique(numbers):
        ids = np.flatnonzero(numbers == z)
        if len(ids) < 2:
            continue
        vectors = xyz[ids, None] - xyz[ids][None]
        _, distance = find_mic(vectors.reshape(-1, 3), cell, pbc=True)
        distance = distance.reshape(len(ids), len(ids))
        np.fill_diagonal(distance, np.inf)
        minimum[ids] = distance.min(axis=1)
    return minimum


def audit_frames(parent, slots, positions, minimum):
    from ase.geometry import find_mic
    slots = np.asarray(slots, int)
    mobile = np.isin(slots, mobile_indices(parent))
    events = []
    certified = 0
    minimum_margin_bound = np.inf
    for frame, xyz in enumerate(positions):
        displacement, _ = find_mic(xyz - parent["positions_A"][slots], parent["cell_A"], pbc=True)
        shift = displacement[~mobile].mean(axis=0) if (~mobile).any() else np.zeros(3)
        aligned, radius = find_mic(displacement - shift, parent["cell_A"], pbc=True)
        bounds = minimum[slots] - 2 * radius
        finite = bounds[np.isfinite(bounds)]
        if len(finite):
            minimum_margin_bound = min(minimum_margin_bound, float(finite.min()))
        if np.all(bounds > 0):
            certified += 1
            continue
        mapping = mapping_diagnostics(parent["numbers"][slots], xyz, parent, slots)
        events.append(dict(frame=frame,
            mobile_assignment_changes=mapping["mobile_assignment_changes"],
            host_assignment_changes=mapping["host_assignment_changes"],
            mobile_nearest_collisions=mapping["mobile_nearest_collisions"],
            host_nearest_collisions=mapping["host_nearest_collisions"],
            endpoint_occupation_orbit=mapping["endpoint_occupation_orbit"],
            nearest_parent_indices=mapping["nearest_parent_indices"],
            assigned_parent_indices=mapping["assigned_parent_indices"],
            nearest_second_gap_A=mapping["nearest_second_gap_A"]))
    return dict(saved_accepted_step_frames=len(positions), identity_certified_frames=certified,
        full_mapping_frames=len(events),
        minimum_triangle_gap_bound_A=None if not np.isfinite(minimum_margin_bound) else minimum_margin_bound,
        frames_with_mobile_reassignment=sum(r["mobile_assignment_changes"] > 0 for r in events),
        frames_with_host_reassignment=sum(r["host_assignment_changes"] > 0 for r in events),
        frames_with_nearest_collisions=sum(r["mobile_nearest_collisions"] + r["host_nearest_collisions"] > 0 for r in events),
        full_mapping_events=events,
        certificate="For each atom r<d_min/2 after common host translation, its original same-species site is uniquely nearest by the periodic-distance triangle inequality. Original sites are distinct, so this also minimizes injective assignment.")


def audit_task(payload):
    """Independent lineage worker: read coordinate/index keys, never energies."""
    task,row,split,queue,parent,minimum,context_hash=payload
    directory=Path(queue)/"results"/task["task_id"]
    header=directory/"status.json"
    if not header.exists():return None,task["task_id"]
    metadata=json.loads(header.read_text());path=directory/"arrays.npz"
    if sha256(path)!=metadata["arrays_sha256"] or metadata["context_hash"]!=context_hash:
        raise ValueError("Mapping source bytes/context changed")
    with np.load(path,allow_pickle=False) as z:
        slots=z["parent_indices"];trajectory=z["trajectory_positions_A"]
    result=audit_frames(parent,slots,trajectory,minimum)
    result.update(task_id=task["task_id"],cohort_row=row,split=split,arrays_sha256=metadata["arrays_sha256"])
    return result,None


def main(argv=None):
    from cathode_cvae.runtime import require_compute_allocation
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--selection", choices=["pilot", "all"], default="all")
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--workers",type=int,default=1,help="Independent CPU lineage processes; set OMP/OPENBLAS threads=1 when parallel")
    args = p.parse_args(argv)
    require_compute_allocation()
    started=time.monotonic()
    if args.workers<1 or args.workers>int(os.environ.get("SLURM_CPUS_PER_TASK",args.workers)):
        raise ValueError("Mapping process count must fit the caller's allocated CPUs")
    prep, context = check_prepared(args.prepared)
    parent = load_npz(args.prepared / "parent.npz")
    parent={k:v for k,v in parent.items() if k in ["numbers","positions_A","cell_A","target_indices","mobile_indices","li_indices","permutations"]}
    cohort = load_npz(args.prepared / "cohort.npz")
    tasks = json.loads((args.prepared / "tasks_all.json").read_text())["tasks"]
    indices = cohort["pilot_indices"] if args.selection == "pilot" else np.arange(len(tasks))
    minimum = nearest_parent_separations(parent)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results, missing = [], []
    payloads=[(tasks[int(i)],int(i),str(cohort["split"][i]),str(args.queue),parent,minimum,context["context_hash"]) for i in indices]
    executor=ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) if args.workers>1 else None
    try:
        evaluated=executor.map(audit_task,payloads,chunksize=8) if executor else map(audit_task,payloads)
        for result,absent in evaluated:
            if absent:missing.append(absent);continue
            atomic_json(args.output_dir/(result["task_id"]+".json"),result)
            results.append({k:v for k,v in result.items() if k not in ["full_mapping_events","certificate"]})
    finally:
        if executor:executor.shutdown(wait=True,cancel_futures=True)
    if results:
        with (args.output_dir / "lineages.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    receipt = dict(status="PASS_COMPLETE" if not missing else "INCOMPLETE", selection=args.selection,
        audited_UTC=datetime.now(timezone.utc).isoformat(), lineages=len(results), missing=missing,
        saved_step_frames=sum(r["saved_accepted_step_frames"] for r in results),
        identity_certified_frames=sum(r["identity_certified_frames"] for r in results),
        full_mapping_frames=sum(r["full_mapping_frames"] for r in results),
        mobile_reassignment_lineages=sum(r["frames_with_mobile_reassignment"] > 0 for r in results),
        host_reassignment_lineages=sum(r["frames_with_host_reassignment"] > 0 for r in results),
        nearest_collision_lineages=sum(r["frames_with_nearest_collisions"] > 0 for r in results),
        source_sha256=sha256(__file__), context_hash=context["context_hash"], energy_labels_read=False,
        workers=args.workers,elapsed_seconds=time.monotonic()-started,
        scope="All saved accepted optimizer steps and last valid failure geometry. Internal line-search states are not a physical-time trajectory; no PES-basin equivalence claimed.")
    atomic_json(args.output_dir / "receipt.json", receipt)
    if missing and not args.allow_incomplete:
        raise RuntimeError("Missing frozen lineage mapping inputs")
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    main()

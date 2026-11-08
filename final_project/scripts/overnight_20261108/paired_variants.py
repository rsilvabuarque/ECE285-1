"""Freeze bounded TRAIN-only tightening and independent-MACE endpoint controls.

Rows are selected from the original prespecified pilot, without reading outcome
energies. Both relaxations start from exactly the same pristine-derived deletion
geometry. MACE energies evaluated on UMA coordinates are a separate operation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from paired_common import atomic_json, check_prepared, identity, load_npz, physical_key, sha256


def prepare_variants(args):
    from cathode_cvae.runtime import require_compute_allocation
    from cathode_cvae.mlip.registry import MODELS
    require_compute_allocation()
    prep, base = check_prepared(args.prepared)
    data = load_npz(args.prepared / "cohort.npz")
    base_tasks = json.loads((args.prepared / "tasks_all.json").read_text())["tasks"]
    selected = {}
    for name, amount in [("tight_0p01", args.tight_per_count), ("mace_independent", args.mace_per_count)]:
        indices = []
        for n in prep["counts"]:
            available = [int(i) for i in data["pilot_indices"] if data["n_target"][i] == n]
            if amount > len(available):
                raise ValueError("Control would exceed the predeclared TRAIN pilot support")
            indices.extend(available[:amount])
        selected[name] = indices
    args.output_dir.mkdir(parents=True, exist_ok=False)
    variants = {}
    for name, indices in selected.items():
        destination = args.output_dir / name
        destination.mkdir()
        context = json.loads(json.dumps({k:v for k,v in base.items() if k != "context_hash"}))
        context["comparison"] = dict(base_context_hash=base["context_hash"],
            selected_cohort_rows=indices, selection="First frozen TRAIN pilot rows per count; no energies read",
            role="TRAIN only; operational sensitivity control, never a TEST-selected adjustment")
        if name == "tight_0p01":
            context["optimizer"]["fmax_eV_A"] = .01
            context["comparison"]["change"] = "Only physical force stopping target .05->.01 eV/A; rerun identical original starts to preserve FIRE history"
        else:
            spec = MODELS["mace"]
            context["teacher"] = dict(model="mace", checkpoint_hashes=spec["files"], head=spec["head"],
                package_version=spec["version"], dtype="float32", tf32=False)
            context["comparison"]["change"] = "Independent MACE relaxation from original pristine deletion; same fixed cell, FIRE and stopping targets"
        context["context_hash"] = identity(context)
        tasks = []
        for i in indices:
            if data["split"][i] != "train":
                raise ValueError("Control subset crosses protected roles")
            task = dict(base_tasks[i])
            task["source_primary_task_id"] = task["task_id"]
            task["physical_key"] = physical_key(context["context_hash"], task["canonical_vacancy_key"])
            task["task_id"] = "relax_" + task["physical_key"][:24]
            task["context_hash"] = context["context_hash"]
            tasks.append(task)
        atomic_json(destination / "context.json", context)
        atomic_json(destination / "tasks.json", dict(context_hash=context["context_hash"], tasks=tasks))
        receipt = dict(status="FROZEN_BEFORE_CONTROL_SCORING", frozen_unix=time.time(),
            frozen_UTC=datetime.now(timezone.utc).isoformat(), name=name, count=len(tasks),
            primary_preparation=dict(path=str((args.prepared / "preparation.json").absolute()),sha256=sha256(args.prepared / "preparation.json")),
            original_roles_sha256=sha256(args.prepared / "cohort.npz"), labels_read=False,
            source_sha256=sha256(__file__), outputs={filename:sha256(destination / filename) for filename in ["context.json","tasks.json"]})
        atomic_json(destination / "preparation.json", receipt)
        variants[name] = dict(count=len(tasks), context_hash=context["context_hash"], preparation_sha256=sha256(destination / "preparation.json"))
    atomic_json(args.output_dir / "receipt.json", dict(status="FROZEN_CONTROLS", variants=variants,
        new_primary_teacher_endpoint_attempts=len(selected["tight_0p01"]), secondary_teacher_endpoint_attempts=len(selected["mace_independent"]),
        source_energy_labels_read=False, scope="Sensitivity and teacher disagreement, not DFT accuracy or calibrated uncertainty"))
    print(json.dumps(variants,indent=2))
    return variants


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--tight-per-count",type=int,default=32)
    p.add_argument("--mace-per-count",type=int,default=8)
    return prepare_variants(p.parse_args(argv))


if __name__=="__main__":main()

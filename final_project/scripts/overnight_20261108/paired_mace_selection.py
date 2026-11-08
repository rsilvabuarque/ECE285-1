"""Freeze a hash-selected MACE-SP subset across ALL independent TRAIN masks.

Only the original masks/roles/context are read. Endpoint energies, convergence,
availability and worker order cannot affect selection. Coordinate export is a
later operation after the complete primary endpoint queue has finished.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from paired_common import atomic_json, check_prepared, identity, load_npz, sha256


def freeze(args):
    from cathode_cvae.runtime import require_compute_allocation
    from cathode_cvae.mlip.registry import MODELS
    require_compute_allocation()
    preparation, context=check_prepared(args.prepared)
    path=args.prepared/"cohort.npz"
    data=load_npz(path)
    if args.per_count<1:raise ValueError("Select a positive number of independent TRAIN rows")
    cohort_sha=sha256(path)
    selected=[];eligible={};ranked_hashes={}
    for count in preparation["counts"]:
        rows=np.flatnonzero((data["split"]=="train") & (data["n_target"]==count))
        if args.per_count>len(rows):raise ValueError("MACE subset exceeds independent TRAIN support")
        ranked=[]
        for row in rows:
            row=int(row);key=str(data["canonical_masks"][row])
            score=identity(dict(namespace="paired-MACE-SP-TRAIN-hash-v1",seed=args.seed,
                parent_sha256=context["parent_sha256"],cohort_sha256=cohort_sha,n_target=count,canonical_key=key))
            ranked.append((score,key,row))
        ranked.sort()
        eligible[str(count)]=len(ranked)
        ranked_hashes[str(count)]=identity([list(r) for r in ranked])
        for score,key,row in ranked[:args.per_count]:
            selected.append(dict(cohort_row=row,structure_id=str(data["structure_ids"][row]),
                canonical_mask=key,n_target=count,split="train",selection_score_sha256=score))
    args.output_dir.mkdir(parents=True,exist_ok=False)
    record=dict(status="FROZEN_BEFORE_SELECTED_ENDPOINT_LABELS_INSPECTED",frozen_unix=time.time(),
        frozen_UTC=datetime.now(timezone.utc).isoformat(),selection_seed=args.seed,per_count=args.per_count,
        eligible_train_per_count=eligible,ranked_eligible_identities_sha256=ranked_hashes,selected=selected,
        primary_context_hash=context["context_hash"],parent_sha256=context["parent_sha256"],
        source_cohort=dict(path=str(path.absolute()),sha256=cohort_sha),
        source_preparation=dict(path=str((args.prepared/"preparation.json").absolute()),sha256=sha256(args.prepared/"preparation.json")),
        secondary_teacher=dict(model="mace",checkpoint_hashes=MODELS["mace"]["files"],head=MODELS["mace"]["head"],
            package_version=MODELS["mace"]["version"],dtype="float32",tf32=False),
        energies_read=False,endpoint_files_read=False,
        scope=f"{args.per_count}/count selected across all TRAIN by deterministic SHA256 order, not only the pilot, before inspecting selected labels; failures will remain in the prespecified denominator",
        source_sha256=sha256(__file__))
    record["identity"]=identity(record)
    atomic_json(args.output_dir/"selection.json",record)
    print(json.dumps({k:record[k] for k in ["status","eligible_train_per_count","per_count","energies_read","endpoint_files_read","identity"]},indent=2))
    return record


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--per-count",type=int,default=32)
    p.add_argument("--seed",type=int,default=2026110808)
    return freeze(p.parse_args(argv))


if __name__=="__main__":main()

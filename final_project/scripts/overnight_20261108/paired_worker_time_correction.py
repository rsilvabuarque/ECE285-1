"""Correct a historical receipt field name without changing any old artifact."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

from paired_common import atomic_json,sha256


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection-dir",type=Path,required=True)
    p.add_argument("--queue",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args(argv)
    original=args.collection_dir/"receipt.json"
    before=sha256(original);receipt=json.loads(original.read_text())
    recorded=receipt["worker_startup_and_idle_seconds"]
    setup={};sources=[]
    for worker in recorded:
        path=args.queue/"workers"/(worker+".json");raw=json.loads(path.read_text())
        setup[worker]=raw.get("load_and_setup_seconds")
        sources.append(dict(worker=worker,path=str(path.absolute()),sha256=sha256(path),
            recorded_total_elapsed_seconds=recorded[worker],
            current_worker_total_elapsed_seconds=raw.get("elapsed_seconds_including_loading_idle"),
            measured_load_and_setup_seconds=setup[worker]))
    correction=dict(status="METADATA_NAME_CORRECTION_ONLY",created_UTC=datetime.now(timezone.utc).isoformat(),
        original_receipt=dict(path=str(original.absolute()),sha256=before),
        historical_misnamed_field="worker_startup_and_idle_seconds",
        corrected_field_name="worker_elapsed_seconds_including_setup_work_idle",
        worker_elapsed_seconds_including_setup_work_idle=recorded,
        worker_load_and_setup_elapsed_seconds=setup,source_worker_receipts=sources,
        explanation="Historical values are WHOLE worker elapsed wall time, including relaxation work. They are not startup+idle-only. Separate setup values come only from measured worker.load_and_setup_seconds; no subtraction-based idle estimate is invented.",
        allocation_cost_authority="Top-level Slurm allocation accounting; never add these process times to allocation times", 
        original_scientific_artifacts_changed=False,source_sha256=sha256(__file__))
    if args.output.exists():raise FileExistsError("Preserve the existing correction receipt")
    atomic_json(args.output,correction)
    if sha256(original)!=before:raise ValueError("Original collection receipt changed")
    print(json.dumps(dict(status=correction["status"],workers=len(recorded),original_receipt_sha256=before),indent=2))
    return correction


if __name__=="__main__":main()

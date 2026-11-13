#!/usr/bin/env python3
"""Snapshot this study's Slurm allocation time without counting job steps.

Node-hours describe allocated job-node time, not unique physical-node time;
multiple shared jobs may have used the same node concurrently. Estimated GPU
node-hours use allocated GPUs / 4 for this Perlmutter study. They are not an
official posted accounting charge. Only the two factor-one interactive GPU
QOS values are supported; unexpected configurations fail closed.
"""

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path


BASE_JOB_IDS = (
    61111318, 61111510, 61112881, 61112882, 61112887, 61114110,
    61114111, 61114446, 61114634, 61115055, 61115846, 61116230,
    61116284, 61116475, 61116476, 61116482, 61116526, 61118220,
    61118827, 61119730, 61120158, 61120204, 61120205, 61120654,
    61121244, 61121865, 61122009, 61122069, 61122072, 61122073,
    61122074, 61122813, 61122892, 61122893, 61122894, 61122895,
    61123503, 61123709, 61123967, 61124074, 61124075, 61124470,
    61124519, 61124520, 61124878, 61125498, 61126767, 61126768,
    61127437,
)
COLUMNS = (
    "JobIDRaw", "JobName", "Account", "QOS", "State", "AllocNodes",
    "ElapsedRaw", "AllocTRES",
)
ALLOWED_QOS = {"gpu_interactive", "gpu_shared_interactive"}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_tres(value):
    fields = {}
    for item in value.split(","):
        if item:
            key, sep, val = item.partition("=")
            if not sep or key in fields:
                raise ValueError(f"Malformed or repeated AllocTRES field: {item!r}")
            fields[key] = val
    return fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new JSON receipt path; parent directory must exist")
    parser.add_argument("--job-id", type=int, action="append", default=[],
                        help="additional top-level job ID; repeat as needed")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.output.parent.is_dir():
        parser.error(f"output directory does not exist: {args.output.parent}")
    ids = sorted(set(BASE_JOB_IDS).union(args.job_id))
    if any(job <= 0 for job in ids):
        parser.error("job IDs must be positive integers")

    command = ["sacct", "-j", ",".join(map(str, ids)), "-n", "-X", "-P",
               "-o", ",".join(COLUMNS)]
    started = utc_now()
    output = subprocess.run(command, check=True, text=True,
                            capture_output=True).stdout
    finished = utc_now()
    jobs = {}
    for line in output.splitlines():
        values = line.split("|")
        if len(values) != len(COLUMNS):
            raise ValueError(f"Unexpected sacct row: {line!r}")
        row = dict(zip(COLUMNS, values))
        job = int(row["JobIDRaw"])
        if job not in ids or job in jobs:
            raise ValueError(f"Unexpected or repeated top-level job ID: {job}")
        if row["Account"] != "m5397_g" or row["QOS"] not in ALLOWED_QOS:
            raise ValueError(f"Unexpected account or QOS for {job}: {row}")
        nodes = int(row["AllocNodes"] or 0)
        elapsed = int(row["ElapsedRaw"] or 0)
        tres = parse_tres(row["AllocTRES"])
        gpus = int(tres.get("gres/gpu", 0))
        if nodes < 0 or elapsed < 0 or gpus < 0:
            raise ValueError(f"Negative allocation or elapsed time for {job}")
        if nodes == 0:
            if elapsed != 0 or gpus != 0:
                raise ValueError(f"Zero-node job has elapsed time or GPUs: {job}")
        elif row["QOS"] == "gpu_interactive":
            if gpus != 4 * nodes:
                raise ValueError(f"Expected 4 GPUs per full interactive node: {job}")
        elif nodes != 1 or gpus not in (1, 2):
            raise ValueError(f"Unexpected shared-interactive allocation: {job}")
        row.update({
            "job_id": job,
            "allocated_nodes": nodes,
            "allocated_gpus": gpus,
            "elapsed_seconds": elapsed,
            "allocated_job_node_hours": nodes * elapsed / 3600,
            "estimated_gpu_node_hours": gpus * elapsed / (4 * 3600),
        })
        jobs[job] = row
    missing = set(ids) - jobs.keys()
    if missing:
        raise ValueError(f"sacct did not return all requested jobs: {sorted(missing)}")

    ordered = [jobs[job] for job in ids]
    by_qos = {}
    for row in ordered:
        bucket = by_qos.setdefault(row["QOS"], {
            "jobs": 0, "allocated_job_node_hours": 0.0,
            "estimated_gpu_node_hours": 0.0,
        })
        bucket["jobs"] += 1
        for key in ("allocated_job_node_hours", "estimated_gpu_node_hours"):
            bucket[key] += row[key]
    receipt = {
        "status": "PASS_SACCT_SNAPSHOT",
        "query_started_utc": started,
        "query_finished_utc": finished,
        "command": command,
        "account": "m5397_g",
        "requested_job_ids": ids,
        "base_job_ids": list(BASE_JOB_IDS),
        "additional_job_ids": sorted(set(args.job_id) - set(BASE_JOB_IDS)),
        "totals": {
            "jobs": len(ordered),
            "allocated_job_node_hours": sum(r["allocated_job_node_hours"] for r in ordered),
            "estimated_gpu_node_hours": sum(r["estimated_gpu_node_hours"] for r in ordered),
        },
        "by_qos": by_qos,
        "jobs": ordered,
        "interpretation": {
            "allocation": "Sum of AllocNodes * ElapsedRaw / 3600; shared nodes may be counted for multiple jobs.",
            "estimate": "Sum of gres/gpu * ElapsedRaw / (4 * 3600); supported QOS factors are 1.",
            "running": "Running job durations are snapshots and grow after the query.",
            "billing": "Estimate from Slurm allocation, not an official posted Iris charge.",
            "scope": "Top-level jobs only; srun steps are not added a second time.",
            "policy": "https://docs.nersc.gov/jobs/policy/",
        },
    }
    # Exclusive creation prevents accidental replacement even if another writer
    # creates this path after the earlier existence check.
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Record top-level Slurm allocations for this goal, including idle time.

GPU-node-hours here mean allocated GPU-hours divided by four A100s per full
Perlmutter node. This is a resource estimate, not an official posted charge.
Job steps are excluded so training and scoring are not counted twice.
"""

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.execution.read_bytes()
    execution = json.loads(raw)
    ids = sorted(set(map(int, execution["allocation_jobs"])))
    if not ids or min(ids) <= 0 or execution["account"] != "m5397_g":
        raise ValueError("Expected this goal's explicit positive job IDs and account")
    columns = ["JobIDRaw", "JobName", "Account", "QOS", "State",
               "AllocNodes", "ElapsedRaw", "AllocTRES"]
    command = ["sacct", "-X", "-j", ",".join(map(str, ids)), "-n", "-P",
               "-o", ",".join(columns)]
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    rows = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != len(columns):
            raise ValueError(f"Malformed sacct row: {line!r}")
        row = dict(zip(columns, parts))
        job = int(row["JobIDRaw"])
        if job not in ids or job in rows or row["Account"] != "m5397_g":
            raise ValueError(f"Unexpected, repeated or wrong-account job: {row}")
        tres = dict(field.split("=", 1) for field in row["AllocTRES"].split(",") if field)
        nodes, seconds = int(row["AllocNodes"]), int(row["ElapsedRaw"])
        gpus = int(tres.get("gres/gpu", "0"))
        if min(nodes, seconds, gpus) < 0:
            raise ValueError("Negative Slurm allocation")
        if nodes == 0:
            if seconds or gpus:
                raise ValueError("Nonempty allocation with no nodes")
        elif row["QOS"] == "gpu_interactive":
            if gpus != 4 * nodes:
                raise ValueError("Expected four GPUs per regular node")
        elif row["QOS"] == "gpu_shared_interactive":
            if nodes != 1 or gpus not in (1, 2):
                raise ValueError("Unexpected shared-interactive allocation")
        else:
            raise ValueError(f"Unexpected QOS: {row['QOS']}")
        row.update(job_id=job, allocated_gpus=gpus, elapsed_seconds=seconds,
                   allocated_job_node_hours=nodes * seconds / 3600,
                   gpu_hours=gpus * seconds / 3600,
                   gpu_node_hours=gpus * seconds / 14400)
        rows[job] = row
    if set(rows) != set(ids):
        raise ValueError(f"Missing jobs: {sorted(set(ids) - set(rows))}")
    jobs = [rows[job] for job in ids]
    total = sum(row["gpu_node_hours"] for row in jobs)
    receipt = dict(
        status="PASS_SACCT_SNAPSHOT", query_started_utc=started,
        query_finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        command=command, execution_sha256=hashlib.sha256(raw).hexdigest(),
        account="m5397_g", jobs=jobs,
        totals={key: sum(row[key] for row in jobs) for key in
                ("allocated_job_node_hours", "gpu_hours", "gpu_node_hours")},
        limit_gpu_node_hours=execution["max_gpu_node_hours"],
        within_limit=total <= execution["max_gpu_node_hours"],
        all_allocations_terminal=all(row["State"].split()[0].rstrip("+") in
                                    {"COMPLETED", "CANCELLED", "TIMEOUT", "FAILED",
                                     "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED",
                                     "BOOT_FAIL", "DEADLINE"} for row in jobs),
        interpretation="Allocation duration includes startup, failed subprocesses and idle workers. Shared-job node time can overlap physically. GPU-node-hours = GPU-hours/4; not an official posted charge. Running-job snapshots grow. Only top-level jobs are summed.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "gpu_node_hours": total,
                      "all_allocations_terminal": receipt["all_allocations_terminal"]}))


if __name__ == "__main__":
    main()

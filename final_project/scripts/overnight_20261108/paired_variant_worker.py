"""Persistent GPU worker for explicitly frozen paired sensitivity contexts.

No allocation is created. Primary worker/context files remain unchanged; this
driver reuses their bounded relaxation implementation with an explicit new
teacher or force criterion and separate physical identities.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import time

import numpy as np

from paired_common import (atomic_json, canonical, check_prepared, identity,
    load_npz, physical_key, publish_result, sha256, verify_teacher_receipt)
from paired_worker import run_attempt


def verify_variant(prepared, variant):
    _, base = check_prepared(prepared)
    receipt = json.loads((variant / "preparation.json").read_text())
    if receipt["status"] != "FROZEN_BEFORE_CONTROL_SCORING":
        raise ValueError("Control must be frozen before scoring")
    for name, expected in receipt["outputs"].items():
        if sha256(variant / name) != expected:
            raise ValueError("Frozen control input changed")
    context = json.loads((variant / "context.json").read_text())
    if identity({k:v for k,v in context.items() if k != "context_hash"}) != context["context_hash"]:
        raise ValueError("Invalid control context hash")
    if context["comparison"]["base_context_hash"] != base["context_hash"]:
        raise ValueError("Control is not bound to this primary context")
    if context["parent_sha256"] != base["parent_sha256"] or context["geometry"] != base["geometry"]:
        raise ValueError("Control changed initial geometry/cell/atom protocol")
    tasks = json.loads((variant / "tasks.json").read_text())
    if tasks["context_hash"] != context["context_hash"]:
        raise ValueError("Control task context differs")
    return receipt, context, tasks["tasks"]


def main(argv=None):
    from cathode_cvae.runtime import require_compute_allocation
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared",type=Path,required=True)
    p.add_argument("--variant",type=Path,required=True)
    p.add_argument("--queue",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--worker-index",type=int,default=0)
    p.add_argument("--max-worker-seconds",type=float,default=3600.)
    args=p.parse_args(argv);require_compute_allocation()
    began=time.perf_counter()
    preparation, context, tasks=verify_variant(args.prepared,args.variant)
    parent=load_npz(args.prepared/"parent.npz")
    cohort=load_npz(args.prepared/"cohort.npz")
    for directory in ["claims","workers","results"]:(args.queue/directory).mkdir(parents=True,exist_ok=True)
    wid=f"{os.environ.get('SLURM_JOB_ID','local')}_{socket.gethostname()}_{os.getpid()}"
    output=args.queue/"workers"/(wid+".json")
    receipt=dict(worker=wid,started_unix=time.time(),status="loading",job_id=os.environ.get("SLURM_JOB_ID"),
        context_hash=context["context_hash"],source_sha256=sha256(__file__),
        relaxation_source_sha256=sha256(Path(__file__).with_name("paired_worker.py")),
        common_source_sha256=sha256(Path(__file__).with_name("paired_common.py")),claimed=[],completed=[])
    atomic_json(output,receipt)
    def stop(signum,frame):raise InterruptedError(f"Variant worker signal {signum}")
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        from cathode_cvae.mlip.calculators import make_calculator
        import torch
        base, model=make_calculator(context["teacher"]["model"],checkpoint=args.checkpoint,device=args.device,threads=args.threads)
        verify_teacher_receipt(model,context)
        receipt.update(model_receipt=model,status="running",load_seconds=time.perf_counter()-began)
        atomic_json(output,receipt)
        sync=(lambda:torch.cuda.synchronize()) if args.device.startswith("cuda") else (lambda:None)
        for index in np.random.default_rng(args.worker_index+20261108).permutation(len(tasks)):
            if time.perf_counter()-began>=args.max_worker_seconds:
                receipt["status"]="worker_wall_limit";break
            task=tasks[int(index)];row=task["input_row"]
            if (cohort["split"][row]!="train" or task["split"]!="train" or
                    task["input_sha256"]!=sha256(args.prepared/"cohort.npz")):
                raise ValueError("Control changed original TRAIN role or input data")
            key,_=canonical(cohort["occupancy"][row],parent["permutations"])
            if key!=task["canonical_vacancy_key"] or physical_key(context["context_hash"],key)!=task["physical_key"]:
                raise ValueError("Control task initial geometry/context binding differs")
            claim=args.queue/"claims"/(task["task_id"]+".json")
            try:
                with claim.open("x") as f:json.dump(dict(worker=wid,claimed_unix=time.time(),task=task),f,indent=2)
            except FileExistsError:continue
            receipt["claimed"].append(task["task_id"]);atomic_json(output,receipt)
            result,interrupted=run_attempt(base,parent,task,context,args.queue/"results"/task["task_id"],synchronize=sync)
            result.update(worker=wid,model_receipt_sha256=identity(model))
            publish_result(args.queue/"results"/task["task_id"],result)
            receipt["completed"].append(dict(task_id=task["task_id"],status=result["status"]))
            atomic_json(output,receipt)
            if interrupted:raise InterruptedError("Control interrupted; last valid geometry retained")
        else:receipt["status"]="finished_task_scan"
    except BaseException as exc:
        receipt.update(status="worker_error",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt.update(finished_unix=time.time(),finished_UTC=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds_including_loading_idle=time.perf_counter()-began)
        atomic_json(output,receipt)
    return 0


if __name__=="__main__":raise SystemExit(main())

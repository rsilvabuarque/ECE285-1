"""Bounded NEB steps inside an OWNER-CREATED two-GPU Slurm allocation.

This stdlib-only controller creates no allocation and never cancels one. Root
must invoke it after the editor/other extensions and release the allocation
after return. Actual sacct allocation time, including startup and idle GPUs,
remains the authority for the two GPU-device-hour limit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024**2),b""):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open("x") as f:f.write(json.dumps(value,indent=2,allow_nan=False)+"\n")


def validate_config(config):
    required=("project_root","uma_python","mace_python","summary_python","uma_checkpoint","mace_checkpoint",
              "prepared","run_root","summary_output")
    if any(name not in config for name in required):raise ValueError("Missing explicit controller paths")
    result=dict(config)
    for name in required:result[name]=str(Path(config[name]).resolve())
    project=Path(result["project_root"])
    if not (project/"scripts/conditioning_20261107/neb_run.py").is_file():raise ValueError("project_root is not the prepared checkout")
    for name in ("uma_python","mace_python","summary_python","uma_checkpoint","mace_checkpoint"):
        if not Path(result[name]).is_file():raise FileNotFoundError(result[name])
    protocol_path=Path(result["prepared"])/"protocol.json"
    protocol=json.loads(protocol_path.read_text())
    if protocol["maximum_events"]!=2 or protocol["per_event_wall_seconds"]!=2400 or protocol["optional_MACE_profile_wall_seconds"]!=600 or protocol["two_GPU_window_maximum_wall_seconds"]!=3600:
        raise ValueError("Controller needs the reviewed prepared_v3 allocated-budget protocol")
    result["prepared_protocol_sha256"]=digest(protocol_path)
    result["events"]=[e["event_id"] for e in protocol["events"]]
    if len(result["events"])!=2 or protocol["refinement"]["event_id"]!=result["events"][0]:raise ValueError("Unexpected marked event/refinement order")
    return result


def worker_command(config,rank):
    if rank not in (0,1):raise ValueError("Only two NEB ranks are allowed")
    root=Path(config["project_root"]);event=config["events"][rank]
    argv=[config["uma_python"],str(root/"scripts/conditioning_20261107/neb_run.py"),
        "--prepared",config["prepared"],"--event",event,"--checkpoint",config["uma_checkpoint"],
        "--output-dir",str(Path(config["run_root"])/event),"--budget-seconds","2400","--device","cuda","--threads","4"]
    if rank==0:argv.append("--refine")
    return argv


def srun_prefix(job,tasks,gpu=True):
    command=["srun","--jobid",str(job),"--exact","-N","1","-n",str(tasks),"-c","4","--kill-on-bad-exit=0"]
    if gpu:command += ["--gpus-per-task=1","--gpu-bind=single:1"]
    return command


def run_bounded(argv,seconds,log,env,cwd):
    """Own only this subprocess group; preserve logs and terminate on timeout."""
    started=time.monotonic();begin=datetime.now(timezone.utc).isoformat()
    with Path(log).open("x") as stream:
        process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=cwd,start_new_session=True)
        timed_out=False;killed=False
        try:code=process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out=True
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:code=process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                killed=True
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                code=process.wait(timeout=10)
        except BaseException:
            # An owner interruption must not leave our child step running.
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.wait(timeout=10)
            raise
    return dict(argv=argv,started_UTC=begin,completed_UTC=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.monotonic()-started,timeout_seconds=seconds,returncode=code,
        timed_out=timed_out,forced_kill_after_grace=killed,log=str(log))


def run_worker(config):
    rank=int(os.environ.get("SLURM_PROCID","-1"))
    if rank not in (0,1) or not os.environ.get("SLURM_JOB_ID"):raise ValueError("Worker requires one of the two owner-launched Slurm ranks")
    command=worker_command(config,rank);run=Path(config["run_root"])
    env=os.environ.copy();env.update(PYTHONPATH=str(Path(config["project_root"])/"src"),OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4")
    started=time.monotonic()
    receipt=dict(status="WORKER_STARTED",argv=command,started_UTC=datetime.now(timezone.utc).isoformat())
    try:
        receipt.update(run_bounded(command,2500,run/f"controller_worker_{rank}.log",env,config["project_root"]))
        receipt["status"]="WORKER_PROCESS_TERMINAL"
        return int(receipt["returncode"])
    except BaseException as error:
        # run_bounded has already terminated its owned subprocess group. Save
        # the worker failure even when a step signal interrupted its wait.
        receipt.update(status="WORKER_INTERRUPTED_OR_FAILED",exception_type=type(error).__name__,
            error=str(error),returncode=2)
        return 2
    finally:
        receipt.update(rank=rank,event_id=config["events"][rank],job_id=os.environ["SLURM_JOB_ID"],
            completed_UTC=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
            GPU_visibility=os.environ.get("CUDA_VISIBLE_DEVICES"),source_sha256=digest(__file__),
            scientific_source_sha256=digest(Path(config["project_root"])/"scripts/conditioning_20261107/neb_run.py"))
        write_json(run/f"controller_worker_{rank}.json",receipt)


def install_signal_handlers():
    def stop(signum,frame):raise InterruptedError(f"Owner/allocation signal {signum}; terminate owned child step")
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)


def profile_eligible(run,event,remaining):
    path=Path(run)/event/"receipt.json"
    if remaining<750:return False,"DEFERRED_REMAINING_CONTROLLER_WALL_BUDGET"
    if not path.is_file():return False,"DEFERRED_NO_UMA_RECEIPT"
    receipt=json.loads(path.read_text())
    if not receipt.get("climbing_band",{}).get("converged") or not receipt.get("endpoint_event_validation",{}).get("valid") or receipt.get("forward_barrier_eV") is None:
        return False,"DEFERRED_UMA_PATH_NOT_SUCCESSFUL"
    return True,"ELIGIBLE_FIXED_UMA_PATH_SINGLE_POINTS_ONLY"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--print-plan",action="store_true",help="Read-only argument review; creates no steps or model calls")
    a=parser.parse_args(argv)
    original=json.loads(a.config.read_text());config=validate_config(original)
    if a.print_plan:
        print(json.dumps(dict(worker_commands=[worker_command(config,i) for i in range(2)],
            prepared_protocol_sha256=config["prepared_protocol_sha256"],new_model_calls=0,new_allocations=0),indent=2))
        return 0
    # Both the parent and rank workers must raise through run_bounded cleanup;
    # a worker's default SIGTERM handler would skip Python finally blocks.
    install_signal_handlers()
    if a.worker:return run_worker(config)
    job=os.environ.get("SLURM_JOB_ID")
    if not job:raise ValueError("Root must first create a dedicated one-hour, two-GPU allocation")
    allocated=os.environ.get("SLURM_GPUS_ON_NODE")
    if allocated and allocated.isdigit() and int(allocated)!=2:raise ValueError("The dedicated pilot allocation must have exactly two GPUs")
    run=Path(config["run_root"])
    if run.exists():raise FileExistsError("Preserve all previous NEB/controller attempts")
    run.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+3300
    # A Slurm-provided end time can make the declared window stricter. The
    # owner-created one-hour allocation remains the hard two-GPU cost boundary.
    if os.environ.get("SLURM_JOB_END_TIME","").isdigit():
        deadline=min(deadline,started+max(0,float(os.environ["SLURM_JOB_END_TIME"])-time.time()-60))
    config_path=run/"controller_config.json";write_json(config_path,config)
    receipt=dict(status="STARTED",started_UTC=datetime.now(timezone.utc).isoformat(),job_id=job,
        source_sha256=digest(__file__),input_config=dict(path=str(a.config.resolve()),sha256=digest(a.config)),
        prepared_protocol_sha256=config["prepared_protocol_sha256"],allocated_GPUs=2,
        controller_wall_limit_seconds=3300,phases=[],MACE_profile_status="NOT_REACHED",
        allocation_owner="root; this helper creates no allocations and does not release/cancel the allocation")
    env=os.environ.copy();env.update(PYTHONPATH=str(Path(config["project_root"])/"src"),OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4",MKL_NUM_THREADS="4")
    try:
        if deadline-time.monotonic()<2700:
            receipt["status"]="DEFERRED_INSUFFICIENT_ALLOCATION_WINDOW";return 3
        scientific=srun_prefix(job,2)+[sys.executable,str(Path(__file__).resolve()),"--config",str(config_path),"--worker"]
        receipt["phases"].append(run_bounded(scientific,min(2600,deadline-time.monotonic()-90),run/"controller_events.log",env,config["project_root"]))
        allowed,reason=profile_eligible(run,config["events"][0],deadline-time.monotonic())
        receipt["MACE_profile_status"]=reason
        if allowed:
            command=srun_prefix(job,1)+[config["mace_python"],str(Path(config["project_root"])/"scripts/conditioning_20261107/neb_mace_profile.py"),
                "--prepared",config["prepared"],"--uma-run",str(run/config["events"][0]),"--checkpoint",config["mace_checkpoint"],
                "--output-dir",str(run/"mace_profile"),"--budget-seconds","600","--device","cuda","--threads","4"]
            receipt["phases"].append(run_bounded(command,min(650,deadline-time.monotonic()-90),run/"controller_MACE.log",env,config["project_root"]))
            receipt["MACE_profile_status"]="ATTEMPTED_SINGLE_POINT_PROFILE_ONLY"
        if deadline-time.monotonic()>90:
            command=srun_prefix(job,1,gpu=False)+["env","CUDA_VISIBLE_DEVICES=",config["summary_python"],
                str(Path(config["project_root"])/"scripts/conditioning_20261107/neb_summarize.py"),
                "--run-root",str(run),"--output-dir",config["summary_output"]]
            receipt["phases"].append(run_bounded(command,min(90,deadline-time.monotonic()-60),run/"controller_summary.log",env,config["project_root"]))
        else:receipt["summary_status"]="DEFERRED_REMAINING_WALL_BUDGET; saved scientific receipts remain authoritative"
        receipt["status"]="CONTROLLER_COMPLETE_REVIEW_SCIENTIFIC_RECEIPTS"
        return 0
    except Exception as error:
        receipt.update(status="CONTROLLER_FAILED_PRESERVED",exception_type=type(error).__name__,error=str(error));return 2
    finally:
        elapsed=time.monotonic()-started
        receipt.update(completed_UTC=datetime.now(timezone.utc).isoformat(),elapsed_seconds=elapsed,
            controller_window_two_GPU_device_hours=2*elapsed/3600,
            cost_scope="Controller window counts both GPUs, including idle second GPU during MACE. Full allocation startup/idle/release cost requires final sacct; it is not inferred from process times.",
            no_allocation_created_or_cancelled=True,owner_must_release_allocation=True)
        write_json(run/"controller_receipt.json",receipt)
        print(json.dumps(dict(status=receipt["status"],job_id=job,elapsed_seconds=elapsed,owner_must_release_allocation=True),indent=2))


if __name__=="__main__":raise SystemExit(main())

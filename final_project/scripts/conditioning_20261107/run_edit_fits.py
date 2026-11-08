"""Freeze or execute exactly three optional editor trajectories, one GPU/task.

Preparation snapshots the executing package and optional-editor scripts. GPU
allocation remains the integrating owner's responsibility; this script never
calls a scheduler. Durable claims prevent duplicate execution of a fit.
"""
import argparse
import contextlib
from datetime import datetime,timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback


GATE_SHA="c6a7105dd9578c5a1caac1c8bc6acfb2bc134e8fd1ce04e2a0c8def232f91bf3"
SEEDS=(20261106,20261107,20261108)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path,value):
    with Path(path).open("x") as f:json.dump(value,f,indent=2);f.write("\n")


def prepare(root,data_dir):
    from cathode_cvae.learning.train_edit import validate_pairs
    import numpy as np
    root=root.resolve();run=root/"runs/conditioning_20261107"
    gate=root/"artifacts/conditioning_20261107/required_gate.json"
    if sha(gate)!=GATE_SHA or json.loads(gate.read_text())["status"]!="COMPLETE":
        raise ValueError("The reviewed required-report gate is not complete/unchanged")
    receipt_path=data_dir/"preparation_receipt.json";receipt=json.loads(receipt_path.read_text())
    if receipt["status"]!="READY_FOR_BOUNDED_EDITOR" or receipt["required_report_gate"]["sha256"]!=GATE_SHA:
        raise ValueError("No ready gate-bound editor data/support receipt")
    pair_path=data_dir/"edit_pairs.npz"
    if sha(pair_path)!=receipt["outputs"][pair_path.name]:raise ValueError("Pair inputs changed")
    with np.load(pair_path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    checked=validate_pairs(data)
    scripts=["prepare_edit_data.py","generate_edit_comparison.py","evaluate_edit_comparison.py","run_edit_fits.py"]
    for name in scripts:
        if not (root/"scripts/conditioning_20261107"/name).is_file():
            raise ValueError("Complete optional data/comparison/evaluation scripts before freezing sources")
    snapshot=run/"executed_sources/editor_v1"
    if snapshot.exists() or (run/"editor_fit_tasks.json").exists():raise FileExistsError("Preserve editor source/task freeze")
    snapshot.mkdir(parents=True)
    package=root/"src/cathode_cvae";package_manifest={}
    for source in sorted(package.rglob("*.py")):
        relative=Path("cathode_cvae")/source.relative_to(package);target=snapshot/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);package_manifest[str(relative)]=sha(target)
        if sha(source)!=package_manifest[str(relative)]:raise ValueError("Package source changed during editor freeze")
    write_once(snapshot/"manifest.json",package_manifest)
    script_manifest={}
    for name in scripts:
        source=root/"scripts/conditioning_20261107"/name
        if not source.exists():raise ValueError("Complete optional comparison scripts before freezing sources")
        relative=Path("scripts/conditioning_20261107")/name;target=snapshot/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);script_manifest[str(relative)]=sha(target)
        if sha(source)!=script_manifest[str(relative)]:raise ValueError("Optional script changed during source freeze")
    write_once(snapshot/"scripts_manifest.json",script_manifest)
    tasks=[]
    for seed in SEEDS:
        argv=["--root",str(root),"--data",str(pair_path.resolve()),"--preparation-receipt",str(receipt_path.resolve()),
            "--parent",str(root/"artifacts/current/core/data/parents/O3_N48_pilot.npz"),
            "--condition-schema-in",str(root/"artifacts/overnight_20261107/data/condition_schema_uma.json"),
            "--required-gate",str(gate),"--required-gate-sha256",GATE_SHA,"--seed",str(seed),"--counts","12","24","36",
            "--latent-dim","4","--hidden-dim","128","--orientation","joint-augment",
            "--max-updates","4000","--epochs","1000","--batch-size","512","--learning-rate",".001",
            "--beta-final",".1","--beta-warmup","20","--validation-beta",".1","--validation-samples","2",
            "--class-balance","--max-seconds","1200","--threads","4","--device","cuda",
            "--output-dir",str(run/f"fits/editor_{seed}")]
        tasks.append(dict(id=f"editor_{seed}",fit_seed=seed,experiment_phase="optional_editor",
            trainer_module="cathode_cvae.learning.train_edit",argv=argv))
    protocol=dict(status="FROZEN_BEFORE_OPTIONAL_EDITOR_FITS",created_UTC=datetime.now(timezone.utc).isoformat(),
        experiment_phase="optional_editor",required_report_gate=dict(path=str(gate.relative_to(root)),sha256=GATE_SHA),
        hypothesis="A seed-conditioned variational distribution improves measured child-bin targeting and energy change over exact same-seed/same-k uniform mutation",
        representation="Jointly oriented raw seed/child occupations; exact-k removal from occupied seed sites, exact-k addition to original vacant seed sites",
        allowed_training="Verified policy/latent/spatial DEVELOPMENT mutation outcomes only; actual measured CHILD labels; no favorable filtering; all core-FINAL candidate and protectedH0VAL/TEST identities excluded",
        roles="Connected seed/child identity components nested train/val/heldout; all shared identities remain in one role",
        training_budget=dict(full_trajectories=3,seeds=list(SEEDS),optimizer_updates=4000,batch_size=512,
            latent_dim=4,hidden_dim=128,learning_rate=.001,beta_final=.1,beta_warmup_epochs=20,
            checkpoint_selection="UnweightedVAL reconstruction +0.1*summedKL; fixed two-MC noise; no heldout selection"),
        assessment=dict(counts=[12,24,36],requested_CHILD_bins=["low","middle","high"],attempts_per_count_bin_fitrep_method=128,
            methods=["learned_edit","mutation"],total_attempts=6912,
            pairing="Exactly the same heldout editor seed identity and uniform k∈{1,2,3} per paired row; shared subset uniforms",
            seed_information="Seed bin from frozen H0, matching requested CHILD bin for both methods; sampling with replacement, all duplicates/failures retained",
            seed_geometries_scored=True,history="Freeze union of all already-known core-final and development mask identities before any editor scores",
            outcomes=["actualCHILDbin", "distinctnovel in-requested-bin/allattempts", "originalq20low", "childminus-seed energy", "raw2k and symmetry-awareeditdistance"],
            descriptor="Posthoc structural diagnostic only, not an input condition"),
        data_check=checked,data_sha256=sha(pair_path),preparation_receipt_sha256=sha(receipt_path),
        source_snapshot=str(snapshot.resolve()),source_manifest_sha256=sha(snapshot/"manifest.json"),
        script_manifest_sha256=sha(snapshot/"scripts_manifest.json"),new_DFT=False,new_MLIP_finetuning=False)
    protocol_path=run/"editor/protocol.json";protocol_path.parent.mkdir(exist_ok=True)
    write_once(protocol_path,protocol)
    task_path=run/"editor_fit_tasks.json"
    write_once(task_path,dict(stage="optional_editor",experiment_phase="optional_editor",tasks=tasks,
        source_snapshot=str(snapshot.resolve()),package_manifest_sha256=sha(snapshot/"manifest.json"),
        script_manifest_sha256=sha(snapshot/"scripts_manifest.json"),protocol_path=str(protocol_path),protocol_sha256=sha(protocol_path)))
    return dict(status="READY_FOR_ROOT_GPU_LAUNCH",tasks=str(task_path),tasks_sha256=sha(task_path),
        source_snapshot=str(snapshot),protocol=str(protocol_path),data_check=checked,full_trajectories=3)


def execute(task_path):
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation();config=json.loads(task_path.read_text());snapshot=Path(config["source_snapshot"])
    manifest_path=snapshot/"manifest.json"
    if sha(manifest_path)!=config["package_manifest_sha256"]:raise ValueError("Editor source manifest changed")
    manifest=json.loads(manifest_path.read_text())
    for name,expected in manifest.items():
        if sha(snapshot/name)!=expected:raise ValueError("Editor source snapshot changed")
    script_path=snapshot/"scripts_manifest.json"
    if sha(script_path)!=config["script_manifest_sha256"]:raise ValueError("Editor script manifest changed")
    scripts=json.loads(script_path.read_text())
    for name,expected in scripts.items():
        if sha(snapshot/name)!=expected:raise ValueError("Editor script snapshot changed")
    if sha(__file__)!=scripts["scripts/conditioning_20261107/run_edit_fits.py"]:
        raise ValueError("Use the preserved matching optional fit runner")
    if sha(config["protocol_path"])!=config["protocol_sha256"]:raise ValueError("Optional protocol changed")
    module=importlib.import_module("cathode_cvae.learning.train_edit")
    if sha(module.__file__)!=manifest["cathode_cvae/learning/train_edit.py"]:
        raise ValueError("Set PYTHONPATH to the declared preserved editor_v1 snapshot before training")
    claims=task_path.with_suffix(".claims");claims.mkdir(exist_ok=True)
    for task in config["tasks"]:
        claim=claims/task["id"]
        try:claim.mkdir()
        except FileExistsError:continue
        start=time.time();owner=dict(pid=os.getpid(),host=os.uname().nodename,job=os.environ.get("SLURM_JOB_ID"),
            rank=os.environ.get("SLURM_PROCID"),cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),started_unix=start,
            tasks_sha256=sha(task_path),interpreter=sys.executable,argv=task["argv"],experiment_phase="optional_editor")
        write_once(claim/"owner.json",owner);status="COMPLETE";error=None;summary=None
        with (claim/"log.txt").open("x",buffering=1) as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                try:
                    summary=module.main(task["argv"])
                    if summary["status"]!="FIT_COMPLETE":status="PARTIAL"
                except BaseException as problem:
                    status="FAILED";error=f"{type(problem).__name__}: {problem}";traceback.print_exc()
        receipt=dict(owner,status=status,error=error,finished_unix=time.time(),elapsed_seconds=time.time()-start,
            optimizer_updates=None if summary is None else summary["optimizer_updates"])
        write_once(claim/"receipt.json",receipt)
        print(json.dumps(dict(task=task["id"],status=status,error=error,seconds=receipt["elapsed_seconds"])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--prepare",action="store_true")
    p.add_argument("--root",type=Path);p.add_argument("--data-dir",type=Path);p.add_argument("--tasks",type=Path)
    a=p.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8");os.environ.setdefault("OMP_NUM_THREADS","4")
    if a.prepare:
        from cathode_cvae.runtime import require_compute_allocation
        require_compute_allocation()
        if not a.root:raise ValueError("--root is required for source/task preparation")
        data_dir=a.data_dir or a.root/"runs/conditioning_20261107/editor/data_v2"
        print(json.dumps(prepare(a.root,data_dir),indent=2))
    else:
        if not a.tasks:raise ValueError("Supply --tasks for execution")
        execute(a.tasks.resolve())


if __name__=="__main__":main()

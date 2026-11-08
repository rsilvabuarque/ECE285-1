"""Freeze/run joint endpoint proposals only after an explicit model selection.

The selection JSON has status FROZEN_DEVELOPMENT_SELECTION, a namespace,
counts, inputs and arms. Required inputs: parent, history, intake_history,
original_thresholds, current_data/current_schema, paired_deletion_data/schema,
paired_endpoint_data/schema, paired_prepared. Each of current_cvae/current_ar/
paired_deletion_cvae/paired_endpoint_cvae contains exactly three explicit
checkpoint, fit_seed and expected_checkpoint_sha256 records. No weights are
chosen, fitted or altered here. ``freeze`` produces commands; only ``run``
generates proposals and publishes a frozen endpoint task union, never MLIP calls.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from paired_common import atomic_json, check_prepared, identity, load_npz, sha256

REQUIRED_INPUTS = ["parent", "history", "intake_history", "original_thresholds", "current_data", "current_schema",
                   "paired_deletion_data", "paired_deletion_schema", "paired_endpoint_data", "paired_endpoint_schema", "paired_prepared", "history_receipt"]
ARMS = ["current_cvae", "current_ar", "paired_deletion_cvae", "paired_endpoint_cvae"]
SCRIPT_SOURCES = {"generator":"scripts/overnight_20261108/generate_controls.py",
    "registrar":"scripts/overnight_20261108/paired_register.py",
    "common":"scripts/overnight_20261108/paired_common.py",
    "planner":"scripts/overnight_20261108/paired_assessment_plan.py"}


def history_keys(path):
    value=json.loads(Path(path).read_text())
    return set(value if isinstance(value,list) else next(value[name] for name in
        ["keys","canonical_masks","known_keys","canonical_keys"] if name in value))


def file_binding(path):
    path=Path(path).absolute()
    return dict(path=str(path),sha256=sha256(path))


def freeze(args):
    from cathode_cvae.runtime import require_compute_allocation
    from cathode_cvae.conditions import load_schema
    require_compute_allocation()
    selection=json.loads(args.selection.read_text())
    if selection.get("status")!="FROZEN_DEVELOPMENT_SELECTION":
        raise ValueError("Explicit frozen DEVELOPMENT selection is required before final planning")
    if len(selection.get("counts",[]))!=3 or len(set(selection["counts"]))!=3:
        raise ValueError("This frozen endpoint design requires three distinct primary compositions")
    if selection.get("count_per_group",64)!=64:
        raise ValueError("This bounded required assessment freezes exactly64 attempts per cell")
    inputs={}
    for name in REQUIRED_INPUTS:
        raw=selection["inputs"][name]
        path=Path(raw["path"] if isinstance(raw,dict) else raw)
        if not path.is_absolute():path=args.root/path
        if name=="paired_prepared":
            inputs[name]=dict(path=str(path.absolute()),preparation_sha256=sha256(path/"preparation.json"))
        else:
            inputs[name]=file_binding(path)
            if isinstance(raw,dict) and raw.get("sha256",inputs[name]["sha256"])!=inputs[name]["sha256"]:
                raise ValueError("Selected input bytes changed: "+name)
    prepared=Path(inputs["paired_prepared"]["path"])
    preparation,context=check_prepared(prepared)
    if set(selection["counts"])!=set(preparation["counts"]):
        raise ValueError("Assessment counts differ from the independently acquired primary compositions")
    if inputs["parent"]["sha256"]!=context["parent_sha256"]:
        raise ValueError("Assessment parent differs from the exact relaxed-label parent")
    cohort=load_npz(prepared/"cohort.npz")
    history=history_keys(inputs["history"]["path"])
    if not history_keys(inputs["intake_history"]["path"])<=history or not set(cohort["canonical_masks"].astype(str))<=history:
        raise ValueError("Final history omits intake or known acquisition masks")
    history_receipt=json.loads(Path(inputs["history_receipt"]["path"]).read_text())
    if (history_receipt.get("status")!="FROZEN_IDENTITIES_ONLY" or
            history_receipt.get("history_sha256")!=inputs["history"]["sha256"] or
            history_receipt.get("intake_history_sha256")!=inputs["intake_history"]["sha256"] or
            history_receipt.get("cohort_sha256")!=sha256(prepared/"cohort.npz") or
            not history_receipt.get("declared_exposure_manifest_verified")):
        raise ValueError("Final identity history lacks the new complete exposure-source receipt")
    paired=[]
    for target in ["deletion","endpoint"]:
        with np.load(inputs[f"paired_{target}_data"]["path"],allow_pickle=False) as z:
            data={name:z[name] for name in ["occupancy","split","weight","structure_ids","n_target","canonical_masks"]}
        if set(data["split"].astype(str))!={"train","val"}:
            raise ValueError("Paired final generators must fit TRAIN/VAL-only labels, with TEST sealed")
        if not set(data["canonical_masks"].astype(str))<=history:
            raise ValueError("Final history omits paired training/validation identities")
        schema=load_schema(inputs[f"paired_{target}_schema"]["path"])
        if (schema["provenance"].get("energy_target")!=target or
                schema["provenance"].get("endpoint_context_hash")!=context["context_hash"]):
            raise ValueError("Paired target/schema physical context differs")
        paired.append(data)
    for name in ["occupancy","split","weight","structure_ids","n_target","canonical_masks"]:
        if not np.array_equal(paired[0][name],paired[1][name]):
            raise ValueError("Deletion/endpoint model inputs or independent roles are not matched: "+name)
    models={};checkpoint_bindings=[]
    from cathode_cvae.learning.runtime import trusted_checkpoint
    for arm in ARMS:
        entries=selection["arms"][arm]
        if len(entries)!=3 or len({r["fit_seed"] for r in entries})!=3:
            raise ValueError("Each retained arm requires exactly three independent explicit fits")
        models[arm]=[]
        for rep,entry in enumerate(entries):
            path=Path(entry["checkpoint"])
            if not path.is_absolute():path=args.root/path
            binding=file_binding(path)
            if binding["sha256"]!=entry["expected_checkpoint_sha256"]:
                raise ValueError("Selected checkpoint changed; do not choose a fallback")
            # Verified local project files only. Read serialized metadata on
            # CPU without constructing a network or calling its forward method.
            payload=trusted_checkpoint(path,"cpu")
            cfg=payload.get("configuration")
            if not isinstance(cfg,dict):raise ValueError("Selected checkpoint needs embedded configuration metadata")
            target="endpoint" if arm=="paired_endpoint_cvae" else "deletion"
            data_name="paired_endpoint_data" if target=="endpoint" else "paired_deletion_data" if arm=="paired_deletion_cvae" else "current_data"
            schema_name=data_name.replace("_data","_schema")
            schema_id=load_schema(inputs[schema_name]["path"])["identity"]
            expected_kind="ar_baseline" if arm=="current_ar" else "ar_cvae"
            if (cfg.get("kind")!=expected_kind or
                    (int(cfg.get("latent_dim",0))!=0 if arm=="current_ar" else int(cfg.get("latent_dim",0))<=0) or
                    cfg.get("energy_target","deletion")!=target or cfg.get("seed")!=entry["fit_seed"] or
                    cfg.get("data_sha256")!=inputs[data_name]["sha256"] or
                    cfg.get("parent_sha256")!=inputs["parent"]["sha256"] or
                    cfg.get("condition_schema_id")!=schema_id or cfg.get("condition_mode")!="multibin" or
                    not cfg.get("graph_context")):
                raise ValueError("Selected checkpoint metadata does not implement its declared scientific arm: "+arm)
            if arm=="current_cvae" and cfg.get("encoder_kind")!="graph":
                raise ValueError("Required current A arm must retain its graph posterior")
            models[arm].append(dict(checkpoint=binding["path"],expected_checkpoint_sha256=binding["sha256"],
                fit_seed=int(entry["fit_seed"]),replicate_id=rep,
                configuration_audit={name:cfg.get(name) for name in ["kind","latent_dim","graph_context","encoder_kind","energy_target","data_sha256","condition_schema_id"]}))
            checkpoint_bindings.append(binding)
    if any(len({r["expected_checkpoint_sha256"] for r in models[arm]})!=3 for arm in ARMS):
        raise ValueError("Copied weights do not constitute independent fitted replicates")
    if [r["fit_seed"] for r in models["paired_deletion_cvae"]]!=[r["fit_seed"] for r in models["paired_endpoint_cvae"]]:
        raise ValueError("Paired label targets must retain exactly the same initialization-seed ordering")
    def sampler(method,kind,target):
        return [dict(method=method,sampler=kind,fit_seed="",replicate_id=i,energy_target=target) for i in range(3)]
    def fitted(arm,method,target):
        return [dict(r,method=method,energy_target=target) for r in models[arm]]
    specifications={
        "current_low":dict(data="current_data",schema="current_schema",bins=["low"],models=
            fitted("current_cvae","current_graph_cvae","deletion")+fitted("current_ar","current_spatial_ar","deletion")+
            sampler("current_mutation","mutation","deletion")+sampler("uniform","uniform","deletion")),
        "paired_deletion":dict(data="paired_deletion_data",schema="paired_deletion_schema",bins=["low","middle","high"],models=
            fitted("paired_deletion_cvae","paired_deletion_cvae","deletion")+sampler("paired_deletion_mutation","mutation","deletion")),
        "paired_endpoint":dict(data="paired_endpoint_data",schema="paired_endpoint_schema",bins=["low","middle","high"],models=
            fitted("paired_endpoint_cvae","paired_endpoint_cvae","endpoint")+sampler("paired_endpoint_mutation","mutation","endpoint")),
        "shared_uniform":dict(data="current_data",schema="current_schema",bins=["low","middle","high"],models=sampler("uniform","uniform","deletion"))}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    stages={}
    for name,spec in specifications.items():
        manifest=dict(status="FROZEN_JOINT_ENDPOINT_ASSESSMENT",namespace=selection["namespace"],
            counts=selection["counts"],requested_bins=spec["bins"],models=spec["models"],diagnostics=False,
            latent_stream_width=32,count_per_composition_bin_replicate=64,
            expected_primary_attempts=len(spec["models"])*len(selection["counts"])*len(spec["bins"])*64,
            inputs={"parent":inputs["parent"],"data":inputs[spec["data"]],"schema":inputs[spec["schema"]],"history":inputs["history"]})
        path=args.output_dir/(name+".json");atomic_json(path,manifest)
        stages[name]=dict(manifest=file_binding(path),data_input=spec["data"],schema_input=spec["schema"],
                          attempts=manifest["expected_primary_attempts"])
    total=sum(v["attempts"] for v in stages.values())+preparation["independent_masks"]+len(cohort["pilot_indices"])
    if total>18000:raise ValueError("Conservative full endpoint queue exceeds the goal cap")
    generator=args.root/"scripts/overnight_20261108/generate_controls.py"
    registrar=args.root/"scripts/overnight_20261108/paired_register.py"
    source_bindings={name:file_binding(args.root/relative) for name,relative in SCRIPT_SOURCES.items()}
    if source_bindings["planner"]["sha256"]!=sha256(__file__):
        raise ValueError("Freezing source root must contain this exact reviewed planner")
    for path in sorted((args.root/"src/cathode_cvae").rglob("*.py")):
        source_bindings["package:"+path.relative_to(args.root).as_posix()]=file_binding(path)
    record=dict(status="FROZEN_AFTER_DEVELOPMENT_SELECTION_BEFORE_ASSESSMENT",frozen_unix=time.time(),
        frozen_UTC=datetime.now(timezone.utc).isoformat(),namespace=selection["namespace"],counts=selection["counts"],
        selected_models=file_binding(args.selection),inputs=inputs,checkpoints=checkpoint_bindings,stages=stages,
        source_bindings=source_bindings,
        source_sha256=sha256(__file__),primary_context_hash=context["context_hash"],
        conservative_endpoint_budget=dict(acquisition=preparation["independent_masks"],proposal_attempts=sum(v["attempts"] for v in stages.values()),
            tightening=len(cohort["pilot_indices"]),total_before_compatible_dedup=total,hard_cap=18000),
        random_policy="Same namespace,replicate,count,64draws,32-wide base latent and AR uniforms across stages; mutation seed pools differ by label data/schema",
        shared_uniform_policy="CurrentLOW and shared-uniformLOW are the exact same draws/history; retain separate comparison scopes without treating them as independent repetition",
        no_model_selection_or_training_performed=True,new_MLIP_calls=0)
    record["identity"]=identity(record)
    atomic_json(args.output_dir/"plan.json",record)
    print(json.dumps({k:record[k] for k in ["status","conservative_endpoint_budget","identity"]},indent=2))
    return record


def verify_plan(path):
    plan=json.loads(Path(path).read_text())
    if plan.get("source_sha256")!=sha256(__file__):
        raise ValueError("Planner runtime source differs from the frozen reviewed version")
    if plan.get("status")!="FROZEN_AFTER_DEVELOPMENT_SELECTION_BEFORE_ASSESSMENT" or identity({k:v for k,v in plan.items() if k!="identity"})!=plan["identity"]:
        raise ValueError("Final endpoint plan is not the explicit immutable selected plan")
    bindings=[*plan["checkpoints"],plan["selected_models"],*plan["source_bindings"].values(),
              *(v for k,v in plan["inputs"].items() if k!="paired_prepared"),
              *(s["manifest"] for s in plan["stages"].values())]
    for b in bindings:
        if sha256(b["path"])!=b["sha256"]:raise ValueError("Frozen assessment source changed: "+b["path"])
    prep=plan["inputs"]["paired_prepared"]
    if sha256(Path(prep["path"])/"preparation.json")!=prep["preparation_sha256"]:
        raise ValueError("Original independent-role preparation changed")
    return plan


def run(args):
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    plan=verify_plan(args.plan)
    runtime_sources={}
    for name,binding in plan["source_bindings"].items():
        relative=SCRIPT_SOURCES[name] if name in SCRIPT_SOURCES else name.split("package:",1)[1]
        actual=args.root/relative
        if sha256(actual)!=binding["sha256"]:
            raise ValueError("Runtime source root differs from the frozen assessment source: "+relative)
        runtime_sources[name]=str(actual.absolute())
    args.output_dir.mkdir(parents=True,exist_ok=False)
    python=str(Path(args.python or sys.executable).absolute()) # preserve venv symlinks
    env=os.environ.copy();env["PYTHONPATH"]=str(args.root/"src")+os.pathsep+env.get("PYTHONPATH","")
    output=args.output_dir;stages={};registrations=[]
    for name,stage in plan["stages"].items():
        destination=output/"generated"/name;destination.parent.mkdir(parents=True,exist_ok=True)
        command=[python,runtime_sources["generator"],"--manifest",stage["manifest"]["path"],
            "--source-root",str(args.root),"--parent",plan["inputs"]["parent"]["path"],
            "--data",plan["inputs"][stage["data_input"]]["path"],"--condition-schema",plan["inputs"][stage["schema_input"]]["path"],
            "--history",plan["inputs"]["history"]["path"],"--namespace",plan["namespace"],
            "--output-dir",str(destination),"--count","64","--device",args.device,"--threads",str(args.threads)]
        with (output/(name+"_generation.log")).open("w") as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        receipt=json.loads((destination/"receipt.json").read_text())
        if receipt["primary_attempts"]!=stage["attempts"]:raise ValueError("Generator denominator differs from frozen plan")
        for file,digest in receipt["outputs"].items():
            if sha256(destination/file)!=digest:raise ValueError("Generated proposal bytes changed")
        stages[name]=dict(directory=str(destination.absolute()),receipt_sha256=sha256(destination/"receipt.json"),argv=command)
    def uniform_rows(name):
        with (output/"generated"/name/"attempts.csv").open(newline="") as f:
            rr=[r for r in csv.DictReader(f) if r["method"]=="uniform" and r["requested_bin"]=="low"]
        return {(r["replicate_id"],r["n_target"],r["attempt"]):
                (r["canonical_mask"],r["uniforms_sha256"],r["sampling_seed"],r["status"]) for r in rr}
    a,b=uniform_rows("current_low"),uniform_rows("shared_uniform")
    if not a or a!=b:raise ValueError("Shared uniform reuse is not exactly the same prespecified draw stream")
    # Registration follows completion of EVERY frozen stage; no worker is
    # launched here. The owner receives one deduplicated physical task list.
    union={};registration_bindings={}
    for name in plan["stages"]:
        generated=output/"generated"/name;registered=output/"registered"/name;registered.parent.mkdir(parents=True,exist_ok=True)
        command=[python,runtime_sources["registrar"],"--prepared",plan["inputs"]["paired_prepared"]["path"],
            "--pool",str(generated/"primary_candidates.npz"),"--attempts",str(generated/"attempts.csv"),"--label",name,
            "--output-dir",str(registered),"--analysis-inputs",*(plan["inputs"][key]["path"] for key in
                ["paired_deletion_schema","paired_endpoint_schema","history","original_thresholds"])]
        if registrations:command.extend(["--prior-task-files",*map(str,registrations)])
        with (output/(name+"_registration.log")).open("w") as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        tasks_path=registered/"tasks.json";registrations.append(tasks_path)
        tasks=json.loads(tasks_path.read_text())
        for task in tasks["tasks"]:union.setdefault(task["physical_key"],task)
        registration_bindings[name]=dict(directory=str(registered.absolute()),receipt_sha256=sha256(registered/"registration.json"))
    verify_plan(args.plan)
    atomic_json(output/"registration_preflight.json",dict(status="ALL_PROPOSALS_FROZEN_BEFORE_NEW_ENDPOINT_SCORING",
        frozen_unix=time.time(),plan_sha256=sha256(args.plan),stages=stages,registrations=registration_bindings,
        shared_uniform_low_exactly_identical=True,shared_uniform_low_draws=len(a),no_worker_launched=True))
    atomic_json(output/"endpoint_union_tasks.json",dict(schema="paired-relaxation-task-list-v1",context_hash=plan["primary_context_hash"],
        tasks=[union[k] for k in sorted(union)]))
    receipt=dict(status="FROZEN_PROPOSALS_AND_ENDPOINT_TASKS_NO_SCFS",completed_UTC=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha256(args.plan),stages=stages,registrations=registration_bindings,
        endpoint_union_tasks= file_binding(output/"endpoint_union_tasks.json"),unique_physical_tasks=len(union),
        generated_attempts=sum(s["attempts"] for s in plan["stages"].values()),
        shared_uniform_low_reused_exactly=True,new_MLIP_calls=0,new_training_updates=0,
        scope="Separate all-attempt comparison files retained. Union only deduplicates exact physical endpoint work; mutation pools remain label-specific.")
    atomic_json(output/"receipt.json",receipt)
    print(json.dumps({k:receipt[k] for k in ["status","generated_attempts","unique_physical_tasks","new_MLIP_calls"]},indent=2))
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    f=sub.add_parser("freeze");f.add_argument("--root",type=Path,required=True);f.add_argument("--selection",type=Path,required=True);f.add_argument("--output-dir",type=Path,required=True)
    r=sub.add_parser("run");r.add_argument("--root",type=Path,required=True);r.add_argument("--plan",type=Path,required=True);r.add_argument("--output-dir",type=Path,required=True)
    r.add_argument("--python");r.add_argument("--device",choices=["cpu","cuda"],default="cpu");r.add_argument("--threads",type=int,default=4)
    args=p.parse_args(argv)
    return freeze(args) if args.command=="freeze" else run(args)


if __name__=="__main__":main()

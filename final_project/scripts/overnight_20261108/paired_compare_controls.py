"""Numerical stopping-target sensitivity and bounded secondary-teacher checks.

UMA .01 controls keep the .05 TRAIN thresholds frozen when measuring crossings.
MACE energies are never assigned UMA classes. MACE-on-UMA single points and
independent MACE endpoint relaxations retain separate geometry identities.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from paired_common import atomic_json, check_prepared, sha256, verify_teacher_receipt
from paired_variant_worker import verify_variant


def write_csv(path, rows):
    if rows:
        with Path(path).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def finite(value):
    return float(value) if np.isfinite(value) else None


def main(argv=None):
    from cathode_cvae.runtime import require_compute_allocation
    from cathode_cvae.conditions import classify,load_schema
    from scipy.stats import spearmanr,kendalltau
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared",type=Path,required=True)
    p.add_argument("--primary-queue",type=Path,required=True)
    p.add_argument("--variants",type=Path,required=True)
    p.add_argument("--variant-queue",type=Path,required=True)
    p.add_argument("--endpoint-schema",type=Path,required=True)
    p.add_argument("--mace-sp-directory",type=Path)
    p.add_argument("--mace-sp-manifest",type=Path)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--allow-incomplete",action="store_true")
    args=p.parse_args(argv);require_compute_allocation()
    _,primary_context=check_prepared(args.prepared)
    schema=load_schema(args.endpoint_schema)
    if schema["provenance"].get("endpoint_context_hash")!=primary_context["context_hash"]:
        raise ValueError("Sensitivity thresholds belong to a different primary endpoint protocol")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    rows=[];sources=[];missing=[];contexts={}
    for variant_name in ["tight_0p01","mace_independent"]:
        _,context,tasks=verify_variant(args.prepared,args.variants/variant_name)
        contexts[variant_name]=context
        for task in tasks:
            original=args.primary_queue/"results"/task["source_primary_task_id"]/"result.json"
            changed=args.variant_queue/"results"/task["task_id"]/"result.json"
            row=dict(variant=variant_name,cohort_row=task["input_row"],structure_id=task["structure_id"],
                n_target=task["n_target"],split="train",primary_status="missing",control_status="missing",
                primary_endpoint_energy_eV_atom=None,control_endpoint_energy_eV_atom=None,
                control_minus_primary_meV_atom=None,primary_bin="unknown",tight_bin="not_applicable",
                primary_fmax_eV_A=None,control_fmax_eV_A=None,primary_steps=None,control_steps=None)
            if not original.exists() or not changed.exists():
                missing.append(task["task_id"]);rows.append(row);continue
            first=json.loads(original.read_text());second=json.loads(changed.read_text())
            if first["context_hash"]!=primary_context["context_hash"] or second["context_hash"]!=context["context_hash"]:
                raise ValueError("Control/primary result context mismatch")
            if first["initial_geometry_hash"]!=second["initial_geometry_hash"]:
                raise ValueError("Control did not start from exactly the same pristine deletion geometry")
            for source,record in [(original,first),(changed,second)]:
                arrays=source.parent/"arrays.npz"
                if sha256(arrays)!=record["arrays_sha256"]:raise ValueError("Saved control arrays changed")
                sources.append(dict(path=str(source.absolute()),sha256=sha256(source),arrays_sha256=record["arrays_sha256"]))
            row.update(primary_status=first["status"],control_status=second["status"],
                primary_fmax_eV_A=first["endpoint_force_max_eV_A"],control_fmax_eV_A=second["endpoint_force_max_eV_A"],
                primary_steps=first["n_steps"],control_steps=second["n_steps"])
            if first["endpoint_labels_usable"]:
                row["primary_endpoint_energy_eV_atom"]=first["endpoint_energy_eV"]/first["n_atoms"]
                row["primary_bin"]=str(classify(schema,[task["n_target"]],[row["primary_endpoint_energy_eV_atom"]])[0])
            if second["endpoint_labels_usable"]:
                row["control_endpoint_energy_eV_atom"]=second["endpoint_energy_eV"]/second["n_atoms"]
            if variant_name=="tight_0p01" and row["control_endpoint_energy_eV_atom"] is not None:
                row["tight_bin"]=str(classify(schema,[task["n_target"]],[row["control_endpoint_energy_eV_atom"]])[0])
                if row["primary_endpoint_energy_eV_atom"] is not None:
                    row["control_minus_primary_meV_atom"]=1000*(row["control_endpoint_energy_eV_atom"]-row["primary_endpoint_energy_eV_atom"])
            rows.append(row)
    write_csv(args.output_dir/"paired_controls.csv",rows)
    summaries=[]
    for name in contexts:
        for n in sorted({r["n_target"] for r in rows if r["variant"]==name}):
            rr=[r for r in rows if r["variant"]==name and r["n_target"]==n]
            good=[r for r in rr if r["primary_endpoint_energy_eV_atom"] is not None and r["control_endpoint_energy_eV_atom"] is not None]
            a=np.array([r["primary_endpoint_energy_eV_atom"] for r in good]);b=np.array([r["control_endpoint_energy_eV_atom"] for r in good])
            summaries.append(dict(variant=name,n_target=n,attempts=len(rr),converged_pairs=len(good),
                primary_statuses=json.dumps(dict(Counter(r["primary_status"] for r in rr))),
                control_statuses=json.dumps(dict(Counter(r["control_status"] for r in rr))),
                spearman=finite(spearmanr(a,b).statistic) if len(good)>1 else None,
                kendall=finite(kendalltau(a,b).statistic) if len(good)>1 else None,
                mean_tightening_change_meV_atom=float(np.mean(1000*(b-a))) if name=="tight_0p01" and len(good) else None,
                maximum_absolute_tightening_change_meV_atom=float(np.max(np.abs(1000*(b-a)))) if name=="tight_0p01" and len(good) else None,
                primary_threshold_bin_crossings=sum(r["primary_bin"]!=r["tight_bin"] for r in good) if name=="tight_0p01" else None,
                scope="TRAIN-only numerical sensitivity" if name=="tight_0p01" else "Small TRAIN-only cross-teacher ranking; different energy zeros not directly subtracted"))
    write_csv(args.output_dir/"composition_summary.csv",summaries)
    sp_rows=[]
    sp_failures=[]
    if bool(args.mace_sp_directory)!=bool(args.mace_sp_manifest):raise ValueError("Supply both MACE SP output and coordinate-export manifest")
    if args.mace_sp_directory:
        exported=json.loads(args.mace_sp_manifest.read_text())
        model=json.loads((args.mace_sp_directory/"model_receipt.json").read_text())
        verify_teacher_receipt(model,contexts["mace_independent"])
        with (args.mace_sp_directory/"results.jsonl").open() as stream:
            raw_sp=[json.loads(line) for line in stream if line.strip()]
        expected_ids={r.get("scoring_structure_id",r["structure_id"]) for r in exported["records"] if r["exported"]}
        recorded_ids={r["structure_id"] for r in raw_sp}
        if len(raw_sp)!=len(recorded_ids) or not expected_ids<=recorded_ids:
            raise ValueError("MACE SP does not cover this complete prespecified scope")
        sp_failures=[r["structure_id"] for r in raw_sp if r["status"]!="success" and r["structure_id"] in expected_ids]
        with np.load(args.mace_sp_directory/"predictions.npz",allow_pickle=False) as z:
            ids=z["structure_ids"].astype(str);energies=z["energy_eV"];natoms=z["n_atoms"]
            offsets=z["offsets"];forces=z["forces_eV_A"];positions=z["positions"];numbers=z["numbers"];cells=z["cell"]
        from paired_common import geometry_hash
        labels={sid:i for i,sid in enumerate(ids)}
        own={r["structure_id"]:r for r in rows if r["variant"]=="mace_independent"}
        for original in exported["records"]:
            sid=original["structure_id"]
            score_id=original.get("scoring_structure_id",sid)
            if not original["exported"] or score_id not in labels:
                sp_rows.append(dict(structure_id=sid,n_target=original["n_target"],status="unknown",mace_on_uma_energy_eV_atom=None,mace_on_uma_fmax_eV_A=None,mace_independent_energy_eV_atom=None,mace_relax_gain_from_uma_meV_atom=None));continue
            i=labels[score_id];lo,hi=offsets[i:i+2]
            if geometry_hash(numbers[lo:hi],positions[lo:hi],cells[i])!=original["literal_geometry_hash"]:
                raise ValueError("MACE SP coordinates differ from exact UMA endpoint export")
            e=float(energies[i]/natoms[i]);independent=own.get(sid,{}).get("control_endpoint_energy_eV_atom")
            sp_rows.append(dict(structure_id=sid,n_target=original["n_target"],status="known",
                mace_on_uma_energy_eV_atom=e,mace_on_uma_fmax_eV_A=float(np.linalg.norm(forces[lo:hi],axis=1).max()),
                mace_independent_energy_eV_atom=independent,
                mace_relax_gain_from_uma_meV_atom=1000*(independent-e) if independent is not None else None))
        write_csv(args.output_dir/"mace_endpoint_checks.csv",sp_rows)
    receipt=dict(status=("PASS_COMPLETE" if not sp_failures else "COMPLETE_WITH_SECONDARY_SP_FAILURES") if not missing else "INCOMPLETE",completed_UTC=datetime.now(timezone.utc).isoformat(),
        controls=len(rows),missing=missing,primary_schema_sha256=sha256(args.endpoint_schema),source_results=sources,
        contexts={k:v["context_hash"] for k,v in contexts.items()},test_energies_read=False,
        MACE_SP_scope="MACE predictions on exact UMA coordinates; independent MACE endpoints are separately identified",
        MACE_SP_failed_ids=sp_failures,
        no_new_model_calls=True,source_sha256=sha256(__file__),
        limitation="Force convergence and small numerical sensitivity do not establish physical ground states, PES basin uniqueness or DFT accuracy")
    atomic_json(args.output_dir/"receipt.json",receipt)
    if missing and not args.allow_incomplete:raise RuntimeError("Missing prescribed controls; failures retained")
    print(json.dumps({k:receipt[k] for k in ["status","controls","missing","test_energies_read"]},indent=2))
    return receipt


if __name__=="__main__":main()

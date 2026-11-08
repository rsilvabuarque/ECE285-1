"""Independent saved-band/ledger/summary review; no calculator or optimizer.

This verifies numerical provenance and records the combined image/tolerance
sensitivity. It cannot establish the lowest saddle, Hessian index or DFT accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import csv,hashlib,json,math
from pathlib import Path

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def js(path):return json.loads(Path(path).read_text())
def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(path,rows):
    with Path(path).open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def scalar_equal(a,b):
    if not math.isclose(float(a),float(b),abs_tol=1e-10,rel_tol=1e-12):raise ValueError("Saved scalar does not reproduce its raw band")
def maximum_force(forces):return np.sqrt(np.sum(np.asarray(forces,float)**2,axis=-1)).max()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--accounting",type=Path,required=True);a=parser.parse_args(argv)
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation();root=a.root.resolve();run=root/"runs/conditioning_20261107/neb"
    controller=js(run/"controller_receipt.json");config=js(run/"controller_config.json");prepared=Path(config["prepared"]);protocol=js(prepared/"protocol.json")
    if controller["status"]!="CONTROLLER_COMPLETE_REVIEW_SCIENTIFIC_RECEIPTS":raise ValueError("Controller is not terminal")
    if controller["prepared_protocol_sha256"]!=sha(prepared/"protocol.json"):raise ValueError("Frozen scientific protocol differs")
    sources={str(run/"controller_receipt.json"):sha(run/"controller_receipt.json"),str(prepared/"protocol.json"):sha(prepared/"protocol.json")}
    rows=[];bands={};all_calls=0;force_error=0.
    for event in protocol["events"]:
        folder=run/event["event_id"];receipt=js(folder/"receipt.json");sources[str(folder/"receipt.json")]=sha(folder/"receipt.json")
        if receipt["status"]!="CONVERGED_NUMERICALLY_UNVALIDATED" or not receipt["endpoint_event_validation"]["valid"]:raise ValueError("Event numerical/endpoint criterion failed")
        if receipt["protocol_sha256"]!=sha(prepared/"protocol.json"):raise ValueError("Event uses another frozen protocol")
        calls=[json.loads(line) for line in (folder/"calculator_calls.jsonl").read_text().splitlines()]
        if [c["call_index"] for c in calls]!=list(range(1,len(calls)+1)):raise ValueError("Actual calculator attempt sequence is incomplete")
        if len(calls)!=receipt["call_accounting"]["actual_calculator_EF_attempts"] or any(c["status"]!="success" for c in calls):raise ValueError("Calculator count/status receipt differs")
        if dict(Counter(c["phase"] for c in calls))!=receipt["call_accounting"]["by_phase"]:raise ValueError("Phase-call accounting differs")
        all_calls+=len(calls);sources[str(folder/"calculator_calls.jsonl")]=sha(folder/"calculator_calls.jsonl")
        for name,file in (("climbing_band","climbing_band_final.npz"),("refinement","refined_band7_final.npz")):
            if name not in receipt:continue
            data=npz(folder/file);record=receipt[name];bands[event["event_id"],name]=data;sources[str(folder/file)]=sha(folder/file)
            if not data["labels_match_geometry"].all() or not np.isfinite(data["energy_eV"]).all() or not np.isfinite(data["forces_eV_A"]).all():raise ValueError("Final band has unknown/nonfinite labels")
            energy=data["energy_eV"];peak=int(np.argmax(energy));forward=float(energy[peak]-energy[0]);reverse=float(energy[peak]-energy[-1])
            scalar_equal(forward,record["discrete_forward_span_eV"]);scalar_equal(reverse,record["discrete_reverse_span_eV"])
            scalar_equal(energy[-1]-energy[0],record["reaction_energy_eV"])
            projected=float(maximum_force(data["NEB_projected_forces_eV_A"]))
            scalar_equal(projected,record["NEB_projected_Fmax_eV_A"])
            physical=float(maximum_force(data["forces_eV_A"][record["climbing_image_index"]]))
            error=abs(physical-record["physical_climbing_image_Fmax_eV_A"]);force_error=max(force_error,error)
            # Physical arrays came from FP32 calculators; a three-component norm
            # may round differently from this double-precision re-evaluation.
            unit=np.finfo(np.float32).eps/2;bound=(8*unit/(1-8*unit))*max(physical,np.finfo(float).tiny)
            if error>bound:raise ValueError("Physical climbing norm differs beyond FP32 arithmetic")
            if projected>record["target_Fmax_eV_A"]:raise ValueError("Declared projected-force target was not met")
            rows.append(dict(event_id=event["event_id"],teacher="uma",path=name,interior_images=len(energy)-2,
                forward_potential_barrier_eV=forward,reverse_potential_barrier_eV=reverse,reaction_energy_eV=float(energy[-1]-energy[0]),
                NEB_projected_Fmax_eV_A=projected,physical_climbing_image_Fmax_eV_A=physical,
                declared_projected_target_eV_A=record["target_Fmax_eV_A"],source_band_sha256=sha(folder/file)))
        for index,e in enumerate(receipt["endpoints"]):
            endpoint=npz(folder/f"endpoint_{index}.npz")
            if not endpoint["labels_match_geometry"].all() or maximum_force(endpoint["forces_eV_A"])>protocol["endpoint_relaxation"]["Fmax_eV_A"]:
                raise ValueError("Raw endpoint physical-force target failed")
            if not np.array_equal(endpoint["positions_A"][0],bands[event["event_id"],"climbing_band"]["positions_A"][0 if index==0 else -1]):
                raise ValueError("NEB changed its quenched endpoint geometry")
            sources[str(folder/f"endpoint_{index}.npz")]=sha(folder/f"endpoint_{index}.npz")
    first=protocol["events"][0]["event_id"];coarse=bands[first,"climbing_band"];refined=bands[first,"refinement"]
    for field in ("positions_A","numbers","cell_A","pbc","parent_indices"):
        left,right=coarse[field],refined[field]
        if field=="positions_A":left,right=left[[0,-1]],right[[0,-1]]
        if not np.array_equal(left,right):raise ValueError("Refinement changed its exact endpoints, cell or atom identities")
    coarse_barrier=rows[0]["forward_potential_barrier_eV"];fine=next(r for r in rows if r["event_id"]==first and r["path"]=="refinement")
    change=fine["forward_potential_barrier_eV"]-coarse_barrier
    source=run/first/"climbing_band_final.npz";mace_receipt=js(run/"mace_profile/receipt.json");mace=npz(run/"mace_profile/MACE_on_UMA_band.npz")
    if mace_receipt["status"]!="COMPLETE_SINGLE_POINT_DISAGREEMENT_ONLY" or mace_receipt["source_UMA_band"]["sha256"]!=sha(source):raise ValueError("MACE did not audit the exact declared five-interior band")
    for field in ("positions_A","numbers","cell_A","pbc","parent_indices"):
        if not np.array_equal(mace[field],coarse[field]):raise ValueError("MACE profile coordinates/IDs differ from the frozen coarse UMA path")
    calls=[json.loads(line) for line in (run/"mace_profile/calculator_calls.jsonl").read_text().splitlines()]
    if len(calls)!=7 or any(c["status"]!="success" for c in calls):raise ValueError("MACE profile does not have seven successful single points")
    all_calls+=len(calls);energy=mace["energy_eV"];peak=int(energy.argmax());span_f=float(energy[peak]-energy[0]);span_r=float(energy[peak]-energy[-1])
    scalar_equal(span_f,mace_receipt["profile_spans_not_MACE_barriers"]["discrete_forward_span_eV"])
    scalar_equal(span_r,mace_receipt["profile_spans_not_MACE_barriers"]["discrete_reverse_span_eV"])
    profile_rows=[dict(image=i,UMA_energy_relative_initial_eV=float(ue-coarse["energy_eV"][0]),MACE_energy_relative_initial_eV=float(me-energy[0]),
        UMA_physical_Fmax_eV_A=float(maximum_force(uf)),MACE_physical_Fmax_eV_A=float(maximum_force(mf)))
        for i,(ue,me,uf,mf) in enumerate(zip(coarse["energy_eV"],energy,coarse["forces_eV_A"],mace["forces_eV_A"]))]
    sources[str(run/"mace_profile/receipt.json")]=sha(run/"mace_profile/receipt.json");sources[str(run/"mace_profile/MACE_on_UMA_band.npz")]=sha(run/"mace_profile/MACE_on_UMA_band.npz")
    summary=Path(config["summary_output"])
    with (summary/"outcomes.csv").open() as f:published=list(csv.DictReader(f))
    if len(published)!=3:raise ValueError("Summary omits a completed path/profile")
    for r in rows:
        if r["path"]=="climbing_band":
            reported=next(p for p in published if p["event"]==r["event_id"] and p["teacher"]=="uma")
            scalar_equal(reported["forward_barrier_eV"],r["forward_potential_barrier_eV"])
    reported_mace=next(p for p in published if p["teacher"]=="mace")
    if reported_mace["forward_barrier_eV"] or reported_mace["reverse_barrier_eV"]:raise ValueError("Summary mislabels fixed MACE profile as optimized barrier")
    with a.accounting.open() as f:account=list(csv.DictReader(f,delimiter="|"))
    alloc=next(row for row in account if row["JobIDRaw"]==controller["job_id"])
    if alloc["State"] not in {"COMPLETED","CANCELLED","TIMEOUT"}:raise ValueError("Allocation is not terminal")
    generic=next((int(v.split("=",1)[1]) for v in alloc["AllocTRES"].split(",") if v.startswith("gres/gpu=")),None)
    if generic!=2:raise ValueError("Dedicated NEB allocation did not have exactly two GPUs")
    device_hours=generic*int(alloc["ElapsedRaw"])/3600
    if device_hours>2:raise ValueError("Dedicated NEB allocation exceeded two GPU-device-hours")
    out=a.output_dir
    if out.exists():raise FileExistsError("Preserve previous independent NEB review")
    out.mkdir(parents=True,exist_ok=False);write(out/"UMA_band_checks.csv",rows);write(out/"MACE_on_coarse_UMA_profile.csv",profile_rows)
    result=dict(status="PASS_SAVED_NUMERICAL_AND_PROVENANCE_CHECKS",audited_UTC=datetime.now(timezone.utc).isoformat(),
        source_hashes=sources,script_sha256=sha(__file__),actual_total_calculator_EF_attempts=all_calls,failed_calculator_attempts=0,
        event1_exact_endpoints_unchanged_under_refinement=True,event1_coarse_forward_eV=coarse_barrier,event1_refined_forward_eV=fine["forward_potential_barrier_eV"],
        event1_forward_change_eV=change,changed_factors=["five to seven interior images","projected-force target0.05 to0.03eV/A"],
        barrier_stability="Not established: a large combined image/tolerance/path change remains despite numerical-force convergence; no universal barrier-accuracy threshold imposed",
        MACE_source="Exact FIVE-interior UMA climbing band, not the refined seven-interior band",MACE_forward_profile_span_eV=span_f,MACE_reverse_profile_span_eV=span_r,
        independently_optimized_MACE_barrier=False,maximum_physical_norm_recalculation_difference_eV_A=force_error,
        summary_numeric_fields_verified=True,allocation=dict(alloc,allocated_GPU_device_hours=device_hours,accounting_sha256=sha(a.accounting),timezone="UTC"),
        new_MLIP_calls=0,new_optimizer_updates=0,physical_scope="Two marked Li hops under pinned MLIPs; no DFT/Hessian validation, no free energies/diffusivity/transport-network claim")
    with (out/"receipt.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps(result,indent=2));return 0


if __name__=="__main__":raise SystemExit(main())

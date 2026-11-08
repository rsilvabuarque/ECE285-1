"""Independent raw editor-pair audit; open new scores only after owner release.

No generation/evaluation helpers are imported. Canonicalization, exact edit
checks, bin/tie logic, distinct/all-attempt yields and paired energy statistics
are recomputed here. --labels-complete is an explicit execution-owner release,
not a command to poll or inspect partial labels. Without it only proposals and
their provenance are checked; no score archive is opened.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


METHODS = ("learned_edit", "mutation")
BINS = ("low", "middle", "high")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024**2), b""):h.update(block)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline="") as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    if not rows:Path(path).write_text("");return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def canonical(occupancy,permutations):
    """Independent uint64 implementation of all192 parent operations (48 sites)."""
    occupancy=np.asarray(occupancy)
    if occupancy.ndim!=2 or occupancy.shape[1]!=48 or not np.isin(occupancy,[0,1]).all():
        raise ValueError("Expected binary48site present=1 occupations")
    permutations=np.asarray(permutations,int)
    if permutations.shape!=(192,48) or not np.all(np.sort(permutations,axis=1)==np.arange(48)):
        raise ValueError("Independent audit requires all192 valid declared parent operations")
    inverse=np.argsort(permutations,axis=1)
    powers=np.left_shift(np.uint64(1),np.arange(47,-1,-1,dtype=np.uint64))
    result=[]
    for start in range(0,len(occupancy),128):
        transformed=(1-occupancy[start:start+128])[:,inverse]
        minimum=np.sum(transformed.astype(np.uint64)*powers,axis=2).min(1)
        result.extend(format(int(code),"048b") for code in minimum)
    return np.asarray(result)


def exact_edit(seed,child,removed,added,k):
    seed,child,removed,added=map(np.asarray,(seed,child,removed,added))
    if any(a.shape!=(48,) or not np.isin(a,[0,1]).all() for a in (seed,child,removed,added)):
        raise ValueError("Malformed generated raw edit")
    actual_remove=((seed==1)&(child==0)).astype(np.uint8)
    actual_add=((seed==0)&(child==1)).astype(np.uint8)
    if not np.array_equal(removed,actual_remove) or not np.array_equal(added,actual_add):
        raise ValueError("Saved removal/addition masks disagree with raw changes")
    if k not in (1,2,3) or removed.sum()!=k or added.sum()!=k or child.sum()!=seed.sum() or np.count_nonzero(child!=seed)!=2*k:
        raise ValueError("Raw edit violates exact k removals, k insertions or2k Hamming distance")


def uniform_subset(eligible,k,uniforms):
    """Independent combinatorial uniform-k sampler: P(take)=remaining/left."""
    chosen=np.zeros(len(eligible),np.uint8);positions=np.flatnonzero(eligible);remaining=int(k)
    for i,site in enumerate(positions):
        if remaining and (remaining==len(positions)-i or uniforms[site]<remaining/(len(positions)-i)):
            chosen[site]=1;remaining-=1
    if remaining:raise ValueError("Uniform subset did not meet count")
    return chosen


def measured_bin(schema,n,energy_per_atom):
    lower,upper=schema["compositions"][str(n)]["edges_eV_atom"]
    return "low" if energy_per_atom<=lower else "middle" if energy_per_atom<=upper else "high"


def random_seed(*items):return int.from_bytes(hashlib.sha256("|".join(map(str,items)).encode()).digest()[:4],"little")


def string_values(value):
    if isinstance(value,str):return {value}
    if isinstance(value,dict):return set().union(*(string_values(v) for v in value.values())) if value else set()
    if isinstance(value,list):return set().union(*(string_values(v) for v in value)) if value else set()
    return set()


def registration_audit(root,generated,preflight,receipt_sha,candidate_sha,rows):
    """Bind explicit preflight time/output hashes to every fresh editor task."""
    before=read_json(preflight)
    if not {receipt_sha,candidate_sha,digest(generated/"attempts.csv"),digest(generated/"raw_attempts.npz"),digest(generated/"protocol.json")}<=string_values(before):
        raise ValueError("Explicit registration preflight does not bind every frozen proposal output")
    time_keys=("created_unix","preflight_unix","checked_unix","timestamp_unix")
    preflight_time=next((float(before[k]) for k in time_keys if k in before),None)
    if preflight_time is None:raise ValueError("Registration preflight lacks explicit Unix time")
    oracle=root/"runs/conditioning_20261107/oracle"
    registry_path=oracle/"registry.json";registry=read_json(registry_path)
    tasks=[t for t in registry["tasks"] if t["input_sha256"]==candidate_sha]
    if any(t["model"]!="uma" for t in tasks):raise ValueError("Unexpected extra teacher for optional editor")
    candidate=load_npz(generated/"candidates.npz");ids=candidate["structure_ids"].astype(str)
    expected_ids=set(ids);seen=set();records=[]
    for task in tasks:
        if task["input_sha256"]!=digest(task["input_npz"]):raise ValueError("Registered editor candidate input changed")
        actual=ids[np.asarray(task["indices"],int)].tolist()
        if actual!=task["expected_structure_ids"] or len(set(actual))!=len(actual):
            raise ValueError("Registered editor task changes exact physical candidate identities")
        if seen&set(actual):raise ValueError("Repeated fresh editor geometry across tasks")
        seen.update(actual)
        owner_path=oracle/"claims"/task["task_id"]/"owner.json"
        owner=read_json(owner_path)
        start=owner.get("started_unix",owner.get("claimed_unix"))
        if start is None or float(start)<preflight_time:raise ValueError("Editor fresh SCF claim predates its frozen preflight")
        records.append(dict(task_id=task["task_id"],fresh_geometries=len(actual),started_unix=float(start),
            seconds_after_preflight=float(start)-preflight_time,owner_sha256=digest(owner_path),worker=owner.get("worker","")))
    if not seen<=expected_ids:raise ValueError("Editor task contains a nonfrozen candidate")
    return dict(preflight_sha256=digest(preflight),preflight_unix=preflight_time,
        registry_sha256=digest(registry_path),new_editor_tasks=len(tasks),fresh_editor_pairs=len(seen),
        other_candidate_identities=len(expected_ids-seen),
        interpretation="Fresh editor tasks are identified by frozen candidate input SHA. Other identities require existing exact-score provenance; they are not fresh editor calls.",
        all_new_claims_after_preflight=True),records


def compare_table(path,rows,keys,fields):
    actual={tuple(str(row[k]) for k in keys):row for row in read_csv(path)}
    if len(actual)!=len(rows):raise ValueError(f"Independent/evaluator row counts differ: {path.name}")
    for row in rows:
        key=tuple(str(row[k]) for k in keys)
        if key not in actual:raise ValueError(f"Missing evaluator group: {key}")
        for field in fields:
            left,right=row[field],actual[key][field]
            if left=="" and right=="":continue
            if isinstance(left,int) and str(left)!=str(right):raise ValueError(f"Exact denominator/count differs: {field}/{key}")
            if not math.isclose(float(left),float(right),abs_tol=1e-12,rel_tol=1e-10):
                raise ValueError(f"Independent metric differs: {field}/{key}: {left} vs {right}")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True);p.add_argument("--generated",type=Path)
    p.add_argument("--preflight",type=Path);p.add_argument("--scores",type=Path);p.add_argument("--evaluation",type=Path)
    p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--labels-complete",action="store_true")
    a=p.parse_args(argv)
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    root=a.root.resolve();run=root/"runs/conditioning_20261107";gen=a.generated or run/"editor/assessment_v1"
    out=a.output_dir.resolve()
    if out.exists():raise FileExistsError("Preserve the preceding independent editor audit")
    protocol=read_json(gen/"protocol.json");generation=read_json(gen/"receipt.json")
    if generation["status"]!="FROZEN_BEFORE_SCORING" or generation["protocol_sha256"]!=digest(gen/"protocol.json"):
        raise ValueError("Editor proposal protocol is not frozen")
    for name,sha in generation["outputs"].items():
        if digest(gen/name)!=sha:raise ValueError("Frozen editor proposal output changed")
    for source in protocol["sources"].values():
        if digest(source["path"])!=source["sha256"]:raise ValueError("Frozen editor input changed")
    for seed,checkpoint in protocol["checkpoints"].items():
        if digest(checkpoint["path"])!=checkpoint["sha256"]:raise ValueError("Editor checkpoint changed")
        folder=Path(checkpoint["path"]).parent;summary=read_json(folder/"summary.json");cfg=read_json(folder/"configuration.json")
        if summary["status"]!="FIT_COMPLETE" or summary["optimizer_updates"]!=4000 or int(cfg.get("seed",cfg.get("fit_seed")))!=int(seed):
            raise ValueError("Editor replicate is not the actual matched4000-update fit")
    parent=load_npz(protocol["sources"]["parent"]["path"]);perms=parent["permutations"]
    bank=load_npz(protocol["sources"]["seedbank"]["path"]);pairdata=load_npz(protocol["sources"]["pair_data"]["path"])
    raw=load_npz(gen/"raw_attempts.npz");candidates=load_npz(gen/"candidates.npz");rows=read_csv(gen/"attempts.csv")
    if len(rows)!=6912 or len(rows)!=protocol["expected_attempts"] or any(len(raw[k])!=len(rows) for k in raw):raise ValueError("All-attempt raw row budget changed")
    if len({r["attempt_id"] for r in rows})!=len(rows):raise ValueError("Attempt IDs repeat")
    valid=np.asarray([r["status"]=="generated" for r in rows]);seedkeys=canonical(raw["seed_occupancy"],perms)
    childkeys=canonical(raw["occupancy"][valid],perms);childkey_lookup=dict(zip(np.flatnonzero(valid),childkeys))
    candidate_keys=canonical(candidates["occupancy"],perms)
    candidate_ids=candidates["structure_ids"].astype(str)
    if len(set(candidate_ids))!=len(candidate_ids) or not np.array_equal(candidates["occupancy"],1-candidates["masks"]):raise ValueError("Candidate pool contains duplicates or inconsistent masks")
    if any(sid!=f"mask48_{int(key,2):012x}" for sid,key in zip(candidate_ids,candidate_keys)):
        raise ValueError("Scoring candidate ID does not bind its actual canonical geometry")
    if any("".join(map(str,mask.tolist()))!=key for mask,key in zip(candidates["masks"],candidate_keys)):
        raise ValueError("Scoring pool is not in the exact canonical physical orientation")
    if str(candidates["parent_sha256"].item())!=protocol["sources"]["parent"]["sha256"]:
        raise ValueError("Scoring pool belongs to another exact parent")
    candidate_set=set(candidate_ids)
    pairs=defaultdict(dict);groups=defaultdict(list);baseline_mismatches=[]
    nested_role={}
    for sk,ck,role in zip(pairdata["canonical_seed_keys"].astype(str),pairdata["canonical_child_keys"].astype(str),pairdata["split"].astype(str)):
        for key in [sk,ck]:
            if key in nested_role and nested_role[key]!=role:raise ValueError("Editor train/val/heldout identity bridge")
            nested_role[key]=role
    for i,row in enumerate(rows):
        if int(row["raw_row"])!=i or row["canonical_seed_mask"]!=seedkeys[i]:raise ValueError("Raw seed row/canonical identity changed")
        bankrow=int(row["heldout_seed_bank_row"]);seed=raw["seed_occupancy"][i];k=int(row["swap_count"]);n=int(row["n_Li"])
        if not np.array_equal(seed,bank["occupancy"][bankrow]) or str(bank["seed_ids"][bankrow])!=row["seed_structure_id"] or bank["H0_source_rows"][bankrow]!=int(row["H0_source_row"]):
            raise ValueError("Actual raw seed does not match its declared heldout bank identity/source")
        if str(bank["split"][bankrow])!="heldout" or nested_role[seedkeys[i]]!="heldout" or str(bank["component_ids"][bankrow])!=row["seed_component_id"]:
            raise ValueError("Editor assessment seed crosses its nested heldout role")
        if int(seed.sum())!=n or int(row["n_target"])!=n or row["seed_bin"]!=row["requested_bin"] or str(bank["seed_bin"][bankrow])!=row["seed_bin"]:raise ValueError("Assessment seed count/bin differs from request")
        if row["seed_scoring_id"]!=f"mask48_{int(seedkeys[i],2):012x}":raise ValueError("Seed energy scoring identity differs from its actual canonical seed")
        if raw["uniforms_remove"][i].shape!=(48,) or not np.isfinite(raw["uniforms_remove"][i]).all():raise ValueError("Malformed common uniform row")
        for uniform in (raw["uniforms_remove"][i],raw["uniforms_add"][i]):
            if not np.all((uniform>=0)&(uniform<1)):raise ValueError("Uniform stream outside[0,1)")
        if row["method"] in pairs[row["paired_trial_id"]]:raise ValueError("Repeated method within paired trial")
        pairs[row["paired_trial_id"]][row["method"]]=i
        groups[(row["method"],int(row["fit_seed"]),n,row["requested_bin"])].append(i)
        if valid[i]:
            exact_edit(seed,raw["occupancy"][i],raw["removed"][i],raw["added"][i],k)
            if childkey_lookup[i]!=row["canonical_mask"] or row["structure_id"]!=f"mask48_{int(childkey_lookup[i],2):012x}":raise ValueError("Raw child canonical identity differs")
            if row["structure_id"] not in candidate_set or row["seed_scoring_id"] not in candidate_set:raise ValueError("Seed/child missing from complete scoring pool")
            for name in ("removed","added"):
                if "|".join(map(str,np.flatnonzero(raw[name][i])))!=row[name+"_sites"]:raise ValueError("Raw edit-site list changed")
            if row["method"]=="mutation":
                remove=uniform_subset(seed==1,k,raw["uniforms_remove"][i]);add=uniform_subset(seed==0,k,raw["uniforms_add"][i])
                if not np.array_equal(remove,raw["removed"][i]) or not np.array_equal(add,raw["added"][i]):baseline_mismatches.append(row["attempt_id"])
        elif any(not np.all(raw[name][i]==-1) for name in ("occupancy","removed","added")) or row["canonical_mask"] or row["structure_id"] or not row["error"]:
            raise ValueError("Failed/unsupported request was partially repaired or silently hidden")
    if baseline_mismatches:raise ValueError(f"Independent uniform-k reconstruction differs; inspect roundoff/streams before accepting: {baseline_mismatches[:5]}")
    used={r["seed_scoring_id"] for r in rows}|{r["structure_id"] for r in rows if r["status"]=="generated"}
    if used!=candidate_set:raise ValueError("Scoring pool differs from exactly the attempted seeds and generated children")
    expected_groups={(m,s,n,b) for m in METHODS for s in (20261106,20261107,20261108) for n in (12,24,36) for b in BINS}
    if set(groups)!=expected_groups or any(len(indices)!=128 for indices in groups.values()):raise ValueError("Missing or altered128-attempt condition/method/replicate cell")
    for pair,methods in pairs.items():
        if set(methods)!=set(METHODS):raise ValueError("Unpaired trial")
        left,right=(rows[methods[m]] for m in METHODS);li,ri=(methods[m] for m in METHODS)
        for name in ("seed_structure_id","seed_scoring_id","heldout_seed_bank_row","H0_source_row","seed_component_id","canonical_seed_mask","swap_count","n_Li","requested_bin","fit_seed","sampling_seed"):
            if left[name]!=right[name]:raise ValueError(f"Paired trial changes actual seed/edit information: {name}")
        for name in ("seed_occupancy","uniforms_remove","uniforms_add"):
            if not np.array_equal(raw[name][li],raw[name][ri]):raise ValueError("Paired trial lacks identical raw seed and random rows")
    for (method,fit,n,b),indices in groups.items():
        ordered=sorted(indices,key=lambda i:int(rows[i]["paired_trial_id"].rsplit("-",1)[1]))
        for name,field in (("uniforms_remove","uniforms_remove_sha256"),("uniforms_add","uniforms_add_sha256")):
            sha=hashlib.sha256(raw[name][ordered].tobytes()).hexdigest()
            if any(rows[i][field]!=sha for i in indices):raise ValueError("Actual common group uniforms differ from their frozen fingerprint")
        eligible=np.flatnonzero((bank["n_Li"]==n)&(bank["seed_bin"].astype(str)==b))
        rng=np.random.default_rng(random_seed(protocol["namespace"],fit,n,b,"seed-and-edit-budget"))
        chosen=rng.choice(eligible,128,replace=True);budgets=rng.integers(1,4,size=128)
        if not np.array_equal(chosen,[int(rows[i]["heldout_seed_bank_row"]) for i in ordered]) or not np.array_equal(budgets,[int(rows[i]["swap_count"]) for i in ordered]):raise ValueError("Actual seed/k draws differ from prespecified uniform streams")
    out.mkdir(parents=True,exist_ok=False)
    result=dict(status="PROPOSALS_PASS_LABELS_NOT_OPENED",audited_UTC=datetime.now(timezone.utc).isoformat(),
        attempts=len(rows),paired_trials=len(pairs),failed_or_unsupported_requests=int((~valid).sum()),
        canonical_parent_operations=len(perms),raw_2k_verified_generated_attempts=int(valid.sum()),
        actual_seed_k_and_common_uniform_rows_equal=True,independent_uniform_mutation_reconstruction=True,
        seed_and_child_nested_role_bridges=0,script_sha256=digest(__file__),new_MLIP_calls=0,new_training_updates=0)
    if a.labels_complete:
        full_metrics(a,root,run,gen,out,protocol,generation,rows,valid,raw,pairs,groups,result)
    result["source_hashes"]={name:digest(gen/name) for name in ("protocol.json","receipt.json","attempts.csv","raw_attempts.npz","candidates.npz")}
    (out/"receipt.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2));return 0


def full_metrics(a,root,run,gen,out,protocol,generation,rows,valid,raw,pairs,groups,result):
    preflight=a.preflight or run/"editor/registration_preflight.json"
    timing,tasks=registration_audit(root,gen,preflight,digest(gen/"receipt.json"),digest(gen/"candidates.npz"),rows)
    write_csv(out/"registration_tasks.csv",tasks);result["frozen_before_fresh_scoring"]=timing
    path=a.scores or run/"collected/editor/uma.npz";collector=read_json(path.with_name("receipt.json"))
    if collector.get("status")!="PASS":raise ValueError("Owner released labels but collector is incomplete")
    scores=load_npz(path);ids=scores["structure_ids"].astype(str)
    if len(set(ids))!=len(ids):raise ValueError("Ambiguous duplicate scalar score identities")
    score={sid:i for i,sid in enumerate(ids)};schema=read_json(protocol["sources"]["schema"]["path"])
    history=set(read_json(protocol["sources"]["history"]["path"]));train=set(read_json(run/"data/training_keys.json"))
    original={int(r["n_Li"]):float(r["q20_eV_atom"]) for r in read_csv(root/"artifacts/current/reports/generation_thresholds.csv") if r["teacher"]=="uma" and r["source"]=="main_train"}
    if set(original)!={12,24,36}:raise ValueError("Original fixed-q20 thresholds are incomplete")
    result["seed_label_parity"]=seed_parity(protocol,rows,scores,score,schema,out)
    outcomes=[];known={};unknown=0
    for i,row in enumerate(rows):
        record=dict(attempt_id=row["attempt_id"],method=row["method"],fit_seed=int(row["fit_seed"]),n_Li=int(row["n_Li"]),
            requested_bin=row["requested_bin"],status=row["status"],canonical_mask=row["canonical_mask"],energy_known=False,
            novel_both=False,novel_H0TRAIN=False,novel_history=False,measured_bin="failed_or_unmeasured",passes_original_q20=False,
            seed_energy_eV="",child_energy_eV="",delta_energy_eV="",delta_energy_meV_atom="")
        if valid[i]:
            si,ci=score.get(row["seed_scoring_id"]),score.get(row["structure_id"]);n=int(row["n_Li"]);atoms=144+n
            ok=si is not None and ci is not None
            if ok:
                for j in (si,ci):
                    if int(scores["n_atoms"][j])!=atoms or int(scores["n_Li"][j])!=n:raise ValueError("Seed/child energy is from a different exact composition")
                    status=str(scores.get("status",np.full(len(ids),"success"))[j]).lower()
                    ok=ok and status in {"success","complete","completed","ok","fresh","cached"} and np.isfinite(scores["energy_eV"][j])
            if ok:
                es,ec=float(scores["energy_eV"][si]),float(scores["energy_eV"][ci]);key=row["canonical_mask"]
                record.update(energy_known=True,novel_H0TRAIN=key not in train,novel_history=key not in history,
                    novel_both=key not in train and key not in history,measured_bin=measured_bin(schema,n,ec/atoms),
                    passes_original_q20=ec/atoms<=original[n],seed_energy_eV=es,child_energy_eV=ec,
                    delta_energy_eV=ec-es,delta_energy_meV_atom=1000*(ec-es)/atoms)
                known[i]=record
            else:unknown+=1
        outcomes.append(record)
    metrics=[]
    for (method,fit,n,b),indices in sorted(groups.items()):
        rr=[outcomes[i] for i in indices];measured=[r for r in rr if r["energy_known"]]
        useful={r["canonical_mask"] for r in measured if r["novel_both"] and r["measured_bin"]==b}
        low={r["canonical_mask"] for r in measured if r["novel_both"] and r["passes_original_q20"]}
        metrics.append(dict(method=method,fit_seed=fit,n_Li=n,requested_bin=b,attempts=len(rr),measured_attempts=len(measured),
            failed_or_unmeasured_attempts=len(rr)-len(measured),distinct_novel_requested_orbits=len(useful),novel_inbin_yield=len(useful)/len(rr),
            distinct_novel_original_q20_orbits=len(low),novel_original_q20_yield=len(low)/len(rr),
            bin_hit_rate=sum(r["measured_bin"]==b for r in measured)/len(rr),
            mean_delta_energy_meV_atom=float(np.mean([r["delta_energy_meV_atom"] for r in measured])) if measured else "",
            fraction_lower_energy_than_seed=sum(r["delta_energy_eV"]<0 for r in measured)/len(rr)))
    seed_metrics=[]
    for method in METHODS:
        for fit in (20261106,20261107,20261108):
            cc=[r for r in metrics if r["method"]==method and r["fit_seed"]==fit]
            seed_metrics.append(dict(method=method,fit_seed=fit,balanced_novel_inbin_yield=float(np.mean([r["novel_inbin_yield"] for r in cc])),
                original_q20_low_request_yield=float(np.mean([r["novel_original_q20_yield"] for r in cc if r["requested_bin"]=="low"])),
                mean_delta_energy_meV_atom=float(np.mean([r["mean_delta_energy_meV_atom"] for r in cc if r["mean_delta_energy_meV_atom"]!=""]))))
    summary=[]
    for method in METHODS:
        ss=[r for r in seed_metrics if r["method"]==method]
        summary.append(dict(method=method,fit_or_paired_sampler_replicates=len(ss),balanced_novel_inbin_yield=float(np.mean([r["balanced_novel_inbin_yield"] for r in ss])),
            balanced_descriptive_seed_SD=float(np.std([r["balanced_novel_inbin_yield"] for r in ss],ddof=1)),
            original_q20_low_request_yield=float(np.mean([r["original_q20_low_request_yield"] for r in ss])),
            mean_delta_energy_meV_atom=float(np.mean([r["mean_delta_energy_meV_atom"] for r in ss]))))
    paired=[]
    for pair,methods in pairs.items():
        li,ri=(methods[m] for m in METHODS);left,right=outcomes[li],outcomes[ri]
        ok=left["energy_known"] and right["energy_known"]
        if ok and left["seed_energy_eV"]!=right["seed_energy_eV"]:raise ValueError("Paired methods use different physical seed energy")
        paired.append(dict(paired_trial_id=pair,fit_seed=left["fit_seed"],n_Li=left["n_Li"],requested_bin=left["requested_bin"],
            swap_count=int(rows[li]["swap_count"]),both_energies_known=ok,
            learned_minus_mutation_meV_atom=1000*(left["child_energy_eV"]-right["child_energy_eV"])/(144+left["n_Li"]) if ok else ""))
    paired_stats=[]
    for fit in (20261106,20261107,20261108):
        rr=[r for r in paired if r["fit_seed"]==fit];d=np.asarray([r["learned_minus_mutation_meV_atom"] for r in rr if r["both_energies_known"]],float)
        paired_stats.append(dict(fit_seed=fit,all_paired_trials=len(rr),known_paired_trials=len(d),
            mean_learned_minus_mutation_meV_atom=float(d.mean()) if len(d) else "",median_learned_minus_mutation_meV_atom=float(np.median(d)) if len(d) else "",
            learned_lower_energy_pairs=int(np.count_nonzero(d<0)),equal_energy_pairs=int(np.count_nonzero(d==0)),mutation_lower_energy_pairs=int(np.count_nonzero(d>0))))
    for name,content in (("attempt_outcomes.csv",outcomes),("metrics.csv",metrics),("seed_metrics.csv",seed_metrics),
                         ("summary.csv",summary),("paired_energy_outcomes.csv",paired),("paired_energy_statistics.csv",paired_stats)):
        write_csv(out/name,content)
    evaluation=a.evaluation or run/"evaluation/editor"
    compare_table(evaluation/"metrics.csv",metrics,["method","fit_seed","n_Li","requested_bin"],
        ["attempts","measured_attempts","failed_or_unmeasured_attempts","distinct_novel_requested_orbits","novel_inbin_yield",
         "distinct_novel_original_q20_orbits","novel_original_q20_yield","bin_hit_rate","mean_delta_energy_meV_atom","fraction_lower_energy_than_seed"])
    compare_table(evaluation/"seed_metrics.csv",seed_metrics,["method","fit_seed"],
        ["balanced_novel_inbin_yield","original_q20_low_request_yield","mean_delta_energy_meV_atom"])
    compare_table(evaluation/"summary.csv",summary,["method"],
        ["fit_or_paired_sampler_replicates","balanced_novel_inbin_yield","balanced_descriptive_seed_SD","original_q20_low_request_yield","mean_delta_energy_meV_atom"])
    compare_table(evaluation/"paired_energy_outcomes.csv",paired,["paired_trial_id"],["learned_minus_mutation_meV_atom"])
    result.update(status="PASS" if not unknown else "INCOMPLETE_SCORE_COVERAGE",owner_released_labels=True,
        scores_sha256=digest(path),collector_receipt_sha256=digest(path.with_name("receipt.json")),unknown_generated_attempts=unknown,
        independently_recomputed_evaluator_tables_identical=True,summary=summary,paired_energy_statistics=paired_stats,
        physical_scope="Same exact canonical UMA teacher protocol for child/seed; no DFT validation; no selection or training from these labels",
        uncertainty="Three editor fits and matched sampler replicates; descriptive spread, no proposal-level confidence intervals",
        numeric_comparison_tolerance=dict(absolute=1e-12,relative=1e-10,scope="floating-point table reproduction only; counts/identities exact"))


def seed_parity(protocol,attempts,scores,lookup,schema,out):
    """Compare actual canonical seed scores with exact original H0 source rows."""
    preparation=read_json(protocol["sources"]["preparation"]["path"])
    binding=preparation["source_hashes"]["H0"]
    if digest(binding["path"])!=binding["sha256"]:raise ValueError("Original H0 seed-label corpus changed")
    data=load_npz(binding["path"]);bank=load_npz(protocol["sources"]["seedbank"]["path"])
    unique=sorted({int(r["heldout_seed_bank_row"]) for r in attempts})
    records=[]
    for bankrow in unique:
        source=int(bank["H0_source_rows"][bankrow]);key=str(bank["canonical_seed_keys"][bankrow]);sid=f"mask48_{int(key,2):012x}"
        if not np.array_equal(bank["occupancy"][bankrow],data["occupancy"][source]) or str(bank["seed_ids"][bankrow])!=str(data["structure_ids"][source]):
            raise ValueError("Seed-parity H0 row does not match the exact actual seed")
        if sid not in lookup:raise ValueError("Seed-parity canonical geometry lacks its declared current UMA score")
        index=lookup[sid];n=int(bank["n_Li"][bankrow]);atoms=144+n
        if int(scores["n_atoms"][index])!=atoms or not np.isfinite(scores["energy_eV"][index]):raise ValueError("Seed-parity current score is invalid")
        old=float(data["energy_eV_atom"][source]);new=float(scores["energy_eV"][index])/atoms
        old_bin,new_bin=measured_bin(schema,n,old),measured_bin(schema,n,new)
        if old_bin!=str(bank["seed_bin"][bankrow]):raise ValueError("Seed bank bin differs from its actual frozen H0 energy")
        lower,upper=schema["compositions"][str(n)]["edges_eV_atom"];width=1000*(upper-lower)
        delta=1000*(new-old)
        records.append(dict(seed_scoring_id=sid,H0_seed_id=str(bank["seed_ids"][bankrow]),H0_source_row=source,
            n_Li=n,H0_energy_eV_atom=old,current_UMA_energy_eV_atom=new,difference_meV_atom=delta,
            absolute_difference_meV_atom=abs(delta),H0_bin=old_bin,current_UMA_bin=new_bin,
            q20_q80_width_meV_atom=width,absolute_difference_over_q20_q80_width=abs(delta)/width if width else ""))
    summaries=[];confusion=[]
    for n in (12,24,36):
        rr=[r for r in records if r["n_Li"]==n];absolute=np.asarray([r["absolute_difference_meV_atom"] for r in rr]);signed=np.asarray([r["difference_meV_atom"] for r in rr])
        width=rr[0]["q20_q80_width_meV_atom"]
        summaries.append(dict(n_Li=n,unique_actual_seeds=len(rr),median_absolute_difference_meV_atom=float(np.median(absolute)),
            p95_absolute_difference_meV_atom=float(np.quantile(absolute,.95)),maximum_absolute_difference_meV_atom=float(absolute.max()),
            median_signed_difference_meV_atom=float(np.median(signed)),q20_q80_width_meV_atom=width,
            median_absolute_difference_over_binwidth=float(np.median(absolute)/width) if width else "",
            maximum_absolute_difference_over_binwidth=float(absolute.max()/width) if width else "",
            seed_bin_changed_count=sum(r["H0_bin"]!=r["current_UMA_bin"] for r in rr)))
        for old_bin in BINS:
            count=sum(r["H0_bin"]==old_bin for r in rr)
            for new_bin in BINS:
                matches=sum(r["H0_bin"]==old_bin and r["current_UMA_bin"]==new_bin for r in rr)
                confusion.append(dict(n_Li=n,H0_bin=old_bin,current_UMA_bin=new_bin,count=matches,
                    original_bin_seeds=count,row_fraction=matches/count if count else ""))
    absolute=np.asarray([r["absolute_difference_meV_atom"] for r in records])
    for filename,content in (("seed_label_parity.csv",records),("seed_label_parity_summary.csv",summaries),("seed_bin_confusion.csv",confusion)):
        write_csv(out/filename,content)
    return dict(unique_actual_seeds=len(records),all_available_heldout_seed_count=len(bank["occupancy"]),
        maximum_absolute_difference_meV_atom=float(absolute.max()),median_absolute_difference_meV_atom=float(np.median(absolute)),
        p95_absolute_difference_meV_atom=float(np.quantile(absolute,.95)),by_composition=summaries,
        classification_changed_seeds=sum(r["H0_bin"]!=r["current_UMA_bin"] for r in records),
        original_H0_binding=binding,no_reselection=True,no_arbitrary_pass_tolerance=True,
        interpretation="Diagnostic parity check only. Report signed errors and their scale relative to within-composition bin width; differences do not authorize relabeling/reselecting the frozen seed pool.")


if __name__=="__main__":raise SystemExit(main())

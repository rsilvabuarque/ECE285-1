"""Posthoc low-request editor performance and stage-specific query accounting.

The required core, editor proposal pool, and all outcomes remain unchanged.
Running minima use novel children of LOW requests only. Three fitted editors
and their matched mutation streams define descriptive replicates, not thousands
of independent fits. Unique-child-query curves exclude shared seed verification.
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from cathode_cvae.data import sha256
from cathode_cvae.pool_scoring import certify_cache
from cathode_cvae.runtime import require_compute_allocation


def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def truth(value):
    if value in [True,"True","true",1,"1"]:return True
    if value in [False,"False","false",0,"0",""]:return False
    raise ValueError("Unexpected saved boolean")


def low_curves(rows,available_before):
    """Retain attempt order; no free success from copied, known or failed masks."""
    unique_children=set();fresh_children=set();best=np.nan;output=[]
    novel_seen=set();low_seen=set()
    for attempt,row in enumerate(rows,1):
        known=truth(row["energy_known"])
        if known:
            sid=row["structure_id"];unique_children.add(sid)
            if sid not in available_before:fresh_children.add(sid)
            if truth(row["novel_both"]):
                e=float(row["child_energy_eV"])/(144+int(row["n_target"]))
                best=e if not np.isfinite(best) else min(best,e)
                novel_seen.add(row["canonical_mask"])
                if truth(row["passes_original_q20"]):low_seen.add(row["canonical_mask"])
        output.append(dict(attempt=attempt,unique_child_queries=len(unique_children),
            standalone_fresh_child_queries=len(fresh_children),best_novel_child_energy_eV_atom=float(best) if np.isfinite(best) else "",
            distinct_novel_children_seen=len(novel_seen),distinct_novel_original_q20_children_seen=len(low_seen)))
    return output


def step_means(group_curves,axis):
    """Mean only on a common integer budget with all three finite minima.

    At duplicate-query plateaus retain the last attempted result. No linear
    interpolation or imputation for a replicate without a novel observation.
    """
    maximum=min(curve[-1][axis] for curve in group_curves)
    output=[]
    for budget in range(1,maximum+1):
        values=[]
        for curve in group_curves:
            eligible=[r for r in curve if r[axis]<=budget]
            if not eligible:break
            value=eligible[-1]["best_novel_child_energy_eV_atom"]
            if value=="":break
            values.append(float(value))
        output.append(dict(axis=axis,budget=budget,replicates_with_finite_minimum=len(values),
            mean_best_novel_child_energy_eV_atom=float(np.mean(values)) if len(values)==len(group_curves) else ""))
    return output


def terminal_minima(rows):
    """Best novel LOW child versus its matched budget and common known minimum."""
    result=[]
    combinations=sorted({(int(r["n_target"]),int(r["fit_seed"])) for r in rows if r["requested_bin"]=="low"})
    for n,seed in combinations:
        subsets={method:[r for r in rows if r["requested_bin"]=="low" and int(r["n_target"])==n
                         and int(r["fit_seed"])==seed and r["method"]==method] for method in ["learned_edit","mutation"]}
        left,right=subsets["learned_edit"],subsets["mutation"]
        if [r["paired_trial_id"] for r in left]!=[r["paired_trial_id"] for r in right]:raise ValueError("Terminal minima use different paired trial budgets")
        if any(l["seed_scoring_id"]!=r["seed_scoring_id"] or float(l["seed_energy_eV"])!=float(r["seed_energy_eV"]) for l,r in zip(left,right)):
            raise ValueError("Terminal comparison does not share exact current-SCF seeds")
        atoms=144+n
        common_seed=min((float(r["seed_energy_eV"])/atoms,r["seed_scoring_id"]) for r in left)
        record=dict(n_Li=n,fit_seed=seed,attempts_per_method=len(left),unique_common_seed_orbits=len({r["seed_scoring_id"] for r in left}),
            best_common_current_seed_energy_eV_atom=common_seed[0],best_common_current_seed_id=common_seed[1])
        values={}
        for method,rr in subsets.items():
            available=[(float(r["child_energy_eV"])/atoms,r["structure_id"]) for r in rr if truth(r["energy_known"]) and truth(r["novel_both"])]
            if not available:raise ValueError("No novel measured child for a declared terminal minimum")
            best=min(available);values[method]=best[0]
            record.update({f"{method}_best_novel_child_energy_eV_atom":best[0],f"{method}_best_novel_child_id":best[1],
                f"{method}_best_child_minus_best_common_seed_meV_atom":1000*(best[0]-common_seed[0])})
        record["learned_minus_mutation_best_novel_child_meV_atom"]=1000*(values["learned_edit"]-values["mutation"])
        result.append(record)
    means=[]
    fields=[k for k in result[0] if "energy_eV_atom" in k or "meV_atom" in k]
    for n in sorted({r["n_Li"] for r in result}):
        rr=[r for r in result if r["n_Li"]==n]
        means.append(dict(n_Li=n,fit_replicates=len(rr),
            **{f"mean_{field}":float(np.mean([r[field] for r in rr])) for field in fields},
            fits_with_learned_lower_novel_minimum=sum(r["learned_minus_mutation_best_novel_child_meV_atom"]<0 for r in rr),
            fits_with_learned_novel_child_below_best_known_seed=sum(r["learned_edit_best_child_minus_best_common_seed_meV_atom"]<0 for r in rr),
            fits_with_mutation_novel_child_below_best_known_seed=sum(r["mutation_best_child_minus_best_common_seed_meV_atom"]<0 for r in rr)))
    return result,means


def cost_audit(generated,oracle,before_path,rows,output):
    before=json.loads(before_path.read_text())
    current_bytes=(oracle/"registry.json").read_bytes();current=json.loads(current_bytes)
    if before["parent_sha256"]!=current["parent_sha256"]:raise ValueError("Oracle parent changed")
    certificate=certify_cache(oracle,"uma",before["parent_sha256"])
    with np.load(oracle/"cache_uma.npz",allow_pickle=False) as z:cached=set(z["structure_ids"].astype(str))
    pre_registered={entry["structure_id"] for entry in before["physical_keys"].values() if entry["model"]=="uma"}
    available=cached|pre_registered
    pool=generated/"candidates.npz";pool_sha=sha256(pool)
    with np.load(pool,allow_pickle=False) as z:
        pool_ids=set(z["structure_ids"].astype(str));roles=dict(zip(z["structure_ids"].astype(str),z["scoring_role"].astype(str)))
    stage_entries={entry["structure_id"]:entry for key,entry in current["physical_keys"].items()
        if entry["model"]=="uma" and key not in before["physical_keys"] and entry["input_sha256"]==pool_sha}
    all_current={entry["structure_id"] for entry in current["physical_keys"].values() if entry["model"]=="uma"}
    if not pool_ids<=all_current:raise ValueError("Some declared editor geometries were never registered")
    fresh={sid for sid,entry in stage_entries.items() if entry["provenance"]=="new_scoring"}
    cache_added={sid for sid,entry in stage_entries.items() if entry["provenance"]=="exact_geometry_cache"}
    if fresh != pool_ids-available:raise ValueError("Stage attribution differs from the pre-editor available exact labels")
    completed=set();task_receipts={}
    for task in current["tasks"]:
        if task["model"]!="uma" or task["input_sha256"]!=pool_sha:continue
        source=oracle/"scores/uma"/(task["task_id"]+".json")
        record=json.loads(source.read_text())
        if record["status"]!="complete" or record["task"]!=task:raise ValueError("An attributed editor task is not complete")
        scored=source.with_suffix(".npz")
        if sha256(scored)!=record["output_sha256"]:raise ValueError("Attributed editor score bytes changed")
        if set(task["expected_structure_ids"]) & completed:raise ValueError("Editor physical query repeated across tasks")
        completed.update(task["expected_structure_ids"]);task_receipts[str(source)]=sha256(source)
    if completed!=fresh:raise ValueError("Stage-specific new labels differ from executed task identities")
    seed_ids={sid for sid,role in roles.items() if "seed" in role.split("|")}
    child_ids={sid for sid,role in roles.items() if "child" in role.split("|")}
    cost_rows=[]
    for scope,rr in [("all_requested_bins",rows),("low_requests_only",[r for r in rows if r["requested_bin"]=="low"])]:
        actual_seed_ids={r["seed_scoring_id"] for r in rr}
        actual_child_ids={r["structure_id"] for r in rr if truth(r["energy_known"])}
        scopes=[("shared_seed_verification",actual_seed_ids)]
        scopes += [(method,{r["structure_id"] for r in rr if r["method"]==method and truth(r["energy_known"])}) for method in ["learned_edit","mutation"]]
        scopes += [("union_children",actual_child_ids),("union_seeds_and_children",actual_seed_ids|actual_child_ids)]
        for name,ids in scopes:
            cost_rows.append(dict(scope=scope,role_or_method=name,unique_geometry_queries=len(ids),
                pre_editor_registry_hits=len(ids&pre_registered),pre_editor_certified_cache_hits=len(ids&cached),
                union_preexisting_exact_labels=len(ids&available),standalone_fresh_queries_after_required=len(ids-available),
                actual_editor_stage_fresh_ids_encountered=len(ids&fresh),
                caution="Method rows share labels; do not sum them. Seed verification is a shared overhead, not charged twice."))
    write_csv(output/"query_accounting.csv",cost_rows)
    record=dict(status="PASS",UMA_query_pool_geometries=len(pool_ids),fresh_completed_editor_EFS_labels=len(fresh),
        already_available_geometry_queries=len(pool_ids&available),newly_registered_exact_cache_hits=len(cache_added),
        fresh_child_only_labels=len(fresh&child_ids-seed_ids),fresh_seed_only_labels=len(fresh&seed_ids-child_ids),
        fresh_seed_child_overlap_labels=len(fresh&seed_ids&child_ids),shared_seed_geometry_queries=len(seed_ids),
        task_count=len(task_receipts),task_receipts=task_receipts,
        pre_editor_registry_sha256=sha256(before_path),registry_sha256_at_audit=hashlib.sha256(current_bytes).hexdigest(),
        certified_cache_sha256=certificate["npz_sha256"],certified_cache_certificate_sha256=sha256(oracle/"cache_uma.json"),
        pool_sha256=pool_sha,measure="Completed logical exact-geometry UMA EFS labels, not uninstrumented neural forward passes",
        distinction="Standalone unique CHILD queries reset per method/fit/composition; actual global new calls are deduplicated once and verified against pre-editor registry and immutable certified cache. Shared seed verification is separate.")
    write_json(output/"cost_audit.json",record)
    return available,record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument("--output-dir",type=Path);p.add_argument("--evaluated",type=Path);p.add_argument("--generated",type=Path)
    a=p.parse_args();require_compute_allocation();root=a.root.resolve();run=root/"runs/conditioning_20261107"
    evaluated=a.evaluated or run/"evaluation/editor";generated=a.generated or run/"editor/assessment_v1"
    out=a.output_dir or root/"reports/conditioning_20261107/editor_low_request_v2"
    source=evaluated/"attempt_outcomes.csv";rows=read_csv(source)
    original_receipt=json.loads((evaluated/"receipt.json").read_text())
    if original_receipt["status"]!="PASS" or len(rows)!=6912:raise ValueError("Complete optional editor outcomes are required")
    if original_receipt["outputs"]["attempt_outcomes.csv"]!=sha256(source):raise ValueError("Audited editor outcomes changed")
    out.mkdir(parents=True,exist_ok=False)
    available,cost=cost_audit(generated,run/"oracle",run/"oracle_snapshots/required/registry.json",rows,out)
    curve_rows=[];means=[];groups={}
    for method in ["learned_edit","mutation"]:
        for n in [12,24,36]:
            method_curves=[]
            for seed in [20261106,20261107,20261108]:
                rr=[r for r in rows if r["method"]==method and int(r["n_target"])==n and int(r["fit_seed"])==seed and r["requested_bin"]=="low"]
                if len(rr)!=128:raise ValueError("Low request budget differs")
                curve=low_curves(rr,available);groups[method,n,seed]=curve;method_curves.append(curve)
                curve_rows.extend(dict(method=method,n_Li=n,fit_seed=seed,**row) for row in curve)
            for axis in ["attempt","unique_child_queries","standalone_fresh_child_queries"]:
                means.extend(dict(method=method,n_Li=n,**row) for row in step_means(method_curves,axis))
    pairs=defaultdict(dict)
    for row in rows:pairs[row["paired_trial_id"]][row["method"]]=row
    paired_rows=[]
    for pair,methods in pairs.items():
        if set(methods)!={"learned_edit","mutation"}:raise ValueError("Unpaired outcomes")
        left,right=methods["learned_edit"],methods["mutation"]
        if left["seed_scoring_id"]!=right["seed_scoring_id"] or left["swap_count"]!=right["swap_count"]:raise ValueError("Starting information changed")
        if not truth(left["energy_known"]) or not truth(right["energy_known"]):raise ValueError("Per-request energy summaries need both declared measured outcomes")
        n=int(left["n_target"]);gap=1000*(float(left["child_energy_eV"])-float(right["child_energy_eV"]))/ (144+n)
        paired_rows.append(dict(paired_trial_id=pair,fit_seed=int(left["fit_seed"]),n_Li=n,requested_bin=left["requested_bin"],
            swap_count=int(left["swap_count"]),learned_minus_mutation_meV_atom=gap,learned_energy_lower=gap<0,
            equal_energy_exact_float=gap==0,learned_child_minus_seed_meV_atom=float(left["delta_energy_meV_atom"]),
            mutation_child_minus_seed_meV_atom=float(right["delta_energy_meV_atom"])))
    summaries=[]
    for b in ["low","middle","high"]:
        for n in [12,24,36,"macro_counts"]:
            for seed in [20261106,20261107,20261108,"macro_fits"]:
                rr=[r for r in paired_rows if r["requested_bin"]==b and (n=="macro_counts" or r["n_Li"]==n) and (seed=="macro_fits" or r["fit_seed"]==seed)]
                gaps=np.array([r["learned_minus_mutation_meV_atom"] for r in rr])
                summaries.append(dict(requested_bin=b,n_Li=n,fit_seed=seed,paired_trials=len(rr),
                    mean_learned_minus_mutation_meV_atom=float(gaps.mean()),median_learned_minus_mutation_meV_atom=float(np.median(gaps)),
                    fraction_learned_lower=float(np.mean(gaps<0)),fraction_exact_ties=float(np.mean(gaps==0)),
                    mean_learned_child_minus_seed_meV_atom=float(np.mean([r["learned_child_minus_seed_meV_atom"] for r in rr])),
                    mean_mutation_child_minus_seed_meV_atom=float(np.mean([r["mutation_child_minus_seed_meV_atom"] for r in rr])),
                    interpretation="Energy lowering is relevant for LOW; middle/high requests are targeting tasks, not unconstrained minimization"))
    write_csv(out/"low_request_curves.csv",curve_rows);write_csv(out/"low_request_curve_means.csv",means)
    write_csv(out/"paired_request_energy.csv",paired_rows);write_csv(out/"request_energy_summary.csv",summaries)
    minima,minima_means=terminal_minima(rows)
    if len(minima)!=9:raise ValueError("Expected all9 fit-by-composition terminal comparisons")
    write_csv(out/"terminal_best_novel_low_children.csv",minima);write_csv(out/"terminal_minima_composition_means.csv",minima_means)
    plot(out,groups,means,paired_rows)
    record=dict(status="PASS",audited_UTC=datetime.now(timezone.utc).isoformat(),attempts=6912,low_request_attempts=2304,
        low_request_paired_trials=1152,fit_replicates=3,new_MLIP_calls=0,new_training_updates=0,selection_changed=False,
        running_best_scope="Novel to H0TRAIN and frozen editor history; LOW requests only; no energy-bin requirement for running minimum",
        curve_cost_scope=cost["distinction"],means="Stepwise common integer budget; mean only where all3replicates have finite minima; faint individual fits retained",
        energy_units="Raw best energy eV/atom; paired changes meV/atom; neither establishes a physical ground state",
        terminal_minima_scope="All9fit-by-count LOW comparisons at128attempts; novel child minima compared with each other and the SAME best current-SCF seed from that paired seed set; secondary posthoc observable",
        source_sha256=sha256(__file__),sources={str(path.relative_to(root)):sha256(path) for path in [source,evaluated/"receipt.json",generated/"protocol.json",generated/"receipt.json",generated/"candidates.npz"]},
        outputs={path.name:sha256(path) for path in out.iterdir() if path.is_file()},
        low_summary=next(r for r in summaries if r["requested_bin"]=="low" and r["n_Li"]=="macro_counts" and r["fit_seed"]=="macro_fits"))
    write_json(out/"receipt.json",record);print(json.dumps(dict(status="PASS",cost=cost,low=record["low_summary"]),indent=2))


def plot(out,groups,means,paired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    colors={"learned_edit":"#5063b2","mutation":"#cf812e"}
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for row,axis in enumerate(["attempt","unique_child_queries"]):
        for ax,n in zip(axes[row],[12,24,36]):
            for method in ["learned_edit","mutation"]:
                for seed in [20261106,20261107,20261108]:
                    curve=groups[method,n,seed]
                    x=[r[axis] for r in curve];y=[float(r["best_novel_child_energy_eV_atom"]) if r["best_novel_child_energy_eV_atom"]!="" else np.nan for r in curve]
                    ax.step(x,y,where="post",color=colors[method],alpha=.25,lw=1)
                mm=[r for r in means if r["method"]==method and r["n_Li"]==n and r["axis"]==axis]
                ax.step([r["budget"] for r in mm],[r["mean_best_novel_child_energy_eV_atom"] if r["mean_best_novel_child_energy_eV_atom"]!="" else np.nan for r in mm],
                    where="post",color=colors[method],lw=2.1,label=method.replace("_"," "))
            ax.set(title=f"LOW request, nLi={n}",xlabel="Attempts" if axis=="attempt" else "Standalone unique CHILD queries",ylabel="Best novel child UMA energy (eV/atom)")
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    axes[0,1].legend(fontsize=8);fig.suptitle("Running minima on LOW requests only; faint =3fits, bold =equal-fit step mean\nUnique-query axis excludes shared seed verification; it is not global fresh-call accounting",fontsize=11)
    fig.savefig(out/"low_request_best_energy.pdf");fig.savefig(out/"low_request_best_energy.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    for ax,n in zip(axes,[12,24,36]):
        for seed in [20261106,20261107,20261108]:
            gaps=np.sort([r["learned_minus_mutation_meV_atom"] for r in paired if r["requested_bin"]=="low" and r["n_Li"]==n and r["fit_seed"]==seed])
            ax.step(gaps,np.arange(1,len(gaps)+1)/len(gaps),where="post",alpha=.8,label=str(seed))
        ax.axvline(0,color="black",ls="--",lw=.8);ax.set(title=f"Paired LOW children, nLi={n}",xlabel="Learned minus mutation energy (meV/atom)",ylabel="Cumulative paired-trial fraction")
    axes[1].legend(title="Editor fit / paired stream",fontsize=7);fig.suptitle("Negative values favor the learned child for the SAME seed and swap budget",fontsize=11)
    fig.savefig(out/"paired_low_energy.pdf");fig.savefig(out/"paired_low_energy.png",dpi=150);plt.close(fig)


if __name__=="__main__":main()

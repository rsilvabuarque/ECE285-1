"""All-attempt scalar evaluation and paired condition/latent diagnostics.

No model is fit or evaluated here. Teachers are supplied score/schema pairs;
their relative-energy classes are never pooled across teachers/compositions.
Three fit/replicate values remain the uncertainty units, not proposal rows.
"""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from cathode_cvae.conditions import classify,load_schema
from cathode_cvae.data import sha256
from cathode_cvae.learning.generate_mask import canonical_masks
from cathode_cvae.runtime import require_compute_allocation


def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    if not rows:Path(path).write_text("");return
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def mappings(items):
    result={}
    for item in items:
        name,path=item.split("=",1)
        if name in result:raise ValueError("Repeated teacher name")
        result[name]=Path(path)
    return result
def read_keys(path):
    value=json.loads(Path(path).read_text())
    if isinstance(value,dict):value=next(value[k] for k in ("keys","canonical_masks","known_keys","canonical_keys") if k in value)
    return set(map(str,value))


def successful_score(status,provenance=None):
    """Recognize collector-certified cached energies without changing values.

    Exact geometry/teacher/precision certification belongs to the producing
    collector. Its scalar rows carry both status='cached' and provenance='cached'.
    A bare unknown status must not become a valid measured energy here.
    """
    return status in {"success","complete","converged","ok"} or (status=="cached" and provenance=="cached")


def paired_tables(outcomes,raw):
    grouped=defaultdict(dict)
    for row in outcomes:
        key=(row["teacher"],row["method"],row["replicate_id"],row["n_target"],row["paired_id"])
        label=row["requested_bin"] if row["diagnostic"]=="primary" else row["diagnostic"]
        if label in grouped[key]:raise ValueError("Repeated paired trial label")
        grouped[key][label]=row
    pairs=[]
    for key,group in grouped.items():
        for right,label in [("middle","condition_middle_minus_low"),("high","condition_high_minus_low"),("latent_B","alternate_latent_minus_low")]:
            if "low" not in group or right not in group:continue
            a,b=group["low"],group[right];i,j=int(a["raw_row"]),int(b["raw_row"])
            if not np.array_equal(raw["uniforms"][i],raw["uniforms"][j]):raise ValueError("Paired AR uniforms differ")
            same_z=np.array_equal(raw["latents"][i],raw["latents"][j])
            if right!="latent_B" and not same_z:raise ValueError("Condition pair changes actual latent")
            known=a["energy_known"] and b["energy_known"]
            pairs.append(dict(teacher=key[0],method=key[1],replicate_id=key[2],n_Li=int(key[3]),paired_id=key[4],contrast=label,
                both_energies_known=known,energy_shift_meV_atom=1000*(b["energy_eV_atom"]-a["energy_eV_atom"]) if known else "",
                raw_Hamming=int(np.count_nonzero(raw["occupancy"][i]!=raw["occupancy"][j])) if a["status"]==b["status"]=="generated" else "",
                same_orbit=a["canonical_mask"]==b["canonical_mask"] if a["canonical_mask"] and b["canonical_mask"] else "",
                actual_z_equal=bool(same_z),AR_uniforms_equal=True,
                random_scope="Shared actual latent and AR uniforms" if right!="latent_B" else "Shared AR uniforms; alternate actual latent"))
    groups=defaultdict(list)
    for r in pairs:groups[r["teacher"],r["method"],r["replicate_id"],r["n_Li"],r["contrast"]].append(r)
    summary=[]
    for (teacher,method,rep,n,contrast),rr in sorted(groups.items()):
        known=[r for r in rr if r["both_energies_known"]]
        summary.append(dict(teacher=teacher,method=method,replicate_id=rep,n_Li=n,contrast=contrast,paired_attempts=len(rr),known_pairs=len(known),
            mean_energy_shift_meV_atom=float(np.mean([r["energy_shift_meV_atom"] for r in known])) if known else "",
            mean_raw_Hamming=float(np.mean([r["raw_Hamming"] for r in known])) if known else "",
            same_orbit_fraction=float(np.mean([r["same_orbit"] for r in known])) if known else ""))
    return pairs,summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("generated","parent","history","training-data","output-dir"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--score",action="append",required=True,help="TEACHER=scalarNPZ")
    p.add_argument("--schema",action="append",required=True,help="TEACHER=frozenJSON")
    p.add_argument("--fixed-thresholds",type=Path,help="Explicit fixed reference q20 CSV; no relaxed/deletion threshold substitution")
    p.add_argument("--fixed-threshold-source",default="main_train");p.add_argument("--plots",action="store_true")
    a=p.parse_args(argv);require_compute_allocation()
    scores=mappings(a.score);schemas=mappings(a.schema)
    if set(scores)!=set(schemas):raise ValueError("Each teacher needs its own score/schema pair")
    gen=a.generated;receipt=json.loads((gen/"receipt.json").read_text());protocol=json.loads((gen/"protocol.json").read_text())
    if receipt["status"]!="FROZEN_BEFORE_SCORING" or receipt["protocol_sha256"]!=sha256(gen/"protocol.json"):raise ValueError("Unfrozen or modified proposal protocol")
    for name,expected in receipt["outputs"].items():
        if sha256(gen/name)!=expected:raise ValueError("Generated evidence changed")
    for key,path in [("parent",a.parent),("history",a.history),("data",a.training_data)]:
        if sha256(path)!=protocol["inputs"][key]["sha256"]:raise ValueError("Evaluation input differs from proposal freeze: "+key)
    rows=read_csv(gen/"attempts_all.csv")
    primary=[r for r in rows if r["diagnostic"]=="primary"]
    if len(primary)!=protocol["expected_primary_attempts"]:raise ValueError("Primary denominator changed")
    with np.load(gen/"raw_attempts.npz",allow_pickle=False) as z:raw={k:z[k] for k in ("occupancy","latents","uniforms")}
    if len(raw["occupancy"])!=len(rows):raise ValueError("Raw attempt alignment differs")
    with np.load(a.parent,allow_pickle=False) as z:parent={k:z[k] for k in z.files}
    sites=parent["permutations"].shape[1];host_atoms=len(parent["numbers"])-sites
    valid=np.asarray([r["status"]=="generated" for r in rows])
    keys,_=canonical_masks(raw["occupancy"][valid],parent["permutations"])
    if keys.tolist()!=[r["canonical_mask"] for r in rows if r["status"]=="generated"]:raise ValueError("Raw/canonical identities differ")
    with np.load(a.training_data,allow_pickle=False) as z:train=z["split"].astype(str)=="train";training_occupancy=z["occupancy"][train]
    train_keys,_=canonical_masks(training_occupancy,parent["permutations"]);training=set(train_keys);history=read_keys(a.history)
    thresholds={}
    if a.fixed_thresholds:
        for r in read_csv(a.fixed_thresholds):
            if r.get("source",a.fixed_threshold_source)==a.fixed_threshold_source:thresholds[r["teacher"],int(r["n_Li"])]=float(r["q20_eV_atom"])
    a.output_dir.mkdir(parents=True,exist_ok=False)
    outcomes=[];missing={};terminal_failures={};candidate_ids={r["scoring_structure_id"] for r in rows if r["status"]=="generated"}
    for teacher,path in scores.items():
        with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
        ids=d["structure_ids"].astype(str)
        if len(set(ids))!=len(ids):raise ValueError("Ambiguous scalar identities")
        lookup={sid:i for i,sid in enumerate(ids)};schema=load_schema(schemas[teacher])
        if schema["n_sites"]!=sites:raise ValueError("Measured schema parent dimensions differ")
        declared_parent=schema.get("provenance",{}).get("parent_sha256")
        if declared_parent and declared_parent!=sha256(a.parent):raise ValueError("Measured schema names another parent")
        unscored=candidate_ids-set(lookup);failed=set()
        for sid in candidate_ids&set(lookup):
            i=lookup[sid];status=str(d["status"][i]) if "status" in d else "success"
            provenance=str(d["provenance"][i]) if "provenance" in d else None
            if successful_score(status,provenance):
                if not np.isfinite(d["energy_eV"][i]):unscored.add(sid)
            elif status in {"failed","nonconverged","collision","mapping_failed","time_cap"} or status.startswith("failed"):
                failed.add(sid)
            else:unscored.add(sid)
        missing[teacher]=len(unscored);terminal_failures[teacher]=len(failed)
        for original in rows:
            row=dict(original,teacher=teacher,energy_known=False,energy_eV_atom="",energy_eV="",measured_bin="unknown_energy",
                novel_train=False,novel_history=False,novel_both=False,passes_fixed_q20=False,score_status="")
            if original["status"]!="generated":row["measured_bin"]=original["status"]
            else:
                sid=original["scoring_structure_id"];n=int(original["n_target"]);key=original["canonical_mask"]
                row.update(novel_train=key not in training,novel_history=key not in history,novel_both=key not in training and key not in history)
                if sid in lookup:
                    i=lookup[sid];atoms=int(d["n_atoms"][i]);total=float(d["energy_eV"][i])
                    status=str(d["status"][i]) if "status" in d else "success"
                    provenance=str(d["provenance"][i]) if "provenance" in d else None
                    row["score_status"]=status
                    if atoms!=host_atoms+n:raise ValueError("Score and actual parent atom count disagree")
                    if "n_Li" in d and int(d["n_Li"][i])!=n:raise ValueError("Score target-ion count disagrees")
                    if sid in failed:row["measured_bin"]="score_failed"
                    if successful_score(status,provenance) and np.isfinite(total):
                        e=total/atoms;row.update(energy_known=True,energy_eV=total,energy_eV_atom=e,
                            measured_bin=str(classify(schema,np.asarray(n),np.asarray(e)).item()),
                            passes_fixed_q20=(teacher,n) in thresholds and e<=thresholds[teacher,n])
            outcomes.append(row)
    grouped=defaultdict(list)
    for r in outcomes:
        if r["diagnostic"]=="primary":grouped[r["teacher"],r["method"],r["replicate_id"],int(r["n_target"]),r["requested_bin"]].append(r)
    metrics=[];confusion=[]
    for (teacher,method,rep,n,b),rr in sorted(grouped.items()):
        if len(rr)!=protocol["count_per_group"]:raise ValueError("Conditional-cell denominator changed")
        known=[r for r in rr if r["energy_known"]];useful={r["canonical_mask"] for r in known if r["novel_both"] and r["measured_bin"]==b}
        fixed={r["canonical_mask"] for r in known if r["novel_both"] and r["passes_fixed_q20"]}
        metrics.append(dict(teacher=teacher,method=method,replicate_id=rep,fit_seed=rr[0]["fit_seed"],n_Li=n,requested_bin=b,attempts=len(rr),
            scored_attempts=len(known),distinct_orbits=len({r["canonical_mask"] for r in rr if r["canonical_mask"]}),
            distinct_novel_inbin_orbits=len(useful),novel_inbin_yield_per_attempt=len(useful)/len(rr),
            bin_hit_rate_per_attempt=sum(r["measured_bin"]==b for r in known)/len(rr),
            distinct_novel_fixed_q20_orbits=len(fixed),novel_fixed_q20_yield_per_attempt=len(fixed)/len(rr),
            fixed_q20_available=(teacher,n) in thresholds))
        for actual in sorted(set(["low","middle","high","unknown_energy","failed_request","invalid_mask"])|{r["measured_bin"] for r in rr}):
            confusion.append(dict(teacher=teacher,method=method,replicate_id=rep,n_Li=n,requested_bin=b,measured_bin=actual,count=sum(r["measured_bin"]==actual for r in rr)))
    by_seed=defaultdict(list)
    for r in metrics:by_seed[r["teacher"],r["method"],r["replicate_id"]].append(r)
    seed_metrics=[]
    for (teacher,method,rep),rr in sorted(by_seed.items()):
        low=[r for r in rr if r["requested_bin"]=="low" and r["fixed_q20_available"]]
        seed_metrics.append(dict(teacher=teacher,method=method,replicate_id=rep,fit_seed=rr[0]["fit_seed"],
            macro_composition_bin_yield=float(np.mean([r["novel_inbin_yield_per_attempt"] for r in rr])),
            fixed_q20_LOW_request_yield=float(np.mean([r["novel_fixed_q20_yield_per_attempt"] for r in low])) if low else ""))
    summary=[]
    for teacher,method in sorted({(r["teacher"],r["method"]) for r in seed_metrics}):
        rr=[r for r in seed_metrics if r["teacher"]==teacher and r["method"]==method];values=[r["macro_composition_bin_yield"] for r in rr]
        low=[r["fixed_q20_LOW_request_yield"] for r in rr if r["fixed_q20_LOW_request_yield"]!=""]
        summary.append(dict(teacher=teacher,method=method,fit_or_sampler_replicates=len(rr),mean_novel_inbin_yield=float(np.mean(values)),
            descriptive_seed_SD=float(np.std(values,ddof=1)) if len(rr)>1 else "",mean_fixed_q20_LOW_request_yield=float(np.mean(low)) if low else ""))
    pairs,pair_summary=paired_tables(outcomes,raw)
    for name,content in [("attempt_outcomes.csv",outcomes),("metrics.csv",metrics),("confusion.csv",confusion),
        ("seed_metrics.csv",seed_metrics),("summary.csv",summary),("paired_attempts.csv",pairs),("paired_summary.csv",pair_summary)]:write_csv(a.output_dir/name,content)
    result=dict(status="PASS" if not any(missing.values()) else "INCOMPLETE_COVERAGE",primary_attempts=len(primary),diagnostic_attempts=len(rows)-len(primary),
        missing_or_nonterminal_candidate_labels=missing,terminal_failed_candidate_labels=terminal_failures,teachers=list(scores),source_sha256=sha256(__file__),proposal_protocol_sha256=sha256(gen/"protocol.json"),
        score_sources={t:dict(path=str(path.resolve()),sha256=sha256(path),schema_sha256=sha256(schemas[t])) for t,path in scores.items()},
        fixed_thresholds=None if a.fixed_thresholds is None else dict(path=str(a.fixed_thresholds.resolve()),sha256=sha256(a.fixed_thresholds),source=a.fixed_threshold_source),
        outputs={p.name:sha256(p) for p in a.output_dir.iterdir() if p.is_file()},new_MLIP_calls=0,new_training_updates=0,
        uncertainty="Descriptive fit/sampler replicate spread; proposal rows are not independent trained models",novelty_scope="Frozen history and supplied TRAIN mask orbits")
    result['cached_score_policy']="status=cached is accepted only with provenance=cached; the producing collector certifies exact parent/geometry/teacher/precision before export"
    (a.output_dir/"receipt.json").write_text(json.dumps(result,indent=2)+"\n")
    if a.plots:plot(a.output_dir,metrics)
    print(json.dumps(dict(status=result["status"],primary_attempts=len(primary),summary=summary),indent=2))
    if result["status"]!="PASS":raise SystemExit(1)
    return result


def plot(output,metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    teachers=sorted({r["teacher"] for r in metrics});methods=sorted({r["method"] for r in metrics})
    fig,axes=plt.subplots(len(teachers),1,figsize=(max(7,len(methods)*1.2),3.3*len(teachers)),squeeze=False)
    for ax,teacher in zip(axes[:,0],teachers):
        for offset,b in zip([-.24,0,.24],["low","middle","high"]):
            means=[]
            for method in methods:
                rr=[r for r in metrics if r["teacher"]==teacher and r["method"]==method and r["requested_bin"]==b]
                means.append(100*np.mean([r["novel_inbin_yield_per_attempt"] for r in rr]) if rr else np.nan)
            ax.bar(np.arange(len(methods))+offset,means,width=.22,label=b)
        ax.set(xticks=np.arange(len(methods)),xticklabels=methods,ylabel="Distinct novel in-bin / all attempts (%)",title=teacher);ax.legend()
        ax.tick_params(axis="x",rotation=20)
    fig.tight_layout();fig.savefig(output/"useful_yield_by_request.pdf");fig.savefig(output/"useful_yield_by_request.png",dpi=150);plt.close(fig)


if __name__=="__main__":main()

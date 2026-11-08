"""Evaluate same-seed/same-edit-budget learned editing against uniform mutation.

All attempted children remain denominators. Seed and child energies are both
measured on the same canonical deletion protocol. Raw alignment is used for edit
identity; actual parent symmetries are used separately for physical novelty and
the minimum seed-child occupation distance. No fitting or model selection occurs.
"""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from cathode_cvae.acquisition import canonical_keys
from cathode_cvae.conditions import classify,load_schema
from cathode_cvae.data import sha256
from cathode_cvae.runtime import require_compute_allocation
from generate_edit_comparison import verify_edit,write_csv,write_json


def summarize_cell(rows):
    """Distinct orbit successes divided by all attempts, including failures."""
    count=len(rows)
    if not count:raise ValueError("Empty assessment cell")
    measured=[r for r in rows if r["energy_known"]]
    useful={r["canonical_mask"] for r in measured if r["novel_both"] and r["measured_bin"]==r["requested_bin"]}
    low={r["canonical_mask"] for r in measured if r["novel_both"] and r["passes_original_q20"]}
    delta=[r["delta_energy_meV_atom"] for r in measured]
    return dict(attempts=count,measured_attempts=len(measured),failed_or_unmeasured_attempts=count-len(measured),
        distinct_generated_orbits=len({r["canonical_mask"] for r in rows if r["canonical_mask"]}),
        distinct_novel_requested_orbits=len(useful),novel_inbin_yield=len(useful)/count,
        distinct_novel_original_q20_orbits=len(low),novel_original_q20_yield=len(low)/count,
        bin_hit_rate=sum(r["measured_bin"]==r["requested_bin"] for r in measured)/count,
        distinct_novel_to_H0TRAIN=len({r["canonical_mask"] for r in measured if r["novel_H0TRAIN"]}),
        distinct_novel_to_frozen_history=len({r["canonical_mask"] for r in measured if r["novel_history"]}),
        mean_delta_energy_meV_atom=float(np.mean(delta)) if delta else "",
        fraction_lower_energy_than_seed=sum(d<0 for d in delta)/count,
        mean_raw_seed_child_Hamming=float(np.mean([r["raw_Hamming"] for r in measured])) if measured else "",
        mean_symmetry_min_seed_child_Hamming=float(np.mean([r["symmetry_min_Hamming"] for r in measured])) if measured else "",
        mean_descriptor_change=float(np.mean([r["descriptor_change"] for r in measured])) if measured else "")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument("--generated",type=Path,required=True);p.add_argument("--scores",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args();require_compute_allocation();root=a.root.resolve();gen=a.generated.resolve();out=a.output_dir.resolve()
    protocol=json.loads((gen/"protocol.json").read_text());receipt=json.loads((gen/"receipt.json").read_text())
    if receipt["protocol_sha256"]!=sha256(gen/"protocol.json"):raise ValueError("Frozen editor protocol changed")
    for name,digest in receipt["outputs"].items():
        if sha256(gen/name)!=digest:raise ValueError("Frozen generated attempts changed")
    for source in protocol["sources"].values():
        if sha256(source["path"])!=source["sha256"]:raise ValueError("Frozen editor source changed")
    score_receipt=a.scores.with_name("receipt.json")
    if score_receipt.exists() and json.loads(score_receipt.read_text())["status"]!="PASS":
        raise ValueError("Oracle collector is not complete; preserve partial results and evaluate after explicit completion")
    with (gen/"attempts.csv").open() as f:rows=list(csv.DictReader(f))
    with np.load(gen/"raw_attempts.npz",allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    if len(rows)!=protocol["expected_attempts"] or len(raw["occupancy"])!=len(rows):raise ValueError("Attempt budget changed")
    with np.load(protocol["sources"]["parent"]["path"],allow_pickle=False) as z:permutations=z["permutations"]
    with np.load(a.scores,allow_pickle=False) as z:scores={k:z[k] for k in ["structure_ids","energy_eV","n_atoms","n_Li"]}
    ids=scores["structure_ids"].astype(str)
    if len(ids)!=len(set(ids)):raise ValueError("Ambiguous physical score identities")
    lookup={sid:i for i,sid in enumerate(ids)}
    schema=load_schema(protocol["sources"]["schema"]["path"])
    history=set(json.loads(Path(protocol["sources"]["history"]["path"]).read_text()))
    train_path=root/"runs/conditioning_20261107/data/training_keys.json"
    training=set(json.loads(train_path.read_text()))
    thresholds_path=root/"artifacts/current/reports/generation_thresholds.csv"
    with thresholds_path.open() as f:
        thresholds={int(r["n_Li"]):float(r["q20_eV_atom"]) for r in csv.DictReader(f) if r["teacher"]=="uma" and r["source"]=="main_train"}
    graph_path=root/"reports/conditioning_20261107/parent_graph/periodic_graph.json"
    from cathode_cvae.graph import PeriodicGraph,vacancy_pair_fraction
    graph=PeriodicGraph.from_dict(json.loads(graph_path.read_text()))
    generated=np.array([r["status"]=="generated" for r in rows]);valid=np.flatnonzero(generated)
    k=np.array([int(r["swap_count"]) for r in rows])
    verify_edit(raw["seed_occupancy"][valid],raw["occupancy"][valid],raw["removed"][valid],raw["added"][valid],k[valid])
    seed_keys=canonical_keys(raw["seed_occupancy"],permutations)
    child_keys=canonical_keys(raw["occupancy"][valid],permutations)
    if not np.array_equal(seed_keys,np.array([r["canonical_seed_mask"] for r in rows])):raise ValueError("Raw seed canonical mapping differs")
    if not np.array_equal(child_keys,np.array([rows[i]["canonical_mask"] for i in valid])):raise ValueError("Raw child canonical mapping differs")
    inverse=np.argsort(permutations,axis=1);distance=np.full(len(rows),-1,int)
    for start in range(0,len(valid),128):
        ii=valid[start:start+128]
        distance[ii]=np.min(np.sum(raw["occupancy"][ii,None,:]!=raw["seed_occupancy"][ii][:,inverse],axis=2),axis=1)
    seed_descriptor=vacancy_pair_fraction(raw["seed_occupancy"],graph)
    child_descriptor=np.full(len(rows),np.nan);child_descriptor[valid]=vacancy_pair_fraction(raw["occupancy"][valid],graph)
    paired=defaultdict(dict)
    for i,row in enumerate(rows):
        if int(row["raw_row"])!=i:raise ValueError("Raw row alignment changed")
        pair=row["paired_trial_id"]
        if row["method"] in paired[pair]:raise ValueError("Duplicate method in paired trial")
        paired[pair][row["method"]]=i
        row.update(energy_known=False,novel_H0TRAIN=False,novel_history=False,novel_both=False,passes_original_q20=False,
            measured_bin="failed_or_unmeasured",child_energy_eV="",seed_energy_eV="",delta_energy_eV="",delta_energy_meV_atom="",
            raw_Hamming=int(np.count_nonzero(raw["occupancy"][i]!=raw["seed_occupancy"][i])) if generated[i] else -1,
            symmetry_min_Hamming=int(distance[i]),descriptor_change="",seed_descriptor=float(seed_descriptor[i]),child_descriptor="")
        if generated[i]:
            if row["seed_scoring_id"] not in lookup or row["structure_id"] not in lookup:raise ValueError("Missing declared seed or child SCF; no cache-only assessment")
            si=lookup[row["seed_scoring_id"]];ci=lookup[row["structure_id"]];n=int(row["n_target"]);atoms=144+n
            if any(scores["n_atoms"][j]!=atoms or scores["n_Li"][j]!=n or not np.isfinite(scores["energy_eV"][j]) for j in [si,ci]):
                raise ValueError("Seed/child score composition or finite-energy context differs")
            seed_energy=float(scores["energy_eV"][si]);child_energy=float(scores["energy_eV"][ci]);key=row["canonical_mask"]
            row.update(energy_known=True,novel_H0TRAIN=key not in training,novel_history=key not in history,
                novel_both=key not in training and key not in history,passes_original_q20=child_energy/atoms<=thresholds[n],
                measured_bin=str(classify(schema,np.array([n]),np.array([child_energy/atoms]))[0]),
                seed_energy_eV=seed_energy,child_energy_eV=child_energy,delta_energy_eV=child_energy-seed_energy,
                delta_energy_meV_atom=1000*(child_energy-seed_energy)/atoms,child_descriptor=float(child_descriptor[i]),
                descriptor_change=float(child_descriptor[i]-seed_descriptor[i]))
            actual_remove="|".join(map(str,np.flatnonzero(raw["removed"][i])))
            actual_add="|".join(map(str,np.flatnonzero(raw["added"][i])))
            if actual_remove!=row["removed_sites"] or actual_add!=row["added_sites"]:raise ValueError("Saved edited sites differ from actual changes")
    pair_outcomes=[]
    for pair,methods in paired.items():
        if set(methods)!={"learned_edit","mutation"}:raise ValueError("Unpaired attempted trial")
        i=methods["learned_edit"];j=methods["mutation"];left=rows[i];right=rows[j]
        for field in ["n_target","requested_bin","canonical_seed_mask","seed_structure_id","swap_count","heldout_seed_bank_row","sampling_seed"]:
            if left[field]!=right[field]:raise ValueError("Pair changes starting information or edit budget: "+field)
        for field in ["seed_occupancy","uniforms_remove","uniforms_add"]:
            if not np.array_equal(raw[field][i],raw[field][j]):raise ValueError("Pair does not share actual raw seed/random arrays")
        known=left["energy_known"] and right["energy_known"]
        pair_outcomes.append(dict(paired_trial_id=pair,fit_seed=int(left["fit_seed"]),n_Li=int(left["n_target"]),
            requested_bin=left["requested_bin"],swap_count=int(left["swap_count"]),seed_structure_id=left["seed_structure_id"],
            both_energies_known=known,learned_minus_mutation_meV_atom=(1000*(left["child_energy_eV"]-right["child_energy_eV"])/(144+int(left["n_target"]))) if known else "",
            learned_child_bin=left["measured_bin"],mutation_child_bin=right["measured_bin"]))
    cells=[]
    for method in ["learned_edit","mutation"]:
        for seed in protocol["fit_seeds"]:
            for n in protocol["counts"]:
                for b in protocol["requested_bins"]:
                    rr=[r for r in rows if r["method"]==method and int(r["fit_seed"])==seed and int(r["n_target"])==n and r["requested_bin"]==b]
                    if len(rr)!=protocol["count_per_composition_bin_fit_method"]:raise ValueError("Cell budget differs")
                    cells.append(dict(method=method,fit_seed=seed,n_Li=n,requested_bin=b,**summarize_cell(rr)))
    seed_metrics=[]
    for method in ["learned_edit","mutation"]:
        for seed in protocol["fit_seeds"]:
            cc=[r for r in cells if r["method"]==method and r["fit_seed"]==seed]
            seed_metrics.append(dict(method=method,fit_seed=seed,
                balanced_novel_inbin_yield=float(np.mean([r["novel_inbin_yield"] for r in cc])),
                original_q20_low_request_yield=float(np.mean([r["novel_original_q20_yield"] for r in cc if r["requested_bin"]=="low"])),
                mean_delta_energy_meV_atom=float(np.mean([r["mean_delta_energy_meV_atom"] for r in cc if r["mean_delta_energy_meV_atom"]!=""]))))
    summary=[]
    for method in ["learned_edit","mutation"]:
        rr=[r for r in seed_metrics if r["method"]==method]
        summary.append(dict(method=method,fit_or_paired_sampler_replicates=len(rr),
            balanced_novel_inbin_yield=float(np.mean([r["balanced_novel_inbin_yield"] for r in rr])),
            balanced_descriptive_seed_SD=float(np.std([r["balanced_novel_inbin_yield"] for r in rr],ddof=1)),
            original_q20_low_request_yield=float(np.mean([r["original_q20_low_request_yield"] for r in rr])),
            mean_delta_energy_meV_atom=float(np.mean([r["mean_delta_energy_meV_atom"] for r in rr]))))
    out.mkdir(parents=True,exist_ok=False)
    for name,content in [("attempt_outcomes.csv",rows),("paired_energy_outcomes.csv",pair_outcomes),("metrics.csv",cells),("seed_metrics.csv",seed_metrics),("summary.csv",summary)]:write_csv(out/name,content)
    sources=dict(protocol=gen/"protocol.json",generation_receipt=gen/"receipt.json",attempts=gen/"attempts.csv",raw_attempts=gen/"raw_attempts.npz",
        scores=a.scores,parent=Path(protocol["sources"]["parent"]["path"]),schema=Path(protocol["sources"]["schema"]["path"]),
        training_keys=train_path,original_q20=thresholds_path,parent_graph=graph_path)
    record=dict(status="PASS",scope="Optional same-seed exact-edit comparison, not de novo or DFT accuracy",attempts=len(rows),
        paired_trials=len(paired),fit_replicates=3,failed_attempts=int((~generated).sum()),
        actual_seed_budget_and_uniform_arrays_equal=True,raw_exact_2k_verified_attempts=int(generated.sum()),
        canonical_mapping_operations=len(permutations),seed_and_child_energies_same_protocol=True,
        novelty_boundary_orbits=len(history),descriptor_scope="Posthoc invariant structural change, not a requested extra condition",
        selection_changed=False,new_training_updates=0,new_MLIP_calls=0,physical_cost_scope="Integrating oracle/cost receipt; attempts and global provenance are not new SCF counts",
        source_sha256=sha256(__file__),sources={name:dict(path=str(path.resolve()),sha256=sha256(path)) for name,path in sources.items()},
        outputs={path.name:sha256(path) for path in out.iterdir() if path.is_file()},
        uncertainty="Three independent editor fits; matched mutation has three paired sampler replicates. No proposal-level confidence interval." )
    write_json(out/"receipt.json",record)
    plot(out,rows,seed_metrics)
    print(json.dumps(dict(status="PASS",attempts=len(rows),paired_trials=len(paired),summary=summary),indent=2))


def plot(out,rows,seed_metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors={"learned_edit":"#5063b2","mutation":"#d1862e"}
    fig,axes=plt.subplots(1,2,figsize=(8,3.6),constrained_layout=True)
    for ax,metric,title in zip(axes,["balanced_novel_inbin_yield","original_q20_low_request_yield"],["Balanced requested-bin yield","Original-q20 low-request yield"]):
        for i,method in enumerate(["learned_edit","mutation"]):
            values=np.array([r[metric] for r in seed_metrics if r["method"]==method])*100
            ax.bar(i,values.mean(),color=colors[method],alpha=.6);ax.scatter(i+np.linspace(-.07,.07,len(values)),values,color="black",s=20,zorder=5)
        ax.set(xticks=[0,1],xticklabels=["Learned edit","Mutation"],ylabel="Distinct novel successes / all attempts (%)",title=title,ylim=(0,100))
    fig.suptitle("Same raw seeds, one/two/three swap budgets and random streams\nPoints are fit or paired sampler replicates")
    fig.savefig(out/"edit_yield_comparison.pdf");fig.savefig(out/"edit_yield_comparison.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.6),constrained_layout=True)
    for ax,n in zip(axes,[12,24,36]):
        for method in ["learned_edit","mutation"]:
            for seed in sorted({int(r["fit_seed"]) for r in rows}):
                values=sorted(r["delta_energy_meV_atom"] for r in rows if r["method"]==method and int(r["fit_seed"])==seed and int(r["n_target"])==n and r["energy_known"])
                if values:ax.step(values,np.arange(1,len(values)+1)/len(values),where="post",color=colors[method],alpha=.65,label=method if seed==min(int(r["fit_seed"]) for r in rows) else None)
        ax.axvline(0,color="black",lw=.7,ls="--");ax.set(title=f"nLi={n}",xlabel="Child minus its actual seed energy (meV/atom)",ylabel="Cumulative attempt fraction")
    axes[1].legend(fontsize=8);fig.suptitle("Per-seed distributions; lower than zero improves on that seed under UMA")
    fig.savefig(out/"edit_energy_changes.pdf");fig.savefig(out/"edit_energy_changes.png",dpi=150);plt.close(fig)


if __name__=="__main__":main()

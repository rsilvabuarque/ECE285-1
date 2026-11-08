"""Audit the confirmed DEVELOPMENT architecture comparison before final freeze.

This rebuilds proposal-level paired diagnostics and requested-bin yields. The
three fitted seeds, rather than the thousands of proposals, are retained as the
replication unit. No final pool or final labels are opened.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

import numpy as np

from cathode_cvae.runtime import require_compute_allocation


PROJECT = Path(__file__).resolve().parents[2]
source = Path(__file__).with_name("analyze_diagnostics.py")
spec = importlib.util.spec_from_file_location("paired_analysis", source)
diagnostics = importlib.util.module_from_spec(spec); spec.loader.exec_module(diagnostics)


def mean(values):
    return float(np.mean(values))


def boolean(value):
    if value not in {"True", "False"}:raise ValueError("Expected explicit saved boolean")
    return value == "True"


def independently_measure(rows, thresholds):
    cells = defaultdict(list)
    for r in rows:
        if r["diagnostic"] == "primary":
            cells[(r["method"],r["replicate_id"],int(r["n_Li"]),r["requested_bin"])].append(r)
    if len(cells) != 6*3*3*3:raise ValueError("Expected six families, three seeds, three counts and three bins")
    seed_cells = defaultdict(list)
    for (method,seed,n,requested), group in sorted(cells.items()):
        if len(group) != 64 or not all(boolean(r["energy_known"]) and boolean(r["mask_valid"]) for r in group):
            raise ValueError("The confirmed development grid is incomplete")
        fresh = [r for r in group if boolean(r["novel_training"]) and boolean(r["novel_history"])]
        hits = {r["canonical_mask"] for r in fresh if r["measured_bin"] == requested}
        low = {r["canonical_mask"] for r in fresh if float(r["energy_eV_atom"]) <= thresholds[n]}
        seed_cells[(method,seed)].append(dict(n_target=n,requested_bin=requested,
            balanced_cell_yield=len(hits)/64,original_q20_yield=len(low)/64))
    seeds = []
    for (method,seed), cells in sorted(seed_cells.items()):
        if len(cells) != 9:raise ValueError("An independent design seed lacks a composition/condition cell")
        seeds.append(dict(method=method,replicate_id=seed,
            balanced_useful_yield=mean([r["balanced_cell_yield"] for r in cells]),
            low_original_q20_yield=mean([r["original_q20_yield"] for r in cells if r["requested_bin"] == "low"])))
    methods = []
    for method in sorted({r["method"] for r in seeds}):
        rr = [r for r in seeds if r["method"] == method]
        if len(rr) != 3:raise ValueError("Every retained family must have three design seeds")
        methods.append(dict(method=method,design_seeds=3,
            balanced_useful_yield=mean([r["balanced_useful_yield"] for r in rr]),
            low_original_q20_yield=mean([r["low_original_q20_yield"] for r in rr]),
            seed_values=rr))
    return methods,seeds


def paired_family_summary(directory):
    records = []
    for table, name in [("condition_summary","condition"),("latent_summary","latent"),
                        ("posterior_prior_summary","posterior_prior")]:
        rows = diagnostics.read_csv(directory/(table+".csv"))
        fields = ["mean_delta_E_eV_atom","mean_absolute_delta_E_eV_atom","fraction_positive_delta",
            "mean_raw_Hamming","same_orbit_fraction","posterior_mean_target_Hamming","prior_mean_target_Hamming"]
        seed_groups = defaultdict(list)
        for r in rows:seed_groups[(r["method"],r["comparison"],r["replicate_id"])].append(r)
        for (method,comparison,seed), group in sorted(seed_groups.items()):
            record = dict(method=method,diagnostic=name,comparison=comparison,replicate_id=seed,
                composition_groups=len(group),pairs=sum(int(r["pairs"]) for r in group))
            for f in fields:
                values=[float(r[f]) for r in group if r.get(f,"") != ""]
                if values:record[f]=mean(values)
            records.append(record)
    return records


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=PROJECT)
    p.add_argument("--output-dir",type=Path)
    a=p.parse_args(argv);require_compute_allocation()
    root=a.root.resolve();run=root/"runs/conditioning_20261107"
    output=a.output_dir or root/"reports/conditioning_20261107/spatial_confirmed_diagnostics"
    selection=root/"artifacts/conditioning_20261107/architecture_selection.json"
    input_manifest=output.with_name(output.name+"_manifest.json")
    if output.exists() or selection.exists() or input_manifest.exists():
        raise FileExistsError("Preserve existing architecture diagnostics and selection")
    evaluation=run/"evaluation/spatial_confirmed"
    receipt=json.loads((evaluation/"receipt.json").read_text())
    if receipt["status"] != "PASS" or receipt["attempt_rows"] != 10368:
        raise ValueError("Confirmed development oracle analysis has not passed")
    manifests=[run/"spatial_generation_manifest.json",run/"graph_context_confirmation_manifest.json"]
    models=[]
    for path in manifests:
        for r in json.loads(path.read_text())["models"]:
            r=dict(r)
            if r["method"] == "graph_context_pilot":r["method"]="graph_context"
            models.append(r)
    if len(models) != 18 or len({(r["method"],str(r["replicate_id"])) for r in models}) != 18:
        raise ValueError("Confirmed manifest must contain six families with three seeds")
    input_manifest.parent.mkdir(parents=True,exist_ok=True)
    with input_manifest.open("x") as f:json.dump(dict(models=models,
        source_manifests={str(p.relative_to(root)):diagnostics.sha256(p) for p in manifests},
        alias={"graph_context_pilot":"graph_context"},scope="Confirmed DEVELOPMENT only"),f,indent=2)
    paths=dict(attempts_all=run/"development/spatial_confirmed/attempts_all.csv",
        raw_path=run/"development/spatial_confirmed/raw_attempts.npz",
        scores_uma=run/"collected/spatial_confirmed/uma.npz")
    result=diagnostics.analyze(**paths,output=output,
        schema_path=root/"artifacts/overnight_20261107/data/condition_schema_uma.json",
        training_data=root/"artifacts/current/improvement_v1/training_round0.npz",
        history_path=run/"data/history_before_development.json",fits_root=run/"fits",manifest=input_manifest,
        original_thresholds=root/"artifacts/current/reports/generation_thresholds.csv")
    threshold_path=root/"artifacts/current/reports/generation_thresholds.csv"
    thresholds={int(r["n_Li"]):float(r["q20_eV_atom"]) for r in diagnostics.read_csv(threshold_path)
                if r["teacher"] == "uma" and r["source"] == "main_train"}
    methods,seeds=independently_measure(diagnostics.read_csv(output/"attempt_outcomes.csv"),thresholds)
    archived_balanced={r["method"]:float(r["mean_novel_inbin_yield"]) for r in diagnostics.read_csv(evaluation/"summary.csv")}
    archived_low={r["method"]:float(r["mean_novel_fixed_q20_yield"]) for r in diagnostics.read_csv(evaluation/"fixed_q20_low_request.csv")}
    checks=[]
    for r in methods:
        for key, reference in [("balanced_useful_yield",archived_balanced),("low_original_q20_yield",archived_low)]:
            difference=abs(r[key]-reference[r["method"]])
            # This compares two arithmetic evaluations of the same dimensionless
            # rational counts; epsilon is a floating-point check, not a new gate.
            if difference>4*np.finfo(float).eps:raise ValueError("Independent DEV metric differs from archived evidence")
            checks.append(dict(method=r["method"],metric=key,value=r[key],archived=reference[r["method"]],absolute_difference=difference))
    ranking=sorted(methods,key=lambda r:(r["balanced_useful_yield"],r["low_original_q20_yield"]),reverse=True)
    if ranking[0]["balanced_useful_yield"] == ranking[1]["balanced_useful_yield"]:
        raise ValueError("Exact macro tie needs the predeclared low/simplicity resolution reviewed explicitly")
    if ranking[0]["method"] != "graph_context":raise ValueError("Observed DEV family does not reproduce the proposed recommendation")
    diagnostics.write_csv(output/"paired_diagnostics_by_family_seed.csv",paired_family_summary(output))
    diagnostics.write_csv(output/"independent_yields_by_seed.csv",seeds)
    rule_path=root/"artifacts/conditioning_20261107/selection_rule.json"
    sources=[*manifests,input_manifest,evaluation/"receipt.json",evaluation/"summary.csv",
        evaluation/"fixed_q20_low_request.csv",output/"receipt.json",output/"attempt_outcomes.csv",
        output/"distinct_metric_scopes.csv",run/"data/history_before_development.json",rule_path,
        root/"artifacts/conditioning_20261107/protocol_amendment_graph_context_replication.json"]
    selected=[r for r in models if r["method"] == "graph_context"]
    checkpoint_records=[]
    for r in selected:
        checkpoint=Path(r["checkpoint"]);folder=checkpoint.parent
        summary=json.loads((folder/"summary.json").read_text())
        cfg=json.loads((folder/"configuration.json").read_text())
        if summary["optimizer_updates"] != 4000 or summary.get("heldout_test_evaluated") is not False:
            raise ValueError("Incomplete selected architecture trajectory or test-boundary violation")
        declared=summary["selection_checkpoints"]["0.1"]
        if checkpoint.name != declared["filename"] or diagnostics.sha256(checkpoint) != declared["sha256"]:
            raise ValueError("Selected checkpoint differs from fit receipt")
        if not cfg.get("graph_context") or cfg.get("encoder_kind") != "graph" or cfg["latent_dim"] != 4:
            raise ValueError("Architecture recommendation is not the confirmed graph-context D4 model")
        checkpoint_records.append(dict(method=r["method"],fit_seed=r["fit_seed"],checkpoint=str(checkpoint),
            checkpoint_sha256=declared["sha256"],configuration_sha256=diagnostics.sha256(folder/"configuration.json"),
            summary_sha256=diagnostics.sha256(folder/"summary.json"),optimizer_updates=4000,
            checkpoint_selection_beta=.1))
    record=dict(status="PASS_DEVELOPMENT_RECOMMENDATION_BEFORE_FINAL",created_UTC=datetime.now(timezone.utc).isoformat(),
        recommendation_family="graph_context",selected_models=checkpoint_records,
        rule=json.loads(rule_path.read_text())["policy"],
        interpretation="Apply the frozen macro-development requested-bin yield rule to architecture families; graph_context is the unique maximum. Original-q20 low yield and simplicity tie-breaks were not needed.",
        architecture="D4 graph encoder with strict-past causal graph features in the autoregressive decoder",
        scope="Confirmed DEVELOPMENT samples only, equal count/bin/fit weights; all final comparisons remain retained",
        methods_ranked=ranking,independent_metric_checks=checks,
        uncertainty="Three fitted seeds are retained; reported spread is descriptive and proposal rows are not independent fits. Improvement on DEV is a hypothesis for untouched FINAL, not confirmation.",
        final_labels_opened=False,protected_test_labels_used=False,new_training_trajectories=0,new_MLIP_calls=0,
        source_hashes={str(q.relative_to(root)):diagnostics.sha256(q) for q in sources},
        analyzer_source_sha256=diagnostics.sha256(source),audit_source_sha256=diagnostics.sha256(__file__))
    with selection.open("x") as f:json.dump(record,f,indent=2);f.write("\n")
    print(json.dumps(dict(status=record["status"],selection_path=str(selection),
        selection_sha256=diagnostics.sha256(selection),diagnostic_receipt=result,ranked_methods=methods),indent=2))


if __name__ == "__main__":main()

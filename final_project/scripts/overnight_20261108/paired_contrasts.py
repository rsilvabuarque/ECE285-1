"""Descriptive paired-replicate contrasts from complete final stage tables.

LOW-only means answer low-energy search; all-bin means answer conditioning.
Neither draw-level repetitions nor reused uniform pools add independent fits.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from paired_common import atomic_json,sha256


def write_csv(path,rows):
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def paired_difference(left,right,metric,scope):
    """Pair composition/request cells within each frozen replicate, then mean."""
    def index(rows):
        result={}
        for row in rows:
            if scope=="LOW" and row["requested_bin"]!="low":continue
            key=(str(row["replicate"]),int(row["n_target"]),row["requested_bin"])
            if key in result:raise ValueError("Duplicate paired comparison cell")
            if int(row["attempts"])!=64:raise ValueError("Prespecified64-attempt cell denominator changed")
            result[key]=float(row[metric])
        return result
    a,b=index(left),index(right)
    if set(a)!=set(b):raise ValueError("Unmatched replicate/composition/request cells")
    rows=[]
    for rep in sorted({key[0] for key in a}):
        keys=[key for key in sorted(a) if key[0]==rep]
        x=float(np.mean([a[key] for key in keys]));y=float(np.mean([b[key] for key in keys]))
        rows.append(dict(replicate=rep,cells=len(keys),attempts_per_method=64*len(keys),left_yield=x,right_yield=y,delta_percentage_points=100*(x-y)))
    if len(rows)!=3:raise ValueError("Three paired replicate streams are required")
    return rows


def main(argv=None):
    from cathode_cvae.runtime import require_compute_allocation
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-root",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args(argv);require_compute_allocation()
    expected=dict(current_low=2304,paired_deletion=3456,paired_endpoint=3456,shared_uniform=1728)
    tables={};sources={};histories=set();facts=set()
    for stage,count in expected.items():
        directory=a.evaluation_root/stage;receipt_path=directory/"receipt.json"
        receipt=json.loads(receipt_path.read_text())
        if receipt["status"]!="PASS_COMPLETE" or receipt["attempts"]!=count or not receipt["all_attempts_retained"]:
            raise ValueError("Final stage is incomplete or its denominator changed: "+stage)
        histories.add(receipt["novelty_history_sha256"]);facts.add(receipt["compact_facts_sha256"])
        with (directory/"cells.csv").open(newline="") as f:tables[stage]=list(csv.DictReader(f))
        sources[stage]=dict(receipt_sha256=sha256(receipt_path),cells_sha256=sha256(directory/"cells.csv"),attempts=count)
    if len(histories)!=1 or len(facts)!=1 or None in facts:raise ValueError("Final stages must use the exact same frozen history and facts")
    def rows(stage,method):
        result=[r for r in tables[stage] if r["method"]==method]
        if not result:raise ValueError("Missing prespecified method: "+method)
        return result
    comparisons=[
        ("current_low","current_graph_cvae","current_low","current_spatial_ar"),
        ("current_low","current_graph_cvae","current_low","current_mutation"),
        ("current_low","current_graph_cvae","current_low","uniform"),
        ("paired_deletion","paired_deletion_cvae","paired_deletion","paired_deletion_mutation"),
        ("paired_deletion","paired_deletion_cvae","shared_uniform","uniform"),
        ("paired_endpoint","paired_endpoint_cvae","paired_endpoint","paired_endpoint_mutation"),
        ("paired_endpoint","paired_endpoint_cvae","shared_uniform","uniform"),
        ("paired_endpoint","paired_endpoint_cvae","paired_deletion","paired_deletion_cvae"),
    ]
    replicate_rows=[];summaries=[]
    for ls,lm,rs,rm in comparisons:
        scopes=["LOW"] if ls=="current_low" else ["LOW","ALL_BINS"]
        for scope in scopes:
            metrics=["deletion_useful_yield","endpoint_useful_yield"]
            if ls=="current_low":metrics.append("original_q20_deletion_useful_yield")
            for metric in metrics:
                values=paired_difference(rows(ls,lm),rows(rs,rm),metric,scope)
                prefix=dict(left_stage=ls,left_method=lm,right_stage=rs,right_method=rm,scope=scope,metric=metric)
                replicate_rows.extend([dict(**prefix,**r) for r in values])
                delta=np.array([r["delta_percentage_points"] for r in values])
                summaries.append(dict(**prefix,replicates=3,left_mean_yield=float(np.mean([r["left_yield"] for r in values])),
                    right_mean_yield=float(np.mean([r["right_yield"] for r in values])),mean_delta_percentage_points=float(delta.mean()),
                    sample_std_delta_percentage_points=float(delta.std(ddof=1)),positive_replicate_differences=int((delta>0).sum()),
                    zero_replicate_differences=int((delta==0).sum()),minimum_delta_percentage_points=float(delta.min()),maximum_delta_percentage_points=float(delta.max())))
    a.output_dir.mkdir(parents=True,exist_ok=False)
    write_csv(a.output_dir/"per_replicate_contrasts.csv",replicate_rows)
    write_csv(a.output_dir/"summary_contrasts.csv",summaries)
    receipt=dict(status="PASS_COMPLETE_DESCRIPTIVE_PAIRED_CONTRASTS",completed_UTC=datetime.now(timezone.utc).isoformat(),
        sources=sources,history_sha256=next(iter(histories)),facts_sha256=next(iter(facts)),source_sha256=sha256(__file__),
        total_stage_attempts=10944,current_LOW_uniform_is_shared_LOW=True,shared_uniform_LOW_reused_attempts=576,
        new_training_updates=0,new_model_calls=0,new_threshold_fits=0,
        statistical_scope="Three independent fitted-seed/paired sampling-stream replicates. Macro means equally weight compositions and requests. Reported sample SD is between three replicate contrasts; it is not a draw-level confidence interval or evidence of significance.",
        comparison_scope="Both TRAIN-frozen assessment targets are shown; LOW is separate from all-bin conditioning. Mutation label-specific TRAIN seed pools differ. Shared uniform draws are reused across target/request comparisons and never counted as independent repetitions.")
    atomic_json(a.output_dir/"receipt.json",receipt)
    print(json.dumps(dict(status=receipt["status"],comparisons=len(summaries),replicate_rows=len(replicate_rows)),indent=2))


if __name__=="__main__":main()

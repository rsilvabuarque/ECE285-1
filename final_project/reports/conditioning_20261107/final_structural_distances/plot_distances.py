"""Render only saved label-free distance tables; no new distance calculations."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cathode_cvae.runtime import require_compute_allocation


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    require_compute_allocation();root=Path(__file__).resolve().parent
    rows=read(root/"metrics_by_seed.csv");attempts=read(root/"attempt_distances.csv")
    groups=defaultdict(list)
    for r in rows:groups[r["method"]].append(r)
    table=[]
    for method,rr in sorted(groups.items()):
        d=np.asarray([float(r["mean_nearest_TRAIN_reference_orbit_Hamming"]) for r in rr])
        out=dict(method=method,design_seeds=len(rr),mean_nearest_TRAIN_reference_orbit_Hamming=float(d.mean()),
            minimum_seed_mean=float(d.min()),maximum_seed_mean=float(d.max()))
        if method=="mutation":
            out.update(mean_raw_seed_child_Hamming=float(np.mean([float(r["mean_raw_seed_child_Hamming"]) for r in rr])),
                mean_symmetry_minimum_seed_child_Hamming=float(np.mean([float(r["mean_symmetry_minimum_seed_child_Hamming"]) for r in rr])))
        table.append(out)
    for r in attempts:
        value=int(r["nearest_TRAIN_reference_orbit_Hamming"])
        if value%2 or not 0<=value<=2*min(int(r["n_Li"]),48-int(r["n_Li"])):
            raise AssertionError("Fixed-count distance violates parity or maximal possible distance")
    names=sorted(groups)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={"width_ratios":[2,1]})
    for i,name in enumerate(names):
        d=[float(r["mean_nearest_TRAIN_reference_orbit_Hamming"]) for r in groups[name]]
        axes[0].scatter(np.full(len(d),i),d,s=24,color="tab:blue")
    axes[0].set(xticks=range(len(names)),xticklabels=names,ylim=(0,8),
        ylabel="Mean differing site occupations",title="Fixed1024-orbit TRAIN reference/count\nOne point per design seed; upper bound on full-TRAIN nearest")
    axes[0].tick_params(axis="x",labelrotation=60,labelsize=8)
    mutation=[r for r in attempts if r["method"]=="mutation"]
    for k in [1,2,3]:
        values=[r for r in mutation if int(r["swap_count"])==k]
        axes[1].scatter([k-.07],[np.mean([int(r["raw_seed_child_Hamming"]) for r in values])],color="tab:orange",label="raw joint orientation" if k==1 else None)
        axes[1].scatter([k+.07],[np.mean([int(r["symmetry_minimum_seed_child_Hamming"]) for r in values])],color="tab:green",label="minimum over192symmetries" if k==1 else None)
    axes[1].set(xticks=[1,2,3],ylim=(0,6.5),xlabel="Declared mutation swap count k",
        ylabel="Mean differing sites from actual TRAIN seed",title="Mutation retains exact raw2k edits")
    axes[1].legend(fontsize=8,loc="upper left")
    fig.tight_layout()
    outputs=[]
    for suffix in ["png","pdf"]:
        path=root/f"structural_distances.{suffix}"
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=170,bbox_inches="tight");outputs.append(path)
    plt.close(fig)
    fields=list(dict.fromkeys(k for r in table for k in r))
    with (root/"family_summary.csv").open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(table)
    receipt=dict(status="PASS",all34560fixedcount_distances_even_and_bounded=True,family_summary=table,
        input_hashes={p.name:sha(p) for p in [root/"metrics_by_seed.csv",root/"attempt_distances.csv"]},
        output_hashes={p.name:sha(p) for p in outputs+[root/"family_summary.csv"]},source_sha256=sha(Path(__file__)),
        scope="Descriptive only; no energy labels or selections; distances to the1024subset are upper bounds")
    with (root/"plot_receipt.json").open("x") as f:json.dump(receipt,f,indent=2);f.write("\n")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":main()

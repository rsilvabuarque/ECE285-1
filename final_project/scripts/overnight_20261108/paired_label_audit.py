"""TRAIN/VAL-only deletion-versus-endpoint ordering and class transitions."""
from __future__ import annotations

import argparse,csv,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

from paired_common import atomic_json,sha256


def write_csv(path,rows):
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main(argv=None):
    from cathode_cvae.runtime import require_compute_allocation
    from cathode_cvae.conditions import classify,load_schema
    from scipy.stats import spearmanr,kendalltau,rankdata
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args(argv);require_compute_allocation()
    paths={target:args.collection/("learning_"+target+".npz") for target in ["deletion","endpoint"]}
    data={};schemas={};bindings={}
    for target,path in paths.items():
        with np.load(path,allow_pickle=False) as z:
            role=z["split"].astype(str)
            if set(role)-{"train","val"}:raise ValueError("Audit refuses TEST labels before opening any energy field")
            data[target]={k:z[k] for k in ["occupancy","split","structure_ids","n_target","energy_eV","energy_eV_atom"]}
        sp=args.collection/("condition_schema_"+target+".json");schemas[target]=load_schema(sp)
        if schemas[target]["provenance"]["data_sha256"]!=sha256(path):raise ValueError("Schema/data binding changed")
        bindings[target]=dict(data_sha256=sha256(path),schema_sha256=sha256(sp),schema_identity=schemas[target]["identity"])
    a,b=data["deletion"],data["endpoint"]
    for name in ["occupancy","split","structure_ids","n_target"]:
        if not np.array_equal(a[name],b[name]):raise ValueError("Label targets do not share exact independent inputs/roles")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    bins={target:classify(schemas[target],a["n_target"],d["energy_eV_atom"]) for target,d in data.items()}
    summaries=[];transitions=[];lineages=[]
    for count in sorted(set(a["n_target"].tolist())):
        for role in ["train","val"]:
            keep=(a["n_target"]==count)&(a["split"]==role);e0=a["energy_eV_atom"][keep];e1=b["energy_eV_atom"][keep]
            lo0=bins["deletion"][keep]=="low";lo1=bins["endpoint"][keep]=="low";inter=int(np.sum(lo0&lo1))
            change=(e1-e0)*1000;total=b["energy_eV"][keep]-a["energy_eV"][keep]
            summaries.append(dict(n_target=int(count),split=role,independent_masks=int(keep.sum()),
                spearman=float(spearmanr(e0,e1).statistic),kendall=float(kendalltau(e0,e1).statistic),
                deletion_low=int(lo0.sum()),endpoint_low=int(lo1.sum()),low_intersection=inter,
                deletion_low_to_endpoint_low_fraction=inter/int(lo0.sum()) if lo0.any() else None,
                endpoint_low_recalled_by_deletion_low=inter/int(lo1.sum()) if lo1.any() else None,
                mean_delta_eV_cell=float(total.mean()),median_delta_eV_cell=float(np.median(total)),
                minimum_delta_eV_cell=float(total.min()),maximum_delta_eV_cell=float(total.max()),
                mean_delta_meV_atom=float(change.mean()),median_delta_meV_atom=float(np.median(change)),
                minimum_delta_meV_atom=float(change.min()),maximum_delta_meV_atom=float(change.max()),
                positive_relaxation_energy_changes=int(np.sum(total>0))))
            for start in ["low","middle","high"]:
                for end in ["low","middle","high"]:
                    denom=int(np.sum(bins["deletion"][keep]==start))
                    n=int(np.sum((bins["deletion"][keep]==start)&(bins["endpoint"][keep]==end)))
                    transitions.append(dict(n_target=int(count),split=role,deletion_bin=start,endpoint_bin=end,
                        masks=n,deletion_bin_masks=denom,row_fraction=n/denom if denom else None))
    for i in range(len(a["split"])):
        lineages.append(dict(structure_id=str(a["structure_ids"][i]),n_target=int(a["n_target"][i]),split=str(a["split"][i]),
            deletion_energy_eV=float(a["energy_eV"][i]),endpoint_energy_eV=float(b["energy_eV"][i]),
            deletion_energy_eV_atom=float(a["energy_eV_atom"][i]),endpoint_energy_eV_atom=float(b["energy_eV_atom"][i]),
            deletion_bin=str(bins["deletion"][i]),endpoint_bin=str(bins["endpoint"][i])))
    write_csv(args.output_dir/"composition_split_summary.csv",summaries)
    write_csv(args.output_dir/"bin_transitions.csv",transitions);write_csv(args.output_dir/"trainval_lineages.csv",lineages)
    import matplotlib;matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts=sorted(set(a["n_target"].tolist()));fig,axes=plt.subplots(1,len(counts),figsize=(4*len(counts),3.8),squeeze=False)
    for ax,count in zip(axes[0],counts):
        keep=(a["n_target"]==count)&(a["split"]=="val");e0=a["energy_eV_atom"][keep];e1=b["energy_eV_atom"][keep]
        x=(rankdata(e0)-.5)/len(e0);y=(rankdata(e1)-.5)/len(e1)
        color=np.where(bins["deletion"][keep]=="low",np.where(bins["endpoint"][keep]=="low","tab:green","tab:orange"),"0.65")
        ax.scatter(x,y,c=color,s=10,alpha=.65);ax.plot([0,1],[0,1],"k--",lw=.8)
        row=next(r for r in summaries if r["n_target"]==count and r["split"]=="val")
        ax.set(title=f"Li count {count}; VAL N={len(e0)}\nSpearman={row['spearman']:.3f}",xlabel="Deletion energy rank fraction",ylabel="Endpoint energy rank fraction",xlim=(0,1),ylim=(0,1))
    fig.suptitle("UMA fixed-cell labels: green LOW→LOW, orange LOW→other (TRAIN-derived thresholds)",fontsize=10)
    fig.tight_layout();fig.savefig(args.output_dir/"validation_rank_comparison.pdf");fig.savefig(args.output_dir/"validation_rank_comparison.png",dpi=160);plt.close(fig)
    receipt=dict(status="PASS_TRAIN_VAL_ONLY",completed_UTC=datetime.now(timezone.utc).isoformat(),
        independent_pairs=len(a["split"]),test_labels_read=False,inputs=bindings,source_sha256=sha256(__file__),
        scope="Same-mask same-UMA fixed-cell endpoints. Separate TRAIN schemas; no pooled cross-composition R², no TEST outcome selection, no global-minimum/DFT claim.")
    atomic_json(args.output_dir/"receipt.json",receipt)
    print(json.dumps(dict(status=receipt["status"],independent_pairs=receipt["independent_pairs"],validation=[r for r in summaries if r["split"]=="val"]),indent=2))
    return receipt


if __name__=="__main__":main()

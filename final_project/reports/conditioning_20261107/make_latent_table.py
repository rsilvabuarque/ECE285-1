"""Write latent-size report and individual-seed tables from measured records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return list(csv.DictReader(Path(path).open()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-dir", type=Path, required=True)
    p.add_argument("--fits-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("docs/midterm1/generated_latent_table.tex"))
    p.add_argument("--evidence-dir", type=Path, default=Path("reports/conditioning_20261107/latent_development"))
    a = p.parse_args()
    summary = {r["method"]:r for r in read(a.evaluation_dir/"summary.csv") if r["teacher"]=="uma"}
    low = {r["method"]:r for r in read(a.evaluation_dir/"fixed_q20_low_request.csv") if r["teacher"]=="uma"}
    cells = [r for r in read(a.evaluation_dir/"metrics.csv") if r["teacher"]=="uma"]
    seed_rows = [r for r in read(a.evaluation_dir/"seed_metrics.csv") if r["teacher"]=="uma"]
    methods = [("latent4","CVAE4","latent_d4"),("latent8","CVAE8","balance_1_d8"),
               ("latent16","CVAE16","latent_d16"),("latent32","CVAE32","latent_d32"),
               ("ar_baseline","AR, no latent","no_latent"),("uniform","Uniform",None),
               ("mutation","Mutation",None)]
    tex, evidence, inputs = [], [], []
    for method, label, prefix in methods:
        parameter_count = "--"
        if prefix:
            config_rows = []
            for seed in [20261106,20261107,20261108]:
                path = a.fits_root / f"{prefix}_{seed}" / "summary.json"
                r = json.loads(path.read_text())
                assert r["status"]=="FIT_COMPLETE" and r["optimizer_updates"]==4000
                config_rows.append(r)
                inputs.append(path)
            assert len({r["parameter_count"] for r in config_rows})==1
            parameter_count = str(config_rows[0]["parameter_count"])
        mean = 100*float(summary[method]["mean_novel_inbin_yield"])
        sd = 100*float(summary[method]["design_seed_standard_deviation"])
        lo = 100*float(low[method]["mean_novel_fixed_q20_yield"])
        label = label.replace("CVAE", "CVAE $D=") + "$" if label.startswith("CVAE") else label
        tex.append(f"{label} & {parameter_count} & {mean:.2f} $\\pm$ {sd:.2f} & {lo:.2f}\\\\")
        for row in [r for r in seed_rows if r["method"]==method]:
            replicate = row["replicate_id"]
            rr = [r for r in cells if r["method"]==method and r["replicate_id"]==replicate and r["requested_bin"]=="low"]
            assert len(rr)==3
            evidence.append(dict(method=method,replicate_id=replicate,
                fitted_model=prefix is not None,parameters=parameter_count,
                balanced_yield=float(row["macro_composition_bin_inbin_yield"]),
                requested_low_original_q20_yield=sum(float(r["novel_fixed_q20_yield_per_attempt"]) for r in rr)/3))
    text = r"""\begin{center}\small
\begin{tabular}{lrrr}
\toprule
Model / sampler & Parameters & Balanced yield (\%) & Fixed-$q_{20}$ low (\%)\\
\midrule
"""+"\n".join(tex)+"\n"+r"""\bottomrule
\end{tabular}
\end{center}
"""
    a.output.write_text(text)
    a.evidence_dir.mkdir(parents=True,exist_ok=True)
    with (a.evidence_dir/"seed_values.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(evidence[0]));w.writeheader();w.writerows(evidence)
    inputs += [a.evaluation_dir/name for name in ["summary.csv","fixed_q20_low_request.csv","metrics.csv","seed_metrics.csv"]]
    receipt = dict(status="PASS",scope="Latent-size development comparison; samehistory, newnamespace, reusedD8fit",
        source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
        seeds=len(evidence),models=len(methods),
        uncertainty="Descriptive SD across independentfits or samplerreplicates; no physical convergence claim")
    (a.evidence_dir/"table_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(dict(status="PASS",models=len(methods),individual_seed_rows=len(evidence)),indent=2))


if __name__=="__main__":
    main()

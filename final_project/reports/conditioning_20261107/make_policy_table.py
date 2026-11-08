"""Write the report's development-policy table from complete measured CSVs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("docs/midterm1/generated_policy_table.tex"))
    a = p.parse_args()
    paths = [a.evaluation_dir / name for name in ["summary.csv", "fixed_q20_low_request.csv"]]
    tables = [{r["method"]: r for r in csv.DictReader(path.open()) if r["teacher"] == "uma"} for path in paths]
    methods = [("balance0_sel1_d8", "Off", "1"),
               ("balance0_sel0p1_d8", "Off", "0.1"),
               ("balance1_sel1_d8", "On", "1"),
               ("balance1_sel0p1_d8", "On", "0.1"),
               ("uniform", "Uniform", "--"), ("mutation", "Mutation", "--")]
    rows = []
    for method, balancing, beta in methods:
        r, low = (table[method] for table in tables)
        assert int(r["design_seed_count"]) == 3
        mean = 100*float(r["mean_novel_inbin_yield"])
        std = 100*float(r["design_seed_standard_deviation"])
        q20 = 100*float(low["mean_novel_fixed_q20_yield"])
        rows.append(f"{balancing} & {beta} & {mean:.2f} $\\pm$ {std:.2f} & {q20:.2f}\\\\")
    text = r"""\begin{center}\small
\begin{tabular}{llrr}
\toprule
Train balancing / sampler & Selection $\beta$ & Balanced yield (\%) & Fixed-$q_{20}$ low (\%)\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}
\end{center}
"""
    a.output.write_text(text)
    receipt = dict(status="PASS", scope="Development policy comparison, not final assessment",
        source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
        rows=len(rows), percentage_rounding=2,
        uncertainty="Descriptive sample standard deviation over three design replicates")
    a.output.with_suffix(".json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main()

"""Write the complete three-seed spatial DEVELOPMENT table, not final results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("docs/midterm1/generated_spatial_table.tex"))
    a=p.parse_args()
    paths=[a.evaluation_dir/n for n in ["summary.csv","fixed_q20_low_request.csv"]]
    tables=[{r["method"]:r for r in csv.DictReader(q.open()) if r["teacher"]=="uma"} for q in paths]
    methods=[("dense","Dense encoder"),("symmetry","Dense + symmetry augmentation"),
             ("graph_encoder","Graph encoder"),("graph_context","Graph encoder + causal context"),
             ("uniform","Uniform"),("mutation","Frozen mutation")]
    rows=[]
    for method,label in methods:
        row,low=(t[method] for t in tables)
        assert int(row["design_seed_count"])==3
        if method not in {"uniform","mutation"}:
            assert int(row["fitted_seed_count"])==3, "Pilot is not a confirmed three-fit comparison"
        mean=100*float(row["mean_novel_inbin_yield"])
        sd=100*float(row["design_seed_standard_deviation"])
        q=100*float(low["mean_novel_fixed_q20_yield"])
        rows.append(f"{label} & {mean:.2f} $\\pm$ {sd:.2f} & {q:.2f}\\\\")
    text=r"""\begin{center}\small
\begin{tabular}{lrr}
\toprule
Development method & Balanced UMA yield (\%) & Fixed-$q_{20}$ low (\%)\\
\midrule
"""+"\n".join(rows)+"\n"+r"""\bottomrule
\end{tabular}
\end{center}
"""
    a.output.write_text(text)
    receipt=dict(status="PASS",scope="Confirmed three-fit spatial development; final pending",
        source_hashes={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in paths},
        output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
        uncertainty="Three fit or sampler replicates, descriptive sample SD")
    a.output.with_suffix(".json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":main()

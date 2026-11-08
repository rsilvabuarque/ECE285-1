"""Build the main report table from the frozen final assessment, using stdlib.

Balanced values average counts/bins equally then independent fit/sampler seeds.
The low column is a different observable: original-q20 success on low requests.
"""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/conditioning_20261107/evaluation"


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    paths = [SOURCE/"final/summary.csv", SOURCE/"final/fixed_q20_low_request.csv",
             SOURCE/"final_matched/summary.csv", SOURCE/"final_matched/fixed_q20_low_request.csv",
             SOURCE/"final/receipt.json", SOURCE/"final_matched/receipt.json",
             ROOT/"runs/conditioning_20261107/final_protocol.json"]
    for path in paths[4:6]:
        assert json.loads(path.read_text())["status"] == "PASS"
    protocol = json.loads(paths[-1].read_text())
    assert protocol["status"] == "FROZEN_BEFORE_FINAL_GENERATION"
    assert protocol["history_orbits"] == 278382
    summaries = {(r["teacher"],r["method"]):r for r in rows(paths[0])+rows(paths[2])}
    low = {(r["teacher"],r["method"]):r for r in rows(paths[1])+rows(paths[3])}
    methods = [("cvae_d4","Dense D4"),("symmetry","Dense + symmetry"),
               ("graph_encoder","Graph encoder"),("graph_context","Graph + causal context"),
               ("uniform","Uniform"),("mutation","Frozen mutation")]
    output = [r"\begin{center}\small", r"\begin{tabular}{lrrr}",r"\toprule",
              r"Frozen-final method & UMA in-bin (\%) & MACE in-bin (\%) & UMA low (\%)\\",r"\midrule"]
    table = []
    for method,label in methods:
        values=[]
        for teacher in ["uma","mace"]:
            r=summaries[teacher,method]
            assert int(r["design_seed_count"]) == 3
            values.append(f"{100*float(r['mean_novel_inbin_yield']):.2f} $\\pm$ {100*float(r['design_seed_standard_deviation']):.2f}")
        low_value = 100*float(low["uma",method]["mean_novel_fixed_q20_yield"])
        output.append(f"{label} & {values[0]} & {values[1]} & {low_value:.2f}"+r"\\")
        table.append(dict(method=method,label=label,uma=summaries["uma",method],
                          mace=summaries["mace",method],uma_low=low["uma",method],mace_low=low["mace",method]))
    output += [r"\bottomrule",r"\end{tabular}",r"\end{center}",""]
    target=ROOT/"docs/midterm1/generated_final_table.tex"
    target.write_text("\n".join(output))
    receipt=dict(status="PASS_FINAL_TABLE",scope="Frozen final, matched six-family MACE audit; not independent DFT validation",
                 history_orbits=protocol["history_orbits"],rows=table,
                 source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                 table_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    target.with_suffix(".json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(dict(status=receipt["status"],rows=len(table),history_orbits=protocol["history_orbits"])))


if __name__ == "__main__":
    main()

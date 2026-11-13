"""Extract small facts/figure inputs from saved audit CSVs; no model inference.

Run from any directory. The default inputs are included snapshots of the audit
CSV tables. Generated support files are also included, so compiling the PDF
does not require Python, access to the large datasets, or this extraction step.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-results", type=Path,
                        default=base / "assets" / "source_tables")
    audit = parser.parse_args().audit_results.resolve()
    assets = base / "assets"
    assets.mkdir(exist_ok=True)
    macros = {}
    source_hashes = {}
    for prefix, filename in [("Historical", "historical_cached_rank_metrics.csv"),
                             ("Hu", "hu_rank_metrics.csv")]:
        path = audit / filename
        source_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as stream:
            rows = [row for row in csv.DictReader(stream)
                    if row["grouping"] == "phase_composition"]
        assert all(row["informative_ordering"] == "True" for row in rows)
        macros[prefix + "Frames"] = f"{sum(int(row['n']) for row in rows):,}"
        macros[prefix + "Groups"] = str(len(rows))
        macros[prefix + "Rho"] = f"{statistics.median(float(row['spearman_rho']) for row in rows):.4f}"
        macros[prefix + "Precision"] = f"{100 * statistics.median(float(row['bottom20_precision']) for row in rows):.1f}"
        macros[prefix + "Exact"] = str(sum(row["exact_order_match"] == "True" for row in rows))

    path = audit / "hu_optimized_Li10_counterexample.csv"
    source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    selected = next(row for row in rows if float(row["UMA_subset_rank"]) == 1)
    macros["CounterRegret"] = f"{float(selected['DFT_above_min_meV_atom']):.2f}"
    shutil.copyfile(path, assets / path.name)
    (assets / "facts.tex").write_text(
        "% Generated from saved ranking audit CSVs by build_support.py.\n"
        + "".join("\\newcommand{\\" + key + "}{" + value + "}\n"
                  for key, value in macros.items()))

    # Match the same measured structure between the two ordinal rank axes.
    lines = [r"\begin{tikzpicture}[x=1cm,y=0.48cm,>=Latex,font=\small]",
             r"\node[font=\bfseries,text=navy] at (0,1.0) {DFT rank};",
             r"\node[font=\bfseries,text=navy] at (7,1.0) {UMA rank};"]
    for row in rows:
        dft = int(row["DFT_subset_rank"])
        uma = int(row["UMA_subset_rank"])
        sid = row["structure_id"].replace("_", r"\_")
        color = "coral" if uma == 1 else ("teal" if dft == 1 else "slate")
        lines.extend([
            f"\\draw[{color},line width=1.1pt] (0.55,{-dft}) -- (6.45,{-uma});",
            f"\\node[anchor=east,text={color}] at (0.45,{-dft}) {{{dft} \\quad {sid}}};",
            f"\\node[anchor=west,text={color}] at (6.55,{-uma}) {{{uma} \\quad {sid}}};",
        ])
    lines.append(r"\end{tikzpicture}")
    (assets / "rank_reversal.tex").write_text("\n".join(lines) + "\n")
    evidence = {
        "macros": macros,
        "raw_combinations_diagnostic_only": {
            str(n): {str(k): math.comb(n, k) for k in ([1,2,3] if n==12 else [2,4,6])}
            for n in [12,24]
        },
        "source_results": str(audit),
        "source_sha256": source_hashes,
        "method": "Read saved CSVs; no inference or new scientific data generation.",
    }
    (assets / "evidence_extract.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(macros, indent=2))


if __name__ == "__main__":
    main()

"""Recompute the fixed 48-site geometry pilot from existing scalar records."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from .data import sha256


def read_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def evaluate(scores_path, relax_path, output, fmax=0.05):
    """Keep direct inference separate from a subsequently converged optimizer.

    fmax reproduces the pilot's operational maximum-force criterion in eV/A.
    It is neither a force RMS threshold nor a universal convergence standard.
    """
    scores, relax = read_rows(scores_path), read_rows(relax_path)
    if not scores or not relax or fmax <= 0:
        raise ValueError("Nonempty records and a positive force criterion are required")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    baseline = {(r["teacher"], r["source_mask_id"]): r for r in scores if r["method"] == "deletion"}
    if len(baseline) != sum(r["method"] == "deletion" for r in scores):
        raise ValueError("Repeated deletion baseline IDs")
    groups = defaultdict(list)
    seen = set()
    for row in scores:
        key = row["teacher"], row["source_mask_id"], row["method"]
        if key in seen or row["valid"].lower() != "true":
            raise ValueError("Duplicate or invalid score; retain failures explicitly before aggregating")
        seen.add(key)
        ref = baseline[key[:2]]
        n = int(row["n_Li"])
        if n != int(ref["n_Li"]):
            raise ValueError("Baseline and decoded composition differ")
        delta = 1000 * (float(row["energy_eV"]) - float(ref["energy_eV"])) / (144 + n)
        ratio = float(row["force_component_rms_eV_A"]) / float(ref["force_component_rms_eV_A"])
        # These compare alternative calculations from the same CSV scalars;
        # their tolerances cover float text roundoff, not physical agreement.
        if not np.isclose(delta, float(row["energy_minus_deletion_meV_atom"]), atol=1e-8, rtol=1e-10):
            raise ValueError("Recorded energy difference disagrees with its baseline")
        if not np.isclose(ratio, float(row["force_rms_over_deletion"]), atol=1e-10, rtol=1e-10):
            raise ValueError("Recorded force ratio disagrees with its baseline")
        groups[row["teacher"], row["method"]].append((delta, ratio, float(row["force_max_eV_A"])))
    direct = []
    for (teacher, method), values in sorted(groups.items()):
        values = np.asarray(values)
        direct.append(dict(teacher=teacher, method=method, masks=len(values),
            median_energy_minus_deletion_meV_atom=float(np.median(values[:, 0])),
            fraction_lower_energy=float(np.mean(values[:, 0] < 0)),
            median_force_rms_ratio=float(np.median(values[:, 1])),
            fraction_fmax_below_criterion=float(np.mean(values[:, 2] < fmax)), fmax_criterion_eV_A=fmax))
    paired = defaultdict(dict)
    for r in relax:
        group = paired[r["source_mask_id"]]
        if r["method"] in group:
            raise ValueError("Repeated method within a relaxation lineage")
        group[r["method"]] = r
    steps = []
    for method in sorted({r["method"] for r in relax}):
        if any(method not in group or "deletion" not in group for group in paired.values()):
            raise ValueError("Matched relaxation support is incomplete")
        rows = [group[method] for group in paired.values()]
        converged = [r["converged"].lower() == "true" for r in rows]
        delta_steps = [int(group[method]["steps"]) - int(group["deletion"]["steps"]) for group in paired.values()]
        steps.append(dict(method=method, matched_lineages=len(rows), numerically_converged=sum(converged),
            median_steps=float(np.median([int(r["steps"]) for r in rows])),
            median_step_difference_vs_deletion=float(np.median(delta_steps)),
            median_wall_seconds=float(np.median([float(r["wall_seconds"]) for r in rows])),
            median_endpoint_energy_difference_meV_atom=float(np.median([float(r["endpoint_energy_difference_meV_atom"]) for r in rows]))))
    write_rows(out / "direct_summary.csv", direct)
    write_rows(out / "paired_relaxation_summary.csv", steps)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = sorted(next(iter(paired.values())))
    fig, ax = plt.subplots(figsize=(6, 4))
    data = np.array([[int(group[m]["steps"]) for m in methods] for group in paired.values()])
    for row in data:
        ax.plot(range(len(methods)), row, color="steelblue", alpha=.25, linewidth=.7)
    ax.plot(range(len(methods)), np.median(data, axis=0), "o-", color="black", label="Method median")
    ax.set(xticks=range(len(methods)), xticklabels=methods, ylabel="Optimizer steps", title=f"Same {len(paired)} masks: fixed-cell UMA relaxation")
    ax.legend(); fig.tight_layout()
    fig.savefig(out / "paired_steps.pdf"); fig.savefig(out / "paired_steps.png", dpi=160); plt.close(fig)
    receipt = dict(status="PASS", direct_score_rows=len(scores), matched_lineages=len(paired),
        source_hashes=dict(scores=sha256(scores_path), relaxations=sha256(relax_path)), fmax_eV_A=fmax,
        scope="Cached direct geometry predictions and separate matched UMA fixed-cell relaxations. No new MLIP inference; no independent DFT or new-model validation.")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--fmax", type=float, default=0.05, help="Pilot maximum-force criterion, eV/Angstrom")
    a = ap.parse_args(argv)
    return evaluate(a.snapshot / "reports/decoded_geometry_scores.csv",
                    a.snapshot / "reports/decoded_matched_relaxations.csv", a.output_dir, a.fmax)

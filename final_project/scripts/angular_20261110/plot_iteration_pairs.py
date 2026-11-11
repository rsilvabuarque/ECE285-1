#!/usr/bin/env python3
"""Plot seed-paired changes from completed iterative-assessment receipts.

R1 compares acquisition to a no-new-data continuation. R2 compares each
acquisition lineage to its own no-second-batch continuation. These rounds
have different novelty histories; their absolute yields must not be joined.
Inputs are aggregate seed_metrics.csv files whose neighbouring receipts pass.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


POLICIES = (
    ("cvae_q16_unit", "CVAE, 16, unit"),
    ("cvae_q128_unit", "CVAE, 128, unit"),
    ("cvae_q128_fraction20", "CVAE, 128, 20% weight"),
    ("mutation_q16_unit", "Mutation, 16, unit"),
    ("mutation_q128_unit", "Mutation, 128, unit"),
    ("mutation_q128_fraction20", "Mutation, 128, 20% weight"),
)
METRICS = (("balanced_novel_inbin", "Balanced novel in-request yield"),
           ("fixed_q20_low", "Historical fixed-q20 LOW yield"))
SEEDS = ("2026110801", "2026110802", "2026110803")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_aggregate(directory):
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "PASS":
        raise ValueError(f"Aggregate did not pass: {receipt_path}")
    csv_path = directory / "seed_metrics.csv"
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {}
    for row in rows:
        key = (row["policy"], row["replicate_id"])
        if key in lookup:
            raise ValueError(f"Duplicate seed/policy: {key} in {csv_path}")
        lookup[key] = row
    return lookup, {"receipt": str(receipt_path),
                    "receipt_sha256": sha(receipt_path),
                    "seed_metrics": str(csv_path),
                    "seed_metrics_sha256": sha(csv_path),
                    "runtime_verifier_source_status": receipt.get("runtime_verifier_source_status")}


def pairs(lookup, round_number, teacher):
    output = []
    for policy, label in POLICIES:
        model = f"{policy}_last" if round_number == 1 else f"r2_{policy}_last"
        reference = "no_new_last" if round_number == 1 else f"r2_{policy}_no_r2_last"
        for seed in SEEDS:
            try:
                actual, baseline = lookup[(model, seed)], lookup[(reference, seed)]
            except KeyError as exc:
                raise ValueError(f"Missing {teacher} R{round_number} paired model: {exc}") from exc
            for metric, _ in METRICS:
                left, right = float(actual[metric]), float(baseline[metric])
                if not math.isfinite(left) or not math.isfinite(right):
                    raise ValueError(f"Nonfinite yield for {model}, {seed}, {metric}")
                output.append({"round": round_number, "teacher": teacher,
                               "policy": policy, "policy_label": label,
                               "model": model, "reference": reference, "seed": seed,
                               "metric": metric, "model_yield": left,
                               "reference_yield": right,
                               "difference_percentage_points": 100 * (left - right)})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1", type=Path, required=True,
                        help="Directory containing comparison_uma_cached and comparison_mace_cached")
    parser.add_argument("--r2", type=Path,
                        help="Optional directory containing comparison_uma_cached_r2 and comparison_mace_cached_r2")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory; existing output is never overwritten")
    args = parser.parse_args()

    all_pairs, inputs = [], []
    for round_number, root in ((1, args.r1), (2, args.r2)):
        if root is None:
            continue
        for teacher in ("uma", "mace"):
            suffix = "_r2" if round_number == 2 else ""
            data, source = read_aggregate(root / f"comparison_{teacher}_cached{suffix}")
            source.update({"round": round_number, "teacher": teacher})
            inputs.append(source)
            all_pairs.extend(pairs(data, round_number, teacher))

    # Imports happen after validation so missing inputs cannot create partial plots.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = args.output_dir / "paired_seed_differences.csv"
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_pairs[0]))
        writer.writeheader()
        writer.writerows(all_pairs)

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), sharey=True)
    colors = {1: "#2265a9", 2: "#c45826"}
    offsets = {1: -0.15 if args.r2 else 0, 2: 0.15}
    for row_index, (metric, metric_label) in enumerate(METRICS):
        for col_index, teacher in enumerate(("uma", "mace")):
            ax = axes[row_index, col_index]
            ax.axvline(0, color="0.45", linewidth=0.9, zorder=0)
            for index, (policy, _) in enumerate(POLICIES):
                for rnd in ((1, 2) if args.r2 else (1,)):
                    vals = [x["difference_percentage_points"] for x in all_pairs
                            if x["round"] == rnd and x["teacher"] == teacher
                            and x["policy"] == policy and x["metric"] == metric]
                    if len(vals) != 3:
                        raise ValueError(f"Expected three paired fits, got {len(vals)}")
                    y = index + offsets[rnd]
                    ax.scatter(vals, [y] * 3, color=colors[rnd], alpha=0.4, s=19, zorder=2)
                    ax.errorbar(statistics.mean(vals), y, xerr=statistics.stdev(vals),
                                fmt="D", color=colors[rnd], markersize=4.5,
                                capsize=2.5, linewidth=1.1, zorder=3)
            ax.set_yticks(range(len(POLICIES)), [label for _, label in POLICIES])
            ax.set_ylim(len(POLICIES) - 0.5, -0.5)
            ax.grid(axis="x", alpha=0.2)
            ax.set_title(f"{teacher.upper()} — {metric_label}", fontsize=10)
            ax.set_xlabel("Acquisition minus matched continuation (percentage points)")
    from matplotlib.lines import Line2D
    legend = [Line2D([], [], color=colors[1], marker="D", linewidth=1,
                     label="Round 1: vs no new batch")]
    if args.r2:
        legend.append(Line2D([], [], color=colors[2], marker="D", linewidth=1,
                             label="Round 2: vs same-lineage no second batch"))
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Effect of adding oracle-labelled proposals", fontsize=14)
    fig.text(0.5, 0.055,
             "Diamonds and bars: mean ± sample SD over 3 matched fits; circles: individual fits. "
             "Thresholds are teacher-specific.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(args.output_dir / "iteration_paired_effects.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "iteration_paired_effects.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    receipt = {"status": "PASS", "inputs": inputs, "rounds": [1, 2] if args.r2 else [1],
               "pairs": len(all_pairs), "fit_seeds": list(SEEDS),
               "output_csv_sha256": sha(csv_path),
               "script_sha256": sha(Path(__file__).resolve()),
               "meaning": "R1 and R2 each compare with their own matched continuation; "
                          "histories differ. Q20 is a fixed historical TRAIN threshold, "
                          "not the bottom 20% of the complete space."}
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()

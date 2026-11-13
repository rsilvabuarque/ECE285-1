"""Regenerate compact report tables from aggregate CSVs; raw files stay intact."""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_table(path, caption, columns, heading, body):
    path.write_text("\n".join([
        r"\begin{table}[htbp]", r"\centering\small", "\\caption{" + caption + "}",
        r"\begin{tabular}{" + columns + r"}\toprule", heading + r"\\\midrule",
        *[" & ".join(row) + r"\\" for row in body],
        r"\bottomrule\end{tabular}", r"\end{table}", ""]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    bindings = {}
    phase_families = [
        ("latent_diagnostics", "table_latent_short.tex", [
            ("ar", "AR"), ("d1", r"CVAE $D=1$"), ("d2", r"CVAE $D=2$"),
            ("d4", r"CVAE $D=4$")]),
        ("latent_diagnostics", "table_latent_primary.tex", [
            ("ar", "AR"), ("d1", r"CVAE $D=1$"), ("d2", r"CVAE $D=2$"),
            ("d4", r"CVAE $D=4$"), ("cyclic_d2", r"Cyclic $D=2$"),
            ("physical_latent_d2", r"Genuine physical head, $\lambda=1$"),
            ("physical_condition_d2", "Condition-only head"),
            ("physical_shuffled_d2", "Shuffled physical target")]),
        ("latent_phase2", "table_latent_strength.tex", [
            ("ar", "AR"), ("d2", r"CVAE $D=2$ / condition controls"),
            ("lambda10_latent_d2", r"Genuine head, $\lambda=10$"),
            ("lambda10_shuffled_d2", r"Shuffled target, $\lambda=10$"),
            ("lambda100_latent_d2", r"Genuine head, $\lambda=100$"),
            ("lambda100_shuffled_d2", r"Shuffled target, $\lambda=100$")]),
    ]
    for phase, filename, families in phase_families:
        path = args.run_root / phase / "analysis_oracle/oracle_yield_seed_balanced.csv"
        data = rows(path)
        body = []
        for family, label in families:
            values = []
            for teacher in ("uma", "mace"):
                fits = [100 * float(row["conditioned_novel_yield"]) for row in data
                        if row["family"] == family and row["teacher"] == teacher
                        and row["intervention"] == "primary"]
                if len(fits) != 5:
                    raise ValueError(f"Expected five paired fits: {phase}/{family}/{teacher}")
                values.append(f"${statistics.mean(fits):.2f}\\pm{statistics.stdev(fits):.2f}$")
            body.append([label, *values])
        write_table(out / filename,
                    "Primary balanced novel correct-bin yield, percent, mean $\\pm$ sample SD over five fit/proposal seeds. Compare arms within this phase's fresh development history.",
                    "lrr", "Model & UMA & MACE", body)
        bindings[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    path = args.run_root / "entropy_boltzmann_1000K/summary/matched_budget_comparison.csv"
    data = rows(path)
    body = []
    for teacher in ("uma", "mace"):
        for temperature in (300, 1000, 5000):
            values = {}
            for method in ("composition_diversity_AR", f"Boltzmann_T1000_train_{teacher}", "uniform_microstate"):
                selected = [row for row in data if row["method"] == method
                            and row["evaluation_teacher"] == teacher
                            and float(row["temperature_K"]) == temperature and int(row["draws"]) == 8192]
                if len(selected) != 3 or {int(row["n_target"]) for row in selected} != {3, 6, 9}:
                    raise ValueError("Expected one aggregate per entropy composition")
                values[method] = (
                    statistics.mean(float(row["mean_absolute_entropy_error_kB"]) for row in selected),
                    statistics.mean(float(row["mean_weight_ESS_fraction"]) for row in selected))
            original, weighted, uniform = (values[key] for key in
                ("composition_diversity_AR", f"Boltzmann_T1000_train_{teacher}", "uniform_microstate"))
            body.append([teacher.upper(), str(temperature), f"{original[0]:.6f}",
                         f"{weighted[0]:.6f}", f"{uniform[0]:.6f}", f"{weighted[1]:.4f}",
                         f"{uniform[1]:.4f}"])
    write_table(out / "table_entropy_temperature.tex",
                "Temperature transfer of the 1,000 K-trained proposal. Entries are descriptive equal-composition averages over $n=3,6,9$ at 8,192 draws; MAE is in $k_{\\rm B}$ per cell and ESS/$M$ is dimensionless. They are not averages of a physical entropy across compositions.",
                "llrrrrr", "Teacher & $T$ (K) & AR MAE & Weighted MAE & Uniform MAE & W. ESS/$M$ & U. ESS/$M$", body)
    bindings[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = args.run_root / "iteration/study_v2/analysis/paired_fit_summary.csv"
    data = rows(path)
    body = []
    for family, family_label in (("ar", "AR"), ("d2", r"CVAE $D=2$")):
        for mode in ("fixed", "adaptive", "range"):
            values = []
            for teacher in ("uma", "mace"):
                for round_index in (1, 2):
                    selected = [row for row in data if row["teacher"] == teacher
                                and int(row["round"]) == round_index
                                and row["family"] == family and row["mode"] == mode
                                and row["metric"] == "balanced_fixed_yield"]
                    if len(selected) != 1 or int(selected[0]["fit_count"]) != 3:
                        raise ValueError("Expected one three-fit acquisition contrast")
                    row = selected[0]
                    values.append(f"${float(row['paired_gain_pp']):+.2f}\\pm{float(row['paired_gain_sd_pp']):.2f}$")
            body.append([family_label + " / " + mode, *values])
    write_table(out / "table_iteration_gains.tex",
                "Change in balanced H0 fixed-bin yield versus an equal-update no-new-data continuation, in percentage points, mean $\\pm$ sample SD over three paired fit/proposal seeds. Round 2 uses the same first-round lineage as its control. The rounds have separate assessment histories.",
                "lrrrr", "Model / rule & UMA r1 & UMA r2 & MACE r1 & MACE r2", body)
    bindings[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    baseline_path = args.run_root / "iteration/study_v2/interpretation_diagnostics/fresh_uniform_mutation_baselines.csv"
    baseline = rows(baseline_path)
    body = []
    for round_index in (1, 2):
        for teacher in ("uma", "mace"):
            values = []
            for method in ("uniform", "mutation"):
                selected = [row for row in baseline if int(row["round"]) == round_index
                            and row["teacher"] == teacher and row["method"] == method
                            and row["metric"] == "balanced_fixed_yield"]
                if len(selected) != 1:
                    raise ValueError("Expected one baseline summary")
                row = selected[0]
                values.append(f"${float(row['mean_percent']):.2f}\\pm{float(row['sample_sd_percent']):.2f}$")
            body.append([str(round_index), teacher.upper(), *values])
    write_table(out / "table_sampling_baselines.tex",
                "Declared broad three-bin sampling baselines: balanced H0 yield in percent, mean $\\pm$ three-seed SD. Mutation uses its frozen original-TRAIN seed bank. These numbers do not replace earlier LOW-only or differently seeded mutation studies.",
                "llrr", "Round & Teacher & Uniform & Mutation", body)
    bindings[str(baseline_path)] = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    path = args.run_root / "iteration/causal_range/paired_summary.csv"
    data = rows(path)
    body = []
    for family, family_label in (("ar", "AR"), ("d2", r"CVAE $D=2$")):
        for contrast, label in (("H0_relabeling", "H0 range versus fixed"),
                                ("added_rows_fixed", "New rows / fixed edges"),
                                ("added_rows_identical_range_edges", "New rows / matched range edges")):
            values = []
            for teacher in ("uma", "mace"):
                selected = [row for row in data if row["family"] == family
                            and row["teacher"] == teacher and row["contrast"] == contrast
                            and row["metric"] == "balanced_fixed_yield"]
                if len(selected) != 1 or int(selected[0]["fit_count"]) != 3:
                    raise ValueError("Expected one three-fit causal contrast")
                row = selected[0]
                values.append(f"${float(row['gain_pp_mean']):+.2f}\\pm{float(row['gain_pp_sample_sd']):.2f}$")
            body.append([family_label + " / " + label, *values])
    write_table(out / "table_causal_controls.tex",
                "Posthoc conditioning/data controls: balanced H0 fixed-bin yield change, percentage points, mean $\\pm$ sample SD over three paired fits. Relabeling also changes class-balanced loss weights. The matched-range comparison holds the physical edges identical; adding rows can still change minibatches and row weights.",
                "lrr", "Model / contrast & UMA & MACE", body)
    bindings[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    (out / "table_sources.json").write_text(json.dumps(bindings, indent=2) + "\n")
    print(json.dumps({"tables": len(phase_families) + 4, "bound_inputs": len(bindings)}))


if __name__ == "__main__":
    main()

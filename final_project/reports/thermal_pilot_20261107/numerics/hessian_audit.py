"""Independent raw-force reconstruction and numerical audit; CPU allocation only."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import null_space, eigvalsh


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def functions(path):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(Path(path).read_text()).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def harmonic(values, temperature):
    if np.any(values <= 0):
        return None, None
    factor_THz = np.sqrt(1.602176634e-19 / (1e-20 * 1.66053906660e-27)) / (2 * np.pi * 1e12)
    frequencies = np.sqrt(values) * factor_THz
    quantum = frequencies * 4.135667696e-3  # h * nu, eV; nu in THz.
    kT = 8.617333262145e-5 * temperature
    y = quantum / kT
    return float(kT * np.sum(.5 * y + np.log(-np.expm1(-y)))), float(kT * np.sum(np.log(y)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    source = args.project / "scripts/overnight_20261107/thermal_dynamics.py"
    old_source = args.project / "runs/thermal_pilot_20261107/executed_sources/thermal_dynamics_hessian_md_v1.py"
    old_functions, current_functions = functions(old_source), functions(source)
    numerical_functions = ["load_structure", "mapping_diagnostics", "evaluate", "fresh_precision_repeats", "relax", "hessian", "analyze_hessian", "assemble", "md"]
    source_unchanged = {name: old_functions[name] == current_functions[name] for name in numerical_functions}
    assert all(source_unchanged.values()), "Unexpected numerical source change after raw Hessian execution"
    rows = []; datasets = {}; source_hashes = {str(old_source): sha(old_source), str(source): sha(source)}
    pending = []; failures = []
    for basin, delta in [("basin_1", .01), ("basin_2", .01), ("basin_3", .01), ("basin_1", .02)]:
        directory = args.results / f"hessian/{basin}/delta_{str(delta)[1:]}"
        minimum_path = args.results / f"relax/{basin}/final.npz"
        minimum = load(minimum_path); source_hashes[str(minimum_path)] = sha(minimum_path)
        n = len(minimum["numbers"]); ndof = 3 * n
        H = np.empty((ndof, ndof)); columns_seen = []; column_error = []; repeat_E = []; repeat_F = []; repeat_stress = []
        contexts = []; raw_files = []
        for shard in sorted(directory.glob("shard_*")):
            path = shard / "hessian_columns.npz"; raw_files.append(str(path)); z = load(path)
            receipt = json.loads((shard / "receipt.json").read_text()); precision = load(shard / "precision_repeats.npz")
            source_hashes[str(path)] = sha(path); source_hashes[str(shard / "receipt.json")] = sha(shard / "receipt.json")
            assert receipt["status"] == "COMPLETE"
            assert receipt["source_sha256"] == sha(old_source)
            assert str(z["input_sha256"]) == sha(minimum_path)
            assert np.array_equal(z["positions_A"], minimum["positions_A"])
            assert np.array_equal(z["cell_A"], minimum["cell_A"])
            assert np.array_equal(z["masses_amu"], minimum["masses_amu"])
            assert float(z["delta_A"]) == delta
            plus = z["forces_plus_eV_A"].reshape(-1, ndof)
            minus = z["forces_minus_eV_A"].reshape(-1, ndof)
            # Independent orientation: each force component i is one row,
            # each displaced coordinate j one column.
            rebuilt = ((minus - plus) / (2 * delta)).T
            columns = z["columns"].astype(int)
            assert rebuilt.shape == (ndof, len(columns))
            column_error.append(float(np.max(np.abs(rebuilt - z["hessian_columns_eV_A2"]))))
            H[:, columns] = rebuilt; columns_seen.extend(map(int, columns))
            pe = float(np.ptp(precision["energy_eV"]))
            pf = float(np.max(np.ptp(precision["forces_eV_A"], axis=0)))
            ps = float(np.max(np.ptp(precision["stress_eV_A3"], axis=0)))
            assert len(precision["energy_eV"]) == 3 and receipt["force_precision_repeats"]["cache_reset_each_repeat"]
            assert pe == receipt["force_precision_repeats"]["energy_span_eV"]
            assert pf == receipt["force_precision_repeats"]["force_component_max_span_eV_A"]
            repeat_E.append(pe); repeat_F.append(pf); repeat_stress.append(ps)
            teacher = receipt["teacher"]
            contexts.append((teacher["checkpoint_name"], teacher["head"], json.dumps(teacher["checkpoint_hashes"], sort_keys=True), teacher["dtype"], teacher["tf32"]))
        assert sorted(columns_seen) == list(range(ndof)), "Missing/duplicate Hessian coordinate"
        assert len(set(contexts)) == 1
        assert contexts[0][0] == "UMA-S-1.2.1" and contexts[0][1] == "omat"
        assert contexts[0][-2:] == ("float32", False)
        assert max(column_error) == 0.
        mass = minimum["masses_amu"]; masses = np.repeat(mass, 3)
        symmetric = .5 * (H + H.T)
        D = symmetric / np.sqrt(masses[:, None] * masses[None, :])
        translations = np.zeros((ndof, 3))
        for axis in range(3): translations[axis::3, axis] = np.sqrt(mass / mass.sum())
        # SVD null-space construction is independent of runner's complete QR.
        basis = null_space(translations.T)
        values = eigvalsh(basis.T @ D @ basis)
        factor_THz = np.sqrt(1.602176634e-19 / (1e-20 * 1.66053906660e-27)) / (2 * np.pi * 1e12)
        frequencies = np.sign(values) * np.sqrt(np.abs(values)) * factor_THz
        full_values = eigvalsh(D)
        fq, fc = harmonic(values, 433.15)
        key = f"{basin}_delta_{delta}"
        row = dict(case=key, basin=basin, delta_A=delta, atoms=n, raw_columns=len(columns_seen), internal_modes=len(values),
            raw_reconstruction_max_error_eV_A2=max(column_error),
            antisymmetric_max_eV_A2=float(np.max(np.abs(H - H.T))),
            antisymmetric_relative_Fro=float(np.linalg.norm(H - H.T) / np.linalg.norm(H)),
            translation_residual_norm_eV_A2_amu=float(np.linalg.norm(D @ translations)),
            internal_basis_translation_overlap=float(np.max(np.abs(translations.T @ basis))),
            strict_negative_internal_modes=int(np.sum(values < 0)), min_internal_lambda=float(values.min()),
            min_signed_frequency_THz=float(frequencies.min()), max_signed_frequency_THz=float(frequencies.max()),
            three_smallest_abs_full_lambdas=json.dumps(full_values[np.argsort(np.abs(full_values))[:3]].tolist()),
            maximum_fresh_energy_span_eV=max(repeat_E), maximum_fresh_force_component_span_eV_A=max(repeat_F),
            maximum_fresh_stress_component_span_eV_A3=max(repeat_stress),
            observed_repeat_span_over_delta_eV_A2=max(repeat_F) / delta,
            relaxed_minimum_eV=float(minimum["energy_eV"]), harmonic_quantum_vib_F_433K_eV=fq,
            harmonic_classical_vib_F_433K_eV=fc)
        assembled = args.results / f"hessian_assembled/{basin}/delta_{str(delta)[1:]}/hessian.npz"
        if assembled.exists():
            saved = load(assembled); source_hashes[str(assembled)] = sha(assembled)
            saved_receipt = json.loads((assembled.parent / "receipt.json").read_text())
            assert saved_receipt["status"] == "COMPLETE" and saved_receipt["source_sha256"] == sha(source)
            matrix_error = float(np.max(np.abs(D - saved["dynamical_matrix_eV_A2_amu"])))
            eigenvalue_error = float(np.max(np.abs(values - saved["eigenvalues_internal_eV_A2_amu"])))
            # Float64 algebra check only, derived from machine epsilon times
            # matrix scale; this is not a physical phonon stability threshold.
            algebra_tolerance = 256 * np.finfo(float).eps * max(float(np.linalg.norm(D, 2)), 1.)
            assert matrix_error < algebra_tolerance and eigenvalue_error < algebra_tolerance
            assert np.array_equal(H, saved["hessian_raw_eV_A2"])
            assert np.allclose(saved["frequencies_signed_THz"], frequencies, rtol=1e-12, atol=1e-12)
            row.update(assembled_comparison="PASS", assembled_D_max_error=matrix_error,
                       assembled_eigenvalue_max_error=eigenvalue_error, algebra_tolerance=algebra_tolerance)
        else:
            pending.append(str(assembled)); row["assembled_comparison"] = "PENDING"
        datasets[key] = dict(H=H, D=D, values=values, frequencies=frequencies, row=row)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ["case", "strict_negative_internal_modes", "min_signed_frequency_THz", "maximum_fresh_force_component_span_eV_A", "assembled_comparison"]}), flush=True)
    a, b = datasets["basin_1_delta_0.01"], datasets["basin_1_delta_0.02"]
    freq_difference = b["frequencies"] - a["frequencies"]
    sensitivity = dict(basin="basin_1", displacements_A=[.01, .02],
        hessian_relative_Fro=float(np.linalg.norm(a["H"] - b["H"]) / np.linalg.norm(a["H"])),
        internal_eigenvalue_max_abs_difference=float(np.max(np.abs(a["values"] - b["values"]))),
        frequency_max_abs_difference_THz=float(np.max(np.abs(freq_difference))),
        frequency_rms_difference_THz=float(np.sqrt(np.mean(freq_difference ** 2))),
        frequency_max_relative_difference=float(np.max(np.abs(freq_difference / a["frequencies"]))),
        min_internal_lambda_difference=float(b["values"].min() - a["values"].min()),
        harmonic_quantum_vib_F_difference_eV=b["row"]["harmonic_quantum_vib_F_433K_eV"] - a["row"]["harmonic_quantum_vib_F_433K_eV"],
        harmonic_classical_vib_F_difference_eV=b["row"]["harmonic_classical_vib_F_433K_eV"] - a["row"]["harmonic_classical_vib_F_433K_eV"],
        uncertainty_interpretation="Two displacement sizes test numerical sensitivity; their difference is not a statistical CI or a bound on MLIP physical error")
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with (args.output / "hessian_audit_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, z in datasets.items():
        axes[0].plot(np.arange(1, len(z["frequencies"]) + 1), z["frequencies"], label=key.replace("_", " "))
    axes[0].axhline(0, color="k", linewidth=.7); axes[0].set(xlabel="Sorted internal mode", ylabel="Signed frequency (THz)", title="Full finite-cell Γ spectrum; 537 internal modes")
    axes[0].legend(fontsize=7)
    axes[1].plot(np.arange(1, len(freq_difference) + 1), freq_difference * 1000)
    axes[1].set(xlabel="Sorted internal mode", ylabel="ν(0.02 Å) − ν(0.01 Å) (GHz)", title="Basin 1 displacement sensitivity")
    fig.tight_layout(); fig.savefig(args.output / "hessian_audit_spectra.pdf"); fig.savefig(args.output / "hessian_audit_spectra.png", dpi=170); plt.close(fig)
    unstable = [row["case"] for row in rows if row["strict_negative_internal_modes"]]
    receipt = dict(status="PASS" if not pending and not unstable else ("ASSEMBLY_PENDING" if pending and not unstable else "INSTABILITY_REQUIRES_REVIEW"),
        checked_unix=time.time(), source_sha256=sha(__file__), numerical_functions_unchanged_after_serialization_fix=source_unchanged,
        source_hashes=source_hashes, cases=rows, delta_sensitivity=sensitivity, missing_assemblies=pending,
        unstable_cases=unstable, assembly_v1_failure="Nested CLI Path list failed JSON serialization before assembly; raw forces unaffected; current v2 changes only path serialization",
        interpretation="Positive projected modes at sampled displacement sizes support a local same-MLIP mechanical stability check for MD, not DFT validation or finite-temperature basin retention",
        repeat_precision_interpretation="Observed fresh force span divided by delta is a diagnostic scale, not a rigorous Hessian error bound; FP32 bias is not estimated",
        constraints="Three mass-weighted periodic translations projected separately; no negative eigenvalue removed or absolute-valued into a stable mode")
    (args.output / "hessian_audit.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(dict(status=receipt["status"], delta_sensitivity=sensitivity)), flush=True)


if __name__ == "__main__": main()

"""Local quadratic residual-force estimate from existing data; no predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    sensitivity_path = args.output / "hessian_audit.json"
    sensitivity = json.loads(sensitivity_path.read_text())["delta_sensitivity"]
    quantum_delta_effect = abs(sensitivity["harmonic_quantum_vib_F_difference_eV"])
    sources = {str(sensitivity_path): sha(sensitivity_path)}; rows = []
    for basin in ["basin_1", "basin_2", "basin_3"]:
        minimum_path = args.results / f"relax/{basin}/final.npz"
        hessian_path = args.results / f"hessian_assembled/{basin}/delta_.01/hessian.npz"
        minimum, hessian = load(minimum_path), load(hessian_path)
        assert str(hessian["input_sha256"]) == sha(minimum_path)
        for key in ["numbers", "masses_amu", "parent_indices", "cell_A", "positions_A"]:
            assert np.array_equal(minimum[key], hessian[key])
        force = minimum["forces_eV_A"].ravel(); mass = np.repeat(minimum["masses_amu"], 3)
        modes = hessian["eigenvectors_mass_weighted"]
        eigenvalues = hessian["eigenvalues_internal_eV_A2_amu"]
        assert modes.shape == (540, 537) and np.all(eigenvalues > 0)
        projected_force = modes.T @ (force / np.sqrt(mass))
        shift = (modes @ (projected_force / eigenvalues)) / np.sqrt(mass)
        correction = float(-.5 * np.sum(projected_force ** 2 / eigenvalues))
        # Independent Cartesian evaluation of the same quadratic expression.
        cartesian = float(-force @ shift + .5 * shift @ hessian["hessian_sym_eV_A2"] @ shift)
        assert abs(cartesian - correction) < 1e-15
        atomic_norms = np.linalg.norm(shift.reshape(-1, 3), axis=1)
        rows.append(dict(basin_id=basin, residual_max_force_eV_A=float(np.linalg.norm(force.reshape(-1, 3), axis=1).max()),
            estimated_energy_correction_eV=correction,
            estimated_max_atom_shift_A=float(atomic_norms.max()), estimated_RMS_atom_shift_A=float(np.sqrt(np.mean(atomic_norms ** 2))),
            estimated_max_shift_over_0p01_A=float(atomic_norms.max() / .01),
            abs_estimated_energy_over_quantum_delta_sensitivity=float(abs(correction) / quantum_delta_effect),
            reference_quantum_delta_sensitivity_eV=quantum_delta_effect,
            quadratic_identity_error_eV=abs(cartesian - correction),
            mass_weighted_COM_shift_norm=float(np.linalg.norm(np.sum(minimum["masses_amu"][:, None] * shift.reshape(-1, 3), axis=0))),
            minimum_source=str(minimum_path), hessian_source=str(hessian_path)))
        sources[str(minimum_path)] = sha(minimum_path); sources[str(hessian_path)] = sha(hessian_path)
    with (args.output / "hessian_audit_residual_forces.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    result = dict(status="CALCULATED_LOCAL_QUADRATIC_ESTIMATE", source_sha256=sha(__file__), inputs=sources, rows=rows,
        equations=dict(projected_force="f_k = u_k.T M^(-1/2) F_final", energy_correction_eV="DeltaE_quadratic = -1/2 sum_k f_k^2 / lambda_k", atomic_shift_A="deltaR = M^(-1/2) U (f/lambda)"),
        interpretation="Local constant-Hessian estimate around the already relaxed endpoint; not a new geometry optimization, actual model energy, rigorous anharmonic error bound, or universal force-tolerance criterion. No force target changed and no new MLIP/DFT calculation performed.",
        comparison="All three minima use existing positive537-mode Hessians at0.01Angstrom. The0.000384eV reference is basin1 harmonic vibrational free-energy sensitivity to0.01 versus0.02Angstrom at433.15K, not a statistical uncertainty bound.")
    (args.output / "hessian_audit_residual_forces.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()

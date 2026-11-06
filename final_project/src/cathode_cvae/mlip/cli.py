"""Portable single-GPU MLIP commands; no scheduler submission is performed."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .calculators import evaluate, make_calculator
from .checkpoints import fetch, sha256, verify
from .registry import MODELS


UNITS = {"energy": "eV/cell", "energy_per_atom": "eV/atom", "forces": "eV/Angstrom",
         "stress": "eV/Angstrom^3", "stress_order": "xx yy zz yz xz xy", "stress_sign": "tensile positive"}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def environment_map(explicit=None):
    if explicit:
        return Path(explicit).expanduser().resolve()
    if os.environ.get("CATHODE_CVAE_ENV_FILE"):
        return Path(os.environ["CATHODE_CVAE_ENV_FILE"]).expanduser().resolve()
    # Inspect only ancestors of cwd, not a shared filesystem search.
    for directory in [Path.cwd(), *Path.cwd().parents]:
        candidate = directory / "environments.json"
        if candidate.is_file():
            return candidate
    return None


def dispatch_if_needed(args, argv):
    if os.environ.get("CATHODE_CVAE_IN_BACKEND") or args.command == "fetch-models":
        return None
    config_path = environment_map(args.environment_map)
    if config_path is None:
        return None
    config = json.loads(config_path.read_text())
    if args.model_dir is None and config.get("model_dir"):
        args.model_dir = Path(config["model_dir"])
    interpreter = config.get("interpreters", {}).get(MODELS[args.model]["backend"])
    if interpreter is None:
        raise RuntimeError(f"No backend interpreter for {args.model} in {config_path}; rerun scripts/install.sh with MLIP backends enabled.")
    # venv executables can be symlinks to the same base Python. Resolving the
    # symlink here would incorrectly skip the isolated dependency environment.
    if Path(interpreter).absolute() == Path(sys.executable).absolute():
        return None
    if not Path(interpreter).is_file():
        raise FileNotFoundError(f"Backend interpreter does not exist: {interpreter}. Reinstall on this machine; environments are not portable directories.")
    child_env = os.environ.copy()
    child_env["CATHODE_CVAE_IN_BACKEND"] = "1"
    child_env["CATHODE_CVAE_ENV_FILE"] = str(config_path)
    if config.get("model_dir"):
        child_env.setdefault("CATHODE_CVAE_MODEL_DIR", config["model_dir"])
    return subprocess.call([interpreter, "-m", "cathode_cvae.mlip.cli", *argv], env=child_env)


def read_inputs(paths, parent_path=None):
    """Yield (id, ASE Atoms, provenance), preserving input coordinates and atom order.

    Mask NPZ occupancy=1 means target present, masks=1 means vacant. Parent
    target_indices (or historical li_indices) defines sites, never element order.
    Geometry NPZ can supply full-parent positions_A per row; absent atoms are
    removed before an ASE calculator sees them.
    """
    from ase import Atoms
    from ase.io import iread
    parent = None
    parent_hash = None
    if parent_path:
        with np.load(parent_path, allow_pickle=False) as archive:
            parent = {key: archive[key] for key in archive.files}
        parent_hash = sha256(parent_path)
        if (parent["numbers"].ndim != 1 or parent["positions_A"].shape != (len(parent["numbers"]), 3)
                or parent["cell_A"].shape != (3, 3) or not np.isfinite(parent["positions_A"]).all()
                or not np.isfinite(parent["cell_A"]).all() or np.linalg.det(parent["cell_A"]) == 0):
            raise ValueError("Invalid parent numbers/positions/cell schema or singular/non-finite parent cell.")
    seen = set()
    for path in map(Path, paths):
        source = {"path": str(path.resolve()), "sha256": sha256(path)}
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            if "parent_sha256" in data:
                expected = np.asarray(data["parent_sha256"])
                if expected.ndim != 0:
                    raise ValueError("parent_sha256 must be a scalar SHA256 string.")
                if parent_hash is None or str(expected.item()) != parent_hash:
                    raise ValueError(f"Parent SHA256 mismatch for {path}; use the exact --parent that generated these configurations.")
                source["parent_sha256"] = parent_hash
            if "offsets" in data:
                count = len(data["offsets"]) - 1
                rows = []
                for i in range(count):
                    lo, hi = data["offsets"][i:i + 2]
                    rows.append(Atoms(numbers=data["numbers"][lo:hi], positions=data["positions"][lo:hi], cell=data["cell"][i], pbc=True))
            elif "occupancy" in data or "masks" in data:
                if parent is None:
                    raise ValueError("Mask/geometry NPZ requires --parent with numbers, positions_A, cell_A and target_indices (or li_indices).")
                indices = np.asarray(parent.get("target_indices", parent.get("li_indices")), dtype=int)
                if (indices.ndim != 1 or len(set(indices.tolist())) != len(indices)
                        or not np.all((indices >= 0) & (indices < len(parent["numbers"])) )):
                    raise ValueError("Parent target indices must be a one-dimensional set.")
                occupancies = data["occupancy"] if "occupancy" in data else 1 - data["masks"]
                if occupancies.ndim != 2 or occupancies.shape[1] != len(indices) or not np.isin(occupancies, [0, 1]).all():
                    raise ValueError("Occupancies must be binary with one column per parent target site.")
                if "numbers_parent" in data and not np.array_equal(data["numbers_parent"], parent["numbers"]):
                    raise ValueError("numbers_parent does not match the supplied parent's atom identities/order.")
                for name in ["presence", "atom_presence"]:
                    if name in data and (data[name].shape != (len(occupancies), len(parent["numbers"]))
                                         or not np.isin(data[name], [0, 1]).all()):
                        raise ValueError(f"{name} must be a binary array with one row per configuration and one column per parent atom.")
                rows = []
                for i, occupancy in enumerate(occupancies):
                    keep = np.ones(len(parent["numbers"]), dtype=bool)
                    keep[indices[occupancy == 0]] = False
                    for name in ["presence", "atom_presence"]:
                        if name in data and not np.array_equal(data[name][i].astype(bool), keep):
                            raise ValueError(f"{name} disagrees with occupancy and the parent's target sites.")
                    cell = data.get("cell_A", parent["cell_A"])
                    cell = cell[i] if cell.ndim == 3 else cell
                    if cell.shape != (3, 3) or not np.isfinite(cell).all() or np.linalg.det(cell) == 0:
                        raise ValueError("Each geometry must provide a finite, nonsingular 3x3 cell_A.")
                    if "positions_A" in data or "positions_parent_A" in data:
                        positions = data.get("positions_A", data.get("positions_parent_A"))
                        positions = positions[i] if positions.ndim == 3 else positions
                    elif "displacement_A" in data:
                        # Geometry training data use displacements from matched
                        # parent fractional sites, expressed in the supplied cell.
                        fractional_parent = parent["positions_A"] @ np.linalg.inv(parent["cell_A"])
                        displacement = data["displacement_A"]
                        displacement = displacement[i] if displacement.ndim == 3 else displacement
                        positions = fractional_parent @ cell + displacement
                    else:
                        positions = parent["positions_A"]
                    if positions.shape != (len(keep), 3):
                        raise ValueError("NPZ positions_A must use full-parent atom slots.")
                    rows.append(Atoms(numbers=parent["numbers"][keep], positions=positions[keep], cell=cell, pbc=True))
            else:
                raise ValueError(f"Unsupported NPZ schema in {path}; use occupancy/masks plus --parent or ragged offsets/numbers/positions/cell.")
            ids = data.get("structure_ids", np.asarray([f"{path.stem}_{i}" for i in range(len(rows))]))
            if len(ids) != len(rows):
                raise ValueError("structure_ids length does not match structure count")
            iterator = ((str(ids[i]), atoms, {**source, "frame": i}) for i, atoms in enumerate(rows))
        else:
            iterator = ((str(atoms.info.get("structure_id", f"{path.stem}_{i}")), atoms,
                         {**source, "frame": i}) for i, atoms in enumerate(iread(str(path), index=":")))
        for identity, atoms, provenance in iterator:
            if identity in seen:
                raise ValueError(f"Repeated structure ID {identity}; provide distinct IDs rather than silently merging labels.")
            seen.add(identity)
            if len(atoms) == 0 or not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
                raise ValueError(f"Invalid input geometry {identity}")
            yield identity, atoms, provenance


def summarize(energy, forces, stress, atoms, target="Li"):
    from ase.stress import voigt_6_to_full_3x3_stress
    stress_tensor = voigt_6_to_full_3x3_stress(stress)
    return {
        "n_atoms": len(atoms), "n_target": atoms.get_chemical_symbols().count(target),
        "energy_eV": float(energy), "energy_eV_atom": float(energy / len(atoms)),
        "force_component_rms_eV_A": float(np.sqrt(np.mean(forces**2))),
        "force_max_eV_A": float(np.linalg.norm(forces, axis=1).max()),
        "stress_eV_A3": np.asarray(stress).tolist(),
        "max_abs_principal_stress_GPa": float(np.abs(np.linalg.eigvalsh(stress_tensor)).max() * 160.21766208),
        "pressure_GPa": float(-np.trace(stress_tensor) / 3 * 160.21766208),
    }


def run_calculations(args):
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write
    if args.target is None:
        args.target = "Li"
        if args.parent:
            with np.load(args.parent, allow_pickle=False) as parent:
                if "target_symbol" in parent:
                    args.target = str(parent["target_symbol"].item())
    args.output.mkdir(parents=True, exist_ok=False)
    calculator, receipt = make_calculator(args.model, args.checkpoint, args.device, args.threads, args.model_dir)
    write_json(args.output / "model_receipt.json", {**receipt, "units": UNITS,
        "cli_source_sha256": sha256(__file__), "target": args.target,
        "parent_sha256": sha256(args.parent) if args.parent else None})
    rows, successes, offsets = [], [], [0]
    numbers, positions, forces, cells, stresses, energies = [], [], [], [], [], []
    for identity, input_atoms, source in read_inputs(args.input, args.parent):
        atoms = input_atoms.copy()
        start = time.perf_counter()
        row = {"structure_id": identity, "model": args.model, "source": source, "operation": args.command}
        try:
            steps = 0
            if args.command == "relax":
                from ase.filters import FrechetCellFilter
                from ase.optimize import FIRE, LBFGS
                atoms.calc = calculator
                if args.cell == "variable":
                    subject = FrechetCellFilter(atoms, exp_cell_factor=len(atoms) / 4)
                else:
                    subject = atoms
                optimizer_type = FIRE if args.model == "uma" else LBFGS
                optimizer = optimizer_type(subject, logfile=None)
                # Inspect physical force/stress criteria after each accepted step.
                # This avoids confusing the cell filter's generalized force with
                # an actual atom force or a Cauchy stress tolerance.
                for steps in range(args.steps + 1):
                    energy, force, stress = evaluate(calculator, atoms)
                    metrics = summarize(energy, force, stress, atoms, args.target)
                    force_ok = metrics["force_max_eV_A"] < args.fmax
                    stress_ok = metrics["max_abs_principal_stress_GPa"] < args.stress_gpa
                    if force_ok and (args.cell == "fixed" or stress_ok):
                        break
                    if steps < args.steps:
                        optimizer.run(fmax=0.0, steps=1)
                row.update(cell_mode=args.cell, optimizer=optimizer_type.__name__, steps=steps,
                    force_target_eV_A=args.fmax, stress_target_GPa=args.stress_gpa,
                    numerically_converged=bool(force_ok and (args.cell == "fixed" or stress_ok)),
                    geometry_source="MLIP minimization; convergence does not establish a global minimum")
            else:
                energy, force, stress = evaluate(calculator, atoms)
                metrics = summarize(energy, force, stress, atoms, args.target)
            row.update(status="success", elapsed_seconds=time.perf_counter() - start, **metrics)
            exported = atoms.copy()
            exported.info.update(structure_id=identity, model=args.model, operation=args.command)
            exported.calc = SinglePointCalculator(exported, energy=energy, forces=force, stress=stress)
            successes.append(exported)
            numbers.append(atoms.numbers.copy()); positions.append(atoms.positions.copy()); forces.append(force)
            cells.append(atoms.cell.array.copy()); stresses.append(stress); energies.append(energy)
            offsets.append(offsets[-1] + len(atoms))
        except Exception as error:
            row.update(status="failed", error_type=type(error).__name__, error=str(error), elapsed_seconds=time.perf_counter() - start)
        rows.append(row)
        # Keep a durable per-attempt record even when later structures fail.
        with (args.output / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if row["status"] == "failed" and args.fail_fast:
            break
    successful_rows = [row for row in rows if row["status"] == "success"]
    if successes:
        write(str(args.output / "predictions.extxyz"), successes, format="extxyz")
        output_arrays = dict(
            structure_ids=np.asarray([row["structure_id"] for row in successful_rows]),
            numbers=np.concatenate(numbers), positions=np.concatenate(positions), forces=np.concatenate(forces),
            offsets=np.asarray(offsets, dtype=np.int64), cell=np.asarray(cells), stress=np.asarray(stresses), energy=np.asarray(energies),
            energy_eV=np.asarray(energies), forces_eV_A=np.concatenate(forces), stress_eV_A3=np.asarray(stresses),
            n_atoms=np.asarray([row["n_atoms"] for row in successful_rows], dtype=np.int64),
            n_target=np.asarray([row["n_target"] for row in successful_rows], dtype=np.int64))
        if args.target == "Li":
            output_arrays["n_Li"] = output_arrays["n_target"]
        np.savez_compressed(args.output / "predictions.npz", **output_arrays)
        scalar_keys = ["structure_id", "model", "operation", "status", "n_atoms", "n_target", "energy_eV", "energy_eV_atom",
                       "force_component_rms_eV_A", "force_max_eV_A", "max_abs_principal_stress_GPa", "pressure_GPa", "elapsed_seconds"]
        with (args.output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=scalar_keys, extrasaction="ignore")
            writer.writeheader(); writer.writerows(successful_rows)
    failed = len(rows) - len(successes)
    receipt = {"operation": args.command, "attempts": len(rows), "successes": len(successes), "failures": failed,
               "units": UNITS, "status": "complete" if rows and failed == 0 else "incomplete",
               "no_new_DFT": True, "physical_validation": "Not established by a successful API call."}
    if args.command == "relax":
        receipt["numerically_converged"] = sum(row.get("numerically_converged", False) for row in successful_rows)
        receipt["caps"] = len(successes) - receipt["numerically_converged"]
    write_json(args.output / "run_receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if rows and failed == 0 else 1


def doctor(args):
    spec = MODELS[args.model]
    info = {"model": args.model, "python": sys.version.split()[0], "interpreter": sys.executable,
            "required_package": spec["package"], "required_version": spec["version"], "head": spec["head"]}
    try:
        info["installed_version"] = metadata.version(spec["package"])
    except metadata.PackageNotFoundError:
        info["installed_version"] = None
    info["package_version_ok"] = info["installed_version"] == spec["version"]
    # pip check verifies transitive requirements, rather than only the top-level import.
    checked = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
    info["pip_check_ok"] = checked.returncode == 0
    info["pip_check_output"] = (checked.stdout + checked.stderr).strip()
    try:
        path, hashes = verify(args.model, args.checkpoint, args.model_dir)
        info.update(checkpoint=str(path), checkpoint_hashes=hashes, checkpoint_ok=True)
    except (FileNotFoundError, ValueError) as error:
        info.update(checkpoint_ok=False, checkpoint_error=str(error))
    if args.load:
        _, loaded = make_calculator(args.model, args.checkpoint, args.device, args.threads, args.model_dir)
        info["loaded"] = loaded
    print(json.dumps(info, indent=2))
    return 0 if info["package_version_ok"] and info["pip_check_ok"] and info["checkpoint_ok"] else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch_parser = subparsers.add_parser("fetch-models", help="Download hash-checked, pinned upstream checkpoints.")
    fetch_parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    fetch_parser.add_argument("--model-dir", type=Path)
    fetch_parser.add_argument("--environment-map", type=Path)
    for command in ["doctor", "score", "relax"]:
        child = subparsers.add_parser(command)
        child.add_argument("--model", required=True, choices=list(MODELS))
        child.add_argument("--checkpoint", type=Path, help="An alternate location of the same hash-checked pinned checkpoint.")
        child.add_argument("--model-dir", type=Path)
        child.add_argument("--device", default="cuda", help="cpu, cuda, or cuda:N in the process-visible GPU list")
        child.add_argument("--threads", type=int, default=4)
        child.add_argument("--environment-map", type=Path)
        if command == "doctor":
            child.add_argument("--load", action="store_true", help="Also load the model; on HPC run this inside an allocation.")
        else:
            child.add_argument("--input", type=Path, nargs="+", required=True, help="POSCAR/CIF/extxyz or NPZ files; input coordinates are preserved.")
            child.add_argument("--parent", type=Path, help="Parent NPZ for occupancy/mask inputs.")
            child.add_argument("--output", type=Path, required=True, help="New output directory; existing results are never overwritten.")
            child.add_argument("--target", help="Removed species for n_target; defaults to parent target_symbol, otherwise Li.")
            child.add_argument("--fail-fast", action="store_true")
            if command == "relax":
                child.add_argument("--cell", choices=["fixed", "variable"], default="fixed")
                child.add_argument("--fmax", type=float, default=0.05, help="Provisional atom-force criterion, eV/Angstrom.")
                child.add_argument("--stress-gpa", type=float, default=0.1, help="Provisional max absolute principal Cauchy stress criterion, GPa; variable cell only.")
                child.add_argument("--steps", type=int, default=500)
    args = parser.parse_args(argv)
    dispatched = dispatch_if_needed(args, argv)
    if dispatched is not None:
        return dispatched
    if args.command == "fetch-models":
        config_path = environment_map(args.environment_map)
        if args.model_dir is None and config_path:
            args.model_dir = json.loads(config_path.read_text()).get("model_dir")
        for name in args.models:
            print(f"{name}: {fetch(name, args.model_dir)}", flush=True)
        return 0
    if args.command == "doctor":
        return doctor(args)
    if args.command == "relax" and (args.fmax <= 0 or args.stress_gpa <= 0 or args.steps < 0):
        parser.error("Force/stress criteria must be positive and step count nonnegative.")
    return run_calculations(args)


if __name__ == "__main__":
    raise SystemExit(main())

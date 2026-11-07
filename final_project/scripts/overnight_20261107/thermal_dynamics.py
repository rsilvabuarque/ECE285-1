"""Fixed-cell, same-UMA numerical pilot; this script never submits a job.

All energies are total-cell eV. Coordinates are Angstrom, forces eV/Angstrom,
and ASE stress is tensile-positive eV/Angstrom^3 in xx,yy,zz,yz,xz,xy order.
Raw finite differences and negative eigenvalues are retained for diagnosis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import traceback

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def json_paths(value):
    """Preserve nested CLI structure while converting filesystem paths to text."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_paths(item) for item in value]
    return value


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_structure(args):
    from ase import Atoms
    parent = load_npz(args.parent)
    if args.input is not None:
        data = load_npz(args.input)
        for key in ["positions_A", "cell_A", "numbers", "parent_indices"]:
            if key not in data:
                raise ValueError(f"Structure NPZ is missing {key}")
        positions = data["positions_A"]
        if positions.ndim != 2:
            raise ValueError("Input must be one structure, not a trajectory")
        numbers, cell, indices = data["numbers"], data["cell_A"], data["parent_indices"]
        if "parent_sha256" in data and str(data["parent_sha256"]) != digest(args.parent):
            raise ValueError("Input structure belongs to a different parent")
        if len(set(map(int, indices))) != len(indices):
            raise ValueError("Duplicate original parent indices")
        if not np.array_equal(numbers, parent["numbers"][indices]):
            raise ValueError("Atomic species do not match preserved parent indices")
    else:
        text = args.vacancy_mask
        if text is None or len(text) != len(parent["li_indices"]) or set(text) - {"0", "1"}:
            raise ValueError("Provide a full vacancy bitstring; 1 means Li removed")
        indices = np.arange(len(parent["numbers"]))
        present = np.ones(len(indices), bool)
        present[parent["li_indices"][np.array(list(text), dtype=int).astype(bool)]] = False
        indices = indices[present]
        positions, numbers, cell = parent["positions_A"][indices], parent["numbers"][indices], parent["cell_A"]
    if not np.allclose(cell, parent["cell_A"], rtol=0, atol=1e-10):
        raise ValueError("This pilot requires the common, unchanged parent cell")
    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    if "masses_amu" in (data if args.input is not None else {}):
        atoms.set_masses(data["masses_amu"])
    return atoms, parent, np.asarray(indices, dtype=int)


def mapping_diagnostics(atoms, parent, indices):
    """Assignment is geometric; a changed assignment alone does not prove a hop.

    A rectangular Hungarian assignment maps present Li atoms to distinct parent
    Li sites. Nearest/next-nearest gaps and collisions expose ambiguous mappings.
    Host mean displacement removes an arbitrary common translation for mapping.
    """
    from ase.geometry import find_mic
    from scipy.optimize import linear_sum_assignment
    cell = atoms.cell.array
    is_li = np.isin(indices, parent["li_indices"])
    host = ~is_li
    displacement, _ = find_mic(atoms.positions - parent["positions_A"][indices], cell, pbc=True)
    shift = displacement[host].mean(axis=0) if np.any(host) else np.zeros(3)
    aligned, _ = find_mic(displacement - shift, cell, pbc=True)
    li_positions = atoms.positions[is_li] - shift
    parent_li = parent["positions_A"][parent["li_indices"]]
    vectors = li_positions[:, None, :] - parent_li[None, :, :]
    _, flat_distance = find_mic(vectors.reshape(-1, 3), cell, pbc=True)
    distances = flat_distance.reshape(len(li_positions), len(parent_li))
    row, col = linear_sum_assignment(distances ** 2)
    assignment = np.empty(len(li_positions), int)
    assignment[row] = col
    nearest = distances.argmin(axis=1)
    sorted_distances = np.sort(distances, axis=1)
    occupancy = np.zeros(len(parent_li), np.uint8)
    occupancy[assignment] = 1
    initial_occupancy = np.isin(parent["li_indices"], indices).astype(np.uint8)
    # The reciprocal normal measures relative Li/host layer shifts even if c is tilted.
    c_normal = np.cross(cell[0], cell[1]); c_normal /= np.linalg.norm(c_normal)
    host_norm = np.linalg.norm(aligned[host], axis=1)
    return dict(
        occupancy_hungarian=occupancy.tolist(),
        vacancy_mask_hungarian="".join(map(str, 1 - occupancy)),
        original_occupancy=initial_occupancy.tolist(),
        changed_occupancy_sites=int(np.count_nonzero(occupancy != initial_occupancy)),
        assigned_parent_li_sites=assignment.tolist(),
        assignment_distances_A=distances[np.arange(len(assignment)), assignment].tolist(),
        nearest_parent_li_sites=nearest.tolist(),
        nearest_distances_A=sorted_distances[:, 0].tolist(),
        nearest_second_gap_A=(sorted_distances[:, 1] - sorted_distances[:, 0]).tolist(),
        nearest_collision_count=int(len(nearest) - len(set(map(int, nearest)))),
        hungarian_differs_from_nearest=int(np.count_nonzero(assignment != nearest)),
        host_translation_A=shift.tolist(),
        host_displacement_rms_A=float(np.sqrt(np.mean(host_norm ** 2))),
        host_displacement_max_A=float(host_norm.max()),
        host_layer_normal_rms_A=float(np.sqrt(np.mean((aligned[host] @ c_normal) ** 2))),
        li_relative_layer_normal_rms_A=float(np.sqrt(np.mean((aligned[is_li] @ c_normal) ** 2))),
        interpretation="Assignment changes and small nearest-site margins require inspection; they are not conclusive hopping events",
    )


def structure_arrays(atoms, indices, parent_path, energy, force, stress):
    return dict(positions_A=atoms.positions.copy(), cell_A=atoms.cell.array.copy(),
                numbers=atoms.numbers.copy(), masses_amu=atoms.get_masses(), parent_indices=indices,
                parent_sha256=np.array(digest(parent_path)), energy_eV=np.array(energy),
                forces_eV_A=np.asarray(force), stress_eV_A3=np.asarray(stress))


def calculator(args):
    from cathode_cvae.mlip.calculators import make_calculator
    return make_calculator("uma", checkpoint=args.checkpoint, device=args.device, threads=args.threads)


def evaluate(atoms):
    energy = float(atoms.get_potential_energy())
    force = atoms.get_forces(apply_constraint=False)
    stress = atoms.get_stress()
    if not all(np.all(np.isfinite(x)) for x in [energy, force, stress]):
        raise FloatingPointError("Nonfinite energy, force, or stress")
    return energy, force, stress


def fresh_precision_repeats(atoms, output, repeats=3):
    """Repeated fresh predictions measure reproducibility, not DFT accuracy.

    Reset the ASE result cache before each prediction so identical cached arrays
    cannot create a false zero-noise result. This does not bound finite precision
    bias or the finite-difference truncation error.
    """
    values = []
    for _ in range(repeats):
        atoms.calc.reset()
        values.append(evaluate(atoms))
    energies = np.asarray([x[0] for x in values])
    forces = np.asarray([x[1] for x in values])
    stresses = np.asarray([x[2] for x in values])
    np.savez_compressed(output / "precision_repeats.npz", energy_eV=energies,
                        forces_eV_A=forces, stress_eV_A3=stresses,
                        positions_A=atoms.positions, cell_A=atoms.cell.array)
    stats = dict(fresh_predictions=repeats, cache_reset_each_repeat=True,
                 energy_span_eV=float(np.ptp(energies)),
                 force_component_max_span_eV_A=float(np.max(np.ptp(forces, axis=0))),
                 stress_component_max_span_eV_A3=float(np.max(np.ptp(stresses, axis=0))),
                 net_force_norm_eV_A=np.linalg.norm(forces.sum(axis=1), axis=1).tolist(),
                 interpretation="Repeated inference reproducibility only; zero span does not establish force derivative accuracy")
    return values[0], stats


def start_record(args):
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(status="RUNNING", command=args.command, basin_id=args.basin_id,
                  started_unix=time.time(), parent_sha256=digest(args.parent),
                  source_sha256=digest(__file__), arguments=json_paths(vars(args)),
                  host=socket.gethostname(), job_id=os.environ.get("SLURM_JOB_ID"),
                  gpu_visibility=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  fixed_cell=True, energy_units="eV/periodic_cell", stress_units="eV/A^3; tensile positive; xx yy zz yz xz xy")
    if args.input is not None:
        record["input_sha256"] = digest(args.input)
    write_json(args.output / "receipt.json", record)
    return record


def relax(args, record):
    from ase.io import write
    from ase.optimize import LBFGS
    atoms, parent, indices = load_structure(args)
    atoms.calc, record["teacher"] = calculator(args)
    initial = evaluate(atoms)
    np.savez_compressed(args.output / "initial.npz", **structure_arrays(atoms, indices, args.parent, *initial))
    write(args.output / "initial.POSCAR", atoms, format="vasp", direct=True, sort=False)
    initial_cell = atoms.cell.array.copy()
    optimizer = LBFGS(atoms, logfile=str(args.output / "relax.log"),
                      trajectory=str(args.output / "relax.traj"), maxstep=args.maxstep_A)
    t0 = time.perf_counter()
    converged = bool(optimizer.run(fmax=args.fmax_eV_A, steps=args.max_steps))
    final = evaluate(atoms)
    if not np.array_equal(initial_cell, atoms.cell.array):
        raise RuntimeError("Unexpected cell change in fixed-cell relaxation")
    np.savez_compressed(args.output / "final.npz", **structure_arrays(atoms, indices, args.parent, *final))
    write(args.output / "final.POSCAR", atoms, format="vasp", direct=True, sort=False)
    write(args.output / "final.cif", atoms)
    record.update(status="CONVERGED" if converged else "MAX_STEPS_NOT_CONVERGED", converged=converged,
                  steps=optimizer.nsteps, relaxation_seconds=time.perf_counter() - t0,
                  initial_energy_eV=initial[0], final_energy_eV=final[0],
                  initial_max_force_eV_A=float(np.linalg.norm(initial[1], axis=1).max()),
                  final_max_force_eV_A=float(np.linalg.norm(final[1], axis=1).max()),
                  initial_mapping=mapping_diagnostics(type(atoms)(numbers=atoms.numbers, positions=load_npz(args.output / "initial.npz")["positions_A"], cell=atoms.cell, pbc=True), parent, indices),
                  final_mapping=mapping_diagnostics(atoms, parent, indices),
                  tolerance_interpretation="Provisional force target; finite-difference and precision sensitivity must be checked before interpreting phonons")


def hessian(args, record):
    atoms, parent, indices = load_structure(args)
    atoms.calc, record["teacher"] = calculator(args)
    if args.delta_A <= 0 or args.shards < 1 or not 0 <= args.shard_index < args.shards:
        raise ValueError("Positive displacement and valid shard index required")
    original = atoms.positions.copy()
    n = len(atoms); ndof = 3 * n
    columns = np.arange(ndof)[args.shard_index::args.shards]
    baseline, record["force_precision_repeats"] = fresh_precision_repeats(atoms, args.output)
    plus = np.empty((len(columns), n, 3)); minus = np.empty_like(plus)
    energies = np.empty((len(columns), 2)); timing = []
    t0 = time.perf_counter()
    for i, column in enumerate(columns):
        before = time.perf_counter()
        for sign, target, slot in [(1, plus, 0), (-1, minus, 1)]:
            positions = original.copy(); positions.flat[column] += sign * args.delta_A
            atoms.set_positions(positions)
            e, f, _ = evaluate(atoms)
            target[i] = f; energies[i, slot] = e
        timing.append(time.perf_counter() - before)
        # Each completed column is restartable evidence, even if a job stops early.
        np.savez_compressed(args.output / f"column_{column:04d}.npz", column=column,
                            forces_plus_eV_A=plus[i], forces_minus_eV_A=minus[i], energies_eV=energies[i],
                            delta_A=args.delta_A, input_sha256=np.array(digest(args.input)))
        if (i + 1) % 10 == 0:
            print(json.dumps(dict(completed_columns=i + 1, columns=len(columns), elapsed_s=time.perf_counter() - t0)), flush=True)
    atoms.set_positions(original)
    raw = -(plus.reshape(len(columns), ndof) - minus.reshape(len(columns), ndof)).T / (2 * args.delta_A)
    np.savez_compressed(args.output / "hessian_columns.npz", columns=columns, hessian_columns_eV_A2=raw,
                        forces_plus_eV_A=plus, forces_minus_eV_A=minus, energies_eV=energies,
                        baseline_forces_eV_A=baseline[1], baseline_energy_eV=baseline[0],
                        delta_A=args.delta_A, parent_sha256=np.array(digest(args.parent)),
                        input_sha256=np.array(digest(args.input)), masses_amu=atoms.get_masses(),
                        positions_A=original, cell_A=atoms.cell.array, numbers=atoms.numbers,
                        parent_indices=indices, column_seconds=np.array(timing))
    record.update(status="COMPLETE", completed_columns=len(columns), ndof=ndof,
                  displacement_A=args.delta_A, hessian_seconds=time.perf_counter() - t0,
                  baseline_max_force_eV_A=float(np.linalg.norm(baseline[1], axis=1).max()),
                  finite_difference="H_ij=-(F_i(R+delta e_j)-F_i(R-delta e_j))/(2delta); no acoustic sum correction or symmetrization in shard")


def analyze_hessian(raw, masses):
    """Project translations in mass-weighted coordinates; never discard a sign."""
    n = len(masses); ndof = 3 * n
    if raw.shape != (ndof, ndof) or np.any(np.asarray(masses) <= 0) or not np.isfinite(raw).all():
        raise ValueError("Finite square Hessian and positive masses required")
    anti = raw - raw.T
    symmetric = (raw + raw.T) / 2
    invsqrt = np.repeat(1 / np.sqrt(masses), 3)
    dynamical = symmetric * invsqrt[:, None] * invsqrt[None, :]
    translations = np.zeros((ndof, 3))
    for axis in range(3):
        translations[axis::3, axis] = np.sqrt(masses) / np.sqrt(np.sum(masses))
    # Complete QR gives an orthonormal internal basis without choosing a frequency cutoff.
    basis = np.linalg.qr(translations, mode="complete")[0][:, 3:]
    reduced = basis.T @ dynamical @ basis
    values, vectors = np.linalg.eigh(reduced)
    # sqrt(eV/(Angstrom^2 amu)) in rad/s, then /2pi and /1e12 -> THz.
    ev_J, amu_kg = 1.602176634e-19, 1.66053906660e-27
    factor = np.sqrt(ev_J / (1e-20 * amu_kg)) / (2 * np.pi * 1e12)
    signed_frequency = np.sign(values) * np.sqrt(np.abs(values)) * factor
    arrays = dict(hessian_raw_eV_A2=raw, hessian_sym_eV_A2=symmetric,
                  antisymmetric_difference_eV_A2=anti, dynamical_matrix_eV_A2_amu=dynamical,
                  translation_basis=translations, internal_basis=basis,
                  eigenvalues_full_eV_A2_amu=np.linalg.eigvalsh(dynamical),
                  eigenvalues_internal_eV_A2_amu=values, eigenvectors_mass_weighted=basis @ vectors,
                  frequencies_signed_THz=signed_frequency)
    stats = dict(total_modes=ndof, removed_translation_modes=3, retained_internal_modes=ndof - 3,
                 raw_antisymmetric_frobenius_eV_A2=float(np.linalg.norm(anti)),
                 raw_antisymmetric_relative_frobenius=float(np.linalg.norm(anti) / max(np.linalg.norm(raw), np.finfo(float).tiny)),
                 raw_antisymmetric_max_eV_A2=float(np.max(np.abs(anti))),
                 translation_residual_norm_eV_A2_amu=float(np.linalg.norm(dynamical @ translations)),
                 most_negative_eigenvalue_eV_A2_amu=float(values.min()),
                 strict_negative_internal_modes=int(np.sum(values < 0)),
                 signed_frequency_min_THz=float(signed_frequency.min()),
                 signed_frequency_max_THz=float(signed_frequency.max()),
                 stability_interpretation="All signed internal eigenvalues retained; negative values require displacement/force precision checks, not automatic absolute-value replacement",
                 mode_convention="Gamma modes of the full finite periodic cell, matching MD cell; three mass-weighted uniform translations projected separately")
    return arrays, stats


def assemble(args, record):
    atoms, parent, indices = load_structure(args)
    ndof = 3 * len(atoms); matrix = np.empty((ndof, ndof)); seen = set(); deltas = set(); sources = []
    teacher_records = []
    for directory in args.shard:
        path = directory / "hessian_columns.npz" if directory.is_dir() else directory
        data = load_npz(path); sources.append(dict(path=str(path), sha256=digest(path)))
        if str(data["input_sha256"]) != digest(args.input) or str(data["parent_sha256"]) != digest(args.parent):
            raise ValueError("Hessian shard geometry/parent mismatch")
        if not np.array_equal(data["masses_amu"], atoms.get_masses()):
            raise ValueError("Hessian shard mass mismatch")
        columns = data["columns"].astype(int)
        if any(int(x) in seen or x < 0 or x >= ndof for x in columns):
            raise ValueError("Duplicate or invalid Hessian columns")
        seen.update(map(int, columns)); matrix[:, columns] = data["hessian_columns_eV_A2"]
        deltas.add(float(data["delta_A"]))
        receipt = json.loads((path.parent / "receipt.json").read_text())
        if receipt["status"] != "COMPLETE":
            raise ValueError("Hessian shard is not complete")
        teacher_records.append(receipt["teacher"])
    if len(seen) != ndof or len(deltas) != 1:
        raise ValueError(f"Need all {ndof} columns at one displacement; got {len(seen)}")
    contexts = {(x["checkpoint_name"], x["head"], json.dumps(x["checkpoint_hashes"], sort_keys=True), x["dtype"], x["tf32"]) for x in teacher_records}
    if len(contexts) != 1:
        raise ValueError("Mixed Hessian teacher or precision contexts")
    arrays, stats = analyze_hessian(matrix, atoms.get_masses())
    np.savez_compressed(args.output / "hessian.npz", **arrays, positions_A=atoms.positions,
                        cell_A=atoms.cell.array, numbers=atoms.numbers, masses_amu=atoms.get_masses(),
                        parent_indices=indices, delta_A=next(iter(deltas)),
                        parent_sha256=np.array(digest(args.parent)), input_sha256=np.array(digest(args.input)))
    record.update(status="COMPLETE", teacher=teacher_records[0], sources=sources, displacement_A=next(iter(deltas)), **stats)
    with (args.output / "signed_frequencies.csv").open("w") as stream:
        writer = csv.writer(stream); writer.writerow(["mode", "eigenvalue_eV_A2_amu", "frequency_signed_THz"])
        writer.writerows(zip(range(ndof - 3), arrays["eigenvalues_internal_eV_A2_amu"], arrays["frequencies_signed_THz"]))


def md(args, record):
    from ase import units
    from ase.constraints import FixCom
    from ase.io.trajectory import Trajectory
    from ase.md.bussi import Bussi
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
    from ase.md.verlet import VelocityVerlet
    atoms, parent, indices = load_structure(args)
    atoms.calc, record["teacher"] = calculator(args)
    if min(args.timestep_fs, args.save_interval_fs, args.temperature_K, args.taut_fs) <= 0:
        raise ValueError("Positive MD times, target temperature and thermostat time required")
    save_steps = int(round(args.save_interval_fs / args.timestep_fs))
    if not np.isclose(save_steps * args.timestep_fs, args.save_interval_fs):
        raise ValueError("Save interval must be an integer number of steps")
    phases = [("equilibration", args.equilibration_ps, "nvt"), ("production", args.production_ps, args.production_ensemble)]
    if any(duration < 0 for _, duration, _ in phases) or args.production_ps <= 0:
        raise ValueError("Nonnegative equilibration and positive production required")
    atoms.set_constraint(FixCom())
    rng = np.random.default_rng(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_K, rng=rng, force_temp=False)
    Stationary(atoms, preserve_temperature=False)
    initial_cell = atoms.cell.array.copy(); minimum_energy = float(load_npz(args.input)["energy_eV"])
    record.update(seed=args.seed, repeat_id=args.repeat_id, target_temperature_K=args.temperature_K,
                  timestep_fs=args.timestep_fs, saved_interval_fs=args.save_interval_fs,
                  constrained_dof=3, COM_removed=True, active_degrees_of_freedom=3 * len(atoms) - 3,
                  velocity_units="ASE internal; velocities_A_ps = velocities_ASE * ase.units.fs * 1000",
                  velocity_conversion_ASE_to_A_ps=float(units.fs * 1000),
                  velocity_interval_fs=args.timestep_fs, detailed_interval_fs=args.save_interval_fs,
                  velocity_energy_source="Current cached calculator potential energy, asserted unchanged geometry; kinetic energy from momenta; no extra MLIP inference",
                  relaxed_minimum_energy_eV=minimum_energy,
                  thermostat="Bussi stochastic velocity rescaling", thermostat_time_fs=args.taut_fs,
                  initial_mapping=mapping_diagnostics(atoms, parent, indices), phases={})
    global_step = 0
    for phase, duration, ensemble in phases:
        steps = int(round(duration * 1000 / args.timestep_fs))
        if not np.isclose(steps * args.timestep_fs, duration * 1000):
            raise ValueError("Phase duration must be an integer number of steps")
        dyn = Bussi(atoms, timestep=args.timestep_fs * units.fs, temperature_K=args.temperature_K,
                    taut=args.taut_fs * units.fs, rng=rng) if ensemble == "nvt" else VelocityVerlet(atoms, timestep=args.timestep_fs * units.fs)
        rows = {k: [] for k in ["time_fs", "positions_A", "detailed_velocities_ASE", "potential_eV", "kinetic_eV", "energy_total_eV", "temperature_K", "stress_eV_A3", "forces_eV_A", "max_force_eV_A", "host_rms_A", "host_max_A", "changed_occupancy_sites", "assignment_distance_max_A", "nearest_margin_min_A", "nearest_collisions", "COM_velocity_ASE"]}
        velocity_rows = {k: [] for k in ["velocity_time_fs", "velocities_ASE", "velocity_potential_eV", "velocity_kinetic_eV", "velocity_energy_total_eV", "velocity_temperature_K"]}
        trajectory = Trajectory(str(args.output / f"{phase}.traj"), "w", atoms)
        trace_stream = (args.output / f"{phase}_trace.csv").open("w", buffering=1)
        mapping_stream = (args.output / f"{phase}_mapping.jsonl").open("w", buffering=1)
        writer = csv.writer(trace_stream); writer.writerow(["step", "time_fs", "potential_eV", "kinetic_eV", "total_eV", "temperature_K", "max_force_eV_A", "host_rms_A", "changed_occupancy_sites", "assignment_distance_max_A", "nearest_margin_min_A"])
        start = time.perf_counter()
        def sample():
            e, f, stress = evaluate(atoms); kinetic = float(atoms.get_kinetic_energy())
            temp = 2 * kinetic / ((3 * len(atoms) - 3) * units.kB)
            mapping = mapping_diagnostics(atoms, parent, indices)
            vmax = float(np.linalg.norm(f, axis=1).max())
            if not np.array_equal(atoms.cell.array, initial_cell):
                raise RuntimeError("MD cell unexpectedly changed")
            native = atoms.get_velocities().copy()
            comv = np.average(native, axis=0, weights=atoms.get_masses())
            data = dict(time_fs=dyn.nsteps * args.timestep_fs, positions_A=atoms.positions.copy(), detailed_velocities_ASE=native,
                        potential_eV=e, kinetic_eV=kinetic, energy_total_eV=e + kinetic, temperature_K=temp,
                        stress_eV_A3=stress.copy(), forces_eV_A=f.copy(), max_force_eV_A=vmax,
                        host_rms_A=mapping["host_displacement_rms_A"], host_max_A=mapping["host_displacement_max_A"],
                        changed_occupancy_sites=mapping["changed_occupancy_sites"],
                        assignment_distance_max_A=max(mapping["assignment_distances_A"]),
                        nearest_margin_min_A=min(mapping["nearest_second_gap_A"]), nearest_collisions=mapping["nearest_collision_count"], COM_velocity_ASE=comv)
            for key, value in data.items(): rows[key].append(value)
            trajectory.write(atoms)
            mapping_stream.write(json.dumps(dict(step=dyn.nsteps, time_fs=data["time_fs"], **mapping)) + "\n")
            writer.writerow([dyn.nsteps, data["time_fs"], e, kinetic, e + kinetic, temp, vmax, data["host_rms_A"], data["changed_occupancy_sites"], data["assignment_distance_max_A"], data["nearest_margin_min_A"]])
            if not np.isfinite(temp) or temp > args.abort_temperature_K or vmax > args.abort_force_eV_A:
                raise FloatingPointError(f"MD diagnostic stop: T={temp}K, Fmax={vmax}eV/A; triggering detailed frame preserved")
            if len(rows["time_fs"]) % 100 == 0:
                print(json.dumps(dict(phase=phase, step=dyn.nsteps, steps=steps, elapsed_s=time.perf_counter() - start, T_K=temp)), flush=True)
        def sample_velocities():
            # Verlet/Bussi have already evaluated forces at these positions.
            # Reading an asserted-current energy result adds no prediction calls.
            if atoms.calc.check_state(atoms) or "energy" not in atoms.calc.results:
                raise RuntimeError("Full-rate velocity energy anchor has no current cached potential; refusing hidden inference")
            potential = float(atoms.calc.results["energy"])
            kinetic = float(atoms.get_kinetic_energy())
            data = dict(velocity_time_fs=dyn.nsteps * args.timestep_fs,
                        velocities_ASE=atoms.get_velocities().copy(), velocity_potential_eV=potential,
                        velocity_kinetic_eV=kinetic, velocity_energy_total_eV=potential + kinetic,
                        velocity_temperature_K=2 * kinetic / ((3 * len(atoms) - 3) * units.kB))
            if not all(np.all(np.isfinite(v)) for v in data.values()):
                raise FloatingPointError("Nonfinite full-rate velocity/energy sample")
            for key, value in data.items(): velocity_rows[key].append(value)
        # Detailed callback first supplies the phase-zero energy cache.
        dyn.attach(sample, interval=save_steps)
        dyn.attach(sample_velocities, interval=1)
        failure = None
        try:
            dyn.run(steps)
        except Exception as error:
            failure = error
        finally:
            trajectory.close(); trace_stream.close(); mapping_stream.close()
            arrays = {key: np.asarray(value) for key, value in rows.items()}
            arrays.update({key: np.asarray(value) for key, value in velocity_rows.items()})
            arrays["velocities_A_ps"] = arrays["velocities_ASE"] * units.fs * 1000
            arrays["detailed_velocities_A_ps"] = arrays["detailed_velocities_ASE"] * units.fs * 1000
            if failure is not None:
                # Includes an endpoint that failed before a regular observer,
                # even if its model evaluation was nonfinite or unavailable.
                np.savez_compressed(args.output / f"{phase}_failure_state.npz", positions_A=atoms.positions,
                                    cell_A=atoms.cell.array, numbers=atoms.numbers, masses_amu=atoms.get_masses(),
                                    velocities_ASE=atoms.get_velocities(), parent_indices=indices,
                                    phase_step=dyn.nsteps, time_fs=dyn.nsteps * args.timestep_fs)
            np.savez_compressed(args.output / f"{phase}.npz", **arrays, cell_A=atoms.cell.array, numbers=atoms.numbers,
                                masses_amu=atoms.get_masses(), parent_indices=indices, parent_sha256=np.array(digest(args.parent)),
                                basin_id=np.array(args.basin_id), repeat_id=np.array(args.repeat_id),
                                timestep_fs=args.timestep_fs, saved_interval_fs=args.save_interval_fs,
                                velocity_interval_fs=args.timestep_fs, detailed_interval_fs=args.save_interval_fs,
                                constrained_dof=3, COM_removed=True, target_temperature_K=args.temperature_K,
                                relaxed_minimum_energy_eV=minimum_energy)
        if failure is not None: raise failure
        elapsed = time.perf_counter() - start
        total = arrays["energy_total_eV"]; times_ps = arrays["time_fs"] / 1000
        slope = float(np.polyfit(times_ps, total, 1)[0]) if len(total) > 1 else 0.
        record["phases"][phase] = dict(ensemble=ensemble, steps=steps, saved_frames=len(total), duration_ps=duration,
            elapsed_seconds=elapsed, seconds_per_step=elapsed / max(steps, 1),
            mean_total_energy_eV=float(np.mean(total)), mean_potential_energy_eV=float(np.mean(arrays["potential_eV"])),
            mean_kinetic_energy_eV=float(np.mean(arrays["kinetic_eV"])), mean_temperature_K=float(np.mean(arrays["temperature_K"])),
            temperature_std_K=float(np.std(arrays["temperature_K"])), total_energy_linear_drift_eV_ps=slope,
            total_energy_linear_drift_eV_atom_ps=slope / len(atoms), total_energy_span_eV=float(np.ptp(total)),
            maximum_force_eV_A=float(np.max(arrays["max_force_eV_A"])),
            changed_occupancy_observed=bool(np.any(arrays["changed_occupancy_sites"])),
            maximum_COM_speed_ASE=float(np.linalg.norm(arrays["COM_velocity_ASE"], axis=1).max()),
            velocity_saved_frames=len(arrays["velocity_time_fs"]),
            full_rate_mean_total_energy_eV=float(np.mean(arrays["velocity_energy_total_eV"])),
            full_rate_mean_temperature_K=float(np.mean(arrays["velocity_temperature_K"])),
            mean_energy_frame_convention="mean_total_energy_eV uses detailed2fs frames; full_rate_mean_total_energy_eV uses exactly velocity_time_fs frames including phase time zero; downstream must recompute on its analyzed window")
        write_json(args.output / "receipt.json", record)
        global_step += steps
    final = evaluate(atoms)
    np.savez_compressed(args.output / "final.npz", **structure_arrays(atoms, indices, args.parent, *final), velocities_ASE=atoms.get_velocities())
    record.update(status="COMPLETE", final_mapping=mapping_diagnostics(atoms, parent, indices),
                  interpretation="Finite fixed-cell same-MLIP trajectory; thermostat/step-size/block/seed and basin-retention checks are required before thermodynamic interpretation")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["relax", "hessian", "assemble-hessian", "md"]:
        p = sub.add_parser(name); p.add_argument("--parent", type=Path, required=True)
        p.add_argument("--input", type=Path); p.add_argument("--vacancy-mask")
        p.add_argument("--basin-id", default="unspecified"); p.add_argument("--output", type=Path, required=True)
        if name != "assemble-hessian":
            p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--device", default="cuda")
            p.add_argument("--threads", type=int, default=4)
        if name == "relax":
            p.add_argument("--fmax-eV-A", dest="fmax_eV_A", type=float, default=1e-3)
            p.add_argument("--max-steps", type=int, default=1500); p.add_argument("--maxstep-A", dest="maxstep_A", type=float, default=.1)
        elif name == "hessian":
            p.add_argument("--delta-A", dest="delta_A", type=float, default=.01)
            p.add_argument("--shards", type=int, default=1); p.add_argument("--shard-index", type=int, default=0)
        elif name == "assemble-hessian": p.add_argument("--shard", type=Path, action="append", required=True)
        else:
            p.add_argument("--seed", type=int, required=True); p.add_argument("--repeat-id", default="0")
            p.add_argument("--temperature-K", dest="temperature_K", type=float, default=433.15)
            p.add_argument("--timestep-fs", type=float, default=.5); p.add_argument("--save-interval-fs", type=float, default=2.)
            p.add_argument("--equilibration-ps", type=float, default=5.); p.add_argument("--production-ps", type=float, default=20.)
            p.add_argument("--taut-fs", type=float, default=100.); p.add_argument("--production-ensemble", choices=["nve", "nvt"], default="nve")
            p.add_argument("--abort-temperature-K", dest="abort_temperature_K", type=float, default=5000.)
            p.add_argument("--abort-force-eV-A", dest="abort_force_eV_A", type=float, default=100.)
    args = parser.parse_args()
    if args.command != "relax" and args.input is None: parser.error("This subcommand requires --input structure.npz")
    return args


def main():
    args = parse_args(); record = start_record(args)
    try:
        {"relax": relax, "hessian": hessian, "assemble-hessian": assemble, "md": md}[args.command](args, record)
    except Exception as error:
        record.update(status="FAILED", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        record["completed_unix"] = time.time(); record["elapsed_seconds"] = time.time() - record["started_unix"]
        record["outputs"] = {str(p.relative_to(args.output)): dict(bytes=p.stat().st_size, sha256=digest(p)) for p in args.output.iterdir() if p.is_file() and p.name != "receipt.json"}
        write_json(args.output / "receipt.json", record)


if __name__ == "__main__": main()

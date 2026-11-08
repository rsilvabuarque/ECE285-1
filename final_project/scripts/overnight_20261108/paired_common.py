"""Paired fixed-parent labels: identities, periodic mapping, and durable receipts.

Occupancy is one for a present mobile atom. Vacancy keys are strings with one
for a missing atom; the key is minimized under the retained parent group.
Coordinates are Å, energies total-cell eV, forces eV/Å and ASE stresses eV/Å³.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def publish_result(output, record):
    """Write labels and a separate energy-free header for sealed TEST collection."""
    output = Path(output)
    atomic_json(output / "result.json", record)
    safe = {key: value for key, value in record.items()
            if key not in {"initial_energy_eV", "endpoint_energy_eV"}}
    safe["result_sha256"] = sha256(output / "result.json")
    safe["energy_fields_omitted"] = True
    atomic_json(output / "status.json", safe)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def mobile_indices(parent):
    return np.asarray(parent.get("mobile_indices", parent.get("li_indices")), dtype=int)


def canonical(occupancy, permutations):
    """Return exact vacancy key and inverse-gather witness, without 64-bit limits."""
    occupancy = np.asarray(occupancy)
    if occupancy.ndim != 1 or not np.isin(occupancy, [0, 1]).all():
        raise ValueError("Expected one binary occupancy vector")
    occupancy = occupancy.astype(np.uint8)
    permutations = np.asarray(permutations, dtype=int)
    if permutations.ndim != 2 or permutations.shape[1] != len(occupancy):
        raise ValueError("Parent group and occupancy dimensions differ")
    inverse = np.argsort(permutations, axis=1)
    vacancies = (1 - occupancy)[inverse]
    packed = np.packbits(vacancies, axis=1, bitorder="big")
    witness = int(np.lexsort(packed[:, ::-1].T)[0])
    return "".join(map(str, vacancies[witness])), inverse[witness]


def occupancy_from_key(key):
    if not key or set(str(key)) - {"0", "1"}:
        raise ValueError("Invalid vacancy key")
    return 1 - np.fromiter(map(int, key), dtype=np.uint8)


def structure_id(key):
    return f"mask{len(key)}_{int(key, 2):0{(len(key) + 3) // 4}x}"


def physical_key(context_hash, key):
    return identity(dict(context_hash=context_hash, canonical_vacancy_key=key))


def task_for_row(prepared, row, key, role, input_path, input_sha):
    context = json.loads((Path(prepared) / "context.json").read_text())
    pk = physical_key(context["context_hash"], key)
    return dict(task_id="relax_" + pk[:24], physical_key=pk,
                context_hash=context["context_hash"], canonical_vacancy_key=key,
                structure_id=structure_id(key), n_target=int(len(key) - key.count("1")),
                split=str(role), input_row=int(row), input_path=str(Path(input_path).absolute()),
                input_sha256=input_sha)


def geometry_hash(numbers, positions, cell):
    """Literal ordered geometry identity; no tolerance or basin equivalence."""
    return identity(dict(numbers=np.asarray(numbers, int).tolist(),
                         positions_A=np.asarray(positions, float).tolist(),
                         cell_A=np.asarray(cell, float).tolist(), pbc=[True] * 3))


def check_prepared(prepared):
    prepared = Path(prepared)
    receipt = json.loads((prepared / "preparation.json").read_text())
    if receipt["status"] != "FROZEN_BEFORE_ANY_LABELS":
        raise ValueError("Paired acquisition must be frozen before scoring")
    for name, expected in receipt["outputs"].items():
        if sha256(prepared / name) != expected:
            raise ValueError("Frozen preparation changed: " + name)
    context = json.loads((prepared / "context.json").read_text())
    if identity({k: v for k, v in context.items() if k != "context_hash"}) != context["context_hash"]:
        raise ValueError("Invalid physical context identity")
    return receipt, context


def mapping_diagnostics(numbers, positions, parent, parent_indices):
    """Nearest and injective same-species mappings after common host alignment.

    Hungarian assignment prevents double counting a site but can conceal a
    nearest-site collision, so both are retained. Neither is a PES-basin test.
    Raw coordinates/atom order are never changed by this diagnostic.
    """
    from ase.geometry import find_mic
    from scipy.optimize import linear_sum_assignment
    numbers = np.asarray(numbers, int)
    positions = np.asarray(positions, float)
    slots = np.asarray(parent_indices, int)
    mobile = mobile_indices(parent)
    is_mobile = np.isin(slots, mobile)
    cell = np.asarray(parent["cell_A"], float)
    parent_xyz = np.asarray(parent["positions_A"], float)
    parent_z = np.asarray(parent["numbers"], int)
    displacement, _ = find_mic(positions - parent_xyz[slots], cell, pbc=True)
    shift = displacement[~is_mobile].mean(axis=0) if (~is_mobile).any() else np.zeros(3)
    aligned, _ = find_mic(displacement - shift, cell, pbc=True)
    nearest = np.empty(len(slots), int)
    assigned = np.empty(len(slots), int)
    nearest_distance = np.empty(len(slots), float)
    assigned_distance = np.empty(len(slots), float)
    gaps = np.empty(len(slots), float)
    for z in np.unique(numbers):
        rows = np.flatnonzero(numbers == z)
        candidates = np.flatnonzero(parent_z == z)
        vectors = positions[rows, None] - shift - parent_xyz[candidates][None]
        _, distances = find_mic(vectors.reshape(-1, 3), cell, pbc=True)
        distances = distances.reshape(len(rows), len(candidates))
        order = np.argsort(distances, axis=1, kind="stable")
        nearest[rows] = candidates[order[:, 0]]
        nearest_distance[rows] = np.take_along_axis(distances, order[:, :1], axis=1)[:, 0]
        gaps[rows] = (np.take_along_axis(distances, order[:, 1:2], axis=1)[:, 0]
                      - nearest_distance[rows]) if len(candidates) > 1 else np.inf
        r, c = linear_sum_assignment(distances ** 2)
        if len(r) != len(rows):
            raise ValueError("Insufficient same-species parent sites")
        assigned[rows[r]] = candidates[c]
        assigned_distance[rows[r]] = distances[r, c]
    occupied = np.isin(mobile, assigned[is_mobile]).astype(np.uint8)
    nearest_occupied = np.isin(mobile, nearest[is_mobile]).astype(np.uint8)
    key, _ = canonical(occupied, parent["permutations"])
    def ncoll(mask):
        return int(mask.sum() - len(set(nearest[mask].tolist())))
    return dict(endpoint_occupation_orbit=key,
                occupancy_hungarian=occupied.tolist(), occupancy_nearest=nearest_occupied.tolist(),
                nearest_parent_indices=nearest.tolist(), assigned_parent_indices=assigned.tolist(),
                nearest_distances_A=nearest_distance.tolist(), assigned_distances_A=assigned_distance.tolist(),
                nearest_second_gap_A=[None if not np.isfinite(x) else float(x) for x in gaps],
                mobile_assignment_changes=int(np.count_nonzero(assigned[is_mobile] != slots[is_mobile])),
                host_assignment_changes=int(np.count_nonzero(assigned[~is_mobile] != slots[~is_mobile])),
                mobile_nearest_collisions=ncoll(is_mobile), host_nearest_collisions=ncoll(~is_mobile),
                hungarian_differs_from_nearest=int(np.count_nonzero(assigned != nearest)),
                host_translation_A=shift.tolist(),
                host_displacement_rms_A=float(np.sqrt(np.mean(np.sum(aligned[~is_mobile] ** 2, axis=1)))) if (~is_mobile).any() else 0.,
                maximum_internal_displacement_A=float(np.linalg.norm(aligned, axis=1).max()),
                scope="Parent-site mapping, not certification of a unique physical minimum")


class AttemptWallLimit(RuntimeError):
    pass


def counted_calculator(base, *, deadline=None, synchronize=lambda: None):
    """ASE cache wrapper counting actual logical E/F/stress evaluations.

    A property request served from this cache is not another evaluation. Backend
    neural forwards are not instrumented and must not be inferred from this count.
    Last successful labels survive a timeout without an extra cleanup call.
    """
    from ase.calculators.calculator import Calculator, all_changes
    class Counted(Calculator):
        implemented_properties = ["energy", "free_energy", "forces", "stress"]
        def __init__(self):
            super().__init__()
            self.attempts = self.successes = self.failures = 0
            self.seconds = 0.
            self.last = None
        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            if deadline is not None and time.monotonic() >= deadline:
                raise AttemptWallLimit("Provisional per-attempt wall cap reached")
            super().calculate(atoms, properties, system_changes)
            self.attempts += 1
            started = time.perf_counter()
            try:
                base.reset()
                synchronize()
                energy = float(base.get_potential_energy(atoms))
                forces = np.asarray(base.get_forces(atoms), float)
                stress = np.asarray(base.get_stress(atoms), float)
                synchronize()
                if forces.shape != (len(atoms), 3) or stress.shape != (6,):
                    raise ValueError("Unexpected force/stress shape")
                if not all(np.isfinite(x).all() for x in [energy, forces, stress]):
                    raise FloatingPointError("Nonfinite calculator output")
                self.results = dict(energy=energy, free_energy=energy, forces=forces.copy(), stress=stress.copy())
                self.last = dict(positions_A=atoms.positions.copy(), energy_eV=energy,
                                 forces_eV_A=forces.copy(), stress_eV_A3=stress.copy())
                self.successes += 1
            except BaseException:
                self.failures += 1
                raise
            finally:
                self.seconds += time.perf_counter() - started
    return Counted()

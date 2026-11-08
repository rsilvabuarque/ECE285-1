"""Geometry, exact-call accounting and write-once exports for the NEB pilot.

No foundation model is imported here. Site assignments use fixed atom identities,
host-aligned periodic distances and a one-to-one Li/site assignment; symmetry
canonicalization is deliberately absent from endpoint construction/validation.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.geometry import find_mic
from ase.io import write
from scipy.optimize import linear_sum_assignment


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b""): h.update(block)
    return h.hexdigest()


def read_json(path): return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_npz(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def atoms_from(data, positions=None):
    return Atoms(numbers=data["numbers"], positions=data["positions_A"] if positions is None else positions,
                 cell=data["cell_A"], pbc=data.get("pbc", np.ones(3, bool)))


def geometry_hash(atoms):
    h = hashlib.sha256()
    for value in (atoms.numbers.astype("<i8"), atoms.positions.astype("<f8"),
                  np.asarray(atoms.cell).astype("<f8"), atoms.pbc.astype(np.uint8)):
        h.update(value.tobytes())
    return h.hexdigest()


def mapping(atoms, parent, parent_indices):
    """Continuous identity -> parent Li sites, after common host translation.

    The assignment itself has no distance threshold. Distances/margins and
    disagreement with nearest sites are reported to expose ambiguous events.
    """
    parent_indices = np.asarray(parent_indices, int)
    if len(parent_indices) != len(atoms) or len(set(parent_indices)) != len(atoms):
        raise ValueError("Atom identities must be unique and conserved")
    if not np.array_equal(atoms.numbers, parent["numbers"][parent_indices]):
        raise ValueError("Atomic order/species disagree with continuous parent identities")
    li_indices = np.asarray(parent.get("li_indices", np.flatnonzero(parent["numbers"] == 3)), int)
    local = np.flatnonzero(atoms.numbers == 3); host = np.flatnonzero(atoms.numbers != 3)
    host_displacement, _ = find_mic(atoms.positions[host] - parent["positions_A"][parent_indices[host]], atoms.cell, atoms.pbc)
    translation = host_displacement.mean(0) if len(host) else np.zeros(3)
    delta = atoms.positions[local, None, :] - translation - parent["positions_A"][li_indices][None, :, :]
    _, distance = find_mic(delta.reshape(-1, 3), atoms.cell, atoms.pbc)
    distance = distance.reshape(len(local), len(li_indices))
    rows, cols = linear_sum_assignment(distance ** 2)
    if not np.array_equal(rows, np.arange(len(local))): raise ValueError("Incomplete Li/site assignment")
    nearest = distance.argmin(1); sorted_dist = np.sort(distance, axis=1)
    occupancy = np.zeros(len(li_indices), np.uint8); occupancy[cols] = 1
    return dict(local_Li_atom_indices=local.tolist(), assigned_parent_Li_sites=cols.tolist(),
        nearest_parent_Li_sites=nearest.tolist(), assignment_distances_A=distance[rows, cols].tolist(),
        nearest_second_gap_A=(sorted_dist[:, 1]-sorted_dist[:, 0]).tolist(),
        nearest_collision_count=int(len(nearest)-len(set(nearest.tolist()))),
        hungarian_differs_from_nearest=int(np.count_nonzero(cols != nearest)),
        host_translation_A=translation.tolist(), occupancy=occupancy.tolist())


def continuous_hop_displacement(positions, numbers, parent_positions, parent_indices, cell, migrating_atom):
    """Resolve the marked image by summing small saved-frame increments.

    For this archive, coordinates are already unwrapped. Agreement of ALL raw
    atom increments with their MIC counterparts verifies that convention before
    a continuous host translation is removed. No endpoint MIC chooses the hop.
    """
    positions = np.asarray(positions, float)
    raw_steps = np.diff(positions, axis=0)
    mic_steps, _ = find_mic(raw_steps.reshape(-1,3), cell, pbc=True)
    mic_steps = mic_steps.reshape(raw_steps.shape)
    agreement = float(np.linalg.norm(raw_steps-mic_steps,axis=2).max())
    # Pure representation consistency in Angstrom, not a physical hopping
    # cutoff: unwrapped 2fs MD coordinates should differ only by roundoff.
    tolerance = 1e-8
    if agreement > tolerance:
        raise ValueError("Saved coordinates are not the claimed continuously unwrapped per-atom branch")
    host = np.flatnonzero(np.asarray(numbers) != 3)
    translation = (positions[:,host]-np.asarray(parent_positions)[np.asarray(parent_indices)[host]]).mean(1)
    aligned = positions[:,migrating_atom]-translation
    increments, lengths = find_mic(np.diff(aligned,axis=0),cell,pbc=True)
    return dict(displacement_A=increments.sum(0).tolist(), frames=len(positions),
        maximum_migrating_frame_step_A=float(np.max(lengths)),
        maximum_all_atom_raw_vs_MIC_increment_difference_A=agreement,
        representation_agreement_tolerance_A=tolerance,
        rule="Sum of migrating-atom MIC increments after continuous mean host translation; every saved frame retained; no endpoint MIC")


def validate_endpoint_event(pre_map, post_map, expected_pre, expected_post, migrating_atom, source_site, destination_site):
    """Require the physical marked hop, not a difference of canonical orbits."""
    rows = pre_map["local_Li_atom_indices"]
    if rows != post_map["local_Li_atom_indices"] or migrating_atom not in rows:
        return False, "atom_identity_mapping_changed"
    ordinal = rows.index(migrating_atom)
    before, after = pre_map["assigned_parent_Li_sites"], post_map["assigned_parent_Li_sites"]
    if before == after:
        return False, "both_endpoints_returned_to_same_site_assignment"
    if before[ordinal] != source_site or after[ordinal] != destination_site:
        return False, "intended_migrating_atom_did_not_retain_source_destination"
    if before != list(expected_pre) or after != list(expected_post):
        return False, "another_site_assignment_changed_during_endpoint_quench"
    return True, "marked_single_event_survives_endpoint_quench"


class BudgetExpired(RuntimeError): pass


class CallLedger:
    """Count actual ASE calculator.calculate attempts, not optimizer iterations."""
    def __init__(self, path, seconds, max_calls=10000, start=None):
        self.path, self.start = Path(path), time.monotonic() if start is None else start
        self.deadline = self.start + seconds
        self.max_calls, self.attempts, self.successes, self.failures = int(max_calls), 0, 0, 0
        self.phase = "startup"; self.by_phase = Counter(); self.elapsed_calculator_seconds = 0.

    def check(self):
        if time.monotonic() >= self.deadline: raise BudgetExpired("Bounded wall-clock budget exhausted")
        if self.attempts >= self.max_calls: raise BudgetExpired("Bounded actual calculator-call budget exhausted")

    def append(self, row):
        with self.path.open("a") as f: f.write(json.dumps(row, allow_nan=False) + "\n")

    def summary(self):
        return dict(actual_calculator_EF_attempts=self.attempts, successful_calculator_EF=self.successes,
            failed_calculator_EF=self.failures, by_phase=dict(self.by_phase),
            calculator_wall_seconds=self.elapsed_calculator_seconds,
            elapsed_seconds=time.monotonic()-self.start,
            counting_scope="Calls to ASE calculator.calculate requesting E/F/stress. Not optimizer steps, get_property cache hits, or instrumented neural-backbone forward counts.")


class CountingCalculator(Calculator):
    """Independent ASE image cache, one sequential shared GPU backend.

    Every image has a distinct wrapper. Its cached results are copied, so a
    subsequent image/backend reset cannot alter another image's energy/forces.
    Serial NEB only: using this shared backend concurrently is forbidden.
    """
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, inner, ledger, image_tag):
        super().__init__(); self.inner, self.ledger, self.image_tag = inner, ledger, image_tag

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        self.ledger.check()
        super().calculate(atoms, properties, system_changes)
        self.ledger.attempts += 1; self.ledger.by_phase[self.ledger.phase] += 1
        started = time.monotonic()
        row = dict(call_index=self.ledger.attempts, phase=self.ledger.phase, image=self.image_tag,
            geometry_sha256=geometry_hash(self.atoms), requested_properties=list(properties),
            evaluated_properties=["energy", "forces", "stress"], started_UTC=datetime.now(timezone.utc).isoformat())
        try:
            self.inner.reset()
            self.inner.calculate(self.atoms, properties=["energy", "forces", "stress"], system_changes=all_changes)
            self.results = {k: v.copy() if isinstance(v, np.ndarray) else v for k,v in self.inner.results.items()}
            energy, forces = float(self.results["energy"]), np.asarray(self.results["forces"])
            if forces.shape != (len(self.atoms), 3) or not np.isfinite(forces).all() or not np.isfinite(energy):
                raise ValueError("Nonfinite/wrong-shaped MLIP energy or force")
            self.ledger.successes += 1
            row.update(status="success", energy_eV=energy, physical_Fmax_eV_A=float(np.linalg.norm(forces, axis=1).max()))
        except Exception as error:
            self.ledger.failures += 1
            self.results = {}
            row.update(status="failed", exception_type=type(error).__name__, error=str(error))
            raise
        finally:
            elapsed = time.monotonic()-started; self.ledger.elapsed_calculator_seconds += elapsed
            row["elapsed_seconds"] = elapsed; self.ledger.append(row)


def attach_calculators(images, inner, ledger, prefix):
    for i, image in enumerate(images): image.calc = CountingCalculator(inner, ledger, f"{prefix}:{i}")


def snapshot(images, destination, *, projected_forces=None, parent_indices=None):
    """Save complete positions and ONLY same-geometry cached physical labels."""
    destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
    energies, forces, stresses, valid = [], [], [], []
    for image in images:
        calc = image.calc
        ok = calc is not None and not calc.check_state(image) and "energy" in calc.results and "forces" in calc.results
        valid.append(ok)
        energies.append(float(calc.results["energy"]) if ok else np.nan)
        forces.append(calc.results["forces"] if ok else np.full((len(image), 3), np.nan))
        stresses.append(calc.results.get("stress", np.full(6, np.nan)) if ok else np.full(6, np.nan))
    kwargs = dict(positions_A=np.asarray([im.positions for im in images]), cell_A=np.asarray(images[0].cell),
        pbc=images[0].pbc, numbers=images[0].numbers, energy_eV=np.asarray(energies),
        forces_eV_A=np.asarray(forces), stress_eV_A3=np.asarray(stresses), labels_match_geometry=np.asarray(valid))
    if projected_forces is not None: kwargs["NEB_projected_forces_eV_A"] = np.asarray(projected_forces)
    if parent_indices is not None: kwargs["parent_indices"] = np.asarray(parent_indices)
    np.savez_compressed(destination, **kwargs)
    return kwargs


def export_images(images, directory, parent_indices):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        # The original atom order is retained in POSCAR. CIF is a viewing export;
        # exact continuous identity and unwrapped coordinates live in NPZ.
        write(directory / f"image_{i:02d}.POSCAR", image, format="vasp", direct=True, sort=False)
        write(directory / f"image_{i:02d}.cif", image, format="cif", wrap=False)
    snapshot(images, directory / "band.npz", parent_indices=parent_indices)


def discrete_barriers(energy):
    energy = np.asarray(energy, float)
    if not np.isfinite(energy).all(): raise ValueError("Barrier profile has unknown energies")
    peak = int(energy.argmax())
    return dict(peak_image=peak, internal_peak=0 < peak < len(energy)-1,
        discrete_forward_span_eV=float(energy[peak]-energy[0]),
        discrete_reverse_span_eV=float(energy[peak]-energy[-1]),
        reaction_energy_eV=float(energy[-1]-energy[0]), units="eV per marked Li hop, never eV/atom")

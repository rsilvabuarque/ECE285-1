"""Reuse four already evaluated Cartesian finite-displacement geometries.

This packages structures and old MLIP force labels only. It performs no new
force evaluation, DFT preparation, relaxation, submission or scheduling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from ase import Atoms
from ase.io import write


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def record(path, source=None):
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size, source=source)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir
    manifest = json.loads((out/"static_candidate_manifest.json").read_text())
    manifest.pop("no_DFT_inputs_prepared", None)
    manifest["no_executable_DFT_deck_prepared"] = True
    candidates = list(manifest["candidates"])
    pairs = []
    for basin, element, atomic_number in [("basin_1", "Li", 3), ("basin_2", "O", 8)]:
        minimum = args.results_root/"relax"/basin/"final.npz"
        data = load(minimum)
        atom = int(np.flatnonzero(data["numbers"] == atomic_number)[0])
        parent_atom = int(data["parent_indices"][atom])
        column = 3*atom  # x is component zero in the producer's flattened array.
        shard = column % 2
        folder = args.results_root/"hessian"/basin/"delta_.01"/f"shard_{shard}"
        raw_path = folder/f"column_{column:04d}.npz"
        raw = load(raw_path)
        producer = json.loads((folder/"receipt.json").read_text())
        assert str(raw["input_sha256"]) == digest(minimum)
        assert int(raw["column"]) == column and float(raw["delta_A"]) == .01
        raw_copy = out/f"finite_difference_{basin}_{element}_x_column{column:04d}.npz"
        shutil.copyfile(raw_path, raw_copy)
        for slot, sign, name in [(0, 1, "plus"), (1, -1, "minus")]:
            positions = data["positions_A"].copy()
            positions.flat[column] += sign*float(raw["delta_A"])
            atoms = Atoms(numbers=data["numbers"], positions=positions, cell=data["cell_A"], pbc=True)
            candidate_id = f"DFT_{len(candidates)+1:03d}"
            poscar = out/f"{candidate_id}_{basin}_{element}_atom{atom}_x_{name}_0p01A.POSCAR"
            write(poscar, atoms, format="vasp", direct=True, sort=False)
            arrays = poscar.with_suffix(".npz")
            forces = raw[f"forces_{name}_eV_A"]
            np.savez_compressed(arrays, positions_A=positions, numbers=data["numbers"], cell_A=data["cell_A"],
                masses_amu=data["masses_amu"], parent_indices=data["parent_indices"], parent_sha256=data["parent_sha256"],
                energy_eV=raw["energies_eV"][slot], forces_eV_A=forces)
            candidates.append(dict(candidate_id=candidate_id, priority=1, category="Cartesian_force_constant_spotcheck",
                status="EXACT_ALREADY_EVALUATED_MLIP_STRUCTURE_PREPARED_DFT_NOT_STARTED",
                composition="Li36Co48O96", n_atoms=180, basin_id=basin, cell_A=data["cell_A"].tolist(),
                element_order=list(dict.fromkeys(atoms.get_chemical_symbols())),
                geometry_protocol="One Cartesian displacement of a completed UMA fixed-cell minimum; not relaxed after displacement",
                displacement=dict(element=element, local_atom_index_zero_based=atom,
                    parent_atom_index_zero_based=parent_atom, axis="x", column_index=column,
                    signed_delta_A=sign*float(raw["delta_A"])),
                existing_teacher=producer["teacher"], existing_energy_eV=float(raw["energies_eV"][slot]),
                existing_max_force_eV_A=float(np.linalg.norm(forces,axis=1).max()),
                existing_stress_status="Not retained for these individual finite-displacement calls; no minimum stress substituted",
                source_minimum=record(minimum), source_raw_column=record(raw_path), source_receipt=record(folder/"receipt.json"),
                required_observables=["all DFT forces in exact POSCAR atom order, eV/A", "DFT extrapolated and smeared energy, eV/cell", "electronic and magnetic convergence record"],
                resolves="Selected force-constant columns at matched geometries; not a full DFT phonon spectrum or phonon free energy",
                files=[record(poscar), record(arrays)]))
        pairs.append(dict(basin_id=basin, displaced_element=element, local_atom_index_zero_based=atom,
            parent_atom_index_zero_based=parent_atom, axis="x", column_index=column, delta_A=.01,
            plus_candidate=candidates[-2]["candidate_id"], minus_candidate=candidates[-1]["candidate_id"],
            formula="Phi[:,column]=-(F_plus-F_minus)/(2*delta_A), units eV/A^2",
            raw_column_copy=record(raw_copy)))
    manifest.update(status="STATIC_AND_FD_HANDOFF_PREPARED_NO_DFT", candidates=candidates,
        finite_difference_pairs=pairs, pending_additions="Two actual production thermal frames, after measurements exist",
        finite_difference_preparer_sha256=digest(__file__))
    (out/"candidate_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(status=manifest["status"], candidates=len(candidates), finite_difference_pairs=pairs),indent=2))


if __name__ == "__main__":
    main()

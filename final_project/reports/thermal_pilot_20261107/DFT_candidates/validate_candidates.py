"""Check candidate serialization, identity and hashes without any force call."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read
from ase.geometry import find_mic


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    parent = load(args.parent)
    checks, rows, geometry_digests = [], [], []
    for candidate in manifest["candidates"]:
        files = candidate["files"]
        for item in files:
            assert digest(item["path"]) == item["sha256"], item["path"]
        poscar = next(Path(item["path"]) for item in files if item["path"].endswith(".POSCAR"))
        arrays = [Path(item["path"]) for item in files if item["path"].endswith(".npz") and "MACE_singlepoint" not in item["path"]]
        if arrays:
            expected = load(arrays[0])
        else:
            vacancy = np.asarray(list(candidate["vacancy_mask"]), dtype=np.uint8)
            keep = np.ones(len(parent["numbers"]), bool)
            target = parent["li_indices"] if "li_indices" in parent else parent["target_indices"]
            keep[target[vacancy.astype(bool)]] = False
            expected = dict(positions_A=parent["positions_A"][keep], numbers=parent["numbers"][keep], cell_A=parent["cell_A"])
        atoms = read(poscar, format="vasp")
        assert Counter(atoms.get_chemical_symbols()) == {"Co":48,"Li":36,"O":96}
        assert np.array_equal(atoms.numbers, expected["numbers"])
        assert list(dict.fromkeys(atoms.get_chemical_symbols())) == candidate["element_order"]
        delta, _ = find_mic(atoms.positions-expected["positions_A"], atoms.cell, pbc=True)
        maximum = float(np.max(np.linalg.norm(delta,axis=1)))
        cell_error = float(np.max(np.abs(atoms.cell.array-expected["cell_A"])))
        # This is a text-serialization check, not a physical structural tolerance.
        # 16-decimal POSCAR coordinates in a ~28-A cell round-trip at ~1e-14 A;
        # 1e-10 A leaves format headroom while detecting order/geometry mistakes.
        assert maximum < 1e-10 and cell_error < 1e-12
        signature=hashlib.sha256(np.ascontiguousarray(expected["numbers"]).tobytes()+np.ascontiguousarray(expected["positions_A"]).tobytes()+np.ascontiguousarray(expected["cell_A"]).tobytes()).hexdigest()
        geometry_digests.append(signature)
        checks.append(dict(candidate_id=candidate["candidate_id"], all_declared_file_hashes_match=True,
            composition="Li36Co48O96", element_order=candidate["element_order"],
            numbers_match_label_array=True, maximum_serialization_displacement_A=maximum,
            maximum_cell_serialization_error_A=cell_error, exact_array_geometry_hash=signature))
        rows.append(dict(candidate_id=candidate["candidate_id"], priority=candidate["priority"],
            category=candidate["category"], status=candidate["status"], basin_id=candidate.get("basin_id",""),
            source_structure_id=candidate.get("structure_id",candidate.get("source_selection",{}).get("source_structure_id","")),
            formula="Li36Co48O96", n_atoms=180, element_order="Co Li O", poscar=str(poscar),
            poscar_sha256=digest(poscar), geometry_protocol=candidate["geometry_protocol"],
            displacement_axis=candidate.get("displacement",{}).get("axis",""),
            signed_displacement_A=candidate.get("displacement",{}).get("signed_delta_A",""),
            existing_UMA_energy_eV=candidate.get("existing_energy_eV",candidate.get("existing_teacher_labels",{}).get("uma",{}).get("energy_eV","")),
            purpose=candidate["resolves"]))
    assert len(set(geometry_digests)) == len(geometry_digests)
    directory = args.manifest.parent
    with (directory/"candidate_manifest.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(status="PASS_CANDIDATE_SERIALIZATION_AND_HASH_CHECK", candidates=len(checks),
        manifest_sha256=digest(args.manifest), parent_sha256=digest(args.parent),
        exact_array_geometries_distinct=True, checks=checks,
        maximum_serialization_displacement_A=max(r["maximum_serialization_displacement_A"] for r in checks),
        tolerance_scope="Numerical serialization only; 1e-10 A positions and 1e-12 A cell, not chemical equivalence criteria",
        no_new_force_calls=True, no_DFT_run_submitted_or_scheduled=True,
        script_sha256=digest(__file__))
    (directory/"candidate_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="checks"},indent=2))


if __name__ == "__main__":
    main()

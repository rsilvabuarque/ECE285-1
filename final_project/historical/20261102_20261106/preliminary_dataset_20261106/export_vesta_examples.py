"""Export three existing low-tail CVAE lineages; no new model evaluations.

Preserve the evaluated orientation of each mask. The canonical key identifies
its symmetry orbit and need not equal that orientation's vacancy-bit string.
"""
import csv
import hashlib
import json
import platform
import zipfile
from pathlib import Path

import ase
import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read, write

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "vesta_examples"
SELECTION = [
    ("sample_01_Li12", "gen48_1bff7cfbb6ef", "uma_cvae_countDP"),
    ("sample_02_Li24", "gen48_00979f4745be", "uma_cvae_beta01_countDP"),
    ("sample_03_Li36", "gen48_00000a0fa0a6", "mace_cvae"),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main():
    OUT.mkdir(exist_ok=False)
    parent = load(ROOT / "data/parents/O3_N48_pilot.npz")
    generated = load(ROOT / "data/generated_candidates.npz")
    decoded = load(ROOT / "data/decoded_geometry_candidates.npz")
    selected_ids = {sid for _, sid, _ in SELECTION}
    proposals = [r for r in csv.DictReader((ROOT / "cvae/proposal_attempts.csv").open())
                 if r["scoring_structure_id"] in selected_ids]
    thresholds = {(r["teacher"], int(r["n_Li"])): float(r["q20_eV_atom"])
                  for r in csv.DictReader((ROOT / "reports/generation_thresholds.csv").open())
                  if r["source"] == "main_train"}
    scores = {(r["source_mask_id"], r["method"]): r for r in
              csv.DictReader((ROOT / "reports/decoded_geometry_scores.csv").open())
              if r["teacher"] == "uma" and r["source_mask_id"] in selected_ids}
    teachers = {}
    for model in ["uma", "mace", "sevennet", "chgnet", "m3gnet"]:
        data = load(ROOT / f"data/generated_scalar_{model}.npz")
        teachers[model] = {}
        for sid in selected_ids:
            i = int(np.flatnonzero(data["structure_ids"] == sid)[0])
            teachers[model][sid] = float(data["energy_eV"][i] / data["n_atoms"][i])
    catalog = {r["canonical_mask"] for r in
               csv.DictReader((ROOT / "data/mask_manifest.csv").open())
               if r["parent"] == "O3_N48_pilot"}
    manifest, rows, checks = [], [], []

    def export_pair(folder, stem, atoms, slots):
        # Explicit species order is Li, Co, O. Save the row mapping separately.
        order = np.argsort([{3: 0, 27: 1, 8: 2}[int(z)] for z in atoms.numbers], kind="stable")
        a = atoms[order]
        slots = np.asarray(slots)[order]
        a.wrap(eps=1e-12)
        folder.mkdir(exist_ok=True)
        poscar = folder / f"{stem}.vasp"
        cif = folder / f"{stem}.cif"
        write(poscar, a, format="vasp", direct=True, vasp5=True, sort=False)
        write(cif, a, format="cif", wrap=False)
        with (folder / f"{stem}_atom_map.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file_atom_row_1based", "element", "parent_slot_0based"])
            writer.writerows((i + 1, symbol, int(slot)) for i, (symbol, slot)
                             in enumerate(zip(a.get_chemical_symbols(), slots)))
        # These are format round-trip tolerances, not physical convergence gates.
        for path, fmt, tolerance in [(poscar, "vasp", 1e-9), (cif, "cif", 1e-6)]:
            b = read(path, format=fmt)
            assert np.array_equal(a.numbers, b.numbers)
            assert np.max(np.abs(a.cell.array - b.cell.array)) < tolerance
            _, distances = find_mic(b.positions - a.positions, a.cell, pbc=True)
            assert distances.max() < tolerance, (path, distances.max())
            checks.append({"file": str(path.relative_to(OUT)), "sha256": digest(path),
                           "max_roundtrip_position_error_A": float(distances.max()),
                           "roundtrip_tolerance_A": tolerance})
        return a

    for name, sid, method in SELECTION:
        folder = OUT / name
        i = int(np.flatnonzero(generated["structure_ids"] == sid)[0])
        mask = generated["masks"][i]
        raw = "".join(map(str, mask))
        inv = np.argsort(parent["permutations"], axis=1)
        canonical = min("".join(map(str, mask[op])) for op in inv)
        assert sid == f"gen48_{int(canonical, 2):012x}" and canonical not in catalog
        origin = [r for r in proposals if r["scoring_structure_id"] == sid and r["method"] == method]
        assert len(origin) == 1 and origin[0]["condition"] == "low"
        origin = origin[0]
        n_li = int(48 - mask.sum())
        keep = np.ones(192, bool)
        keep[parent["li_indices"]] = ~mask.astype(bool)
        slots = np.flatnonzero(keep)
        cfg = ROOT / "cvae/runs" / origin["training_run"] / "configuration.json"
        checkpoint = cfg.parent / "best.pt"
        record = {"sample": name, "source_mask_id": sid, "n_Li": n_li,
                  "formula": f"Li{n_li}Co48O96", "lithiation_fraction": n_li / 48,
                  "vacancy_mask_evaluated_orientation": raw, "canonical_vacancy_mask": canonical,
                  "novel_to_entire_initial_catalog": True, "proposal": origin,
                  "mask_checkpoint": str(checkpoint.relative_to(ROOT)),
                  "mask_checkpoint_sha256": digest(checkpoint),
                  "mask_configuration_sha256": digest(cfg), "teachers_deletion_geometry": {}}
        for model in teachers:
            e, q = teachers[model][sid], thresholds[model, n_li]
            record["teachers_deletion_geometry"][model] = {
                "energy_eV_atom": e, "training_q20_eV_atom": q,
                "energy_minus_q20_meV_atom": 1000 * (e - q), "passes_low_tail": e <= q}
        assert record["teachers_deletion_geometry"]["uma"]["passes_low_tail"]
        geometry_checkpoint = ROOT / "cvae/geometry_runs/cvae_20261400/best.pt"
        record.update(geometry_checkpoint=str(geometry_checkpoint.relative_to(ROOT)),
                      geometry_checkpoint_sha256=digest(geometry_checkpoint), versions=[])
        for stem, start, relaxed in [
            ("01_vacancy_only", "deletion", False),
            ("02_geometry_CVAE_unrelaxed", "cvae_draw0", False),
            ("03_UMA_relaxed_from_vacancy", "deletion", True),
            ("04_UMA_relaxed_from_geometry_CVAE", "cvae_draw0", True),
        ]:
            j = int(np.flatnonzero((decoded["source_mask_ids"] == sid) &
                                  (decoded["methods"] == start))[0])
            assert np.array_equal(decoded["atom_presence"][j], keep)
            item = {"version": stem, "optimized": relaxed, "cell_relaxed": False}
            if relaxed:
                base = ROOT / "data/decoded_relax/uma" / f"decoded_relax__uma__{sid}__{start}"
                arrays = load(base.with_suffix(".npz"))
                receipt = json.loads(base.with_suffix(".json").read_text())
                assert receipt["converged"] and not receipt["cell_relaxed"]
                assert np.array_equal(arrays["parent_slot"], slots)
                assert np.array_equal(arrays["numbers"], parent["numbers"][keep])
                assert digest(receipt["input_file"]) == receipt["input_sha256"]
                actual_input = read(receipt["input_file"], format="extxyz")
                assert np.array_equal(arrays["initial_positions_A"], actual_input.positions)
                # ASE's executed extxyz input printed coordinates to 8 decimals:
                # 5e-9 A is the rounding bound, plus floating-point slack.
                assert np.allclose(arrays["initial_positions_A"], decoded["positions_parent_A"][j, keep],
                                   atol=5e-9 + 1e-13, rtol=0)
                xyz, cell = arrays["positions_A"][-1], arrays["cell_A"][-1]
                energy = float(arrays["energy_eV"][-1])
                fmax = float(np.linalg.norm(arrays["forces_eV_A"][-1], axis=1).max())
                assert abs(energy - receipt["final_energy_eV"]) < 1e-8 and fmax < 0.05
                item.update(source_npz=str(base.with_suffix(".npz").relative_to(ROOT)),
                            source_sha256=digest(base.with_suffix(".npz")),
                            optimizer="UMA-S-1.2.1 omat / FIRE / fixed cell",
                            steps=receipt["steps"], force_target_eV_A=0.05)
            else:
                xyz, cell = decoded["positions_parent_A"][j, keep], decoded["cell_A"][j]
                score = scores[sid, start]
                energy, fmax = float(score["energy_eV"]), float(score["force_max_eV_A"])
                item.update(source_npz="data/decoded_geometry_candidates.npz", source_index=j)
            assert np.allclose(cell, parent["cell_A"], rtol=0, atol=1e-10)
            atoms = Atoms(numbers=parent["numbers"][keep], positions=xyz, cell=cell, pbc=True)
            a = export_pair(folder, stem, atoms, slots)
            distances = a.get_all_distances(mic=True)
            np.fill_diagonal(distances, np.inf)
            item.update(energy_UMA_eV=energy, energy_UMA_eV_atom=energy / len(a),
                        force_max_UMA_eV_A=fmax, minimum_pair_distance_A=float(distances.min()))
            record["versions"].append(item)
            rows.append({"sample": name, "source_mask_id": sid, "n_Li": n_li,
                         "lithiation_percent": 100 * n_li / 48, "version": stem,
                         "optimized": relaxed, "energy_UMA_eV": energy,
                         "energy_UMA_eV_atom": energy / len(a), "force_max_UMA_eV_A": fmax,
                         "steps": item.get("steps", ""),
                         "deletion_energy_minus_UMA_q20_meV_atom": record["teachers_deletion_geometry"]["uma"]["energy_minus_q20_meV_atom"],
                         "minimum_pair_distance_A": item["minimum_pair_distance_A"]})
        manifest.append(record)
    # Visual reference only; it is not a generated sample.
    export_pair(OUT / "pristine_reference", "Li48Co48O96",
                Atoms(numbers=parent["numbers"], positions=parent["positions_A"],
                      cell=parent["cell_A"], pbc=True), np.arange(192))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (OUT / "energies_and_forces.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    readme = """Three existing, novel, low-tail CVAE mask lineages for VESTA

Open any .cif or .vasp file. Both formats contain the same geometry. CIFs use
P1 and explicit atoms: no symmetry expansion, average occupancies or ghost Li.
The atom-map CSVs map file rows to the original 192-atom parent slots.

sample_01_Li12: 25% lithiation; UMA standard CVAE, exact-count Bernoulli draw.
sample_02_Li24: 50% lithiation; UMA beta=0.1 CVAE, exact-count Bernoulli draw.
sample_03_Li36: 75% lithiation; MACE CVAE, top-count decoding.

Each folder has four versions of exactly the same ancestral vacancy mask:
01_vacancy_only: Mask CVAE output, Li deleted at pristine coordinates.
02_geometry_CVAE_unrelaxed: First prior draw from the separate UMA-trained
   displacement CVAE, applied to that mask; no subsequent minimization.
03_UMA_relaxed_from_vacancy: Existing fixed-cell UMA/FIRE endpoint from 01.
04_UMA_relaxed_from_geometry_CVAE: Existing fixed-cell UMA/FIRE endpoint from 02.

All optimized endpoints meet maximum atomic force <0.05 eV/Angstrom. The cell
was fixed, so stress need not be small. They are MLIP local minima to this
provisional force tolerance; they are not DFT-optimized or proven ground states.
No new MLIP evaluation or optimization was performed for this export.

Selection: one energy-conditioned CVAE lineage per Li count 12/24/36, chosen
from the existing 32-mask matched-relaxation diagnostic because all four
versions already existed and the DELETION geometry passes UMA's frozen
composition-specific training q20. This is an illustrative, favorable subset,
not a new performance test or the three lowest-energy configurations overall.
The full per-teacher pass/fail values and source checkpoint hashes are in
manifest.json; all measured version energies/forces are in energies_and_forces.csv.

The low-tail label applies to the deletion geometry and its deletion-energy
reference distribution. It must not be reinterpreted as a relaxed-energy
percentile. Compare absolute energies only at equal composition and teacher.
The symmetry-orbit key and evaluated-orientation vacancy mask can differ;
the original evaluated orientation is preserved, so force/coordinate rows match.

All 3 masks are distinct under the retained 192 parent permutations and absent
from the entire original 21,294-mask main catalog. Different Li counts are
necessarily different orbits. A pristine Li48 reference is included separately.
VESTA bond cutoffs affect display only. Visual plausibility and the force gate
do not establish DFT accuracy or thermal/phase stability.
"""
    (OUT / "README.txt").write_text(readme)
    validation = {"status": "PASS", "sample_lineages": 3, "versions_per_lineage": 4,
                  "structure_files": len(checks), "includes_pristine_reference": True,
                  "checks": checks, "ASE": ase.__version__, "Python": platform.python_version(),
                  "numpy": np.__version__, "script_sha256": digest(__file__),
                  "scope": "Existing-coordinate export and format round trip; no new chemistry or physical validation."}
    (OUT / "export_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    archive = ROOT / "vesta_examples.zip"
    assert not archive.exists()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob("*")):
            if path.is_file(): z.write(path, path.relative_to(ROOT))
        z.write(Path(__file__), "vesta_examples/export_vesta_examples.py")
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    Path(str(archive) + ".sha256").write_text(f"{digest(archive)}  {archive.name}\n")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "sha256": digest(archive), "examples": manifest}, indent=2))


if __name__ == "__main__":
    main()

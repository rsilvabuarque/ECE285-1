"""Inventory archived x=0.75 references without running an electronic calculation.

Run CSV/XML analysis inside an authorized compute allocation. Sources remain
read-only. POTCAR content is never copied; only file hashes and TITEL metadata
are recorded. All errors are existing matched-geometry predictions, in FP64.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pool-metrics", type=Path, required=True)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--hu-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read(args.source)
    selected = [r for r in rows if 4*int(r["n_Li"]) == 3*int(r["n_Co"])]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    columns = ["geometry_key", "dataset", "phase", "cohort", "structure_id", "source", "frame",
               "case_id", "temperature_K", "pressure_MPa", "n_atoms", "n_Co", "n_Li", "x",
               "volume_A3", "model", "dft_energy_eV", "mlip_energy_eV", "energy_error_meV_atom",
               "force_RMSE_eV_A", "force_MAE_eV_A", "force_RMSE_Li_eV_A", "force_RMSE_Co_eV_A",
               "force_RMSE_O_eV_A", "stress_RMSE_GPa", "stress_MAE_GPa"]
    write(out/"archived_x075_EFS.csv", [{key: r[key] for key in columns} for r in selected])
    pools = {(r["dataset"], r["phase"], r["cohort"], r["n_Co"], r["n_Li"], r["model"]): r
             for r in read(args.pool_metrics) if r["grouping"] == "phase_composition_cohort"}
    groups = defaultdict(list)
    for r in selected:
        groups[r["dataset"], r["phase"], r["cohort"], r["n_Co"], r["n_Li"], r["model"]].append(r)
    summary = []
    for key, rr in sorted(groups.items()):
        values = dict(zip(["dataset", "phase", "cohort", "n_Co", "n_Li", "model"], key))
        values.update(n_frames=len(rr), n_atoms=int(rr[0]["n_atoms"]))
        for name in ["force_RMSE_eV_A", "force_MAE_eV_A", "force_RMSE_Li_eV_A",
                     "force_RMSE_Co_eV_A", "force_RMSE_O_eV_A", "stress_RMSE_GPa"]:
            array = np.asarray([float(r[name]) for r in rr])
            values[f"median_{name}"] = float(np.median(array))
            values[f"p90_{name}"] = float(np.quantile(array, .9))
        pool = pools[key]
        for name in ["spearman_rho", "top1_tie_mean_regret_meV_atom", "pair_gap_RMSE_meV_atom",
                     "mean_offset_meV_atom", "DFT_span_meV_atom"]:
            values[name] = pool[name]
        summary.append(values)
    write(out/"archived_x075_summary.csv", summary)

    protocols = []
    historical = {r["case_id"]: r for r in selected if r["dataset"] == "historical" and r["model"] == "uma"}
    for case_id, r in sorted(historical.items()):
        folder = args.case_root/case_id
        xml = ET.parse(folder/"vasprun.xml").getroot()
        last = xml.findall("./calculation")[-1]
        energy = {item.attrib["name"]: float(item.text) for item in last.findall("./energy/i")}
        incar = (folder/"INCAR").read_text()
        potcar = (folder/"POTCAR").read_text()
        protocols.append(dict(case_id=case_id, source_folder=str(folder),
            files={name: dict(path=str(folder/name), sha256=digest(folder/name))
                   for name in ["INCAR", "KPOINTS", "POSCAR", "POTCAR", "vasprun.xml", "OUTCAR", "run_manifest.json"]},
            incar_text=incar, kpoints_text=(folder/"KPOINTS").read_text(),
            potential_titles=[line.strip().split("=", 1)[1].strip() for line in potcar.splitlines() if "TITEL" in line],
            vasp_version=(folder/"OUTCAR").read_text().splitlines()[0].strip(),
            final_xml_energies_eV=energy, archived_energy_eV=float(r["dft_energy_eV"]),
            archived_energy_matches_e0=abs(float(r["dft_energy_eV"])-energy["e_0_energy"]) <= 1e-8,
            final_electronic_steps=len(last.findall("./scstep")),
            electronic_ediff_message_present="aborting loop because EDIFF is reached" in (folder/"OUTCAR").read_text(),
            magnetic_state_validation="Initial ISPIN/MAGMOM verified; no alternate-spin ground-state comparison certified"))
    hu = json.loads(args.hu_config.read_text())
    hu_parameters = dict(source=str(args.hu_config), sha256=digest(args.hu_config),
        fp_style=hu.get("fp_style"), fp_incar=hu.get("fp_incar"), fp_pp_files=hu.get("fp_pp_files"),
        referenced_incar_exists=(args.hu_config.parent/hu["fp_incar"]).exists(),
        protocol_status="Original frame-level VASP inputs/SCF and spin comparisons not present in this archive; unresolved")
    payload = dict(historical=protocols, hu=hu_parameters,
        no_POTCAR_content_copied=True, no_inputs_prepared=True, no_DFT_run_submitted_or_scheduled=True)
    (out/"existing_dft_protocols.json").write_text(json.dumps(payload, indent=2)+"\n")
    keys = {r["geometry_key"] for r in rows}
    n48 = {r["geometry_key"] for r in rows if r["n_Co"] == "48" and r["n_Li"] == "36"}
    receipt = dict(status="PASS_EXISTING_REFERENCE_AUDIT", source_sha256=digest(args.source),
        source_rows=len(rows), source_unique_geometries=len(keys),
        composition_x075_unique_geometries=len({r["geometry_key"] for r in selected}),
        main_Li36Co48O96_matched_DFT_count=len(n48),
        cobalt_sizes_present=sorted({int(r["n_Co"]) for r in rows}),
        historical_protocol_cases=len(protocols),
        distinct_historical_incar_hashes=len({r["files"]["INCAR"]["sha256"] for r in protocols}),
        distinct_historical_kpoint_hashes=len({r["files"]["KPOINTS"]["sha256"] for r in protocols}),
        all_historical_archive_energies_match_xml_e0=all(r["archived_energy_matches_e0"] for r in protocols),
        source_geometry_conventions="Exact archived geometries; FP64 older model audit; ASE tensile-positive stress",
        limitation="Smaller cells and different geometry cohorts do not validate new main-parent minima, forces, phonons or finite-T free energies",
        source_script_sha256=digest(__file__), outputs={name: digest(out/name) for name in
            ["archived_x075_EFS.csv", "archived_x075_summary.csv", "existing_dft_protocols.json"]})
    old_analysis=args.source.parent.parent/"analyze_models.py"
    if old_analysis.is_file():
        assert "force_RMSE_eV_A=float(np.sqrt(np.mean(delta**2)))" in old_analysis.read_text()
        receipt["archived_force_definition"] = dict(source=str(old_analysis),sha256=digest(old_analysis),
            quantity="sqrt(mean((F_model-F_DFT)^2)) across Cartesian force components; table reports median across frames",
            units="eV/A")
    (out/"existing_reference_receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

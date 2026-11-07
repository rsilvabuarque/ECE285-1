"""Package exact completed MLIP structures for a manual, unscheduled DFT handoff."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def file_record(path, source=None):
    return dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size, source=source)


def species_order(numbers):
    atoms = Atoms(numbers=numbers)
    return list(dict.fromkeys(atoms.get_chemical_symbols()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--relax-root", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--mace-check", type=Path)
    p.add_argument("--harmonic-audit", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads(args.selection.read_text())
    sym = json.loads((out/"selected_occupancy_symmetry.json").read_text())
    parent = load(args.parent)
    assert digest(args.parent) == selection["parent_sha256"]
    inverse = np.argsort(parent["permutations"], axis=1)
    candidates, relaxed = [], []
    for k, row in enumerate(selection["rows"], 1):
        source = args.relax_root/row["basin_id"]
        receipt = json.loads((source/"receipt.json").read_text())
        assert receipt["status"] == "CONVERGED" and receipt["fixed_cell"]
        geom = load(source/"final.npz")
        assert np.array_equal(geom["cell_A"], parent["cell_A"])
        mapping = receipt["final_mapping"]
        assigned = mapping["vacancy_mask_hungarian"]
        mask = np.asarray(list(assigned), dtype=np.uint8)
        canonical = min("".join(map(str, value)) for value in mask[inverse])
        assert canonical == row["vacancy_mask"]
        assert mapping["nearest_collision_count"] == 0
        assert mapping["hungarian_differs_from_nearest"] == 0
        destination = out/f"DFT_{k:03d}_{row['basin_id']}_UMA_fixedcell_minimum.POSCAR"
        assert digest(source/"final.POSCAR") == receipt["outputs"]["final.POSCAR"]["sha256"]
        shutil.copyfile(source/"final.POSCAR", destination)
        npz = destination.with_suffix(".npz")
        shutil.copyfile(source/"final.npz", npz)
        record = dict(candidate_id=f"DFT_{k:03d}", priority=0, category="competing_fixedcell_minimum",
            status="EXACT_STRUCTURE_PREPARED_DFT_NOT_STARTED", basin_id=row["basin_id"],
            composition="Li36Co48O96", n_atoms=180, cell_A=geom["cell_A"].tolist(),
            element_order=species_order(geom["numbers"]),
            geometry_protocol="Completed UMA fixed-cell local relaxation, not DFT relaxed",
            source_selection=row, existing_teacher=receipt["teacher"],
            existing_energy_eV=float(geom["energy_eV"]), existing_max_force_eV_A=receipt["final_max_force_eV_A"],
            existing_stress_eV_A3=geom["stress_eV_A3"].tolist(),
            required_observables=["DFT e_0_energy and smeared free energy, eV/cell", "all atomic forces, eV/A", "stress tensor with documented sign/units", "total/site magnetic moments", "electronic convergence record"],
            resolves="Relative energies at matched UMA stationary geometries and DFT force residuals; does not establish DFT-relaxed basin free energies",
            source_receipt=file_record(source/"receipt.json"), files=[file_record(destination, str(source/"final.POSCAR")), file_record(npz, str(source/"final.npz"))])
        candidates.append(record)
        relaxed.append(dict(basin_id=row["basin_id"], canonical_assigned_vacancy_mask=canonical,
            occupancy_changed_sites=mapping["changed_occupancy_sites"],
            nearest_collision_count=mapping["nearest_collision_count"],
            hungarian_differs_from_nearest=mapping["hungarian_differs_from_nearest"],
            maximum_assigned_distance_A=max(mapping["assignment_distances_A"]),
            minimum_nearest_second_margin_A=min(mapping["nearest_second_gap_A"]),
            host_displacement_rms_A=mapping["host_displacement_rms_A"],
            final_energy_eV=float(geom["energy_eV"]), max_force_eV_A=receipt["final_max_force_eV_A"],
            steps=receipt["steps"], relaxation_seconds=receipt["relaxation_seconds"],
            parent_action_orbit_size=192,
            orbit_interpretation="Trivial parent-mask stabilizer plus unique equivariant parent-site assignment; no nonidentity retained parent operation can fix this geometry"))

    if args.mace_check is not None:
        mace_receipt = json.loads((args.mace_check/"receipt.json").read_text())
        assert mace_receipt["status"] == "COMPLETE"
        shutil.copyfile(args.mace_check/"comparison.csv", out/"mace_minima_comparison.csv")
        shutil.copyfile(args.mace_check/"receipt.json", out/"mace_minima_receipt.json")
        for candidate, comparison in zip(candidates, mace_receipt["rows"]):
            assert candidate["basin_id"] == comparison["basin_id"]
            original_npz = args.relax_root/candidate["basin_id"]/"final.npz"
            assert digest(original_npz) == comparison["input_sha256"]
            uma_geometry = load(original_npz)
            mace_source = args.mace_check/f"{candidate['basin_id']}.npz"
            mace_data = load(mace_source)
            for name in ["numbers", "positions_A", "cell_A"]:
                assert np.array_equal(uma_geometry[name], mace_data[name])
            mace_destination = out/f"{candidate['candidate_id']}_MACE_singlepoint.npz"
            shutil.copyfile(mace_source, mace_destination)
            candidate["existing_MACE_singlepoint"] = dict(teacher=mace_receipt["teacher"],
                energy_eV=float(mace_data["energy_eV"]), maximum_force_eV_A=comparison["MACE_Fmax_eV_A"],
                force_teacher_difference_component_RMSE_eV_A=comparison["force_teacher_difference_component_RMSE_eV_A"],
                stress_eV_A3=mace_data["stress_eV_A3"].tolist(), comparison=comparison,
                interpretation="Exact UMA geometry; not a MACE-relaxed minimum or DFT reference")
            candidate["files"].append(file_record(mace_destination, str(mace_source)))
            candidate["source_MACE_receipt"] = file_record(args.mace_check/"receipt.json")

    benchmark = load(args.benchmark/"candidates.npz")
    scores = {name: load(args.benchmark/f"scores_{name}.npz") for name in ["uma", "mace"]}
    teacher_metadata = {}
    for name in scores:
        schema_file = args.benchmark.parent/"data"/f"condition_schema_{name}.json"
        schema = json.loads(schema_file.read_text())
        teacher_metadata[name] = dict(schema_id=schema["identity"],
            provenance=schema["provenance"], schema_sha256=digest(schema_file),
            precision="float32", tf32=False,
            label_source="Completed Phase1 exact canonical deletion scoring")
    lookups = {name: {str(value): i for i, value in enumerate(data["structure_ids"])} for name, data in scores.items()}
    eligible = np.flatnonzero(benchmark["n_Li"] == 36)
    ids = [str(benchmark["structure_ids"][i]) for i in eligible]
    energy = {name: np.asarray([scores[name]["energy_eV"][lookups[name][key]] for key in ids]) for name in scores}
    delta = energy["mace"]-energy["uma"]
    center = float(np.median(delta))
    extremes = [("minimum_centered_MACE_minus_UMA", int(np.argmin(delta))),
                ("maximum_centered_MACE_minus_UMA", int(np.argmax(delta)))]
    for label, local in extremes:
        index = int(eligible[local]); key = ids[local]
        mask = benchmark["masks"][index]
        target = parent["li_indices"] if "li_indices" in parent else parent["target_indices"]
        keep = np.ones(len(parent["numbers"]), bool)
        keep[target[np.asarray(mask, bool)]] = False
        atoms = Atoms(numbers=parent["numbers"][keep], positions=parent["positions_A"][keep], cell=parent["cell_A"], pbc=True)
        number = len(candidates)+1
        destination = out/f"DFT_{number:03d}_{label}_deletion.POSCAR"
        write(destination, atoms, format="vasp", direct=True, sort=False)
        teacher_labels = {}
        for name, data in scores.items():
            j = lookups[name][key]
            teacher_labels[name] = dict(energy_eV=float(data["energy_eV"][j]),
                force_component_rms_eV_A=float(data["force_component_rms_eV_A"][j]),
                max_force_eV_A=float(data["force_max_eV_A"][j]), stress_eV_A3=data["stress_eV_A3"][j].tolist())
        candidates.append(dict(candidate_id=f"DFT_{number:03d}", priority=2,
            category="centered_teacher_disagreement", status="EXACT_STRUCTURE_PREPARED_DFT_NOT_STARTED",
            structure_id=key, composition="Li36Co48O96", n_atoms=180, cell_A=atoms.cell.array.tolist(),
            element_order=species_order(atoms.numbers), geometry_protocol="Unrelaxed canonical parent deletion; no geometry optimization",
            vacancy_mask="".join(map(str, mask)), existing_teacher_labels=teacher_labels,
            existing_teacher_metadata=teacher_metadata,
            disagreement=dict(selection=label, complete_candidate_pool=len(ids),
                median_MACE_minus_UMA_eV=center, centered_MACE_minus_UMA_eV=float(delta[local]-center),
                offset_removal="Subtract median model difference within the same composition/geometry cohort"),
            required_observables=["DFT relative energy against the other disagreement endpoint, eV/cell", "all forces, eV/A", "stress with documented sign/units", "magnetic and electronic convergence record"],
            resolves="Which teacher describes this within-composition energy-gap contrast; large forces are expected in unrelaxed deletions",
            files=[file_record(destination)]))

    emin = min(r["final_energy_eV"] for r in relaxed)
    kb = 8.617333262145e-5; temperature = 433.15
    mass = np.exp(-np.array([r["final_energy_eV"]-emin for r in relaxed])/(kb*temperature))
    mass /= mass.sum()
    for r, probability in zip(relaxed, mass):
        r.update(relative_energy_eV=r["final_energy_eV"]-emin, static_selected_three_probability_equal_representatives=float(probability))
    symmetry = dict(status="DISTINCT_RELAXED_OCCUPANCY_ORBITS_UNDER_DECLARED_PARENT_ACTION",
        rows=relaxed, initial_symmetry=sym,
        distinct_canonical_assigned_masks=len({r["canonical_assigned_vacancy_mask"] for r in relaxed}),
        all_initial_orbit_multiplicities_equal_192=True,
        interpretation="No relaxed endpoint merge under the retained parent symmetry action with unique site assignments. This is not a proof of finite-temperature basin retention or global phase identity.",
        weight_scope="Static context normalized over these three representatives only; a thermal two-basin comparison must renormalize the static baseline over the same two basins.")
    (out/"relaxed_basin_symmetry.json").write_text(json.dumps(symmetry, indent=2)+"\n")
    if args.harmonic_audit is not None:
        with args.harmonic_audit.open(newline="") as handle:
            harmonic = {r["basin"]: r for r in csv.DictReader(handle) if float(r["delta_A"]) == .01}
        context = []
        for candidate, endpoint in zip(candidates[:3], relaxed):
            row = harmonic[endpoint["basin_id"]]
            assert int(row["strict_negative_internal_modes"]) == 0 and int(row["internal_modes"]) == 537
            assert float(row["relaxed_minimum_eV"]) == endpoint["final_energy_eV"]
            context.append(dict(basin_id=endpoint["basin_id"], n_atoms=180, temperature_K=temperature,
                UMA_static_E_eV=endpoint["final_energy_eV"],
                MACE_singlepoint_E_eV=candidate["existing_MACE_singlepoint"]["energy_eV"],
                UMA_harmonic_classical_F_eV=endpoint["final_energy_eV"]+float(row["harmonic_classical_vib_F_433K_eV"]),
                UMA_harmonic_quantum_F_eV=endpoint["final_energy_eV"]+float(row["harmonic_quantum_vib_F_433K_eV"])))
        methods=["UMA_static_E_eV", "MACE_singlepoint_E_eV", "UMA_harmonic_classical_F_eV", "UMA_harmonic_quantum_F_eV"]
        for method in methods:
            energies=np.asarray([r[method] for r in context]); probabilities=np.exp(-(energies-energies.min())/(kb*temperature)); probabilities/=probabilities.sum()
            for r, energy, probability in zip(context, energies, probabilities):
                r[method+"_relative_to_basin1"] = float(energy-energies[0])
                r[method+"_selected_three_probability"] = float(probability)
        with (out/"basin_energy_context.csv").open("w", newline="") as handle:
            writer=csv.DictWriter(handle, fieldnames=list(context[0])); writer.writeheader(); writer.writerows(context)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
        x=np.arange(3)
        for name,label in [(methods[0],"UMA static minimum"),(methods[1],"MACE single point, same geometry")]:
            axs[0].plot(x,[r[name+"_relative_to_basin1"] for r in context],"o-",label=label)
        for name,label in [(methods[0],"UMA static"),(methods[2],"UMA classical harmonic"),(methods[3],"UMA quantum harmonic")]:
            axs[1].plot(x,[r[name+"_relative_to_basin1"] for r in context],"o-",label=label)
        for ax in axs:
            ax.axhline(0,c="gray",lw=.6);ax.set_xticks(x,["basin 1","basin 2","basin 3"]);ax.set_ylabel("Energy or F minus basin 1 (eV/cell)");ax.grid(alpha=.2);ax.legend(fontsize=8)
        axs[0].set_title("Teacher disagreement at UMA minima");axs[1].set_title("Finite-cell harmonic comparison, 433.15 K")
        for extension in ["pdf","png"]:fig.savefig(out/f"basin_energy_context.{extension}",dpi=170)
        plt.close(fig)
    receipt = dict(status="STATIC_HANDOFF_PREPARED_NO_DFT", candidates=candidates,
        source_sha256={str(path): digest(path) for path in [args.selection, args.parent,
            args.benchmark/"candidates.npz", args.benchmark/"scores_uma.npz", args.benchmark/"scores_mace.npz"]},
        protocol_status="Historical source protocol audited; main-cell k mesh, species ordering and magnetic-state strategy require manual review",
        no_POTCAR_redistributed=True, no_executable_DFT_deck_prepared=True, no_DFT_run_submitted_or_scheduled=True,
        pending_additions="Select actual thermal or instability/soft-mode structures only after those measurements exist",
        script_sha256=digest(__file__))
    (out/"static_candidate_manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(dict(status=receipt["status"], candidate_count=len(candidates), relaxed=relaxed,
        disagreement_candidates=[r["disagreement"] for r in candidates if r["category"] == "centered_teacher_disagreement"]), indent=2))


if __name__ == "__main__":
    main()

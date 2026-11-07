"""Apply a frozen two-frame selection rule to completed production only.

Heavy NPZ/JSON analysis belongs in an authorized compute allocation. No
calculator is constructed: every scalar, force and stress is an existing label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write


def digest(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def file_record(path, source=None):
    return dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size,source=source)


def write_frame(data,index,path):
    atoms=Atoms(numbers=data["numbers"],positions=data["positions_A"][index],cell=data["cell_A"],pbc=True)
    write(path,atoms,format="vasp",direct=True,sort=False)
    arrays=path.with_suffix(".npz")
    np.savez_compressed(arrays,numbers=data["numbers"],cell_A=data["cell_A"],positions_A=data["positions_A"][index],
        masses_amu=data["masses_amu"],parent_indices=data["parent_indices"],parent_sha256=data["parent_sha256"],
        energy_eV=data["potential_eV"][index],forces_eV_A=data["forces_eV_A"][index],stress_eV_A3=data["stress_eV_A3"][index],
        production_time_fs=data["time_fs"][index],frame_temperature_K=data["temperature_K"][index])
    return [file_record(path),file_record(arrays)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--basin-1-repeat1",type=Path,required=True)
    p.add_argument("--basin-2-repeat1",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args();out=args.output_dir
    policy_path=out/"thermal_selection_policy.json"
    policy=json.loads(policy_path.read_text())
    manifest_path=out/"candidate_manifest.json"
    manifest=json.loads(manifest_path.read_text())
    assert len(manifest["candidates"])==9, "Start from the static+FD manifest; preserve any completed thermal package"
    keys=["numbers","cell_A","positions_A","masses_amu","parent_indices","parent_sha256","time_fs",
          "potential_eV","forces_eV_A","stress_eV_A3","temperature_K","max_force_eV_A",
          "changed_occupancy_sites","nearest_collisions","nearest_margin_min_A"]
    selections=[];events=[]
    for task,source in zip(policy["candidates"],[args.basin_1_repeat1,args.basin_2_repeat1]):
        receipt_path=source.parent/"receipt.json";receipt=json.loads(receipt_path.read_text())
        assert receipt["status"]=="COMPLETE"
        assert receipt["basin_id"]==task["basin_id"]
        assert receipt["phases"]["production"]["duration_ps"]==20.0
        assert receipt["phases"]["equilibration"]["duration_ps"]==5.0
        assert str(receipt["repeat_id"]) in {"1","repeat1","repeat_1","r1"}
        with np.load(source,allow_pickle=False) as archive:
            data={key:archive[key] for key in keys}
        mapping_path=source.parent/"production_mapping.jsonl"
        times=[];nearest_agrees=[];atom_changes=[];diagnostic_flags=[]
        initial=np.asarray(receipt["initial_mapping"]["assigned_parent_li_sites"])
        with mapping_path.open() as handle:
            for line in handle:
                row=json.loads(line);times.append(row["time_fs"])
                unique=(row["hungarian_differs_from_nearest"]==0 and row["nearest_collision_count"]==0 and min(row["nearest_second_gap_A"])>0)
                nearest_agrees.append(unique)
                changes=int(np.count_nonzero(np.asarray(row["assigned_parent_li_sites"])!=initial))
                atom_changes.append(changes)
                diagnostic_flags.append(bool(row["changed_occupancy_sites"] or changes or not unique))
        assert np.array_equal(np.asarray(times),data["time_fs"])
        assert len(times)==10001 and data["time_fs"][-1]==20000.
        unique=np.asarray(nearest_agrees,bool)
        eligible=(data["time_fs"]>=policy["minimum_production_time_fs"]) & (data["changed_occupancy_sites"]==0) & (data["nearest_collisions"]==0) & unique
        indices=np.flatnonzero(eligible)
        if not len(indices):raise ValueError(f"No admissible production frame for {task['basin_id']}; preserve diagnostics")
        if task["candidate_id"]=="DFT_010":
            median=float(np.median(data["potential_eV"][indices]));index=int(indices[np.argmin(np.abs(data["potential_eV"][indices]-median))])
            target=dict(median_admissible_potential_eV=median,selection_distance_eV=float(abs(data["potential_eV"][index]-median)))
        else:
            index=int(indices[np.argmax(data["max_force_eV_A"][indices])]);target={}
        poscar=out/f"{task['candidate_id']}_{task['basin_id']}_repeat1_thermal_frame{index:05d}.POSCAR"
        files=write_frame(data,index,poscar)
        candidate=dict(candidate_id=task["candidate_id"],priority=1,category="selected_production_force_environment",
            status="EXACT_COMPLETE_PRODUCTION_FRAME_PREPARED_DFT_NOT_STARTED",composition="Li36Co48O96",n_atoms=180,
            basin_id=task["basin_id"],repeat_id=receipt["repeat_id"],cell_A=data["cell_A"].tolist(),
            element_order=["Co","Li","O"],geometry_protocol="Exact 2-fs detailed frame from completed fixed-cell UMA NVE production; not relaxed",
            selection=task["selection"],selection_policy_sha256=digest(policy_path),selection_frame_count=len(indices),
            production_frame_index_zero_based=index,production_time_fs=float(data["time_fs"][index]),
            target_temperature_K=receipt["target_temperature_K"],instantaneous_kinetic_temperature_K=float(data["temperature_K"][index]),
            full_rate_production_mean_temperature_K=receipt["phases"]["production"]["full_rate_mean_temperature_K"],
            minimum_nearest_second_margin_A=float(data["nearest_margin_min_A"][index]),
            atom_resolved_assignment_changes_from_initial=int(atom_changes[index]),
            changed_occupancy_sites=int(data["changed_occupancy_sites"][index]),
            existing_teacher=receipt["teacher"],existing_energy_eV=float(data["potential_eV"][index]),
            existing_max_force_eV_A=float(data["max_force_eV_A"][index]),existing_stress_eV_A3=data["stress_eV_A3"][index].tolist(),
            existing_stress_interpretation="Calculator/configurational stress; no ionic kinetic term, not full finite-temperature pressure",
            source_trajectory=file_record(source),source_mapping=file_record(mapping_path),source_receipt=file_record(receipt_path),
            required_observables=["DFT extrapolated and smeared single-point energies, eV/cell","all force components in exact POSCAR order, eV/A","stress tensor with documented sign/units","magnetic/electronic convergence record"],
            resolves="Targeted finite-temperature force/stress reference at one exact sampled geometry; not a mean DFT error or an equilibrium ensemble estimate",
            files=files,**target)
        manifest["candidates"].append(candidate)
        selections.append(dict(candidate_id=task["candidate_id"],basin_id=task["basin_id"],source=str(source),
            all_detailed_frames=len(times),admissible_frames=len(indices),
            excluded_by_time_floor=int(np.count_nonzero(data["time_fs"]<policy["minimum_production_time_fs"])),
            excluded_by_changed_mask=int(np.count_nonzero(data["changed_occupancy_sites"])),
            frames_with_atom_resolved_changes=int(np.count_nonzero(atom_changes)),
            frames_with_assignment_uniqueness_flags=int(np.count_nonzero(~unique)),
            selected_frame=index,production_time_fs=candidate["production_time_fs"],**target))
        flagged=np.flatnonzero(diagnostic_flags)
        if len(flagged):
            first=int(flagged[0]);path=out/f"diagnostic_event_{task['basin_id']}_production_frame{first:05d}.POSCAR"
            events.append(dict(basin_id=task["basin_id"],scope="First production diagnostic; equilibration audited separately",
                frame_index=first,time_fs=float(data["time_fs"][first]),atom_resolved_changes=int(atom_changes[first]),
                changed_mask_sites=int(data["changed_occupancy_sites"][first]),unique_assignment=bool(unique[first]),
                interpretation="Requires event review; a diagnostic assignment change alone is not a proven Li hop",
                files=write_frame(data,first,path)))
    manifest.update(status="DFT_HANDOFF_COMPLETE_NO_DFT_EXECUTED" if not events else "THERMAL_FRAMES_PREPARED_EVENT_REVIEW_REQUIRED",
        thermal_selection_policy=file_record(policy_path),thermal_selection_receipt=selections,
        production_diagnostic_events=events,pending_additions="None; event review required only when recorded events are nonempty",
        thermal_preparer_sha256=digest(__file__),no_new_MLIP_calls_for_handoff=True)
    manifest_path.write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(status=manifest["status"],candidate_count=len(manifest["candidates"]),thermal=selections,events=events),indent=2))


if __name__=="__main__":main()

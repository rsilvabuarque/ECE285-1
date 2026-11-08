"""Verify two archived MD events and freeze bounded NEB inputs; no MLIP calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
from pathlib import Path

import ase
import numpy as np
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.mep import NEB
from ase.mep.neb import idpp_interpolate
from ase.optimize import FIRE

from neb_common import atoms_from, continuous_hop_displacement, digest, load_npz, mapping, read_json, write_json


EVENTS = (("basin_1/repeat2", 18, 31, 65, 16.702), ("basin_2/repeat1", 11, 23, 59, 16.882))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--md-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from cathode_cvae.runtime import require_compute_allocation
    from cathode_cvae.mlip.registry import MODELS
    require_compute_allocation()
    root, out = args.root.resolve(), args.output_dir.resolve()
    if out.exists(): raise FileExistsError("Preserve the existing prepared NEB protocol")
    gate_path = root / "artifacts/conditioning_20261107/required_gate.json"
    gate = read_json(gate_path)
    if gate.get("status") != "COMPLETE": raise ValueError("Required packet must precede optional NEB")
    if digest(root / gate["report_pdf"]["path"]) != gate["report_pdf"]["sha256"]:
        raise ValueError("Frozen required report changed")
    audit_root = root / "reports/thermal_pilot_20261107/numerics"
    audit_path = audit_root / "md_audit_events.json"
    geometry_path = audit_root / "md_audit_event_geometry_manifest.json"
    audit, geometry = read_json(audit_path), read_json(geometry_path)
    parent_path = root / "artifacts/current/core/data/parents/O3_N48_pilot.npz"
    parent = load_npz(parent_path); parent_hash = digest(parent_path)
    sources = {str(p): digest(p) for p in (gate_path, audit_path, geometry_path, parent_path,
        root / "src/cathode_cvae/mlip/calculators.py", root / "src/cathode_cvae/mlip/registry.py")}
    if ase.__version__ != "3.26.0": raise ValueError("This frozen pilot was reviewed for ASE 3.26.0")
    api = {obj.__name__: dict(signature=str(inspect.signature(obj)), source_sha256=digest(inspect.getsourcefile(obj)))
           for obj in (NEB, FIRE, idpp_interpolate)}
    prepared = []
    out.mkdir(parents=True, exist_ok=False)
    for name, source_site, destination, expected_atom, first_time in EVENTS:
        folder = args.md_root / name; receipt_path = folder / "receipt.json"
        receipt = read_json(receipt_path); sources[str(receipt_path)] = digest(receipt_path)
        if receipt["status"] != "COMPLETE" or not receipt["fixed_cell"] or receipt["parent_sha256"] != parent_hash:
            raise ValueError(f"Incompatible archived trajectory: {name}")
        if receipt["teacher"]["checkpoint_name"] != MODELS["uma"]["name"] or receipt["teacher"]["head"] != "omat" or receipt["teacher"]["dtype"] != "float32" or receipt["teacher"]["tf32"]:
            raise ValueError("Archived event uses a different UMA context")
        archived_hashes = set(receipt["teacher"]["checkpoint_hashes"].values())
        if archived_hashes != set(MODELS["uma"]["files"].values()): raise ValueError("Archived UMA checkpoint changed")
        for filename in ("production.npz", "production.traj", "production_mapping.jsonl"):
            path = folder / filename; actual = digest(path)
            if actual != receipt["outputs"][filename]["sha256"]: raise ValueError(f"Archived source changed: {path}")
            sources[str(path)] = actual
        entry = next(z for z in audit["runs"] if z["identity"] == name)["phases"]["production"]
        if entry["first_atom_assignment_change_ps"] != first_time: raise ValueError("Reported first assignment time differs")
        affected = entry["affected_atoms"]
        if len(affected) != 1: raise ValueError("Pilot requires one marked migrating atom")
        actual_atom = affected[0]
        if (actual_atom["atom_index_0based"], actual_atom["original_parent_Li_site_0based"], actual_atom["final_assigned_parent_Li_site_0based"]) != (expected_atom, source_site, destination):
            raise ValueError("Actual archived event differs from the proposed event indices")
        extracts = [z for z in geometry["frames"] if z["identity"] == name]
        selected = {tag: next(z for z in extracts if z["tag"] == tag)
                    for tag in ("before_200fs", "first_changed_assignment", "settled_after_100fs")}
        frames, continuous_indices, common_cell = {}, None, None
        wanted = {int(e["detailed_frame_0based"]): tag for tag,e in selected.items()}
        maps = {}
        with (folder / "production_mapping.jsonl").open() as stream:
            import json
            for index, line in enumerate(stream):
                if index in wanted: maps[wanted[index]] = json.loads(line)
        for tag, extraction in selected.items():
            # The archived manifest has old absolute worktree paths. Resolve the
            # exact relative event export in this checkout and verify its bytes.
            path = audit_root / "md_audit_event_geometry" / name / (tag + ".npz")
            if digest(path) != extraction["NPZ_sha256"]: raise ValueError("Observed-frame extraction changed")
            sources[str(path)] = digest(path); frame = load_npz(path)
            atom_frame = read(folder / "production.traj", index=int(extraction["detailed_frame_0based"]))
            if not np.array_equal(atom_frame.numbers, frame["numbers"]) or not np.array_equal(atom_frame.positions, frame["positions_A"]) or not np.array_equal(atom_frame.cell, frame["cell_A"]):
                raise ValueError("Extracted coordinates differ from the original saved trajectory frame")
            if str(frame["source_MD_sha256"].item()) != sources[str(folder / "production.npz")]:
                raise ValueError("Frame provenance points to another detailed MD archive")
            if not np.array_equal(frame["cell_A"], parent["cell_A"]) or not atom_frame.pbc.all():
                raise ValueError("Events must use one exact common periodic fixed cell")
            if continuous_indices is None:
                continuous_indices, common_cell = frame["parent_indices"], frame["cell_A"]
            if not np.array_equal(frame["parent_indices"], continuous_indices) or not np.array_equal(frame["cell_A"], common_cell):
                raise ValueError("Independent atom reordering/cell variation is forbidden")
            atoms = atoms_from(frame); fresh_map = mapping(atoms, parent, continuous_indices)
            if fresh_map["assigned_parent_Li_sites"] != maps[tag]["assigned_parent_li_sites"]:
                raise ValueError("Independent coordinate mapping disagrees with archived assignment")
            frames[tag] = (frame, atoms, fresh_map, extraction)
        before, after = frames["before_200fs"], frames["settled_after_100fs"]
        li_rows = before[2]["local_Li_atom_indices"]; ordinal = li_rows.index(expected_atom)
        pre_assignment, post_assignment = before[2]["assigned_parent_Li_sites"], after[2]["assigned_parent_Li_sites"]
        changed = np.flatnonzero(np.asarray(pre_assignment) != np.asarray(post_assignment))
        if changed.tolist() != [ordinal] or pre_assignment[ordinal] != source_site or post_assignment[ordinal] != destination:
            raise ValueError("Observed endpoints are not the declared single-atom event")
        if source_site in post_assignment or destination in pre_assignment:
            raise ValueError("Marked source/destination occupancy is not occupied-to-empty")
        li_indices = parent["li_indices"]
        raw_hop = parent["positions_A"][li_indices[destination]]-parent["positions_A"][li_indices[source_site]]
        first_frame = int(before[3]["detailed_frame_0based"])
        last_frame = int(after[3]["detailed_frame_0based"])
        with Trajectory(folder / "production.traj", "r") as trajectory:
            continuous = []
            for index in range(first_frame,last_frame+1):
                im = trajectory[index]
                if not np.array_equal(im.numbers,before[1].numbers) or not np.array_equal(im.cell,common_cell) or not im.pbc.all():
                    raise ValueError("Continuous event segment changed atom order, PBC or common cell")
                continuous.append(im.positions)
        continuous = np.asarray(continuous)
        branch = continuous_hop_displacement(continuous,before[1].numbers,parent["positions_A"],
            continuous_indices,common_cell,expected_atom)
        observed = np.asarray(branch["displacement_A"])
        image = np.rint((observed-raw_hop) @ np.linalg.inv(common_cell)).astype(int)
        hop = raw_hop + image @ common_cell
        event_id = name.replace("/", "_"); event_dir = out / event_id; event_dir.mkdir()
        for label, tag in (("initial", "before_200fs"), ("transit", "first_changed_assignment"), ("final", "settled_after_100fs")):
            frame = frames[tag][0]
            np.savez_compressed(event_dir / (label + ".npz"), positions_A=frame["positions_A"],
                numbers=frame["numbers"], cell_A=frame["cell_A"], pbc=np.ones(3, bool), parent_indices=continuous_indices,
                source_time_ps=frame["time_ps"], source_detailed_frame=frame["source_detailed_frame"])
        event = dict(event_id=event_id, source_run=name, migrating_atom_index=expected_atom,
            original_parent_atom_index=int(continuous_indices[expected_atom]), Li_ordinal=ordinal,
            source_parent_Li_site=source_site, destination_parent_Li_site=destination,
            periodic_hop_image=image.tolist(), ideal_hop_vector_A=hop.tolist(), ideal_hop_length_A=float(np.linalg.norm(hop)),
            observed_host_aligned_displacement_A=observed.tolist(), expected_initial_assignment=pre_assignment,
            continuous_image_audit=branch,
            continuous_segment_positions_sha256=__import__("hashlib").sha256(continuous.astype("<f8").tobytes()).hexdigest(),
            n_Li=len(li_rows), n_atoms=len(before[1]), composition=before[1].get_chemical_formula(),
            expected_final_assignment=post_assignment,
            first_assignment_change_ps=first_time, persistent_destination_start_ps=entry["state_segments"][-1]["first_ps"],
            transit_assignment=frames["first_changed_assignment"][2]["assigned_parent_Li_sites"][ordinal],
            frame_times_ps={tag:float(value[0]["time_ps"]) for tag,value in frames.items()},
            frame_source_indices={tag:int(value[0]["source_detailed_frame"]) for tag,value in frames.items()},
            atom_order="Original saved MD atom order and continuous unwrapped coordinates; no independent canonicalization or wrapping",
            endpoint_inputs={label:dict(path=f"{event_id}/{label}.npz",sha256=digest(event_dir/(label+".npz"))) for label in ("initial","transit","final")})
        write_json(event_dir / "event.json", event); prepared.append(event)
    # The numerical targets below are provisional optimization criteria from
    # the goal, not a claim that MLIP or DFT saddle forces are physically exact.
    protocol = dict(status="PREPARED_NO_MLIP_EVALUATION", frozen_UTC=datetime.now(timezone.utc).isoformat(),
        required_report_gate=dict(path=str(gate_path),sha256=sources[str(gate_path)],completed_UTC=gate["completed_UTC"]),
        parent=dict(path=str(parent_path),sha256=parent_hash), events=prepared,
        maximum_events=2, maximum_allocated_GPU_device_hours=2., per_event_wall_seconds=2400,
        maximum_calculator_EF_attempts_per_event=10000,
        optional_MACE_profile_wall_seconds=600, two_GPU_window_maximum_wall_seconds=3600,
        planned_two_GPU_window_seconds=3000, planned_maximum_allocated_GPU_device_hours=5/3,
        reserve_allocated_GPU_device_hours=1/3, root_controls_allocations_and_stops=True,
        allocated_budget_rule="Both GPUs count throughout the NEB window, including startup/cleanup and the idle second GPU during optional MACE profiling. The owner must end the two-GPU window within3600s, even if optimizers remain unfinished.",
        teachers={name:MODELS[name] for name in ("uma","mace")}, installed_ASE=ase.__version__, installed_API=api,
        endpoint_relaxation=dict(cell="fixed common cell", movable="all atoms", constraints="none",
            optimizer="FIRE", max_steps_per_endpoint=500, Fmax_eV_A=.03, maxstep_A=.05, dt_ASE=.1, dtmax_ASE=1.),
        initial_band=dict(interior_images=5, method="improvedtangent", k_eV_A2=.1, parallel=False,
            linear_interpolation_MIC=False, branch="Saved continuous unwrapped MD branch retained for the marked atom",
            IDPP_pair_distances_MIC=True, IDPP_max_steps=100, IDPP_optimizer_fmax=.1,
            IDPP_threshold_scope="Auxiliary IDPP force-like optimizer residual, not a physical MLIP force target; ASE3.26 documented default used provisionally",
            initial_Fmax_eV_A=.10, initial_max_steps=300, climbing_Fmax_eV_A=.05, climbing_max_steps=700,
            optimizer="FIRE", maxstep_A=.05, dt_ASE=.1, dtmax_ASE=1.),
        refinement=dict(event_id=prepared[0]["event_id"], interior_images=7, climbing_Fmax_eV_A=.03,
            max_steps=400, initialize="Piecewise arclength resampling of the converged five-interior-image band",
            minimum_remaining_wall_seconds=600, comparison="Report barrier difference; do not silently relax tolerances"),
        numerical_scope="Provisional endpoint and projected NEB maximum-force targets from goal section12. Successful exit/force convergence is not DFT accuracy.",
        k_units_note="Installed ImprovedTangentMethod multiplies spring length (A) by k to form force (eV/A), hence k has eV/A^2. The current web page labels k eV/Ang; implementation dimensional analysis is used here.",
        official_reference=dict(url="https://docs.ase-lib.org/ase/neb.html",accessed_UTC=datetime.now(timezone.utc).isoformat(),
            support="NEB/IDPP APIs, serial image calculators, initial band before climbing, FIRE suitability; current online default differs from installed3.26, so method is explicit",
            publications=["https://doi.org/10.1063/1.1323224","https://doi.org/10.1063/1.1329672"]),
        outcomes="Forward/reverse potential-energy barriers in eV per marked hop; not free energies, diffusivity, or transport-network validation",
        invalid_endpoint_policy="Save failure if intended occupied-to-empty identity event disappears after quench. Equal symmetry-orbit keys alone are not a failure; never force endpoints or substitute barrier zero.",
        MACE_scope="Optional single-point disagreement profile on UMA path, never an optimized MACE NEB barrier",
        no_new_MD=True, no_DFT_or_VASP_inputs=True, actual_MLIP_calls=0,
        source_hashes=sources,
        script_hashes={p.name:digest(p) for p in sorted(Path(__file__).parent.glob("neb_*.py"))})
    changed = [path for path,sha in sources.items() if digest(path) != sha]
    if changed: raise ValueError(f"Source changed during preparation: {changed}")
    write_json(out / "protocol.json", protocol)
    print(__import__("json").dumps(dict(status=protocol["status"],events=[e["event_id"] for e in prepared],actual_MLIP_calls=0),indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())

"""Promote two independently reviewed exact Li-hop frames to the DFT handoff.

Original prespecified thermal selections remain intact. This uses existing
event exports and labels only, with no inference, DFT or geometry optimization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from ase import Atoms


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path,source=None):
    return dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size,source=source)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--candidate-manifest",type=Path,required=True)
    p.add_argument("--event-audit",type=Path,required=True)
    p.add_argument("--event-geometries",type=Path,required=True)
    args=p.parse_args();out=args.candidate_manifest.parent
    manifest=json.loads(args.candidate_manifest.read_text())
    assert len(manifest["candidates"])==11
    events=json.loads(args.event_audit.read_text())
    geometries=json.loads(args.event_geometries.read_text())
    assert geometries["event_audit_sha256"]==digest(args.event_audit)
    event=next(r["phases"]["production"] for r in events["runs"] if r["identity"]=="basin_2/repeat1")
    assert event["affected_atoms"][0]["final_assigned_parent_Li_site_0based"]==23
    assert not event["returned_to_original_after_first_change"]
    chosen=[next(r for r in geometries["frames"] if r["identity"]=="basin_2/repeat1" and r["tag"]==tag)
            for tag in ["first_changed_assignment","settled_after_100fs"]]
    for number,(tag,row) in enumerate(zip(["first_transition","clear_posthop"],chosen),12):
        assert digest(row["POSCAR"])==row["POSCAR_sha256"]
        assert digest(row["NPZ"])==row["NPZ_sha256"]
        time_tag=f"{row['time_ps']:.3f}".replace(".","p")
        poscar=out/f"DFT_{number:03d}_basin2_repeat1_{tag}_{time_tag}ps.POSCAR"
        arrays=poscar.with_suffix(".npz")
        shutil.copyfile(row["POSCAR"],poscar);shutil.copyfile(row["NPZ"],arrays)
        with np.load(arrays,allow_pickle=False) as z:data={key:z[key] for key in z.files}
        assert str(data["source_MD_sha256"])==row["source_production_sha256"]
        assert float(data["energy_eV"])==row["observed_potential_eV"]
        symbols=Atoms(numbers=data["numbers"]).get_chemical_symbols()
        manifest["candidates"].append(dict(candidate_id=f"DFT_{number:03d}",priority=1 if number==12 else 2,
            category="observed_UMA_Li_transition" if number==12 else "observed_UMA_posthop_environment",
            status="EXACT_REVIEWED_PRODUCTION_EVENT_FRAME_PREPARED_DFT_NOT_STARTED",
            composition="Li36Co48O96",n_atoms=len(data["numbers"]),basin_id="basin_2",repeat_id="repeat1",
            cell_A=data["cell_A"].tolist(),element_order=list(dict.fromkeys(symbols)),
            geometry_protocol="Exact saved thermal frame; no new inference or minimization",
            event_tag=row["tag"],production_frame_index_zero_based=row["detailed_frame_0based"],
            production_time_ps=row["time_ps"],affected_local_atom_index_zero_based=row["affected_atom_index_0based"],
            assigned_parent_Li_site_zero_based=row["parent_Li_site_0based"],nearest_second_margin_A=row["nearest_margin_A"],
            distance_original_site_A=row["distance_original_site_A"],distance_final_site_A=row["distance_final_site_A"],
            existing_teacher=row["teacher"],existing_energy_eV=float(data["energy_eV"]),
            existing_max_force_eV_A=float(np.linalg.norm(data["forces_eV_A"],axis=1).max()),
            existing_stress_eV_A3=data["stress_eV_A3"].tolist(),
            existing_stress_interpretation="Calculator/configurational stress only; no ionic kinetic term",
            source_trajectory=dict(path=row["source_production_NPZ"],sha256=row["source_production_sha256"]),
            source_event_audit=record(args.event_audit),source_geometry_manifest=record(args.event_geometries),
            selection_reason="Observed persistent nearest-neighbor Li hop makes the full20ps stream unsuitable as initial-basin vibrational entropy; retain the first transition and a clear post-hop environment",
            interpretation="Transition geometry has ambiguous nearest-site assignment during a transit, not a separate confirmed intermediate basin" if number==12 else "Clearly assigned new site, 100fs after first persistent final-site assignment; still an unrelaxed thermal snapshot",
            required_observables=["DFT extrapolated and smeared energy at exact geometry, eV/cell","all forces in POSCAR atom order, eV/A","configurational stress with sign/units recorded","magnetic/electronic convergence record"],
            resolves="Electronic energy and force response at the model's observed transition/new-site geometry; not a migration barrier, rate, diffusion coefficient or DFT free energy",
            files=[record(poscar,row["POSCAR"]),record(arrays,row["NPZ"])]))
    manifest.update(status="DFT_HANDOFF_COMPLETE_OBSERVED_EVENTS_PRESERVED_NO_DFT_EXECUTED",
        event_review=dict(status="REVIEWED_PERSISTENT_LI_NEIGHBOR_HOP_UNDER_UMA",audit=record(args.event_audit),
            interpretation="Transient32/40fs nearest-site segments are transit labels; persistent final site and continuous coordinates support a hop under UMA, not validated real-LCO diffusion",
            original_high_force_selection_preserved=True,extra_transition_and_posthop_candidates=["DFT_012","DFT_013"],
            other_event_preserved_at="reports/thermal_pilot_20261107/numerics/md_audit_event_geometry/basin_1/repeat2"),
        pending_additions="None",reviewed_event_preparer_sha256=digest(__file__),
        no_DFT_run_submitted_or_scheduled=True,no_executable_DFT_deck_prepared=True)
    args.candidate_manifest.write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(status=manifest["status"],candidates=len(manifest["candidates"]),extra_frames=[dict(id=f"DFT_{i:03d}",time_ps=r["time_ps"],tag=r["tag"],nearest_margin_A=r["nearest_margin_A"]) for i,r in enumerate(chosen,12)]),indent=2))


if __name__=="__main__":main()

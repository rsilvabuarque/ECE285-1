"""Extract observed event frames and plot continuous Li geometry; no predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.io import write


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); events_path = args.output / "md_audit_events.json"
    events = json.loads(events_path.read_text()); extracts = []; plot_data = []
    destination = args.output / "md_audit_event_geometry"; destination.mkdir(exist_ok=False)
    for run in events["runs"]:
        event = run["phases"]["production"]
        if event["first_atom_assignment_change_ps"] is None: continue
        root = args.md_root / run["identity"]; receipt = json.loads((root / "receipt.json").read_text())
        assert receipt["status"] == "COMPLETE"
        with np.load(root / "production.npz") as z:
            times = z["time_fs"] / 1000.; positions = z["positions_A"]; numbers = z["numbers"]; cell = z["cell_A"]
            indices = z["parent_indices"]; forces = z["forces_eV_A"]; stress = z["stress_eV_A3"]; energies = z["potential_eV"]
            velocities = z["detailed_velocities_ASE"]; host_rms = z["host_rms_A"]
        with np.load(receipt["arguments"]["parent"]) as z:
            parent_positions = z["positions_A"]; parent_li = z["li_indices"]
        affected = event["affected_atoms"]; assert len(affected) == 1
        atom = affected[0]; row = atom["atom_index_0based"]; old = atom["original_parent_Li_site_0based"]; new = atom["final_assigned_parent_Li_site_0based"]
        is_li = np.isin(indices, parent_li)
        # The MD runner does not wrap positions, so common host translation is
        # continuous. Per-atom Li distances below still use periodic MIC.
        host_translation = (positions[:, ~is_li] - parent_positions[indices[~is_li]][None, :, :]).mean(axis=1)
        aligned_Li = positions[:, row] - host_translation
        _, distance_old = find_mic(aligned_Li - parent_positions[parent_li[old]], cell, pbc=True)
        _, distance_new = find_mic(aligned_Li - parent_positions[parent_li[new]], cell, pbc=True)
        mapping_times = []; margins = []; assignments = []; collisions = []
        with (root / "production_mapping.jsonl").open() as stream:
            for line in stream:
                item = json.loads(line); mapping_times.append(item["time_fs"] / 1000.)
                ordinal = atom["Li_ordinal_0based"]
                margins.append(item["nearest_second_gap_A"][ordinal]); assignments.append(item["assigned_parent_li_sites"][ordinal]); collisions.append(item["nearest_collision_count"])
        assert np.allclose(mapping_times, times, rtol=0, atol=1e-12)
        first = event["first_atom_assignment_change_ps"]
        final_start = event["state_segments"][-1]["first_ps"]
        requested = [("before_200fs", first - .2), ("last_original_assignment", first - .002),
                     ("first_changed_assignment", first), ("settled_after_100fs", final_start + .1), ("final", float(times[-1]))]
        folder = destination / run["identity"]; folder.mkdir(parents=True)
        for tag, requested_time in requested:
            frame = int(np.argmin(np.abs(times - requested_time)))
            atoms = Atoms(numbers=numbers, positions=positions[frame], cell=cell, pbc=True)
            poscar = folder / f"{tag}.POSCAR"; npz = folder / f"{tag}.npz"
            write(poscar, atoms, format="vasp", direct=True, sort=False)
            np.savez_compressed(npz, positions_A=positions[frame], numbers=numbers, cell_A=cell, parent_indices=indices,
                                energy_eV=energies[frame], forces_eV_A=forces[frame], stress_eV_A3=stress[frame],
                                velocities_ASE=velocities[frame], time_ps=times[frame], source_detailed_frame=frame,
                                source_MD_sha256=np.array(receipt["outputs"]["production.npz"]["sha256"]))
            extracts.append(dict(identity=run["identity"], tag=tag, time_ps=float(times[frame]), detailed_frame_0based=frame,
                observed_potential_eV=float(energies[frame]), affected_atom_index_0based=row,
                parent_Li_site_0based=int(assignments[frame]), nearest_margin_A=float(margins[frame]),
                distance_original_site_A=float(distance_old[frame]), distance_final_site_A=float(distance_new[frame]),
                host_RMS_A=float(host_rms[frame]), POSCAR=str(poscar), POSCAR_sha256=sha(poscar), NPZ=str(npz), NPZ_sha256=sha(npz),
                source_production_NPZ=str(root / "production.npz"), source_production_sha256=receipt["outputs"]["production.npz"]["sha256"],
                teacher=receipt["teacher"], geometry_protocol="Exact saved unrelaxed MD frame; no new inference or minimization"))
        late = times >= final_start + .1
        plot_data.append(dict(identity=run["identity"], times=times, first=first, distance_old=distance_old, distance_new=distance_new,
                              margins=np.asarray(margins), assignments=np.asarray(assignments), host_rms=host_rms,
                              atom=atom, late_mean_original_distance_A=float(np.mean(distance_old[late])),
                              late_mean_final_distance_A=float(np.mean(distance_new[late]))))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, len(plot_data), figsize=(6 * len(plot_data), 8), squeeze=False)
    geometry_summaries = []
    for column, z in enumerate(plot_data):
        t = z["times"]
        axes[0,column].plot(t,z["distance_old"],label="Distance to original site"); axes[0,column].plot(t,z["distance_new"],label="Distance to final site")
        axes[0,column].set(title=z["identity"] + f"; atom index {z['atom']['atom_index_0based']} (zero based)",ylabel="Li–parent-site distance (Å)")
        axes[0,column].legend(fontsize=8)
        axes[1,column].plot(t,z["margins"]); axes[1,column].set(ylabel="Nearest / second-site gap (Å)")
        axes[2,column].plot(t,z["host_rms"]); axes[2,column].set(ylabel="Host RMS displacement (Å)",xlabel="Production time (ps)")
        for ax in axes[:,column]: ax.axvline(z["first"],color="k",linestyle="--",linewidth=.8); ax.set_xlim(z["first"] - 1.7,20)
        geometry_summaries.append({k:z[k] for k in ["identity","atom","late_mean_original_distance_A","late_mean_final_distance_A"]})
    fig.tight_layout(); fig.savefig(args.output / "md_audit_event_geometry.pdf"); fig.savefig(args.output / "md_audit_event_geometry.png",dpi=170);plt.close(fig)
    receipt = dict(status="EXTRACTED_OBSERVED_EVENT_GEOMETRIES", script_sha256=sha(__file__), event_audit_sha256=sha(events_path),
                   geometry_summaries=geometry_summaries, frames=extracts,
                   interpretation="Continuous site-distance changes and multi-ps new-site residence corroborate hopping; first few nearly tied site assignments describe a transit region, not additional equilibrated basins. All frames are observed snapshots, not geometry optimizations or DFT results.")
    (args.output / "md_audit_event_geometry_manifest.json").write_text(json.dumps(receipt,indent=2) + "\n")
    print(json.dumps(geometry_summaries,indent=2))


if __name__ == "__main__": main()

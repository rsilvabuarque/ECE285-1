"""Independent completed-file MD audit; no model calls and no submissions.

Window and mode-bin summaries are descriptive. Correlated frames, a stable
energy trace, or a matching mean temperature do not establish equilibrium.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from ase import units


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def write_csv(path, rows):
    if not rows: return
    keys = sorted(set().union(*(x.keys() for x in rows)))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def bins(times_ps, series, width_ps):
    edges = np.arange(0., float(times_ps[-1]) + width_ps * .5, width_ps)
    if len(edges) < 2: edges = np.array([0., float(times_ps[-1])])
    if edges[-1] < times_ps[-1] - 1e-10: edges = np.r_[edges, times_ps[-1]]
    answer = []
    for j, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (times_ps >= left - 1e-10) & ((times_ps < right - 1e-10) if j < len(edges) - 2 else (times_ps <= right + 1e-10))
        if np.any(selected): answer.append((float(left), float(right), selected))
    return answer


def mode_diagnostic(d, hessian, identity, phase, max_frames):
    """Project saved 2fs velocities onto fixed relaxed harmonic modes.

    A normalized mass-weighted eigenvector u gives mode velocity
    qdot = u.T sqrt(M) v_ASE; its kinetic energy is qdot²/2 in eV.
    This tests possible modal imbalance, not a universal equilibration gate.
    """
    v = d["detailed_velocities_ASE"]
    stride = max(1, int(np.ceil(len(v) / max_frames)))
    selected = np.arange(0, len(v), stride)
    vectors = hessian["eigenvectors_mass_weighted"]
    frequencies = hessian["frequencies_signed_THz"]
    if np.any(frequencies <= 0): raise ValueError("Unresolved harmonic modes cannot define this diagnostic")
    sqrt_mass = np.repeat(np.sqrt(d["masses_amu"]), 3)
    nmodes = vectors.shape[1]
    assert vectors.shape == (3 * len(d["masses_amu"]), 3 * len(d["masses_amu"]) - 3)
    quartiles = np.array_split(np.arange(nmodes), 4)
    sums = np.zeros((2, 4)); counts = np.zeros(2, int); max_kinetic_error = 0.
    sums_all = np.zeros(4); times = d["time_fs"][selected]
    split_time = .5 * (float(d["time_fs"][0]) + float(d["time_fs"][-1]))
    for begin in range(0, len(selected), 128):
        ix = selected[begin:begin + 128]
        weighted = v[ix].reshape(len(ix), -1) * sqrt_mass
        velocities = weighted @ vectors
        kinetic = .5 * velocities ** 2
        max_kinetic_error = max(max_kinetic_error, float(np.max(np.abs(kinetic.sum(axis=1) - d["kinetic_eV"][ix]))))
        for q, modes in enumerate(quartiles):
            energy = kinetic[:, modes].sum(axis=1)
            sums_all[q] += float(energy.sum())
            for half in [0, 1]:
                mask = d["time_fs"][ix] < split_time if half == 0 else d["time_fs"][ix] >= split_time
                sums[half, q] += float(energy[mask].sum())
        for half in [0, 1]:
            mask = d["time_fs"][ix] < split_time if half == 0 else d["time_fs"][ix] >= split_time
            counts[half] += int(mask.sum())
    # Pure float64 orthonormal projection tolerance, not physical acceptance.
    assert max_kinetic_error < 1e-9
    rows = []
    global_temperature = float(np.mean(d["temperature_K"][selected]))
    for q, modes in enumerate(quartiles):
        temperature = 2 * sums_all[q] / (len(selected) * len(modes) * units.kB)
        early = 2 * sums[0, q] / (max(counts[0], 1) * len(modes) * units.kB)
        late = 2 * sums[1, q] / (max(counts[1], 1) * len(modes) * units.kB)
        rows.append(dict(identity=identity, phase=phase, frequency_quartile=q + 1, modes=len(modes),
                         minimum_frequency_THz=float(frequencies[modes].min()), maximum_frequency_THz=float(frequencies[modes].max()),
                         selected_frames=len(selected), detailed_frame_stride=stride,
                         mode_temperature_K=float(temperature), early_half_mode_temperature_K=float(early), late_half_mode_temperature_K=float(late),
                         sampled_global_temperature_K=global_temperature, mode_to_global_temperature=float(temperature / global_temperature),
                         kinetic_projection_max_error_eV=max_kinetic_error))
    return rows


def trajectory_audit(d, receipt, phase, identity):
    n = len(d["masses_amu"]); ndof = 3 * n - 3
    detailed_t, velocity_t = d["time_fs"], d["velocity_time_fs"]
    for key, value in d.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all(): raise ValueError(f"Nonfinite {identity}:{phase}:{key}")
    assert n == 180 and ndof == 537
    assert int(d["constrained_dof"]) == 3 and bool(d["COM_removed"])
    assert len(detailed_t) == len(d["positions_A"]) == len(d["detailed_velocities_ASE"])
    assert len(velocity_t) == len(d["velocities_ASE"]) == len(d["velocity_energy_total_eV"])
    assert np.isclose(detailed_t[0], 0.) and np.isclose(velocity_t[0], 0.)
    assert np.isclose(detailed_t[-1], velocity_t[-1])
    assert np.allclose(np.diff(detailed_t), float(d["detailed_interval_fs"]), rtol=0, atol=1e-10)
    assert np.allclose(np.diff(velocity_t), float(d["timestep_fs"]), rtol=0, atol=1e-10)
    conversion_error = float(np.max(np.abs(d["velocities_A_ps"] - d["velocities_ASE"] * units.fs * 1000)))
    kinetic = .5 * np.einsum("fai,a,fai->f", d["velocities_ASE"], d["masses_amu"], d["velocities_ASE"], optimize=True)
    kinetic_error = float(np.max(np.abs(kinetic - d["velocity_kinetic_eV"])))
    temperature_error = float(np.max(np.abs(2 * kinetic / (ndof * units.kB) - d["velocity_temperature_K"])))
    energy_error = float(np.max(np.abs(d["velocity_energy_total_eV"] - d["velocity_potential_eV"] - d["velocity_kinetic_eV"])))
    com = np.einsum("fai,a->fi", d["velocities_ASE"], d["masses_amu"], optimize=True) / d["masses_amu"].sum()
    max_com = float(np.linalg.norm(com, axis=1).max())
    # Algebraic zero-COM check: a summation-roundoff scale grows with atom
    # count and velocity scale. This is a software constraint check, not a
    # material-dependent transport tolerance.
    com_roundoff_scale = 4 * n * np.finfo(float).eps * max(float(np.max(np.abs(d["velocities_ASE"]))), np.finfo(float).tiny)
    assert max_com < com_roundoff_scale
    match_indices = np.rint(detailed_t / float(d["timestep_fs"])).astype(int)
    detail_velocity_error = float(np.max(np.abs(d["detailed_velocities_ASE"] - d["velocities_ASE"][match_indices])))
    detail_energy_error = float(np.max(np.abs(d["energy_total_eV"] - d["velocity_energy_total_eV"][match_indices])))
    # These test stored-array consistency in float64, not physical accuracy.
    assert conversion_error == 0. and kinetic_error < 1e-10 and temperature_error < 1e-8 and energy_error < 1e-10
    assert detail_velocity_error == 0. and detail_energy_error < 1e-10
    times = velocity_t / 1000.; energies = d["velocity_energy_total_eV"]
    temp = d["velocity_temperature_K"]; potential = d["velocity_potential_eV"]
    fit = np.polyfit(times, energies, 1)
    detrended = energies - np.polyval(fit, times)
    duration = float(times[-1]); early_width = min(1., duration / 4)
    early = times < early_width; late = times >= duration - early_width
    before_late = (times >= duration - 2 * early_width) & (times < duration - early_width)
    row = dict(identity=identity, phase=phase, basin_id=str(d["basin_id"]), repeat_id=str(d["repeat_id"]),
        ensemble=receipt["phases"][phase]["ensemble"], duration_ps=duration,
        detailed_frames=len(detailed_t), velocity_frames=len(velocity_t),
        velocity_interval_fs=float(d["timestep_fs"]), detailed_interval_fs=float(d["detailed_interval_fs"]),
        target_temperature_K=float(d["target_temperature_K"]), mean_temperature_K=float(temp.mean()), temperature_std_K=float(temp.std()),
        early_window_ps=early_width, early_temperature_K=float(temp[early].mean()), tail_temperature_K=float(temp[late].mean()),
        previous_tail_temperature_K=float(temp[before_late].mean()), tail_minus_previous_window_K=float(temp[late].mean() - temp[before_late].mean()),
        tail_minus_target_K=float(temp[late].mean() - float(d["target_temperature_K"])),
        mean_total_energy_eV=float(energies.mean()), mean_potential_energy_eV=float(potential.mean()),
        mean_kinetic_eV=float(kinetic.mean()), total_energy_slope_eV_ps=float(fit[0]), total_energy_slope_eV_atom_ps=float(fit[0] / n),
        total_energy_span_eV=float(np.ptp(energies)), total_energy_detrended_rms_eV=float(np.sqrt(np.mean(detrended ** 2))),
        max_COM_speed_ASE=max_com, COM_roundoff_check_scale_ASE=com_roundoff_scale, velocity_conversion_max_error_A_ps=conversion_error,
        kinetic_max_error_eV=kinetic_error, temperature_max_error_K=temperature_error,
        energy_identity_max_error_eV=energy_error, detailed_velocity_match_error_ASE=detail_velocity_error,
        detailed_energy_match_error_eV=detail_energy_error,
        maximum_force_eV_A=float(d["max_force_eV_A"].max()), maximum_host_RMS_A=float(d["host_rms_A"].max()),
        maximum_host_displacement_A=float(d["host_max_A"].max()), minimum_nearest_site_margin_A=float(d["nearest_margin_min_A"].min()),
        maximum_assignment_distance_A=float(d["assignment_distance_max_A"].max()), maximum_nearest_collisions=int(d["nearest_collisions"].max()),
        any_changed_occupation=bool(np.any(d["changed_occupancy_sites"])),
        fraction_detailed_frames_changed=float(np.mean(d["changed_occupancy_sites"] != 0)))
    source_mean = receipt["phases"][phase]["full_rate_mean_total_energy_eV"]
    assert abs(row["mean_total_energy_eV"] - source_mean) < 1e-10
    windows = []
    width = min(1., duration / 4) if phase == "equilibration" else min(5., duration / 4)
    for left, right, chosen in bins(times, temp, width):
        windows.append(dict(identity=identity, phase=phase, start_ps=left, stop_ps=right, frames=int(chosen.sum()),
            mean_temperature_K=float(temp[chosen].mean()), temperature_std_K=float(temp[chosen].std()),
            mean_potential_eV=float(potential[chosen].mean()), mean_total_eV=float(energies[chosen].mean())))
    first5 = times <= min(5., duration)
    first_fit = np.polyfit(times[first5], energies[first5], 1)
    row.update(first_common_window_ps=float(times[first5][-1]), first_common_window_energy_slope_eV_ps=float(first_fit[0]),
               first_common_window_energy_span_eV=float(np.ptp(energies[first5])))
    plot = dict(times=times, temperature=temp, energy=energies, detailed_times=detailed_t / 1000., host_rms=d["host_rms_A"], ensemble=row["ensemble"])
    return row, windows, plot


def mapping_audit(path, identity, phase, original_sites):
    masks = set(); changes = 0; count = 0; minimum_margin = np.inf; maximum_assigned = 0.; max_nearest_difference = 0
    assignments = set(); original_sites = np.asarray(original_sites, dtype=int)
    changed_ever = np.zeros(len(original_sites), bool); transitions = np.zeros(len(original_sites), int)
    original_changed_frames = 0; phase_initial_changed_frames = 0; max_changed_atoms = 0
    previous = None; first = None; changed_assignment_minimum_margin = None
    with path.open() as stream:
        for line in stream:
            item = json.loads(line); count += 1; masks.add(item["vacancy_mask_hungarian"])
            current = np.asarray(item["assigned_parent_li_sites"], dtype=int)
            assert current.shape == original_sites.shape
            assignments.add(tuple(map(int, current)))
            if first is None: first = current.copy()
            if previous is not None: transitions += current != previous
            changed = current != original_sites
            changed_ever |= changed
            original_changed_frames += bool(np.any(changed))
            phase_initial_changed_frames += bool(np.any(current != first))
            max_changed_atoms = max(max_changed_atoms, int(changed.sum()))
            if np.any(changed):
                margin = min(item["nearest_second_gap_A"])
                changed_assignment_minimum_margin = margin if changed_assignment_minimum_margin is None else min(changed_assignment_minimum_margin, margin)
            previous = current
            changes += item["changed_occupancy_sites"] != 0
            minimum_margin = min(minimum_margin, min(item["nearest_second_gap_A"]))
            maximum_assigned = max(maximum_assigned, max(item["assignment_distances_A"]))
            max_nearest_difference = max(max_nearest_difference, item["hungarian_differs_from_nearest"])
    return dict(identity=identity, phase=phase, detailed_frames=count, distinct_assigned_masks=len(masks),
                fraction_changed_masks=changes / count, minimum_nearest_margin_A=float(minimum_margin),
                maximum_assigned_distance_A=float(maximum_assigned), maximum_hungarian_nearest_difference=max_nearest_difference,
                unique_atom_resolved_assignments=len(assignments),
                Li_atoms_with_assignment_different_from_original=int(changed_ever.sum()),
                fraction_frames_atom_assignment_differs_from_original=original_changed_frames / count,
                fraction_frames_atom_assignment_differs_from_phase_initial=phase_initial_changed_frames / count,
                maximum_atom_assignments_changed_from_original=max_changed_atoms,
                atom_resolved_assignment_transition_count=int(transitions.sum()),
                assignment_transitions_per_Li_atom=json.dumps(transitions.tolist()),
                changed_assignment_minimum_nearest_margin_A=changed_assignment_minimum_margin,
                interpretation="Occupation masks and atom-resolved assignments are separate. Assignment transitions can reveal identical-Li exchanges, but geometric changes alone are not conclusive hopping when mapping is ambiguous")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--md-root", type=Path, required=True); p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--prefix", default="md_audit")
    p.add_argument("--max-mode-frames", type=int, default=2048)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rows = []; windows = []; modes = []; mappings = []; pending = []; paths = {}; plots = {}
    candidates = sorted(args.md_root.glob("**/receipt.json"))
    for path in candidates:
        receipt = json.loads(path.read_text()); identity = str(path.parent.relative_to(args.md_root))
        if receipt["status"] != "COMPLETE": pending.append(dict(identity=identity,status=receipt["status"])); continue
        assert receipt["command"] == "md"
        teacher = receipt["teacher"]
        assert teacher["checkpoint_name"] == "UMA-S-1.2.1" and teacher["head"] == "omat" and teacher["dtype"] == "float32" and not teacher["tf32"]
        paths[str(path)] = sha(path)
        basin = receipt["basin_id"]
        parent = load(receipt["arguments"]["parent"])
        parent_site_lookup = {int(index): j for j, index in enumerate(parent["li_indices"])}
        hessian_path = args.results / f"hessian_assembled/{basin}/delta_.01/hessian.npz"
        hessian = load(hessian_path); paths[str(hessian_path)] = sha(hessian_path)
        assert str(hessian["input_sha256"]) == receipt["input_sha256"]
        hessian_receipt_path = hessian_path.parent / "receipt.json"
        hessian_receipt = json.loads(hessian_receipt_path.read_text())
        paths[str(hessian_receipt_path)] = sha(hessian_receipt_path)
        assert hessian_receipt["teacher"]["checkpoint_hashes"] == teacher["checkpoint_hashes"]
        for phase in ["equilibration", "production"]:
            phase_path = path.parent / f"{phase}.npz"; d = load(phase_path); paths[str(phase_path)] = sha(phase_path)
            assert paths[str(phase_path)] == receipt["outputs"][f"{phase}.npz"]["sha256"]
            for key in ["cell_A", "numbers", "masses_amu", "parent_indices"]:
                assert np.array_equal(d[key], hessian[key]), f"MD/Hessian {key} mapping differs"
            row, current_windows, plot = trajectory_audit(d, receipt, phase, identity)
            rows.append(row); windows.extend(current_windows); plots[(identity, phase)] = plot
            modes.extend(mode_diagnostic(d, hessian, identity, phase, args.max_mode_frames))
            original_sites = [parent_site_lookup[int(index)] for index in d["parent_indices"] if int(index) in parent_site_lookup]
            mappings.append(mapping_audit(path.parent / f"{phase}_mapping.jsonl", identity, phase, original_sites))
            print(json.dumps({k:row[k] for k in ["identity","phase","mean_temperature_K","tail_temperature_K","total_energy_slope_eV_ps","any_changed_occupation"]}), flush=True)
    if not rows: raise RuntimeError("No COMPLETE MD trajectories; refusing to audit partial files")
    write_csv(args.output / f"{args.prefix}_summary.csv", rows)
    write_csv(args.output / f"{args.prefix}_windows.csv", windows)
    write_csv(args.output / f"{args.prefix}_mode_temperatures.csv", modes)
    write_csv(args.output / f"{args.prefix}_mapping.csv", mappings)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for (identity, phase), z in plots.items():
        stride = max(1, int(np.ceil(len(z["times"]) / 1500)))
        ax = axes[0, 0] if phase == "equilibration" else axes[0, 1]
        ax.plot(z["times"][::stride], z["temperature"][::stride], alpha=.65, linewidth=.8, label=identity)
        if phase == "production":
            if z["ensemble"] == "nve": axes[1, 0].plot(z["times"][::stride], (z["energy"][::stride] - z["energy"].mean()) * 1000 / 180, label=identity)
            axes[1, 1].plot(z["detailed_times"], z["host_rms"], linewidth=.8, label=identity)
    for ax in axes[0]: ax.axhline(433.15, color="k", linestyle="--", linewidth=.7); ax.set_ylabel("Kinetic temperature (K)"); ax.set_xlabel("Phase time (ps)")
    axes[0,0].set_title("Bussi equilibration; startup and tail are separate")
    axes[0,1].set_title("Actual production temperature")
    axes[1,0].set(xlabel="Production time (ps)", ylabel="Etotal − own mean (meV/atom)", title="NVE energy conservation")
    axes[1,1].set(xlabel="Production time (ps)", ylabel="Host RMS displacement (Å)", title="Host geometry relative to parent")
    for ax in axes.flat: ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(args.output / f"{args.prefix}_traces.pdf"); fig.savefig(args.output / f"{args.prefix}_traces.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    identities = sorted({r["identity"] for r in modes}); width = .8 / len(identities)
    for i, identity in enumerate(identities):
        chosen = sorted([r for r in modes if r["identity"] == identity and r["phase"] == "production"], key=lambda r:r["frequency_quartile"])
        ax.bar(np.arange(4) + (i - .5 * (len(identities)-1)) * width, [r["mode_to_global_temperature"] for r in chosen], width=width, label=identity)
    ax.axhline(1, color="k", linestyle="--", linewidth=.8)
    ax.set(xticks=range(4), xticklabels=["Lowest quarter", "Second quarter", "Third quarter", "Highest quarter"], ylabel="Mode-group T / sampled global T", title="Relaxed-harmonic velocity projection; descriptive equipartition diagnostic")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(args.output / f"{args.prefix}_modes.pdf"); fig.savefig(args.output / f"{args.prefix}_modes.png", dpi=160); plt.close(fig)
    production = [r for r in rows if r["phase"] == "production"]
    independent = [r for r in production if r["repeat_id"] in ["repeat1", "repeat2"]]
    seed_groups = []
    for basin in sorted({r["basin_id"] for r in independent}):
        selected = [r for r in independent if r["basin_id"] == basin]
        seed_groups.append(dict(basin_id=basin,repeats=len(selected), mean_temperature_range_K=[min(r["mean_temperature_K"] for r in selected),max(r["mean_temperature_K"] for r in selected)],mean_total_energy_range_eV=[min(r["mean_total_energy_eV"] for r in selected),max(r["mean_total_energy_eV"] for r in selected)],interpretation="Independent seed spread is descriptive; two repeats do not establish a calibrated confidence interval"))
    receipt = dict(status="PASS_COMPLETED_ARRAY_CONTRACT" if not pending else "PARTIAL_COMPLETE_FILES_ONLY",
        checked_unix=time.time(), source_sha256=sha(__file__), inputs=paths, pending=pending,
        completed_runs=len(production), phase_summaries=rows, mapping_summaries=mappings, independent_seed_descriptives=seed_groups,
        normal_mode_diagnostic=dict(reference="Same fixed-cell relaxed UMA Hessian eigenvectors; four nearly equal mode-count frequency quartiles",maximum_frames_per_phase=args.max_mode_frames,kinetic_projection="K_s = (u_s.T sqrt(M) v_ASE)^2 / 2; T_group = 2 mean(sum_s K_s)/(number_modes kB)",interpretation="Differences can reflect finite-time mode sampling, non-equipartition or anharmonic mixing; this is not a universal pass/fail equilibrium criterion"),
        time_step_control_limitation="The0.25fs control and0.5fs run use the same initial seed but independently evolved thermostatted warmups; their NVE starting states differ. Matched first5ps drift/span is a diagnostic, not a clean same-state error-order test.",
        temperature_interpretation="Whole equilibration includes startup from a relaxed minimum. Tail/window summaries, production windows and independent seeds must be assessed separately; target433.15K is not substituted for observed temperature.",
        statistical_interpretation="Time-window and seed differences are descriptive, without treating adjacent frames as independent or manufacturing confidence intervals")
    (args.output / f"{args.prefix}.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__": main()

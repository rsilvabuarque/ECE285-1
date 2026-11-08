"""Audit paired deletion and fixed-cell endpoint labels without new inference.

The portable input is the frozen geometry NPZ. Optional historical source
receipts establish the executed teacher/optimizer settings; they are provenance,
not runtime defaults. Parent-site occupation classes are not certified PES basins.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ase import Atoms
from scipy.stats import kendalltau, spearmanr


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def rows_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def canonical(occupancy, permutations):
    """Byte-string vacancy convention avoids integer mask size limitations."""
    vacancy = 1 - np.asarray(occupancy, dtype=np.uint8)
    return min(vacancy[p].tobytes() for p in permutations).hex()


def q20(values):
    return float(np.sort(values)[int(np.ceil(.2 * len(values))) - 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--source-root", type=Path)
    a = ap.parse_args()
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=False)
    source = a.snapshot / "core/data/geometry_uma_fixed.npz"
    parent_path = a.snapshot / "core/data/parents/O3_N48_pilot.npz"
    with np.load(source, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    with np.load(parent_path, allow_pickle=False) as z:
        parent = {k: z[k] for k in z.files}
    numbers = parent["numbers"]
    xyz = parent["positions_A"]
    cell = parent["cell_A"]
    mobile = parent["li_indices"].astype(int)
    permutations = parent["permutations"].astype(int)
    assert d["occupancy"].shape[1] == len(mobile)
    assert np.all((d["occupancy"] == 0) | (d["occupancy"] == 1))
    assert np.all(d["occupancy"].sum(1) == d["n_Li"])
    assert np.array_equal(d["presence"][:, mobile], d["occupancy"].astype(bool))
    host = np.setdiff1d(np.arange(len(numbers)), mobile)
    assert d["presence"][:, host].all()
    assert np.array_equal(d["cell_A"], np.broadcast_to(cell, d["cell_A"].shape))
    assert np.isfinite(d["energy_eV"]).all()
    assert np.all(d["displacement_A"][~d["presence"]] == 0)

    # No geometry-merging tolerance is introduced. A nearest-parent-site
    # assignment is checked directly and ambiguities/collisions remain visible.
    by_id = defaultdict(list)
    for i, sid in enumerate(d["structure_ids"]):
        by_id[str(sid)].append(i)
    groups = []
    start_keys, end_keys = [], []
    for sid, ids in sorted(by_id.items()):
        ids = np.array(ids)
        deletion = ids[d["cohort"][ids] == "deletion"]
        endpoint = ids[d["cohort"][ids] == "fixed_endpoint"]
        assert len(deletion) == len(endpoint) == 1
        i, j = int(deletion[0]), int(endpoint[0])
        assert len(set(d["split"][ids])) == 1
        assert np.array_equal(d["occupancy"][ids], np.broadcast_to(d["occupancy"][i], d["occupancy"][ids].shape))
        assert np.all(d["step"][i] == 0) and d["step"][j] > 0
        slots = np.flatnonzero(d["presence"][j])
        positions = xyz[slots] + d["displacement_A"][j, slots]
        combined = Atoms(numbers=np.r_[numbers[slots], numbers],
                         positions=np.r_[positions, xyz], cell=cell, pbc=True)
        distances = combined.get_all_distances(mic=True)[:len(slots), len(slots):]
        assigned = []
        margins = []
        for local, slot in enumerate(slots):
            eligible = np.flatnonzero(numbers == numbers[slot])
            order = np.argsort(distances[local, eligible], kind="stable")
            assigned.append(int(eligible[order[0]]))
            margins.append(float(distances[local, eligible[order[1]]] - distances[local, eligible[order[0]]]))
        assigned = np.array(assigned)
        mobile_local = np.isin(slots, mobile)
        assigned_mobile = assigned[mobile_local]
        new_occupancy = np.isin(mobile, assigned_mobile).astype(np.uint8)
        collisions = len(assigned_mobile) - len(set(assigned_mobile))
        key0 = canonical(d["occupancy"][i], permutations)
        key1 = canonical(new_occupancy, permutations)
        start_keys.append(key0)
        end_keys.append(key1)
        n = int(d["n_Li"][i])
        r = dict(structure_id=sid, n_Li=n, split=str(d["split"][i]),
                 deletion_frame=i, endpoint_frame=j,
                 deletion_energy_eV=float(d["energy_eV"][i]),
                 endpoint_energy_eV=float(d["energy_eV"][j]),
                 relaxation_gain_meV_atom=float(1000*(d["energy_eV"][j]-d["energy_eV"][i])/len(slots)),
                 endpoint_fmax_eV_A=float(d["force_max_eV_A"][j]),
                 endpoint_converged=bool(d["endpoint_converged"][j]),
                 all_saved_frames_localized=bool(d["localized"][ids].all()),
                 endpoint_Li_assignment_changes=int(np.sum(assigned[mobile_local] != slots[mobile_local])),
                 endpoint_host_assignment_changes=int(np.sum(assigned[~mobile_local] != slots[~mobile_local])),
                 endpoint_Li_site_collisions=collisions,
                 minimum_nearest_second_site_margin_A=min(margins),
                 initial_occupation_orbit=key0, endpoint_occupation_orbit=key1,
                 endpoint_occupation_orbit_changed=key0 != key1,
                 endpoint_step=int(d["step"][j]))
        groups.append(r)
    rows_csv(out / "paired_lineages.csv", groups)

    summaries = []
    support = []
    for n in sorted({r["n_Li"] for r in groups}):
        rr = [r for r in groups if r["n_Li"] == n]
        accepted = [r for r in rr if r["endpoint_converged"]]
        e0 = np.array([r["deletion_energy_eV"] for r in accepted])
        e1 = np.array([r["endpoint_energy_eV"] for r in accepted])
        lo0, lo1 = e0 <= q20(e0), e1 <= q20(e1)
        summaries.append(dict(n_Li=n, attempted_lineages=len(rr), converged=len(accepted),
            failed_or_step_limited=len(rr)-len(accepted),
            spearman=float(spearmanr(e0, e1).statistic), kendall=float(kendalltau(e0, e1).statistic),
            deletion_low20_count=int(lo0.sum()), endpoint_low20_count=int(lo1.sum()),
            low20_intersection=int((lo0 & lo1).sum()),
            deletion_low20_precision=float((lo0 & lo1).sum()/lo0.sum()),
            deletion_low20_recall=float((lo0 & lo1).sum()/lo1.sum()),
            median_relaxation_gain_meV_atom=float(np.median([r["relaxation_gain_meV_atom"] for r in accepted])),
            changed_Li_assignment=sum(r["endpoint_Li_assignment_changes"] > 0 for r in rr),
            changed_occupation_orbit=sum(r["endpoint_occupation_orbit_changed"] for r in rr)))
        train = [r for r in rr if r["split"] == "train"]
        for target in ["deletion", "endpoint"]:
            edge = q20(np.array([r[f"{target}_energy_eV"] for r in train]))
            for split in ["train", "val", "test"]:
                ss = [r for r in rr if r["split"] == split]
                support.append(dict(n_Li=n, target=target, split=split, lineages=len(ss),
                                    train_q20_eV=edge, low_lineages=sum(r[f"{target}_energy_eV"] <= edge for r in ss)))
    rows_csv(out / "composition_ranking.csv", summaries)
    rows_csv(out / "label_target_support.csv", support)

    source_receipts = {}
    if a.source_root:
        # Verify source scalar rows against the exported portable corpus. The
        # original attempt remains authoritative rather than selecting by energy.
        logical = list(csv.DictReader((a.source_root / "data/logical_completion_uma_fixed.csv").open()))
        by_sid = {r["structure_id"]: r for r in groups}
        model_receipt_keys = []
        worker_receipts = {}
        for row in logical:
            raw_path = Path(row["source_npz"])
            record_path = raw_path.with_suffix(".json")
            record = json.loads(record_path.read_text())
            assert record["model"] == "uma" and record["parent"] == "O3_N48_pilot"
            assert record["cell"] is False and record["cell_relaxed"] is False
            assert record["optimizer"] == "FIRE" and record["max_steps"] == 500
            worker = record["worker"]
            if worker not in worker_receipts:
                worker_path = a.source_root / "workers" / (worker + ".json")
                w = json.loads(worker_path.read_text())
                assert w["model"] == "uma" and w["task"] == "omat"
                assert w["dtype"] == "float32" and w["tf32"] is False
                worker_receipts[worker] = {k: w[k] for k in ["model", "checkpoint_hashes", "dtype", "tf32", "package_version", "torch", "ase", "task", "settings"]}
                worker_receipts[worker]["receipt_sha256"] = digest(worker_path)
            sid = record["structure_id"]
            paired = by_sid[sid]
            assert float(record["initial_energy_eV"]) == paired["deletion_energy_eV"]
            assert float(record["final_energy_eV"]) == paired["endpoint_energy_eV"]
            assert bool(record["converged"]) == paired["endpoint_converged"]
            model_receipt_keys.append(record)
        # Keep the full first representative receipt separately; package/hash
        # consistency is explicitly audited from every executed source record.
        source_receipts["records_checked"] = len(model_receipt_keys)
        source_receipts["first_record"] = model_receipt_keys[0]
        source_receipts["used_worker_receipts"] = worker_receipts
        teacher_identities = {json.dumps({k:v for k,v in w.items() if k != "receipt_sha256"}, sort_keys=True) for w in worker_receipts.values()}
        assert len(teacher_identities) == 1
        for filename in ["calculators.py", "export_geometry.py", "prepare_relaxations.py", "config.json", "validation/geometry_uma_fixed.json", "data/logical_completion_uma_fixed.csv"]:
            source_receipts[filename] = digest(a.source_root / filename)
        (out / "historical_source_receipts.json").write_text(json.dumps(source_receipts, indent=2) + "\n")

    counts = Counter(end_keys)
    split_counts = Counter(r["split"] for r in groups)
    primary_support = [r for r in support if r["n_Li"] in [12,24,36]]
    receipt = dict(status="PASS", input_hashes={str(source):digest(source),str(parent_path):digest(parent_path)},
        frames=len(d["structure_ids"]), lineages=len(groups), splits=dict(split_counts),
        exact_fixed_cell=True, numerical_convergence_criterion_eV_A=.05,
        converged_lineages=sum(r["endpoint_converged"] for r in groups),
        endpoint_fmax_max_eV_A=max(r["endpoint_fmax_eV_A"] for r in groups),
        changed_Li_assignment_lineages=sum(r["endpoint_Li_assignment_changes"] > 0 for r in groups),
        changed_host_assignment_lineages=sum(r["endpoint_host_assignment_changes"] > 0 for r in groups),
        mobile_collision_lineages=sum(r["endpoint_Li_site_collisions"] > 0 for r in groups),
        changed_occupation_orbits=sum(r["endpoint_occupation_orbit_changed"] for r in groups),
        unique_initial_occupation_orbits=len(set(start_keys)), unique_endpoint_occupation_orbits=len(counts),
        endpoint_orbit_merge_groups=sum(v>1 for v in counts.values()),
        primary_composition_lineages=sum(r["n_Li"] in [12,24,36] for r in groups),
        primary_label_support=primary_support,
        median_composition_spearman=float(np.median([r["spearman"] for r in summaries])),
        minimum_composition_spearman=float(min(r["spearman"] for r in summaries)),
        minimum_composition_low20_recall=float(min(r["deletion_low20_recall"] for r in summaries)),
        minimum_nearest_site_margin_A=min(r["minimum_nearest_second_site_margin_A"] for r in groups),
        source_raw_scalar_records_checked=source_receipts.get("records_checked",0),
        sampling="Previously acquisition-selected: about50%random,25%lowUMA,25%teacherdisagreement,balancedcomposition; notpopulationrepresentative.",
        scope="Sameinput deletion/fixedcellUMA endpoints. All attempts included; assignmentorbit collisions are a parent-site structural audit, not proof of unique energy minima or physical basin convergence.",
        label_target_fit="DEFERRED: primary12/24/36cohort has69total lineages; exactprotectedlowclasssupport saved. Do not broadencompositions or resplit to make comparisonlookpowered.")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.5))
    x = np.array([r["n_Li"] for r in summaries])/len(mobile)
    axes[0].plot(x,[r["spearman"] for r in summaries],"o-",label="Spearman rank")
    axes[0].plot(x,[r["deletion_low20_recall"] for r in summaries],"s-",label="Low-quintile recall")
    axes[0].set(xlabel="Li fraction",ylabel="Dimensionless agreement",ylim=(-.05,1.05))
    axes[0].legend(fontsize=8)
    for n in [12,24,36]:
        rr=[r for r in groups if r["n_Li"]==n]
        e0=np.array([r["deletion_energy_eV"] for r in rr]);e1=np.array([r["endpoint_energy_eV"] for r in rr])
        axes[1].scatter(1000*(e0-e0.min())/(len(host)+n),1000*(e1-e1.min())/(len(host)+n),label=f"nLi={n} ({len(rr)} masks)",s=18)
    axes[1].set(xlabel="Deletion energy above cohort minimum (meV/atom)",ylabel="Relaxed energy above cohort minimum (meV/atom)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Existing paired fixed-cell UMA cohort; thresholds fitted within each plotted cohort",fontsize=10)
    fig.tight_layout();fig.savefig(out/"deletion_vs_relaxed_audit.pdf");fig.savefig(out/"deletion_vs_relaxed_audit.png",dpi=160)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ["primary_label_support","input_hashes"]},indent=2))


if __name__ == "__main__":
    main()

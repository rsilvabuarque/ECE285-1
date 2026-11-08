"""Matched-Hamiltonian comparison on the frozen 96 UMA endpoint poses.

The per-composition energy shift is descriptive: fitting one constant removes
the different teachers' reference-energy zero, without fitting a slope or
claiming either teacher is a DFT reference. No classes are calibrated here.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from paired_common import atomic_json, check_prepared, geometry_hash, sha256, verify_teacher_receipt
from paired_variant_worker import verify_variant


def summarize_energy(uma, mace):
    """Within one composition, fit only mean(MACE-UMA), in eV/atom."""
    from scipy.stats import kendalltau, spearmanr

    u, m = np.asarray(uma, float), np.asarray(mace, float)
    if u.ndim != 1 or u.shape != m.shape or not len(u) or not np.isfinite([u, m]).all():
        raise ValueError("Matched finite one-dimensional energy arrays are required")
    delta = m-u
    residual = delta-delta.mean()
    du = u[:, None]-u[None, :]
    dm = m[:, None]-m[None, :]
    i, j = np.triu_indices(len(u), 1)
    product = du[i, j]*dm[i, j]
    return dict(
        pairs=len(u), spearman=float(spearmanr(u, m).statistic),
        kendall_tau_b=float(kendalltau(u, m).statistic),
        mean_MACE_minus_UMA_offset_meV_atom=float(1000*delta.mean()),
        offset_removed_MAE_meV_atom=float(1000*np.mean(np.abs(residual))),
        offset_removed_RMSE_meV_atom=float(1000*np.sqrt(np.mean(residual**2))),
        offset_removed_max_absolute_meV_atom=float(1000*np.max(np.abs(residual))),
        pairwise_concordant=int((product > 0).sum()),
        pairwise_discordant=int((product < 0).sum()),
        pairwise_tied_in_at_least_one_teacher=int((product == 0).sum()),
    )


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    from cathode_cvae.runtime import require_compute_allocation
    from scipy.stats import rankdata

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--primary-queue", type=Path, required=True)
    p.add_argument("--variants", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--mace-sp-directory", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(argv)
    require_compute_allocation()
    _, primary_context = check_prepared(a.prepared)
    _, mace_context, _ = verify_variant(a.prepared, a.variants/"mace_independent")
    manifest = json.loads(a.manifest.read_text())
    if manifest["primary_context_hash"] != primary_context["context_hash"]:
        raise ValueError("Endpoint-export context differs from the primary teacher")
    expected = manifest["records"]
    if manifest["prespecified"] != 96 or len(expected) != 96:
        raise ValueError("This audit is only the prespecified hash-selected 96 scope")
    if any(r["split"] != "train" for r in expected):
        raise ValueError("Only prespecified TRAIN endpoints may be read")
    model = json.loads((a.mace_sp_directory/"model_receipt.json").read_text())
    verify_teacher_receipt(model, mace_context)
    completion = json.loads((a.mace_sp_directory/"run_receipt.json").read_text())
    if completion["status"] != "complete":
        raise ValueError("MACE score run is not terminal")
    raw_sp = [json.loads(s) for s in (a.mace_sp_directory/"results.jsonl").read_text().splitlines() if s.strip()]
    by_sp = {r["structure_id"]: r for r in raw_sp}
    if len(by_sp) != len(raw_sp):
        raise ValueError("Duplicate MACE score identities")
    pred_path = a.mace_sp_directory/"predictions.npz"
    with np.load(pred_path, allow_pickle=False) as z:
        pred = {k: z[k] for k in ["structure_ids", "offsets", "positions", "numbers", "cell", "energy_eV", "forces_eV_A", "stress_eV_A3", "n_atoms"]}
    ids = pred["structure_ids"].astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate successful prediction identities")
    index = {s: i for i, s in enumerate(ids)}
    rows, sources, force_differences, stress_differences = [], [], {}, {}
    for r in expected:
        n, sid = int(r["n_target"]), r["structure_id"]
        score_id = r.get("scoring_structure_id", sid)
        row = dict(structure_id=sid, n_target=n, split="train", status="unknown",
                   UMA_energy_eV_atom=None, MACE_energy_eV_atom=None,
                   UMA_rank=None, MACE_rank=None, UMA_fmax_eV_A=None, MACE_fmax_eV_A=None,
                   force_component_MAE_eV_A=None, force_component_RMSE_eV_A=None,
                   stress_component_MAE_eV_A3=None)
        if not r["exported"] or score_id not in by_sp or by_sp[score_id]["status"] != "success":
            rows.append(row)
            continue
        if score_id not in index:
            raise ValueError("Success record missing from prediction arrays")
        result_path = a.primary_queue/"results"/r["primary_task_id"]/"result.json"
        result = json.loads(result_path.read_text())
        arrays_path = result_path.with_name("arrays.npz")
        arrays_sha = sha256(arrays_path)
        if arrays_sha != r["source_arrays_sha256"] or arrays_sha != result["arrays_sha256"]:
            raise ValueError("Original endpoint evidence changed")
        if result["task"]["split"] != "train" or result["context_hash"] != primary_context["context_hash"]:
            raise ValueError("Unexpected role or primary context")
        if result["task"]["structure_id"] != sid or result["n_target"] != n:
            raise ValueError("Endpoint identity/count mismatch")
        if not result["endpoint_labels_usable"]:
            rows.append(row)
            continue
        with np.load(arrays_path, allow_pickle=False) as z:
            u_pos, u_num, cell = z["endpoint_positions_A"], z["numbers"], z["cell_A"]
            u_energy = float(z["endpoint_energy_eV"])
            u_force, u_stress = z["endpoint_forces_eV_A"], z["endpoint_stress_eV_A3"]
        i = index[score_id]
        lo, hi = pred["offsets"][i:i+2]
        m_pos, m_num = pred["positions"][lo:hi], pred["numbers"][lo:hi]
        literal = geometry_hash(u_num, u_pos, cell)
        if literal != r["literal_geometry_hash"] or literal != geometry_hash(m_num, m_pos, pred["cell"][i]):
            raise ValueError("Teachers were not evaluated on exactly the same ordered coordinates")
        if result["endpoint_literal_geometry_hash"] != literal or u_energy != result["endpoint_energy_eV"]:
            raise ValueError("Primary scalar/header evidence mismatch")
        if int(pred["n_atoms"][i]) != len(u_num) or len(u_num) != result["n_atoms"]:
            raise ValueError("Atom count mismatch")
        m_force, m_stress = pred["forces_eV_A"][lo:hi], pred["stress_eV_A3"][i]
        df, ds = m_force-u_force, m_stress-u_stress
        if df.shape != u_force.shape or ds.shape != u_stress.shape:
            raise ValueError("Force/stress shape mismatch")
        row.update(status="known", UMA_energy_eV_atom=u_energy/len(u_num),
                   MACE_energy_eV_atom=float(pred["energy_eV"][i]/len(u_num)),
                   UMA_fmax_eV_A=float(np.linalg.norm(u_force, axis=1).max()),
                   MACE_fmax_eV_A=float(np.linalg.norm(m_force, axis=1).max()),
                   force_component_MAE_eV_A=float(np.abs(df).mean()),
                   force_component_RMSE_eV_A=float(np.sqrt((df**2).mean())),
                   stress_component_MAE_eV_A3=float(np.abs(ds).mean()))
        rows.append(row)
        force_differences.setdefault(n, []).append(df.ravel())
        stress_differences.setdefault(n, []).append(ds.ravel())
        sources.append(dict(structure_id=sid, result_sha256=sha256(result_path), arrays_sha256=arrays_sha,
                            literal_geometry_hash=literal))
    summaries = []
    for n in sorted({r["n_target"] for r in rows}):
        attempted = [r for r in rows if r["n_target"] == n]
        known = [r for r in attempted if r["status"] == "known"]
        if len(attempted) != 32:
            raise ValueError("Prespecified denominator changed")
        if not known:
            summaries.append(dict(n_target=n, attempts=32, known=0))
            continue
        u = np.array([r["UMA_energy_eV_atom"] for r in known])
        m = np.array([r["MACE_energy_eV_atom"] for r in known])
        for row, urank, mrank in zip(known, rankdata(u), rankdata(m)):
            row.update(UMA_rank=float(urank), MACE_rank=float(mrank))
        df, ds = np.concatenate(force_differences[n]), np.concatenate(stress_differences[n])
        summaries.append(dict(n_target=n, attempts=32, known=len(known), **summarize_energy(u, m),
            UMA_physical_fmax_median_eV_A=float(np.median([r["UMA_fmax_eV_A"] for r in known])),
            MACE_physical_fmax_median_eV_A=float(np.median([r["MACE_fmax_eV_A"] for r in known])),
            force_component_MAE_eV_A=float(np.abs(df).mean()),
            force_component_RMSE_eV_A=float(np.sqrt((df**2).mean())),
            stress_component_MAE_eV_A3=float(np.abs(ds).mean()),
            stress_component_RMSE_eV_A3=float(np.sqrt((ds**2).mean()))))
    a.output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(a.output_dir/"matched_endpoint_rows.csv", rows)
    write_csv(a.output_dir/"composition_summary.csv", summaries)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for j, s in enumerate(summaries):
        known = [r for r in rows if r["n_target"] == s["n_target"] and r["status"] == "known"]
        u = np.array([r["UMA_energy_eV_atom"] for r in known])
        m = np.array([r["MACE_energy_eV_atom"] for r in known])
        axes[0, j].scatter([r["UMA_rank"] for r in known], [r["MACE_rank"] for r in known], s=20)
        axes[0, j].plot([1, 32], [1, 32], "k--", lw=1)
        axes[0, j].set(title=f"Li{s['n_target']} / 48; N={len(known)}\nSpearman={s['spearman']:.3f}", xlabel="UMA rank (low first)", ylabel="MACE rank (low first)")
        axes[1, j].scatter(1000*(u-u.mean()), 1000*(m-m.mean()), s=20)
        lim = np.max(np.abs(1000*np.r_[u-u.mean(), m-m.mean()]))
        axes[1, j].plot([-lim, lim], [-lim, lim], "k--", lw=1)
        axes[1, j].set(xlabel="Centered UMA energy (meV/atom)", ylabel="Centered MACE energy (meV/atom)")
    fig.suptitle("Two teachers on exactly the same UMA-relaxed poses; 32 preselected TRAIN masks/composition")
    fig.savefig(a.output_dir/"matched_endpoint_ranks.pdf")
    fig.savefig(a.output_dir/"matched_endpoint_ranks.png", dpi=180)
    plt.close(fig)
    receipt = dict(status="PASS_COMPLETE" if all(r["status"] == "known" for r in rows) else "COMPLETE_WITH_UNKNOWN_LABELS",
        completed_UTC=datetime.now(timezone.utc).isoformat(), attempts=len(rows), known=sum(r["status"] == "known" for r in rows),
        manifest=dict(path=str(a.manifest.absolute()), sha256=sha256(a.manifest)),
        prediction_sha256=sha256(pred_path), primary_context_hash=primary_context["context_hash"],
        secondary_context_hash=mace_context["context_hash"], source_sha256=sha256(__file__),
        primary_sources=sources, test_energies_read=False, new_model_calls=0,
        definitions=dict(energy_shift="Within each composition, subtract mean(MACE E/N - UMA E/N); no slope fit",
                         force_error="MACE force vector minus UMA force vector on identical ordered atom positions",
                         stress_error="MACE ASE Voigt stress minus UMA ASE Voigt stress; eV/Angstrom^3; tensile-positive",
                         rank="Ascending energy among these 32 prespecified poses only; average ranks for exact ties"),
        limitations="Teacher agreement is not DFT validation. No MACE endpoint thresholds are fitted or borrowed from deletion labels; these force residuals do not represent MACE-relaxed geometries.")
    atomic_json(a.output_dir/"receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ["status", "attempts", "known", "test_energies_read", "new_model_calls"]}, indent=2))


if __name__ == "__main__":
    main()

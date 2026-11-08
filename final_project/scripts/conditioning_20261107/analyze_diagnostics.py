"""Paired CVAE diagnostics from complete frozen proposal and oracle records.

One row is a proposal, not an independent model fit. Energy changes compare
identical latent/uniform streams for condition pairs, or identical uniforms for
latent pairs. Posterior targets belong to protected validation, never test.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from cathode_cvae.conditions import classify, load_schema
from cathode_cvae.runtime import require_compute_allocation


PROJECT = Path(__file__).resolve().parents[2]
GROUP = ("method", "replicate_id", "fit_seed", "n_Li")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)


def keys_from_json(path):
    if path is None:
        return None
    x = json.loads(Path(path).read_text())
    if isinstance(x, dict):
        x = next((x[k] for k in ["canonical_masks", "canonical_keys", "known_keys", "keys"] if k in x), None)
    if not isinstance(x, list):
        raise ValueError("History keys must be a JSON list or a declared key-list field")
    return set(map(str, x))


def load_training(path):
    with np.load(path, allow_pickle=False) as archive:
        d = {k: archive[k] for k in archive.files}
    roles = d["split"].astype(str)
    ids = d["structure_ids"].astype(str)
    target = {s: dict(energy=float(e), split=r) for s, e, r in zip(ids, d["energy_eV_atom"], roles)}
    if "canonical_masks" not in d:
        return None, target
    keys = d["canonical_masks"].astype(str)
    counts = d.get("n_target", d.get("n_Li"))
    if keys.ndim != 1 or len(keys) != len(ids) or any(k.count("0") != int(n) for k, n in zip(keys, counts)):
        raise ValueError("Training canonical_masks must be vacancy-one strings with the declared count")
    for sid, key, role in zip(ids, keys, roles):
        if role in {"val", "validation"}:
            target[sid]["canonical_occupancy"] = 1 - np.asarray([int(b) for b in key], np.uint8)
    return set(keys[roles == "train"]), target


def join_scores(rows, raw, score_path, schema, training_keys=None, history=None, *, allow_incomplete=False):
    """Retain every attempt; do not treat cache-missing proposals as failures or successes."""
    with np.load(score_path, allow_pickle=False) as archive:
        ids = archive["structure_ids"].astype(str)
        total = np.asarray(archive["energy_eV"], float)
        atoms = np.asarray(archive["n_atoms"], float)
        status = archive["status"].astype(str) if "status" in archive else np.full(len(ids), "success")
    if len(set(ids)) != len(ids) or not (ids.shape == total.shape == atoms.shape == status.shape):
        raise ValueError("Oracle scores require unique IDs and matching scalar arrays")
    energy = {s: e / n for s, e, n, ok in zip(ids, total, atoms, status)
              if ok in {"success", "complete", "completed", "ok", "cached", "fresh"}
              and np.isfinite(e) and np.isfinite(n) and n > 0}
    cell_energy = {s: float(e) for s, e, n, ok in zip(ids, total, atoms, status)
                   if s in energy}
    result, missing = [], []
    for source in rows:
        row = dict(source)
        row["n_Li"] = int(row["n_Li"])
        row["replicate_id"] = str(row.get("replicate_id", row.get("fit_seed", "")))
        row["fit_seed"] = str(row.get("fit_seed", ""))
        row["diagnostic"] = row.get("diagnostic", "primary")
        index = int(row["raw_row"]) if row.get("raw_row", "") != "" else None
        valid = bool(index is not None and 0 <= index < len(raw)
                     and np.isin(raw[index], [0, 1]).all() and int(raw[index].sum()) == row["n_Li"])
        sid, key = row.get("scoring_structure_id", ""), row.get("canonical_mask", "")
        valid = valid and len(key) == raw.shape[1] and set(key) <= {"0", "1"} and key.count("0") == row["n_Li"]
        generated = row.get("status", "generated") == "generated"
        known = bool(valid and generated and sid in energy)
        if valid and generated and not known:
            missing.append(row.get("attempt_id", sid))
        measured = str(classify(schema, row["n_Li"], energy[sid])) if known else "unknown_energy"
        row.update(mask_valid=valid, energy_known=known,
            energy_eV_atom=float(energy[sid]) if known else "",
            energy_eV_cell=cell_energy[sid] if known else "", measured_bin=measured,
            requested_bin_hit=bool(known and measured == row["requested_bin"]),
            novel_training="" if training_keys is None else bool(valid and key not in training_keys),
            novel_history="" if history is None else bool(valid and key not in history))
        result.append(row)
    if missing and not allow_incomplete:
        raise ValueError(f"Incomplete exact oracle coverage: {len(missing)} generated attempts; first={missing[:3]}")
    return result, missing


def meta(row):
    return {k: row[k] for k in GROUP}


def pair_difference(a, b, raw, comparison):
    known = bool(a["energy_known"] and b["energy_known"])
    valid = bool(a["mask_valid"] and b["mask_valid"])
    ia, ib = (int(a["raw_row"]), int(b["raw_row"])) if valid else (None, None)
    if a.get("checkpoint_sha256") in {None, "", "not-applicable"}:
        stream_scope = "Sampler control: attempt-index pairing; not common conditional draws"
    elif comparison.startswith("low_to_"):
        stream_scope = "Shared latent and AR uniforms; requested condition changes"
    elif comparison == "change_latent_same_uniforms":
        stream_scope = "Shared AR uniforms; latent is resampled (empty for no-latent AR)"
    else:
        stream_scope = "Shared condition and AR uniforms; posterior latent versus matched prior latent"
    return dict(**meta(a), comparison=comparison, paired_id=a.get("paired_id", ""),
        attempt_a=a.get("attempt_id", ""), attempt_b=b.get("attempt_id", ""),
        requested_a=a["requested_bin"], requested_b=b["requested_bin"], pair_scored=known,
        delta_E_eV_atom=float(b["energy_eV_atom"]-a["energy_eV_atom"]) if known else "",
        delta_E_eV_cell=float(b["energy_eV_cell"]-a["energy_eV_cell"]) if known else "",
        raw_Hamming=int(np.count_nonzero(raw[ia] != raw[ib])) if valid else "",
        same_orbit=bool(valid and a["canonical_mask"] == b["canonical_mask"]),
        random_stream_scope=stream_scope,
        a_bin_hit=a["requested_bin_hit"], b_bin_hit=b["requested_bin_hit"],
        a_novel_training=a["novel_training"], b_novel_training=b["novel_training"],
        a_novel_history=a["novel_history"], b_novel_history=b["novel_history"],
        uniform_fingerprints_equal=(a.get("uniforms_sha256", "") == b.get("uniforms_sha256", ""))
            if a.get("uniforms_sha256") and b.get("uniforms_sha256") else "",
        latent_fingerprints_equal=(a.get("latents_sha256", "") == b.get("latents_sha256", ""))
            if a.get("latents_sha256") and b.get("latents_sha256") else "")


def paired_diagnostics(rows, raw):
    primary, latent = {}, {}
    posterior = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in GROUP) + (row.get("paired_id", ""),)
        diag = row["diagnostic"]
        if diag == "primary":
            k = key + (row["requested_bin"],)
            if k in primary:
                raise ValueError("Primary condition pair is ambiguous")
            primary[k] = row
        elif diag == "latent_B":
            if key in latent:
                raise ValueError("Latent-B pair is ambiguous")
            latent[key] = row
        elif diag.startswith(("posterior_", "prior_matched_")):
            target = row.get("validation_target_id", "")
            k = tuple(row[g] for g in GROUP) + (target,)
            kind = "posterior" if diag.startswith("posterior_") else "prior"
            if kind in posterior[k]:
                raise ValueError("Posterior/prior target pair is ambiguous")
            posterior[k][kind] = row
    conditions, latents, posteriors = [], [], []
    for key, low in primary.items():
        if key[-1] != "low":
            continue
        for target in ["middle", "high"]:
            other = primary.get(key[:-1] + (target,))
            if other is not None:
                conditions.append(pair_difference(low, other, raw, "low_to_" + target))
        other = latent.get(key[:-1])
        if other is not None:
            latents.append(pair_difference(low, other, raw, "change_latent_same_uniforms"))
    for key, pair in sorted(posterior.items()):
        if set(pair) != {"posterior", "prior"}:
            raise ValueError("Posterior diagnostic lacks its matched prior proposal")
        a, b = pair["posterior"], pair["prior"]
        p = pair_difference(a, b, raw, "posterior_to_prior_same_uniforms")
        p.update(validation_target_id=key[-1],
                 posterior_target_Hamming=float(a["posterior_target_hamming"]),
                 prior_target_Hamming=float(a["prior_target_hamming"]))
        target = a.get("validation_target_energy_eV_atom", "")
        if target != "" and p["pair_scored"]:
            target = float(target)
            p.update(target_energy_eV_atom=target,
                posterior_minus_target_eV_atom=float(a["energy_eV_atom"]-target),
                prior_minus_target_eV_atom=float(b["energy_eV_atom"]-target))
        posteriors.append(p)
    return conditions, latents, posteriors


def summarize_pairs(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in GROUP) + (row["comparison"],)].append(row)
    result = []
    for key, rr in sorted(groups.items()):
        known = [r for r in rr if r["pair_scored"]]
        valid = [r for r in rr if r["raw_Hamming"] != ""]
        values = np.asarray([r["delta_E_eV_atom"] for r in known])
        result.append(dict(**dict(zip((*GROUP, "comparison"), key)), pairs=len(rr), scored_pairs=len(known),
            mean_delta_E_eV_atom=float(values.mean()) if len(values) else "",
            median_delta_E_eV_atom=float(np.median(values)) if len(values) else "",
            mean_absolute_delta_E_eV_atom=float(np.abs(values).mean()) if len(values) else "",
            fraction_positive_delta=float(np.mean(values > 0)) if len(values) else "",
            valid_mask_pairs=len(valid),
            mean_raw_Hamming=float(np.mean([r["raw_Hamming"] for r in valid])) if valid else "",
            same_orbit_fraction=float(np.mean([r["same_orbit"] for r in valid])) if valid else "",
            a_bin_hit_rate=float(np.mean([r["a_bin_hit"] for r in rr])),
            b_bin_hit_rate=float(np.mean([r["b_bin_hit"] for r in rr])),
            a_novel_training_fraction=float(np.mean([r["a_novel_training"] for r in rr])) if all(r["a_novel_training"] != "" for r in rr) else "",
            b_novel_training_fraction=float(np.mean([r["b_novel_training"] for r in rr])) if all(r["b_novel_training"] != "" for r in rr) else "",
            a_novel_history_fraction=float(np.mean([r["a_novel_history"] for r in rr])) if all(r["a_novel_history"] != "" for r in rr) else "",
            b_novel_history_fraction=float(np.mean([r["b_novel_history"] for r in rr])) if all(r["b_novel_history"] != "" for r in rr) else "",
            posterior_mean_target_Hamming=float(np.mean([r["posterior_target_Hamming"] for r in rr])) if "posterior_target_Hamming" in rr[0] else "",
            prior_mean_target_Hamming=float(np.mean([r["prior_target_Hamming"] for r in rr])) if "prior_target_Hamming" in rr[0] else ""))
    return result


def outcome_summaries(rows, original_thresholds):
    groups = defaultdict(list)
    for row in rows:
        if row["diagnostic"] == "primary":
            groups[tuple(row[k] for k in GROUP) + (row["requested_bin"],)].append(row)
    cells = []
    for key, rr in sorted(groups.items()):
        novelty_known = all(r["novel_training"] != "" and r["novel_history"] != "" for r in rr)
        complete = all(r["energy_known"] or not r["mask_valid"] or r.get("status", "generated") != "generated" for r in rr)
        useful = {r["canonical_mask"] for r in rr if r["requested_bin_hit"] and r["novel_training"] is True and r["novel_history"] is True}
        q = original_thresholds.get(int(key[3]))
        original = {r["canonical_mask"] for r in rr if q is not None and r["energy_known"]
                    and r["energy_eV_atom"] <= q and r["novel_training"] is True and r["novel_history"] is True}
        cells.append(dict(**dict(zip((*GROUP, "requested_bin"), key)), attempts=len(rr),
            known_energies=sum(r["energy_known"] for r in rr),
            complete_energy_coverage=complete,
            requested_bin_hit_rate=sum(r["requested_bin_hit"] for r in rr)/len(rr) if complete else "",
            observed_hit_lower_bound=sum(r["requested_bin_hit"] for r in rr)/len(rr),
            distinct_novel_inbin_yield=len(useful)/len(rr) if novelty_known and complete else "",
            original_fixed_q20_yield=len(original)/len(rr) if novelty_known and complete and q is not None else "",
            unique_orbits=len({r["canonical_mask"] for r in rr if r["mask_valid"]})))
    summaries = []
    by_method = defaultdict(list)
    for row in cells:
        by_method[row["method"]].append(row)
    for method, rr in sorted(by_method.items()):
        for scope, chosen in [("balanced_low_middle_high", rr), ("requested_low_only_original_q20", [r for r in rr if r["requested_bin"] == "low"])]:
            by_seed = defaultdict(list)
            for row in chosen:
                by_seed[row["replicate_id"]].append(row)
            for seed, ss in sorted(by_seed.items()):
                field = "distinct_novel_inbin_yield" if scope.startswith("balanced") else "original_fixed_q20_yield"
                summaries.append(dict(method=method, replicate_id=seed, scope=scope,
                    composition_request_cells=len(ss), attempts=sum(r["attempts"] for r in ss),
                    macro_useful_yield=float(np.mean([r[field] for r in ss])) if all(r[field] != "" for r in ss) else "",
                    macro_requested_bin_hit_rate=float(np.mean([r["requested_bin_hit_rate"] for r in ss])) if all(r["requested_bin_hit_rate"] != "" for r in ss) else ""))
    return cells, summaries


def confusion_rows(outcomes):
    groups = defaultdict(list)
    for row in outcomes:
        if row["diagnostic"] == "primary":
            groups[tuple(row[k] for k in GROUP) + (row["requested_bin"],)].append(row)
    result = []
    for key, rr in sorted(groups.items()):
        outcomes = ["failed_request" if r.get("status", "generated") != "generated" else
                    "invalid_mask" if not r["mask_valid"] else r["measured_bin"] for r in rr]
        for measured in ["low", "middle", "high", "unknown_energy", "invalid_mask", "failed_request"]:
            count = outcomes.count(measured)
            result.append(dict(**dict(zip((*GROUP, "requested_bin"), key)), measured_bin=measured,
                               count=count, all_attempts=len(rr), fraction=count/len(rr)))
    return result


def aggregate_fits(fits_root, manifest=None):
    histories, selections, components = [], [], []
    if fits_root is None:
        return histories, selections, components
    labels = defaultdict(list)
    allowed_fit_names = None
    if manifest is not None:
        allowed_fit_names = set()
        for row in json.loads(Path(manifest).read_text()).get("models", []):
            if row.get("checkpoint"):
                checkpoint = Path(row["checkpoint"])
                allowed_fit_names.add(checkpoint.parent.name)
                labels[(checkpoint.parent.name, checkpoint.name)].append(row["method"])
    for config_path in sorted(Path(fits_root).glob("*/configuration.json")):
        directory = config_path.parent
        if allowed_fit_names is not None and directory.name not in allowed_fit_names:
            continue
        cfg = json.loads(config_path.read_text())
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        base = dict(fit=directory.name, fit_seed=cfg["seed"], latent_dim=cfg["latent_dim"],
            model_kind=cfg.get("kind", "unknown"),
            latent_role="AR baseline without latent variables" if cfg.get("kind") == "ar_baseline" else "Variational model with Gaussian latent" if cfg["latent_dim"] > 0 else "Unspecified legacy model",
            encoder_kind=cfg.get("encoder_kind", "dense"), graph_context=cfg.get("graph_context", False),
            orientation=cfg.get("orientation", "canonical"), class_balance=cfg.get("class_balance", False),
            parameter_count=cfg.get("parameter_count", ""), actual_optimizer_updates=summary["optimizer_updates"],
            train_seconds=summary["training_seconds"], configuration_sha256=sha256(config_path),
            summary_sha256=sha256(summary_path), history_sha256=sha256(directory/"history.csv"),
            latent_history_sha256=sha256(directory/"latent_history.jsonl") if (directory/"latent_history.jsonl").exists() else "")
        history = read_csv(directory / "history.csv")
        histories.extend({**base, **row} for row in history)
        by_epoch = {int(row["epoch"]): row for row in history}
        latent_path = directory / "latent_history.jsonl"
        latent = {int(r["epoch"]): r for r in map(json.loads, latent_path.read_text().splitlines())} if latent_path.exists() else {}
        for beta, selected in summary.get("selection_checkpoints", {}).items():
            checkpoint = directory / selected["filename"]
            if sha256(checkpoint) != selected["sha256"]:
                raise ValueError("Selected checkpoint differs from fit summary")
            epoch = int(selected["epoch"])
            record = latent.get(epoch, {})
            stat = record.get("pooled", {})
            out = {**base, "selection_beta": float(beta), "selected_epoch": epoch,
                "checkpoint_sha256": selected["sha256"], "methods": "|".join(labels[(directory.name, checkpoint.name)]),
                "val_reconstruction": float(by_epoch[epoch]["val_reconstruction"]),
                "val_KL": float(by_epoch[epoch]["val_KL"]),
                "total_KL_nats": stat.get("total_KL_nats", ""),
                "mean_KL_nats_per_dimension": stat.get("mean_KL_nats_per_dimension", ""),
                **{"active_at_"+k: v for k, v in stat.get("active_components_by_cutoff", {}).items()}}
            selections.append(out)
            for scope, stats in [("pooled", stat), *record.get("by_composition", {}).items()]:
                for j, kl in enumerate(stats.get("KL_nats_by_dimension", [])):
                    components.append({**base, "selection_beta": float(beta), "selected_epoch": epoch,
                        "scope": scope, "component": j, "KL_nats": kl,
                        "posterior_mean": stats["posterior_mean_by_dimension"][j],
                        "posterior_mean_variance": stats["posterior_mean_variance_by_dimension"][j],
                        "posterior_variance_mean": stats["posterior_variance_mean_by_dimension"][j]})
    return histories, selections, components


def plot_diagnostics(output, outcomes, condition_summary, latent_summary, posterior_summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    colors = dict(low="tab:blue", middle="tab:orange", high="tab:red")
    methods = sorted({r["method"] for r in outcomes if r["diagnostic"] == "primary"})
    for method in methods:
        rr = [r for r in outcomes if r["method"] == method and r["diagnostic"] == "primary"]
        counts = sorted({r["n_Li"] for r in rr})
        fig, axes = plt.subplots(1, len(counts), figsize=(4.4*len(counts), 3.4), squeeze=False)
        for ax, n in zip(axes[0], counts):
            for requested in ["low", "middle", "high"]:
                seeds = sorted({r["replicate_id"] for r in rr})
                for j, seed in enumerate(seeds):
                    e = sorted(r["energy_eV_atom"] for r in rr if r["n_Li"] == n and r["requested_bin"] == requested and r["replicate_id"] == seed and r["energy_known"])
                    if e:
                        ax.step(e, np.arange(1, len(e)+1)/len(e), where="post", color=colors[requested], alpha=.65,
                                label=requested if j == 0 else None)
            ax.set(title=f"n={n}", xlabel="UMA energy (eV/atom)", ylabel="Empirical cumulative fraction")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
            ax.tick_params(labelsize=9)
            ax.xaxis.label.set_fontsize(10); ax.yaxis.label.set_fontsize(10)
            ax.legend(fontsize=8)
        fig.suptitle(method + "; thin curves are design replicates", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, .96))
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in method)
        for ext in ["png", "pdf"]:
            fig.savefig(output / f"energy_distributions_{safe}.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)

    for name, table, field, ylabel in [
        ("condition_response", condition_summary, "mean_delta_E_eV_atom", "Mean paired energy shift (meV/atom)"),
        ("latent_response", latent_summary, "mean_raw_Hamming", "Mean differing physical site occupations")]:
        if not table:
            continue
        names = sorted({r["method"] for r in table})
        fig, ax = plt.subplots(figsize=(max(8, .9*len(names)), 4.5))
        for k, method in enumerate(names):
            for comparison, color in [("low_to_middle", "tab:orange"), ("low_to_high", "tab:red"), ("change_latent_same_uniforms", "tab:blue")]:
                grouped = defaultdict(list)
                for row in table:
                    if row["method"] == method and row["comparison"] == comparison and row[field] != "":
                        grouped[row["replicate_id"]].append(float(row[field]))
                values = [np.mean(v)*(1000 if field.endswith("eV_atom") else 1) for v in grouped.values()]
                if not values:
                    continue
                ax.scatter(np.full(len(values), k) + (.1 if comparison == "low_to_high" else -.1), values, color=color, s=25,
                           label={"low_to_middle": "low → middle", "low_to_high": "low → high", "change_latent_same_uniforms": "change latent, same uniforms"}[comparison] if k == 0 else None)
        ax.axhline(0, color="0.5", lw=.8)
        ax.set(xticks=range(len(names)), xticklabels=names, ylabel=ylabel,
               title="One point per design replicate; equal composition weights")
        ax.title.set_fontsize(11); ax.yaxis.label.set_fontsize(10)
        ax.tick_params(axis="x", labelrotation=60, labelsize=9)
        ax.legend(fontsize=8); fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(output / f"{name}.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)
    if posterior_summary:
        names = sorted({r["method"] for r in posterior_summary})
        fig, ax = plt.subplots(figsize=(max(8, .9*len(names)), 4.5))
        for k, method in enumerate(names):
            grouped = defaultdict(list)
            for row in posterior_summary:
                if row["method"] == method:
                    grouped[row["replicate_id"]].append(row)
            for seed, rr in grouped.items():
                for offset, field, label, color in [(-.12, "posterior_mean_target_Hamming", "posterior", "tab:blue"),
                                                   (.12, "prior_mean_target_Hamming", "prior", "tab:orange")]:
                    ax.scatter(k+offset, np.mean([r[field] for r in rr]), color=color, label=label if k == 0 and seed == sorted(grouped)[0] else None)
        ax.set(xticks=range(len(names)), xticklabels=names, ylabel="Differing sites from VAL target",
               title="Posterior versus matched prior; one point per design replicate")
        ax.title.set_fontsize(11); ax.yaxis.label.set_fontsize(10)
        ax.tick_params(axis="x", labelrotation=60, labelsize=9); ax.legend(); fig.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(output / f"posterior_vs_prior.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)


def plot_fit_diagnostics(output, histories, selections, components):
    """Small multipage supplements; no pooled fit-seed confidence claims."""
    if not histories:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    names = sorted({r["fit"] for r in histories})
    with PdfPages(output / "fit_learning_curves.pdf") as pdf:
        for name in names:
            rr = [r for r in histories if r["fit"] == name]
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
            updates = [int(r["optimizer_updates"]) for r in rr]
            axes[0].plot(updates, [float(r["train_reconstruction"]) for r in rr], label="train weighted")
            axes[0].plot(updates, [float(r["val_reconstruction"]) for r in rr], label="validation")
            axes[1].plot(updates, [float(r["val_KL"]) for r in rr], label="validation KL, sum over latent dimensions")
            epoch_updates = {int(r["epoch"]): int(r["optimizer_updates"]) for r in rr}
            for selected in [s for s in selections if s["fit"] == name]:
                u = epoch_updates[selected["selected_epoch"]]
                for ax in axes:
                    ax.axvline(u, ls="--", alpha=.5, label=f"selected βval={selected['selection_beta']:g}")
            axes[0].set(ylabel="Reconstruction nats per weighted row")
            axes[1].set(ylabel="KL nats per weighted row")
            for ax in axes:
                ax.set_xlabel("Completed optimizer updates"); ax.legend(fontsize=7)
            fig.suptitle(name + "; training and validation weights are recorded in configuration", fontsize=10)
            fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    if not components:
        return
    with PdfPages(output / "selected_latent_components.pdf") as pdf:
        for name in sorted({r["fit"] for r in components}):
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
            for beta in sorted({r["selection_beta"] for r in components if r["fit"] == name}):
                rr = [r for r in components if r["fit"] == name and r["scope"] == "pooled" and r["selection_beta"] == beta]
                axes[0].plot([r["component"] for r in rr], [r["KL_nats"] for r in rr], "o-", label=f"βval={beta:g}")
                axes[1].plot([r["component"] for r in rr], [r["posterior_mean_variance"] for r in rr], "o-", label=f"βval={beta:g}")
            for cutoff in [.001, .01, .1]:
                axes[1].axhline(cutoff, color="0.5", ls=":", lw=.8, label=f"descriptive cutoff {cutoff}")
            axes[0].set_ylabel("Validation mean KL (nats/component)")
            axes[1].set_ylabel("Variance of posterior mean across VAL masks")
            for ax in axes:
                ax.set_xlabel("Latent component"); ax.legend(fontsize=7)
            fig.suptitle(name + "; thresholds are diagnostics, not physical convergence", fontsize=10)
            fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def analyze(attempts_all, raw_path, scores_uma, output, *, schema_path, training_data,
            history_path=None, fits_root=None, manifest=None, original_thresholds=None,
            allow_incomplete=False, plots=True):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    rows = read_csv(attempts_all)
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = archive["occupancy"]
        raw_targets = archive["seed_occupancy"] if "seed_occupancy" in archive else None
    if raw.ndim != 2:
        raise ValueError("Raw occupancy must be a two-dimensional proposal array")
    schema = load_schema(schema_path)
    if raw.shape[1] != schema["n_sites"]:
        raise ValueError("Raw site count differs from the frozen condition schema")
    training_keys, targets = load_training(training_data)
    history = keys_from_json(history_path)
    outcomes, missing = join_scores(rows, raw, scores_uma, schema, training_keys, history, allow_incomplete=allow_incomplete)
    for row in outcomes:
        target = row.get("validation_target_id")
        if target:
            if target not in targets or targets[target]["split"] not in {"val", "validation"}:
                raise ValueError("Posterior target is not a protected validation identity")
            expected = targets[target]["energy"]
            if row.get("validation_target_energy_eV_atom", "") != "" and float(row["validation_target_energy_eV_atom"]) != expected:
                raise ValueError("Saved posterior target energy differs from exact training-data record")
            row["validation_target_energy_eV_atom"] = expected
            index = int(row["raw_row"])
            if raw_targets is not None and raw_targets.shape == raw.shape and int(raw_targets[index].sum()) == row["n_Li"]:
                target_mask = raw_targets[index]
                if not np.isin(target_mask, [0, 1]).all():
                    raise ValueError("Saved posterior target occupancy is not binary")
                expected_mask = targets[target].get("canonical_occupancy")
                if expected_mask is not None and not np.array_equal(target_mask, expected_mask):
                    raise ValueError("Saved posterior target mask differs from the exact canonical validation target")
                distance = int(np.count_nonzero(raw[index] != target_mask))
                field = "posterior_target_hamming" if row["diagnostic"].startswith("posterior_") else "prior_target_hamming"
                if distance != int(row[field]):
                    raise ValueError("Saved posterior/prior Hamming distance differs from the raw masks")
                row["target_Hamming_verified_from_raw"] = True
            else:
                row["target_Hamming_verified_from_raw"] = False
    condition, latent, posterior = paired_diagnostics(outcomes, raw)
    condition_summary, latent_summary, posterior_summary = map(summarize_pairs, [condition, latent, posterior])
    thresholds = {}
    if original_thresholds is not None:
        thresholds = {int(r["n_Li"]): float(r["q20_eV_atom"]) for r in read_csv(original_thresholds)
                      if r["teacher"] == "uma" and r["source"] == "main_train"}
    cells, scopes = outcome_summaries(outcomes, thresholds)
    histories, selections, components = aggregate_fits(fits_root, manifest)
    tables = dict(attempt_outcomes=outcomes, paired_conditions=condition, condition_summary=condition_summary,
        paired_latents=latent, latent_summary=latent_summary, posterior_prior_pairs=posterior,
        posterior_prior_summary=posterior_summary, primary_condition_cells=cells,
        bin_confusion=confusion_rows(outcomes),
        distinct_metric_scopes=scopes, fit_history=histories, selected_latent_diagnostics=selections,
        latent_components=components)
    for name, values in tables.items():
        write_csv(output / (name + ".csv"), values)
    if plots:
        plot_diagnostics(output, outcomes, condition_summary, latent_summary, posterior_summary)
        plot_fit_diagnostics(output, histories, selections, components)
    inputs = dict(attempts_all=attempts_all, raw=raw_path, scores_uma=scores_uma, schema=schema_path,
                  training_data=training_data, history=history_path, original_thresholds=original_thresholds,
                  model_manifest=manifest)
    receipt = dict(status="PASS_COMPLETE_ORACLE_COVERAGE" if not missing else "INCOMPLETE_UNKNOWN_ENERGIES",
        attempts=len(rows), primary_attempts=sum(r["diagnostic"] == "primary" for r in rows),
        diagnostic_attempts=sum(r["diagnostic"] != "primary" for r in rows), unknown_generated_attempts=len(missing),
        invalid_mask_attempts=sum(not r["mask_valid"] for r in outcomes),
        condition_pairs=len(condition), latent_pairs=len(latent), posterior_prior_pairs=len(posterior),
        posterior_or_prior_raw_target_Hamming_verified=sum(r.get("target_Hamming_verified_from_raw", False) for r in outcomes),
        training_novelty_known=training_keys is not None, history_novelty_known=history is not None,
        posterior_targets_verified="Protected validation identities and exact source energies; never test",
        comparison_scope="CVAE paired conditions use shared latent and AR uniforms; sampler controls are paired by attempt index without common conditional draws; latent pairs hold uniforms fixed; posterior/prior share target condition and uniforms",
        uncertainty="Individual design replicates retained. No confidence interval from proposal draws or trajectory frames.",
        active_unit_definition="Validation posterior-mean variance > cutoff; .001/.01/.1 sensitivity is descriptive, not physical convergence",
        fit_diagnostic_scope="Only model-manifest fit directories when supplied; other concurrently completed fits cannot enter this stage",
        fit_input_hashes={r["fit"]: {k: r[k] for k in ["configuration_sha256", "summary_sha256", "history_sha256", "latent_history_sha256"]} for r in histories},
        metric_scope="Balanced low/middle/high useful yield is separate from requested-low original fixed-q20 yield",
        input_hashes={k: sha256(v) for k, v in inputs.items() if v is not None},
        source_sha256=sha256(__file__), output_hashes={p.name: sha256(p) for p in sorted(output.glob("*.csv"))})
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attempts-all", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--scores-uma", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--schema", type=Path, default=PROJECT/"artifacts/overnight_20261107/data/condition_schema_uma.json")
    p.add_argument("--training-data", type=Path, default=PROJECT/"artifacts/current/improvement_v1/training_round0.npz")
    p.add_argument("--history", type=Path)
    p.add_argument("--fits-root", type=Path)
    p.add_argument("--model-manifest", type=Path)
    p.add_argument("--original-thresholds", type=Path, default=PROJECT/"artifacts/current/reports/generation_thresholds.csv")
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args(argv); require_compute_allocation()
    receipt = analyze(a.attempts_all, a.raw, a.scores_uma, a.output_dir,
        schema_path=a.schema, training_data=a.training_data, history_path=a.history,
        fits_root=a.fits_root, manifest=a.model_manifest, original_thresholds=a.original_thresholds,
        allow_incomplete=a.allow_incomplete, plots=not a.no_plots)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

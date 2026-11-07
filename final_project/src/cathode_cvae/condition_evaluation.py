"""Evaluate requested energy ranges without dropping failed or unscored attempts.

This evaluator is explicitly scoped to the current 48-site O3 parent: a retained
Li count n has 144+n atoms. All energy comparisons remain within composition,
teacher and its own frozen training-derived schema. Oracle-query curves count
distinct exact geometries a method would query; shared campaign call costs must
be read from the separate physical execution receipt, not summed over methods.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .conditions import BIN_NAMES, classify, load_schema, schema_identity
from .data import sha256, target_indices


GROUP_FIELDS = ["teacher", "method", "replicate_id", "fit_seed", "sampling_seed", "n_Li", "requested_bin",
                "temperature_K", "allocation_policy", "request_schema_id", "measured_schema_id",
                "checkpoint_sha256"]
OUTCOMES = [*BIN_NAMES, "failed_request", "invalid_mask", "unknown_energy"]


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def load_keys(path):
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict):
        for name in ["canonical_masks", "canonical_keys", "known_keys", "keys"]:
            if name in value:
                value = value[name]
                break
    if not isinstance(value, list):
        raise ValueError("Key JSON must be a list or contain canonical_masks/canonical_keys/known_keys/keys")
    keys = set(map(str, value))
    if any(len(k) != 48 or set(k) - {"0", "1"} for k in keys):
        raise ValueError("Novelty keys must be 48-bit vacancy strings")
    return keys


def load_candidates(path, parent_path=None):
    with np.load(path, allow_pickle=False) as archive:
        d = {k: archive[k] for k in archive.files}
    x = d["canonical_occupancy"] if "canonical_occupancy" in d else d.get("occupancy")
    if x is None:
        x = 1 - d["masks"]
    x = np.asarray(x)
    ids = d["structure_ids"].astype(str)
    if x.ndim != 2 or x.shape[1] != 48 or len(x) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Candidates require unique IDs and one 48-site occupancy row each")
    binary = np.isin(x, [0, 1]).all(1)
    keys = d.get("canonical_vacancy_mask", d.get("canonical_masks"))
    if keys is None:
        if not binary.all():
            raise ValueError("Invalid occupancy rows need an explicit candidate key for failure accounting")
        keys = np.asarray(["".join(map(str, (1-row).astype(int).tolist())) for row in x])
    keys = np.asarray(keys).astype(str)
    if keys.shape != ids.shape or len(set(keys)) != len(keys):
        raise ValueError("Candidate orbit keys must be unique and match IDs")
    valid_key = np.asarray([len(k) == 48 and not (set(k) - {"0", "1"}) for k in keys])
    n = x.sum(1)
    valid = binary & valid_key & np.asarray([k.count("0") == count for k, count in zip(keys, n)])
    parent = None
    layer = np.full(len(x), np.nan)
    layer_counts = [""] * len(x)
    layer_info = None
    if parent_path is not None:
        with np.load(parent_path, allow_pickle=False) as archive:
            parent = {k: archive[k] for k in archive.files}
        indices = target_indices(parent)
        if len(indices) != 48 or len(parent["numbers"]) != 192:
            raise ValueError("This evaluator currently requires the 48-target,192-atom main parent")
        if "parent_sha256" in d and str(d["parent_sha256"].item()) != sha256(parent_path):
            raise ValueError("Candidates belong to a different parent")
        from .acquisition import canonical_keys
        if binary.any():
            actual_keys = canonical_keys(x[binary], parent["permutations"])
            if not np.array_equal(actual_keys, keys[binary]):
                raise ValueError("Candidate keys are not canonical under the supplied parent action")
        # This is an explicit coordinate-frame descriptor, not a phase criterion.
        fractional = parent["positions_A"][indices] @ np.linalg.inv(parent["cell_A"])
        rounded = np.round(fractional[:, 2] % 1, 6) % 1
        z, labels = np.unique(rounded, return_inverse=True)
        sites = np.bincount(labels)
        for i in np.flatnonzero(binary):
            occupied = np.bincount(labels, weights=x[i], minlength=len(sites))
            layer[i] = float(np.std(occupied / sites))
            layer_counts[i] = "|".join(map(str, occupied.astype(int).tolist()))
        layer_info = dict(axis="third fractional cell coordinate", fractional_rounding_decimals=6,
            layer_fractional_positions=z.tolist(), sites_per_layer=sites.tolist(),
            interpretation="Standard deviation of layer filling fractions; coordinate-frame diagnostic, not phase identification")
    return {sid: dict(key=k, n_actual=int(count) if np.isfinite(count) else -1,
                     mask_valid=bool(ok), layer_imbalance=float(imbalance), layer_counts=counts)
            for sid, k, count, ok, imbalance, counts in zip(ids, keys, n, valid, layer, layer_counts)}, layer_info


def load_scores(path, candidates):
    with np.load(path, allow_pickle=False) as archive:
        d = {k: archive[k] for k in archive.files}
    ids = d["structure_ids"].astype(str)
    if ids.ndim != 1 or len(set(ids)) != len(ids):
        raise ValueError("Score IDs must be unique, even when a table includes extra reference rows")
    total = d.get("energy_eV", d.get("energy"))
    if total is None or "n_atoms" not in d:
        raise ValueError("Scores need total energy_eV and actual n_atoms")
    total, atoms = np.asarray(total, dtype=float), np.asarray(d["n_atoms"])
    if total.shape != ids.shape or atoms.shape != ids.shape:
        raise ValueError("Score energy/count shapes differ from IDs")
    provenance = np.asarray(d.get("provenance", d.get("cached_source", np.full(len(ids), "unspecified")))).astype(str)
    status = np.asarray(d.get("status", np.full(len(ids), "success"))).astype(str)
    seconds = np.asarray(d.get("seconds", np.full(len(ids), np.nan)), dtype=float)
    if provenance.ndim == 0: provenance = np.repeat(provenance, len(ids))
    if status.ndim == 0: status = np.repeat(status, len(ids))
    if seconds.ndim == 0: seconds = np.repeat(seconds, len(ids))
    if any(v.shape != ids.shape for v in [provenance, status, seconds]):
        raise ValueError("Score status/provenance/timing shape differs from IDs")
    result = {}
    for i, sid in enumerate(ids):
        if sid not in candidates:
            continue
        candidate = candidates[sid]
        claimed = status[i].lower() in {"success", "complete", "completed", "ok", "fresh", "cached"}
        if claimed and candidate["mask_valid"] and np.isfinite(atoms[i]) and atoms[i] > 0:
            if atoms[i] != int(atoms[i]) or int(atoms[i]) != 144 + candidate["n_actual"]:
                raise ValueError(f"Score atom count disagrees with exact candidate geometry: {sid}")
            if "n_Li" in d and int(d["n_Li"][i]) != candidate["n_actual"]:
                raise ValueError(f"Score composition disagrees with candidate: {sid}")
        known = (claimed
                 and np.isfinite(total[i]) and np.isfinite(atoms[i]) and atoms[i] > 0)
        result[sid] = dict(known=bool(known), energy_eV_atom=float(total[i] / atoms[i]) if known else None,
                           provenance=provenance[i], seconds=float(seconds[i]) if np.isfinite(seconds[i]) else None)
    return result


def _meta(row, teacher, measured_schema_id):
    value = dict(teacher=teacher, method=row["method"], replicate_id=str(row.get("replicate_id", row.get("fit_seed", ""))), fit_seed=str(row.get("fit_seed", "")),
        sampling_seed=str(row.get("sampling_seed", row.get("seed", ""))), n_Li=int(row["n_Li"]),
        requested_bin=row["requested_bin"], temperature_K=row.get("temperature_K", ""),
        allocation_policy=row.get("allocation_policy", "fixed-request"),
        request_schema_id=row.get("condition_schema_id", ""), measured_schema_id=measured_schema_id,
        checkpoint_sha256=row.get("checkpoint_sha256", ""))
    if value["temperature_K"] != "":
        temperature = float(value["temperature_K"])
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Acquisition temperature must be positive kelvin or empty for fixed requests")
        value["temperature_K"] = format(temperature, ".12g")
    return value


def _key(meta, fields=GROUP_FIELDS):
    return tuple(meta[k] for k in fields)


def _safe_quantile(values, q):
    return float(np.quantile(values, q)) if len(values) else ""


def _macro_cells(cells, field):
    values = []
    for rows in cells:
        known = [r[field] for r in rows if r[field] != ""]
        if known: values.append(float(np.mean(known)))
    return float(np.mean(values)) if values else ""


def _declared_fitted_model(row):
    absent = {"", "none", "null", "n/a", "not-applicable"}
    return (str(row["fit_seed"]).lower() not in absent
            and str(row["checkpoint_sha256"]).lower() not in absent)


def _summaries(metrics):
    fields = ["teacher", "method", "temperature_K", "allocation_policy", "request_schema_id", "measured_schema_id"]
    strata = defaultdict(list)
    for r in metrics:
        strata[tuple(r[k] for k in fields)].append(r)
    summaries, seeds, per_request = [], [], []
    for key, rows in sorted(strata.items()):
        base = dict(zip(fields, key))
        design = defaultdict(list)
        for r in rows:
            design[r["replicate_id"]].append(r)
        values = []
        for replicate_id, rr in sorted(design.items()):
            # Equal condition cells, then equal sampling replicates in each cell.
            cells = defaultdict(list)
            for r in rr: cells[r["n_Li"], r["requested_bin"]].append(r)
            macro = float(np.mean([np.mean([v["novel_inbin_yield_per_attempt"] for v in cell]) for cell in cells.values()]))
            macro_low = _macro_cells(cells.values(), "novel_fixed_q20_yield_per_attempt")
            values.append(macro)
            seeds.append(dict(**base, replicate_id=replicate_id, fit_seeds="|".join(sorted({r["fit_seed"] for r in rr})), condition_cells=len(cells),
                sampling_seeds="|".join(sorted({r["sampling_seed"] for r in rr})),
                macro_composition_bin_inbin_yield=macro, macro_fixed_q20_yield_all_requested_bins=macro_low,
                pooled_inbin_yield=sum(r["distinct_novel_inbin_orbits"] for r in rr) / sum(r["attempts"] for r in rr),
                has_fitted_checkpoint=any(_declared_fitted_model(r) for r in rr)))
        summaries.append(dict(**base, replicate_count=len(design), design_seed_count=len(design),
            fitted_seed_count=len({r["fit_seed"] for r in rows if _declared_fitted_model(r)}),
            sampling_seed_count=len({r["sampling_seed"] for r in rows}),
            mean_novel_inbin_yield=float(np.mean(values)), minimum_design_seed_yield=float(min(values)),
            maximum_design_seed_yield=float(max(values)),
            design_seed_standard_deviation=float(np.std(values, ddof=1)) if len(values) > 1 else "",
            condition_grids=";".join(f"{s}:" + "|".join(f"{n}:{b}" for n,b in sorted({(r['n_Li'],r['requested_bin']) for r in rr})) for s,rr in sorted(design.items()))))
        for requested in sorted({r["requested_bin"] for r in rows}):
            by_seed = defaultdict(list)
            for r in rows:
                if r["requested_bin"] == requested: by_seed[r["replicate_id"]].append(r)
            inbin, low, hit = [], [], []
            for rr in by_seed.values():
                cells = defaultdict(list)
                for r in rr: cells[r["n_Li"]].append(r)
                inbin.append(float(np.mean([np.mean([x["novel_inbin_yield_per_attempt"] for x in cc]) for cc in cells.values()])))
                observed_low = _macro_cells(cells.values(), "novel_fixed_q20_yield_per_attempt")
                if observed_low != "": low.append(observed_low)
                hit.append(float(np.mean([np.mean([x["bin_hit_rate_per_attempt"] for x in cc]) for cc in cells.values()])))
            per_request.append(dict(**base, requested_bin=requested, design_seed_count=len(by_seed),
                mean_novel_inbin_yield=float(np.mean(inbin)), mean_novel_fixed_q20_yield=float(np.mean(low)) if low else "",
                mean_bin_hit_rate_per_attempt=float(np.mean(hit))))
    return summaries, seeds, per_request


def evaluate(attempts_path, candidates_path, score_paths, schema_paths, training_path,
             history_path, historical_thresholds, output_dir, parent_path=None,
             reference_plans=None, cost_receipt=None):
    rows = read_csv(attempts_path)
    if not rows or not {"method", "n_Li", "requested_bin", "status", "scoring_structure_id", "canonical_mask"} <= set(rows[0]):
        raise ValueError("Attempt CSV is empty or lacks condition-aware identity/status fields")
    if set(score_paths) != set(schema_paths):
        raise ValueError("Every teacher must have its own measured-energy schema")
    candidates, layer_info = load_candidates(candidates_path, parent_path)
    train, history = load_keys(training_path), load_keys(history_path)
    schemas = {t: load_schema(p) for t, p in schema_paths.items()}
    if any(s["n_sites"] != 48 for s in schemas.values()):
        raise ValueError("This evaluator handles the current 48-site parent only")
    scores = {t: load_scores(p, candidates) for t, p in score_paths.items()}
    thresholds = {(r["teacher"], int(r["n_Li"])): float(r["q20_eV_atom"])
                  for r in read_csv(historical_thresholds) if r["source"] == "main_train"}
    attempt_ids = [r.get("attempt_id", str(i)) for i, r in enumerate(rows)]
    if len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError("Attempt IDs must be globally unique")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    detailed, candidate_rows = [], []
    for teacher, schema in schemas.items():
        sid_schema = schema_identity(schema)
        for sid, candidate in candidates.items():
            record = scores[teacher].get(sid)
            known = bool(record and record["known"] and candidate["mask_valid"])
            measured = (str(classify(schema, candidate["n_actual"], record["energy_eV_atom"]).item())
                        if known and str(candidate["n_actual"]) in schema["compositions"] else "unknown_energy")
            candidate_rows.append(dict(teacher=teacher, scoring_structure_id=sid, canonical_mask=candidate["key"],
                n_Li=candidate["n_actual"], mask_valid=candidate["mask_valid"], energy_known=known,
                energy_eV_atom=record["energy_eV_atom"] if known else "", measured_bin=measured,
                novel_to_training=candidate["key"] not in train, novel_to_history=candidate["key"] not in history,
                measured_schema_id=sid_schema, layer_imbalance_std_fraction=candidate["layer_imbalance"] if np.isfinite(candidate["layer_imbalance"]) else "",
                layer_occupied_site_counts=candidate["layer_counts"]))
        for position, (row, attempt_id) in enumerate(zip(rows, attempt_ids)):
            meta = _meta(row, teacher, sid_schema)
            candidate = candidates.get(row["scoring_structure_id"])
            generated = row["status"] == "generated"
            if generated and candidate is not None and row["canonical_mask"] != candidate["key"]:
                raise ValueError("Attempt/candidate identity mismatch; refusing an ambiguous energy join")
            valid = bool(generated and candidate is not None and candidate["mask_valid"])
            exact = bool(valid and candidate["n_actual"] == meta["n_Li"])
            record = scores[teacher].get(row["scoring_structure_id"]) if exact else None
            known = bool(record and record["known"])
            if not generated: outcome = "failed_request"
            elif not exact: outcome = "invalid_mask"
            elif not known: outcome = "unknown_energy"
            else: outcome = str(classify(schema, meta["n_Li"], record["energy_eV_atom"]).item())
            energy = record["energy_eV_atom"] if known else None
            key = candidate["key"] if exact else ""
            q = thresholds.get((teacher, meta["n_Li"]))
            item = dict(meta, attempt_id=attempt_id, source_position=position,
                scoring_structure_id=row["scoring_structure_id"], canonical_mask=key,
                generated=generated, mask_valid=valid, exact_count=exact, energy_known=known,
                measured_bin=outcome, in_requested_bin=known and outcome == meta["requested_bin"],
                novel_training=bool(exact and key not in train), novel_history=bool(exact and key not in history),
                novel_both=bool(exact and key not in train and key not in history),
                energy_eV_atom=energy if known else "", fixed_q20_eV_atom=q if q is not None else "",
                passes_fixed_q20=bool(known and q is not None and energy <= q),
                score_provenance=record["provenance"] if known else "unknown",
                score_seconds=record["seconds"] if known and record["seconds"] is not None else "",
                allocation_weight=row.get("allocation_weight", ""),
                allocation_run_id=row.get("allocation_run_id", ""),
                layer_imbalance_std_fraction=candidate["layer_imbalance"] if exact and np.isfinite(candidate["layer_imbalance"]) else "")
            groups[_key(meta)].append(item); detailed.append(item)
    metrics, confusion, curves = [], [], []
    for group, rr in sorted(groups.items()):
        meta = dict(zip(GROUP_FIELDS, group)); attempts = len(rr)
        unique = {r["canonical_mask"] for r in rr if r["exact_count"]}
        known_ids = {r["scoring_structure_id"] for r in rr if r["energy_known"]}
        novel_train = {r["canonical_mask"] for r in rr if r["novel_training"]}
        novel_history = {r["canonical_mask"] for r in rr if r["novel_history"]}
        novel_both = {r["canonical_mask"] for r in rr if r["novel_both"]}
        hits = {r["canonical_mask"] for r in rr if r["novel_both"] and r["in_requested_bin"]}
        low = {r["canonical_mask"] for r in rr if r["novel_both"] and r["passes_fixed_q20"]}
        energies = np.asarray([r["energy_eV_atom"] for r in rr if r["energy_known"]])
        scored = sum(r["energy_known"] for r in rr)
        exact_attempts = sum(r["exact_count"] for r in rr)
        layer = [r["layer_imbalance_std_fraction"] for r in rr if r["layer_imbalance_std_fraction"] != ""]
        outcomes = Counter(r["measured_bin"] for r in rr)
        historical_available = all(r["fixed_q20_eV_atom"] != "" for r in rr)
        metrics.append(dict(meta, attempts=attempts, generated=sum(r["generated"] for r in rr),
            request_failures=outcomes["failed_request"], invalid_masks=outcomes["invalid_mask"],
            mask_validity_rate=sum(r["mask_valid"] for r in rr)/attempts,
            exact_count_rate=sum(r["exact_count"] for r in rr)/attempts,
            scored_attempts=scored, score_coverage_per_attempt=scored/attempts,
            distinct_scored_geometries=len(known_ids), unique_orbits=len(unique),
            unique_orbit_yield_per_attempt=len(unique)/attempts,
            orbit_uniqueness_among_exact_count=len(unique)/exact_attempts if exact_attempts else "",
            duplicate_fraction_among_exact_count=1-len(unique)/exact_attempts if exact_attempts else "",
            distinct_novel_training_orbits=len(novel_train), distinct_novel_history_orbits=len(novel_history),
            distinct_novel_both_orbits=len(novel_both),
            novel_training_yield_per_attempt=len(novel_train)/attempts,
            novel_history_yield_per_attempt=len(novel_history)/attempts,
            training_novel_fraction_of_unique=len(novel_train)/len(unique) if unique else "",
            history_novel_fraction_of_unique=len(novel_history)/len(unique) if unique else "",
            bin_hits=sum(r["in_requested_bin"] for r in rr),
            bin_hit_rate_per_attempt=sum(r["in_requested_bin"] for r in rr)/attempts,
            bin_hit_rate_scored_valid=sum(r["in_requested_bin"] for r in rr)/scored if scored else "",
            distinct_novel_inbin_orbits=len(hits), novel_inbin_yield_per_attempt=len(hits)/attempts,
            historical_q20_available=historical_available,
            distinct_novel_fixed_q20_orbits=len(low) if historical_available else "",
            novel_fixed_q20_yield_per_attempt=len(low)/attempts if historical_available else "",
            best_observed_energy_eV_atom=float(energies.min()) if len(energies) else "",
            energy_p10_eV_atom=_safe_quantile(energies,.1), energy_median_eV_atom=_safe_quantile(energies,.5),
            energy_p90_eV_atom=_safe_quantile(energies,.9),
            mean_layer_imbalance_std_fraction=float(np.mean(layer)) if layer else ""))
        for outcome in OUTCOMES:
            confusion.append(dict(meta, measured_outcome=outcome, count=outcomes[outcome], fraction_all_attempts=outcomes[outcome]/attempts))
        seen_queries, seen_fresh, seen_hits, seen_low, seen_novel = set(), set(), set(), set(), set()
        best = None; observed = 0
        for i, r in enumerate(rr, 1):
            if r["exact_count"]: seen_queries.add(r["scoring_structure_id"])
            if r["novel_both"]: seen_novel.add(r["canonical_mask"])
            if r["energy_known"]:
                observed += 1; energy = r["energy_eV_atom"]
                best = energy if best is None else min(best, energy)
                if r["score_provenance"].lower() in {"fresh", "new"}: seen_fresh.add(r["scoring_structure_id"])
                if r["novel_both"] and r["in_requested_bin"]: seen_hits.add(r["canonical_mask"])
                if r["novel_both"] and r["passes_fixed_q20"]: seen_low.add(r["canonical_mask"])
            curves.append(dict(meta, proposal_attempts=i, unique_score_queries_seen=len(seen_queries),
                unique_fresh_labels_encountered=len(seen_fresh), score_coverage_per_attempt=observed/i,
                distinct_novel_orbits_seen=len(seen_novel), distinct_novel_inbin_orbits_seen=len(seen_hits),
                distinct_novel_fixed_q20_orbits_seen=len(seen_low),
                best_observed_energy_eV_atom=best if best is not None else ""))
    summary, seed_rows, by_request = _summaries(metrics)
    fixed_q20_low_request = [dict(r, comparison_scope="Requested low only; original teacher-specific q20. New pool/history/budget remain distinct from the historical experiment.")
                            for r in by_request if r["requested_bin"] == "low"]
    allocation_fields = [k for k in GROUP_FIELDS if k not in {"sampling_seed", "requested_bin"}] + ["allocation_run_id"]
    allocation_groups = defaultdict(list)
    for r in detailed: allocation_groups[tuple(r[k] for k in allocation_fields)].append(r)
    allocation = []
    for key, rr in sorted(allocation_groups.items()):
        base = dict(zip(allocation_fields, key)); total = len(rr)
        for b in OUTCOMES:
            weights = {str(r["allocation_weight"]) for r in rr if r["requested_bin"] == b and r["allocation_weight"] != ""}
            allocation.append(dict(base, energy_bin=b, total_attempts=total,
                requested_attempts=sum(r["requested_bin"]==b for r in rr),
                requested_fraction=sum(r["requested_bin"]==b for r in rr)/total,
                measured_attempts=sum(r["measured_bin"]==b for r in rr),
                measured_fraction_all_attempts=sum(r["measured_bin"]==b for r in rr)/total,
                declared_allocation_weights="|".join(sorted(weights)),
                sampling_seeds="|".join(sorted({r["sampling_seed"] for r in rr}))))
    joint_fields = [k for k in allocation_fields if k != "temperature_K"]
    joint_groups = defaultdict(list)
    for r in detailed: joint_groups[tuple(r[k] for k in joint_fields)].append(r)
    joint_allocation, joint_summary = [], []
    for key, rr in sorted(joint_groups.items()):
        base = dict(zip(joint_fields, key)); total = len(rr)
        for temperature in sorted({r["temperature_K"] for r in rr}):
            for b in OUTCOMES:
                requested = sum(r["temperature_K"] == temperature and r["requested_bin"] == b for r in rr)
                measured = sum(r["temperature_K"] == temperature and r["measured_bin"] == b for r in rr)
                joint_allocation.append(dict(base, temperature_K=temperature, energy_bin=b,
                    total_schedule_attempts=total, requested_attempts=requested,
                    requested_joint_fraction=requested/total, measured_attempts=measured,
                    measured_joint_fraction_all_attempts=measured/total))
        successful = {r["canonical_mask"] for r in rr if r["novel_both"] and r["in_requested_bin"]}
        joint_summary.append(dict(base, total_schedule_attempts=total,
            temperatures_K="|".join(sorted({r["temperature_K"] for r in rr})),
            distinct_novel_inbin_orbits=len(successful), novel_inbin_orbits_per_schedule_attempt=len(successful)/total,
            measured_energy_coverage=sum(r["energy_known"] for r in rr)/total))
    for name, table in [("metrics.csv",metrics),("confusion.csv",confusion),("attempt_outcomes.csv",detailed),
                        ("candidate_measurements.csv",candidate_rows),("budget_curves.csv",curves),
                        ("summary.csv",summary),("seed_metrics.csv",seed_rows),("summary_by_request.csv",by_request),
                        ("fixed_q20_low_request.csv",fixed_q20_low_request),
                        ("allocation_realization.csv",allocation),
                        ("allocation_joint_realization.csv",joint_allocation),
                        ("allocation_joint_summary.csv",joint_summary)]: write_csv(out/name,table)
    referenced = {r["scoring_structure_id"] for r in detailed if r["exact_count"]}
    valid_attempts = [r for r in detailed if r["exact_count"]]
    receipt = dict(status=("NO_VALID_GENERATED_STRUCTURES" if not valid_attempts else
                           "PASS" if all(r["energy_known"] for r in valid_attempts) else "INCOMPLETE_SCORE_COVERAGE"),
        scope="Current 48-site pristine-derived deletion benchmark; not relaxed energies or independent DFT validation",
        attempt_rows=len(rows), teacher_attempt_rows=len(detailed), unique_candidates=len(candidates), groups=len(metrics),
        teachers=list(scores), canonical_keys_recomputed=parent_path is not None,
        layer_diagnostic=layer_info, training_keys=len(train), frozen_history_keys=len(history),
        training_keys_absent_from_history=len(train-history), novelty_success_requires="Absent from both training and frozen prior history",
        global_unique_fresh_scored_geometries={t:sum(sid in referenced and r["known"] and r["provenance"].lower() in {"fresh","new"} for sid,r in table.items()) for t,table in scores.items()},
        summary_weighting="Within replicate_id: equal observed composition/requested-bin cells and equal sampling replicates; then equal replicate_id values. Actual fitted_seed_count requires a nonempty checkpoint identity. Condition grids recorded. Allocation table pools sampling seeds within its explicit schedule scope.",
        curve_cost_interpretation="unique_score_queries_seen is a per-method stand-alone deduplicated query budget. unique_fresh_labels_encountered describes encountered labels; summing it across methods double-counts shared campaign calls. Actual acquisition/training/dispatch costs require the execution receipt.",
        source_hashes=dict(attempts=sha256(attempts_path), candidates=sha256(candidates_path), training_keys=sha256(training_path),history=sha256(history_path),historical_thresholds=sha256(historical_thresholds),
            **{f"scores_{t}":sha256(p) for t,p in score_paths.items()},**{f"schema_{t}":sha256(p) for t,p in schema_paths.items()}),
        limitations=["Unknown energies and failed requests remain in all-attempt denominators; no neighbor energy is filled in.","Energy quantiles and best values describe observed scored-valid attempts; coverage is separate.","Design-seed spread is descriptive, not a confidence interval; baseline design seeds are not fitted models.","Layer filling is a coordinate-frame descriptor, not proof of phase diversity or mechanical stability."])
    if parent_path: receipt["source_hashes"]["parent"] = sha256(parent_path)
    if reference_plans: receipt["source_hashes"]["reference_plans"] = sha256(reference_plans)
    if cost_receipt:
        receipt["source_hashes"]["cost_receipt"] = sha256(cost_receipt)
        receipt["external_cost_receipt"] = json.loads(Path(cost_receipt).read_text())
    (out/"receipt.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n")
    print(json.dumps(receipt,indent=2,allow_nan=False))
    return receipt


def _mapping(values):
    if any("=" not in v for v in values): raise ValueError("Use teacher=path for each input")
    result = dict(v.split("=",1) for v in values)
    if len(result) != len(values): raise ValueError("Repeated teacher input")
    return result


def plot_tables(output_dir):
    """Render condition quality and low-request budget curves from saved tables.

    Curve averaging is within one composition/teacher/method and independent
    replicate IDs. Query-budget curves stop where every replicate has support;
    they do not extrapolate a short curve to a larger query budget.
    """
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output_dir)
    figures = output/"figures";figures.mkdir(exist_ok=True)
    summary = read_csv(output/"summary_by_request.csv")
    strata = defaultdict(list)
    for r in summary:
        strata[r["teacher"],r["temperature_K"],r["allocation_policy"],r["request_schema_id"],r["measured_schema_id"]].append(r)
    for (teacher, temperature, policy, request_schema, measured_schema), rows in strata.items():
        methods = sorted({r["method"] for r in rows})
        fig,axs=plt.subplots(1,2,figsize=(10,max(3,len(methods)*.42)),sharey=True)
        for ax,field,title in zip(axs,["mean_bin_hit_rate_per_attempt","mean_novel_inbin_yield"],
                                 ["Requested-bin hit / all attempts","Distinct novel in-bin / all attempts"]):
            values=np.full((len(methods),3),np.nan)
            for r in rows:
                if r["requested_bin"] in BIN_NAMES:
                    values[methods.index(r["method"]),BIN_NAMES.index(r["requested_bin"])]=float(r[field])
            im=ax.imshow(values,vmin=0,vmax=1,cmap="viridis",aspect="auto")
            ax.set_xticks(range(3),BIN_NAMES);ax.set_yticks(range(len(methods)),methods);ax.set_title(title,fontsize=9)
            for i,j in np.ndindex(values.shape):
                if np.isfinite(values[i,j]):ax.text(j,i,f"{100*values[i,j]:.1f}%",ha="center",va="center",color="white" if values[i,j]<.6 else "black",fontsize=8)
        fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.03,pad=.03,label="Fraction")
        fig.suptitle(f"{teacher}: {policy}; T={temperature or 'fixed request'}; equal composition/replicate means",fontsize=10)
        fig.subplots_adjust(left=.23,right=.86,bottom=.14,top=.82,wspace=.3)
        tag=re.sub(r"[^A-Za-z0-9_.-]","_",f"{teacher}_{policy}_{temperature or 'fixed'}_{request_schema[:8]}_{measured_schema[:8]}")[:160]
        for ext in ["pdf","png"]:fig.savefig(figures/f"condition_quality_{tag}.{ext}",dpi=160)
        plt.close(fig)
    curves=read_csv(output/"budget_curves.csv")
    strata=defaultdict(list)
    for r in curves:
        if r["requested_bin"]=="low" and r["temperature_K"]=="":
            strata[r["teacher"],r["allocation_policy"],r["request_schema_id"],r["measured_schema_id"]].append(r)
    for (teacher,policy,request_schema,measured_schema),rows in strata.items():
        counts=sorted({int(r["n_Li"]) for r in rows});methods=sorted({r["method"] for r in rows})
        for xfield,xlabel,kind in [("proposal_attempts","Proposal attempts","proposal"),
                                  ("unique_score_queries_seen","Distinct exact-geometry queries (stand-alone budget)","query")]:
            fig,axs=plt.subplots(1,len(counts),figsize=(4.3*len(counts),3.7),squeeze=False)
            for ax,n in zip(axs[0],counts):
                for method in methods:
                    by_rep=defaultdict(list)
                    for r in rows:
                        if int(r["n_Li"])==n and r["method"]==method:
                            by_rep[r["replicate_id"],r["sampling_seed"]].append(r)
                    if not by_rep:continue
                    limit=min(max(int(r[xfield]) for r in rr) for rr in by_rep.values())
                    grid=np.arange(limit+1);ys=[]
                    for rr in by_rep.values():
                        seen={0:0}
                        for r in rr:seen[int(r[xfield])]=int(r["distinct_novel_inbin_orbits_seen"])
                        xx=np.asarray(sorted(seen));yy=np.asarray([seen[x] for x in xx])
                        ys.append(yy[np.maximum(0,np.searchsorted(xx,grid,side="right")-1)])
                    mean=np.mean(ys,axis=0);line=ax.step(grid,mean,where="post",label=method)[0]
                    for y in ys:ax.step(grid,y,where="post",color=line.get_color(),alpha=.18,linewidth=.7)
                ax.set(title=f"Li{n}Co48O96",xlabel=xlabel);ax.grid(alpha=.2)
            axs[0,0].set_ylabel("Distinct novel low-bin orbits (mean replicate)")
            axs[0,-1].legend(fontsize=7)
            fig.suptitle(f"{teacher}: requested-low benchmark; {policy}; faint curves are replicates",fontsize=10);fig.tight_layout()
            tag=re.sub(r"[^A-Za-z0-9_.-]","_",f"{teacher}_{policy}_{request_schema[:8]}_{measured_schema[:8]}")[:160]
            for ext in ["pdf","png"]:fig.savefig(figures/f"low_yield_{kind}_budget_{tag}.{ext}",dpi=160)
            plt.close(fig)
    return figures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempts",type=Path,required=True)
    parser.add_argument("--candidates",type=Path,required=True)
    parser.add_argument("--scores",nargs="+",required=True)
    parser.add_argument("--schemas",nargs="+",required=True)
    parser.add_argument("--training-keys",type=Path,required=True)
    parser.add_argument("--history",type=Path,required=True)
    parser.add_argument("--historical-thresholds",type=Path,required=True)
    parser.add_argument("--parent",type=Path)
    parser.add_argument("--reference-plans",type=Path)
    parser.add_argument("--cost-receipt",type=Path)
    parser.add_argument("--plots",action="store_true",help="Also render condition-quality and low-request budget figures")
    parser.add_argument("--output-dir",type=Path,required=True)
    a=parser.parse_args(argv)
    from .runtime import require_compute_allocation
    require_compute_allocation()
    receipt=evaluate(a.attempts,a.candidates,_mapping(a.scores),_mapping(a.schemas),a.training_keys,
        a.history,a.historical_thresholds,a.output_dir,a.parent,a.reference_plans,a.cost_receipt)
    if a.plots:plot_tables(a.output_dir)
    return receipt


if __name__=="__main__":main()

"""Frozen middle-energy / structural-descriptor pilot, then complete evaluation.

One fitted descriptor CVAE is preliminary evidence, not three independent fits.
This script never evaluates an MLIP. Generate all declared attempts first, score
the exact canonical pool externally, then join every attempt (including repeats
and missing scores). Mutation is frozen, uniformly seeded within the SAME
training energy/descriptor request and uses one/two/three swaps uniformly.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from cathode_cvae.conditions import classify as energy_classify, load_schema as load_energy_schema
from cathode_cvae.data import sha256, target_indices
from cathode_cvae.descriptor import classify as descriptor_classify, load_schema as load_descriptor_schema
from cathode_cvae.graph import build_periodic_graph, vacancy_pair_fraction
from cathode_cvae.learning.generate_mask import attempt_records, load_model, resolve_request
from cathode_cvae.learning.train_mask import canonicalize
from cathode_cvae.parent_spec import ParentSpec
from cathode_cvae.runtime import require_compute_allocation


PROTOCOL = "pristine-derived deletion; unrelaxed; exact canonical parent coordinates"


def _seed(*tokens):
    return int.from_bytes(hashlib.sha256("|".join(map(str, tokens)).encode()).digest()[:4], "little")


def _write_csv(path, records):
    if not records:
        return
    fields = list(dict.fromkeys(k for row in records for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(records)


def _load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _keys_from_codes(codes, n_sites):
    return np.asarray([format(int(x), f"0{n_sites}b") for x in codes])


def _ids_from_codes(codes, n_sites):
    width = (n_sites + 3) // 4
    return np.asarray([f"mask{n_sites}_{int(x):0{width}x}" for x in codes])


def frozen_swaps(seed, rng):
    """One exactly k-swap child, with all actual site selections returned."""
    seed = np.asarray(seed, dtype=np.uint8)
    swaps = int(rng.integers(1, 4))
    if min(int(seed.sum()), int((1 - seed).sum())) < 3:
        raise ValueError("the declared uniform 1/2/3-swap baseline requires at least three occupied and empty sites")
    removed = rng.choice(np.flatnonzero(seed), swaps, replace=False)
    added = rng.choice(np.flatnonzero(1 - seed), swaps, replace=False)
    child = seed.copy(); child[removed] = 0; child[added] = 1
    if int(np.sum(child != seed)) != 2 * swaps or child.sum() != seed.sum():
        raise AssertionError("frozen swap construction broke exact count/locality")
    return child, swaps, removed, added


def _graph(parent, descriptor_schema, parent_sha):
    spec = ParentSpec.from_legacy(parent, parent_sha)
    graph = build_periodic_graph(spec, 5.)
    provenance = descriptor_schema["provenance"]
    if provenance.get("parent_hash") != spec.parent_hash or provenance.get("graph_hash") != graph.to_dict()["graph_hash"]:
        raise ValueError("descriptor schema belongs to another parent/graph construction")
    return graph


def _canonical(raw, inv, device):
    x = torch.as_tensor(raw, dtype=torch.float32, device=device)
    canonical, codes = canonicalize(x, inv)
    return canonical.cpu().numpy().astype(np.uint8), codes


def generate(a):
    require_compute_allocation()
    started = time.perf_counter()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    parent, data = _load(a.parent), _load(a.data)
    parent_sha = sha256(a.parent)
    data_sha = sha256(a.data)
    n_sites = len(target_indices(parent))
    if not 1 <= n_sites <= 62:
        raise ValueError("this benchmark's integer scoring IDs support 1..62 mobile sites; ParentSpec itself is not limited")
    energy_schema, desc_schema = load_energy_schema(a.energy_schema), load_descriptor_schema(a.descriptor_schema)
    graph = _graph(parent, desc_schema, parent_sha)
    values = _load(a.descriptor_values)
    if not np.array_equal(values["structure_ids"].astype(str), data["structure_ids"].astype(str)):
        raise ValueError("descriptor and learning structure IDs/order disagree")
    if str(values["parent_hash"].item()) != graph.parent.parent_hash or str(values["graph_hash"].item()) != graph.to_dict()["graph_hash"]:
        raise ValueError("descriptor array provenance differs from requested graph")
    descriptor = values["descriptor_values"].astype(float)
    if not np.array_equal(descriptor, vacancy_pair_fraction(data["occupancy"], graph)):
        raise ValueError("descriptor values do not match actual masks")
    counts = data["occupancy"].sum(1).astype(int)
    labels = energy_classify(energy_schema, counts, data["energy_eV_atom"])
    ranges = descriptor_classify(desc_schema, counts, descriptor)
    train = data["split"].astype(str) == "train"
    history = json.loads(a.history.read_text())
    if not isinstance(history, list) or any(len(str(k)) != n_sites or set(str(k)) - {"0", "1"} for k in history):
        raise ValueError("history must be a frozen JSON list of canonical vacancy strings")
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = a.device
    models = {}
    for method, path in (("descriptor_cvae", a.checkpoint), ("plain_cvae", a.control_checkpoint)):
        model, cfg = load_model(path, device)
        if cfg.get("parent_sha256") != parent_sha:
            raise ValueError("checkpoint parent differs from benchmark")
        if cfg.get("condition_schema", {}).get("identity") != energy_schema["identity"]:
            raise ValueError("checkpoint energy boundaries differ from benchmark")
        if method == "descriptor_cvae" and cfg.get("descriptor_schema", {}).get("identity") != desc_schema["identity"]:
            raise ValueError("descriptor checkpoint and requested schema differ")
        if method == "plain_cvae" and cfg.get("descriptor_schema"):
            raise ValueError("plain control must not already have descriptor conditions")
        models[method] = (model, cfg, sha256(path), path)
    if models["descriptor_cvae"][0].latent_dim != models["plain_cvae"][0].latent_dim:
        raise ValueError("paired descriptor/plain comparison requires the same latent dimension")
    inv = torch.as_tensor(np.argsort(parent["permutations"], axis=1), dtype=torch.long, device=device)
    records, all_codes, seed_codes, reference_records = [], [], [], []
    raw_children, raw_seeds = [], []
    methods = ["descriptor_cvae", "plain_cvae", "mutation_joint"] + (["uniform"] if a.include_uniform else [])
    for n in a.counts:
        reference_candidates = np.flatnonzero(train & (counts == n))
        # A bounded reference, selected without any energy ranking. Its nearest
        # orbit distance is an upper bound on distance to the FULL training set.
        ranked = sorted(reference_candidates, key=lambda i: hashlib.sha256(str(data["structure_ids"][i]).encode()).digest())
        for i in ranked[:a.distance_reference_size]:
            reference_records.append(dict(n_target=n, structure_id=str(data["structure_ids"][i]),
                                          raw_occupancy="".join(map(str, data["occupancy"][i].astype(int)))))
        common_seed = _seed("descriptor-pilot-v1", a.sampling_seed, n)
        common = torch.Generator(device=device).manual_seed(common_seed)
        latent_dim = models["descriptor_cvae"][0].latent_dim
        z = torch.randn((a.count, latent_dim), device=device, generator=common)
        uniforms = torch.rand((a.count, n_sites), device=device, generator=common)
        for requested_range in ("low", "high"):
            available = np.flatnonzero(train & (counts == n) & (labels == "middle") & (ranges == requested_range))
            if not len(available):
                raise ValueError("primary middle/descriptor request has no protected training support")
            for method in methods:
                seed_rows, swap_counts, removals, additions = [], [], [], []
                rng_seed = _seed("descriptor-pilot-v1", a.sampling_seed, n, requested_range, method)
                rng = np.random.default_rng(rng_seed)
                if method in models:
                    model, cfg, ck_sha, _ = models[method]
                    vector, _, effective, _, _ = resolve_request(cfg, n, "middle", request_schema=energy_schema,
                        descriptor_range=requested_range if method == "descriptor_cvae" else None)
                    c = torch.as_tensor(vector, dtype=torch.float32, device=device).repeat(a.count, 1)
                    with torch.no_grad():
                        children = model.sample(c, n, z=z, uniforms=uniforms).cpu().numpy().astype(np.uint8)
                    fit_seed = cfg["seed"]
                elif method == "mutation_joint":
                    chosen = available[rng.integers(len(available), size=a.count)]
                    children = []
                    for index in chosen:
                        child, k, removed, added = frozen_swaps(data["occupancy"][index], rng)
                        children.append(child); seed_rows.append(int(index)); swap_counts.append(k)
                        removals.append(removed.tolist()); additions.append(added.tolist())
                    children = np.asarray(children, dtype=np.uint8)
                    effective, ck_sha, fit_seed = "training middle+descriptor seed; frozen uniform one-to-three swaps", "not-applicable", ""
                else:
                    children = np.zeros((a.count, n_sites), np.uint8)
                    for row in children:
                        row[rng.choice(n_sites, n, replace=False)] = 1
                    effective, ck_sha, fit_seed = "unconditioned fixed-count", "not-applicable", ""
                if not np.isin(children, [0, 1]).all() or not np.all(children.sum(1) == n):
                    raise ValueError("generated mask violated binary/exact-composition rule")
                canonical, codes = _canonical(children, inv, device)
                keys, ids = _keys_from_codes(codes, n_sites), _ids_from_codes(codes, n_sites)
                all_codes.extend(map(int, codes))
                rr = attempt_records(a.count, ids, keys, method=method, fit_seed=fit_seed,
                    sampling_seed=common_seed if method in models else rng_seed,
                    n_target=n, requested_bin="middle", effective_model_class=effective,
                    schema_id=energy_schema["identity"], checkpoint_sha256=ck_sha,
                    parent_sha256=parent_sha, geometry_protocol=PROTOCOL,
                    descriptor_range=requested_range, descriptor_schema_id=desc_schema["identity"])
                if seed_rows:
                    raw_seed_array = data["occupancy"][seed_rows]
                    _, canonical_seed_codes = _canonical(raw_seed_array, inv, device)
                    seed_codes.extend(map(int, canonical_seed_codes))
                for i, row in enumerate(rr):
                    index = seed_rows[i] if seed_rows else None
                    raw_seed = data["occupancy"][index] if index is not None else np.zeros(n_sites, np.uint8)
                    raw_children.append(children[i]); raw_seeds.append(raw_seed)
                    row.update(attempt_id=f"descriptor-{a.sampling_seed}-{method}-{n}-{requested_range}-{i}",
                        replicate_id=str(a.sampling_seed), common_latent_uniform_seed=common_seed,
                        requested_descriptor_range=requested_range,
                        seed_source_sha256=data_sha if index is not None else "",
                        seed_structure_id=str(data["structure_ids"][index]) if index is not None else "",
                        seed_scoring_structure_id=str(_ids_from_codes([canonical_seed_codes[i]], n_sites)[0]) if index is not None else "",
                        seed_canonical_mask=str(_keys_from_codes([canonical_seed_codes[i]], n_sites)[0]) if index is not None else "",
                        seed_raw_occupancy="".join(map(str, raw_seed.astype(int))) if index is not None else "",
                        child_raw_occupancy="".join(map(str, children[i].astype(int))),
                        swap_count=swap_counts[i] if index is not None else "",
                        removed_sites="|".join(map(str, removals[i])) if index is not None else "",
                        added_sites="|".join(map(str, additions[i])) if index is not None else "",
                        raw_seed_child_hamming=2 * swap_counts[i] if index is not None else "",
                        seed_training_energy_eV_atom=float(data["energy_eV_atom"][index]) if index is not None else "",
                        generation_elapsed_seconds="")
                records.extend(rr)
    unique = np.unique(np.asarray(all_codes + seed_codes, dtype=np.int64))
    masks = ((unique[:, None] >> np.arange(n_sites - 1, -1, -1)) & 1).astype(np.uint8)
    ids = _ids_from_codes(unique, n_sites)
    proposed = set(all_codes); seeds = set(seed_codes)
    np.savez_compressed(a.output_dir / "candidates.npz", structure_ids=ids, masks=masks, occupancy=1 - masks,
        n_Li=n_sites - masks.sum(1), n_atoms=len(parent["numbers"]) - masks.sum(1),
        proposed_candidate=np.asarray([int(k) in proposed for k in unique]),
        mutation_seed_candidate=np.asarray([int(k) in seeds for k in unique]), parent_sha256=np.asarray(parent_sha))
    np.savez_compressed(a.output_dir / "raw_attempts.npz", child_occupancy=np.asarray(raw_children), seed_occupancy=np.asarray(raw_seeds),
                        attempt_ids=np.asarray([r["attempt_id"] for r in records]), parent_sha256=np.asarray(parent_sha))
    _write_csv(a.output_dir / "attempts.csv", records)
    _write_csv(a.output_dir / "distance_reference.csv", reference_records)
    receipt = dict(status="GENERATED_FROZEN_BEFORE_SCORING", scope="Preliminary one-fit descriptor pilot; middle-energy requests only",
        attempts=len(records), unique_proposed_orbits=len(proposed), unique_mutation_seed_orbits=len(seeds),
        registered_candidate_pool_including_seeds=len(unique), per_group_attempts=a.count,
        counts=a.counts, requested_energy="middle", descriptor_requests=["low", "high"], methods=methods,
        parent_sha256=parent_sha, data_sha256=data_sha, energy_schema_sha256=sha256(a.energy_schema),
        descriptor_schema_sha256=sha256(a.descriptor_schema), descriptor_values_sha256=sha256(a.descriptor_values),
        history_sha256=sha256(a.history), history_keys=len(history), history_source=str(a.history),
        checkpoint_sha256={k: v[2] for k, v in models.items()}, checkpoint_configurations={k: v[1] for k, v in models.items()},
        generator_sha256=sha256(__file__), elapsed_seconds=time.perf_counter() - started,
        no_energy_filtering=True, all_attempts_retained=True,
        seed_protocol="Uniform protected TRAIN seed in requested middle-energy and descriptor bin; uniform k∈{1,2,3}; distinct occupied removals and vacant additions; no energy acceptance/population update",
        paired_generation="Same latent vectors and AR uniform streams across descriptor requests and descriptor/plain models at each composition",
        distance_reference="Deterministic SHA256-ranked bounded training subset per composition; nearest orbit distance is an upper bound on nearest distance to all training",
        distance_reference_size_per_count=a.distance_reference_size,
        outputs={f: sha256(a.output_dir / f) for f in ("candidates.npz", "attempts.csv", "raw_attempts.npz", "distance_reference.csv")})
    (a.output_dir / "generation_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "attempts", "unique_proposed_orbits", "registered_candidate_pool_including_seeds", "elapsed_seconds")}, indent=2))


def _orbit_distances(occupancy, reference, permutations):
    """Exact Hamming distance minimized over the provided reference orbits."""
    x, reference = np.asarray(occupancy, np.uint8), np.asarray(reference, np.uint8)
    inv = np.argsort(permutations, axis=1)
    # Packed bytes keep the bounded 256-reference calculation small on CPUs.
    transformed = np.packbits(reference[:, inv].reshape(-1, x.shape[1]), axis=-1)
    packed = np.packbits(x, axis=-1)
    lookup = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    result = np.empty(len(x), dtype=int)
    for start in range(0, len(x), 16):
        xor = np.bitwise_xor(packed[start:start + 16, None], transformed[None])
        result[start:start + 16] = lookup[xor].sum(axis=-1).min(axis=-1)
    return result


def _score_table(path, candidates):
    d = _load(path); ids = d["structure_ids"].astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate score IDs")
    result = {}
    for i, sid in enumerate(ids):
        if sid not in candidates:
            continue
        energy = float(d["energy_eV"][i]); atoms = int(d["n_atoms"][i])
        status = str(d.get("status", np.full(len(ids), "success"))[i])
        known = np.isfinite(energy) and status in {"success", "complete", "cached", "fresh"} and atoms > 0
        if known and atoms != candidates[sid]["n_atoms"]:
            raise ValueError("score geometry atom count differs from candidate")
        result[sid] = dict(known=bool(known), energy_eV_atom=energy / atoms if known else None,
                           energy_eV=energy if known else None, seconds=float(d.get("seconds", np.zeros(len(ids)))[i]))
    return result


def evaluate(a):
    require_compute_allocation()
    begin = time.perf_counter()
    receipt = json.loads((a.generated / "generation_receipt.json").read_text())
    if sha256(a.data) != receipt["data_sha256"] or sha256(a.parent) != receipt["parent_sha256"] or sha256(a.history) != receipt["history_sha256"]:
        raise ValueError("data, parent or frozen history changed after proposal generation")
    for f, digest in receipt["outputs"].items():
        if sha256(a.generated / f) != digest:
            raise ValueError("frozen generated output mutated before evaluation")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    parent, data, pool = _load(a.parent), _load(a.data), _load(a.generated / "candidates.npz")
    desc_schema = load_descriptor_schema(a.descriptor_schema)
    if sha256(a.descriptor_schema) != receipt["descriptor_schema_sha256"]:
        raise ValueError("descriptor schema changed after generation")
    graph = _graph(parent, desc_schema, sha256(a.parent))
    n_sites = graph.n_sites
    masks = pool["masks"].astype(np.uint8); occupancy = 1 - masks
    keys = np.array(["".join(map(str, row.astype(int))) for row in masks])
    descriptor = vacancy_pair_fraction(occupancy, graph)
    ranges = descriptor_classify(desc_schema, pool["n_Li"], descriptor)
    training_keys = set(data["canonical_masks"][data["split"].astype(str) == "train"].astype(str))
    history = set(json.loads(a.history.read_text()))
    candidates = {str(s): dict(index=i, key=keys[i], n=int(pool["n_Li"][i]), n_atoms=int(pool["n_atoms"][i]),
                              descriptor=float(descriptor[i]), descriptor_range=str(ranges[i]),
                              novel_train=keys[i] not in training_keys, novel_history=keys[i] not in history)
                  for i, s in enumerate(pool["structure_ids"])}
    rows = list(csv.DictReader((a.generated / "attempts.csv").open()))
    reference_rows = list(csv.DictReader((a.generated / "distance_reference.csv").open()))
    for n in receipt["counts"]:
        indices = np.flatnonzero(pool["n_Li"] == n)
        ref = np.array([[int(c) for c in r["raw_occupancy"]] for r in reference_rows if int(r["n_target"]) == n], np.uint8)
        distances = _orbit_distances(occupancy[indices], ref, parent["permutations"])
        for i, distance in zip(indices, distances):
            candidates[str(pool["structure_ids"][i])]["nearest_training_reference_orbit_hamming"] = int(distance)
    schemas = {k: load_energy_schema(v) for k, v in a.schemas.items()}
    scores = {k: _score_table(v, candidates) for k, v in a.scores.items()}
    if set(schemas) != set(scores):
        raise ValueError("one teacher-specific energy schema per score table is required")
    detailed, metrics, confusion, curves = [], [], [], []
    for teacher, teacher_scores in scores.items():
        groups = defaultdict(list)
        for row in rows:
            candidate = candidates[row["scoring_structure_id"]]
            record = teacher_scores.get(row["scoring_structure_id"], {})
            known = bool(record.get("known"))
            measured = str(energy_classify(schemas[teacher], candidate["n"], record["energy_eV_atom"]).item()) if known else "unknown_energy"
            seed_record = teacher_scores.get(row.get("seed_scoring_structure_id", ""), {})
            seed_known = bool(seed_record.get("known"))
            row_out = dict(row, teacher=teacher, energy_known=known, measured_energy_bin=measured,
                measured_descriptor=candidate["descriptor"], measured_descriptor_range=candidate["descriptor_range"],
                energy_eV_atom=record.get("energy_eV_atom", "") if known else "",
                energy_hit=known and measured == row["requested_bin"],
                descriptor_hit=candidate["descriptor_range"] == row["requested_descriptor_range"],
                joint_hit=known and measured == row["requested_bin"] and candidate["descriptor_range"] == row["requested_descriptor_range"],
                novel_training=candidate["novel_train"], novel_history=candidate["novel_history"],
                novel_both=candidate["novel_train"] and candidate["novel_history"],
                nearest_training_reference_orbit_hamming=candidate["nearest_training_reference_orbit_hamming"],
                seed_energy_known=seed_known,
                seed_energy_eV_atom=seed_record.get("energy_eV_atom", "") if seed_known else "",
                energy_change_from_seed_eV_atom=record["energy_eV_atom"] - seed_record["energy_eV_atom"] if known and seed_known else "")
            if row.get("seed_raw_occupancy"):
                seed = np.array([[int(c) for c in row["seed_raw_occupancy"]]], np.uint8)
                child = np.array([[int(c) for c in row["child_raw_occupancy"]]], np.uint8)
                row_out["seed_child_orbit_hamming"] = int(_orbit_distances(child, seed, parent["permutations"])[0])
            else:
                row_out["seed_child_orbit_hamming"] = ""
            groups[(row["method"], int(row["n_Li"]), row["requested_descriptor_range"])].append(row_out)
            detailed.append(row_out)
        for (method, n, requested_range), rr in sorted(groups.items()):
            attempts = len(rr)
            unique = {r["canonical_mask"] for r in rr}
            novel_train = {r["canonical_mask"] for r in rr if r["novel_training"]}
            novel_both = {r["canonical_mask"] for r in rr if r["novel_both"]}
            success = {r["canonical_mask"] for r in rr if r["joint_hit"] and r["novel_both"]}
            energy = np.array([r["energy_eV_atom"] for r in rr if r["energy_known"]], float)
            pair_values = np.array([r["measured_descriptor"] for r in rr], float)
            base = dict(teacher=teacher, method=method, n_Li=n, requested_energy_bin="middle", requested_descriptor_range=requested_range,
                        fit_scope="one fitted model per neural method; preliminary", sampling_seed=a.sampling_seed if hasattr(a, "sampling_seed") else rows[0]["replicate_id"])
            metrics.append(dict(base, attempts=attempts, exact_count_rate=1., unique_orbits=len(unique),
                unique_yield_per_attempt=len(unique) / attempts, novel_training_orbits=len(novel_train),
                novel_training_yield=len(novel_train) / attempts, novel_both_orbits=len(novel_both),
                novel_both_yield=len(novel_both) / attempts, scored_attempts=sum(r["energy_known"] for r in rr),
                score_coverage=sum(r["energy_known"] for r in rr) / attempts,
                energy_hit_rate=sum(r["energy_hit"] for r in rr) / attempts,
                descriptor_hit_rate=sum(r["descriptor_hit"] for r in rr) / attempts,
                joint_hit_rate=sum(r["joint_hit"] for r in rr) / attempts,
                distinct_novel_joint_orbits=len(success), novel_joint_yield_per_all_attempts=len(success) / attempts,
                energy_p10=float(np.quantile(energy, .1)) if len(energy) else "",
                energy_median=float(np.median(energy)) if len(energy) else "",
                energy_p90=float(np.quantile(energy, .9)) if len(energy) else "",
                best_energy_eV_atom=float(energy.min()) if len(energy) else "",
                descriptor_mean=float(pair_values.mean()), descriptor_std=float(pair_values.std()),
                mean_nearest_training_reference_orbit_hamming=float(np.mean([r["nearest_training_reference_orbit_hamming"] for r in rr])),
                standalone_unique_child_oracle_queries=len({r["scoring_structure_id"] for r in rr})))
            outcomes = Counter((r["measured_energy_bin"], r["measured_descriptor_range"]) for r in rr)
            for measured_energy in ("low", "middle", "high", "unknown_energy"):
                for measured_range in ("low", "high"):
                    total = outcomes[measured_energy, measured_range]
                    confusion.append(dict(base, measured_energy_bin=measured_energy, measured_descriptor_range=measured_range,
                                          attempts=total, fraction_all_attempts=total / attempts))
            seen, hits, best = set(), set(), None
            for i, r in enumerate(rr, 1):
                seen.add(r["scoring_structure_id"])
                if r["joint_hit"] and r["novel_both"]:
                    hits.add(r["canonical_mask"])
                if r["energy_known"]:
                    best = r["energy_eV_atom"] if best is None else min(best, r["energy_eV_atom"])
                curves.append(dict(base, attempts=i, unique_geometry_queries=len(seen), distinct_novel_joint_orbits=len(hits),
                                   best_observed_energy_eV_atom=best if best is not None else ""))
    summary = []
    for teacher in sorted(scores):
        for method in sorted({r["method"] for r in metrics}):
            rr = [r for r in metrics if r["teacher"] == teacher and r["method"] == method]
            summary.append(dict(teacher=teacher, method=method, condition_cells=len(rr), fit_replicates=1 if "cvae" in method else 0,
                sampling_replicates=1, mean_novel_joint_yield=float(np.mean([r["novel_joint_yield_per_all_attempts"] for r in rr])),
                mean_joint_hit_rate=float(np.mean([r["joint_hit_rate"] for r in rr])),
                scope="Equal mean across 3 compositions ×2 descriptor requests; middle energy only; one-fit preliminary"))
    for name, table in (("attempt_outcomes.csv", detailed), ("metrics.csv", metrics), ("confusion.csv", confusion),
                        ("budget_curves.csv", curves), ("summary.csv", summary)):
        _write_csv(a.output_dir / name, table)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(scores), figsize=(5. * len(scores), 3.8), squeeze=False)
    for ax, teacher in zip(axes[0], sorted(scores)):
        rr = [r for r in summary if r["teacher"] == teacher]
        ax.bar([r["method"] for r in rr], [100 * r["mean_novel_joint_yield"] for r in rr])
        ax.set(title=f"{teacher}: one-fit structural-condition pilot", ylabel="Novel joint successes / all attempts (%)")
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(a.output_dir / f"joint_yield.{ext}", dpi=180)
    plt.close(fig)
    unknown = {t: sorted(sid for sid in candidates if not scores[t].get(sid, {}).get("known")) for t in scores}
    final = dict(status="PASS" if all(not x for x in unknown.values()) else "INCOMPLETE_UNKNOWN_ENERGIES",
        unknown_by_teacher=unknown, all_attempts_in_denominator=True,
        attempts=len(rows), candidate_pool_including_mutation_seeds=len(candidates),
        energy_scope="Teacher-specific frozen middle bin; descriptor schema is the same training-only structural definition for all teachers",
        neural_replication="One independently fitted descriptor model and one plain control; descriptive pilot, no fit-variation confidence interval",
        distance_scope="Exact symmetry-aware distance to the bounded recorded training reference; upper bound on full-training nearest distance",
        scoring_cost_scope="Unique per-method queries are stand-alone costs and overlap; use global oracle receipts for actual NEW calls and allocation time",
        input_hashes=dict(generation_receipt=sha256(a.generated / "generation_receipt.json"),
                         scores={t: sha256(v) for t, v in a.scores.items()}, schemas={t: sha256(v) for t, v in a.schemas.items()}),
        evaluator_sha256=sha256(__file__), elapsed_seconds=time.perf_counter() - begin,
        no_new_oracle_calls=True)
    (a.output_dir / "receipt.json").write_text(json.dumps(final, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: final[k] for k in ("status", "attempts", "candidate_pool_including_mutation_seeds", "elapsed_seconds")}, indent=2))


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    for name in ("checkpoint", "control-checkpoint", "parent", "data", "energy-schema", "descriptor-schema", "descriptor-values", "history", "output-dir"):
        g.add_argument("--" + name, type=Path, required=True)
    g.add_argument("--count", type=int, default=128); g.add_argument("--counts", type=int, nargs="+", default=[12, 24, 36])
    g.add_argument("--sampling-seed", type=int, default=20261117)
    g.add_argument("--distance-reference-size", type=int, default=256)
    g.add_argument("--include-uniform", action="store_true")
    g.add_argument("--device", choices=["cpu", "cuda"], default="cuda"); g.add_argument("--threads", type=int, default=4)
    e = sub.add_parser("evaluate")
    for name in ("generated", "parent", "data", "descriptor-schema", "history", "output-dir"):
        e.add_argument("--" + name, type=Path, required=True)
    e.add_argument("--scores", nargs="+", required=True, help="teacher=collected_scores.npz")
    e.add_argument("--schemas", nargs="+", required=True, help="teacher=training_energy_schema.json")
    args = p.parse_args(argv)
    if args.command == "generate" and (args.count < 1 or args.distance_reference_size < 1):
        p.error("attempt and reference counts must be positive")
    if args.command == "evaluate":
        args.scores = {k: Path(v) for k, v in (x.split("=", 1) for x in args.scores)}
        args.schemas = {k: Path(v) for k, v in (x.split("=", 1) for x in args.schemas)}
    return args


if __name__ == "__main__":
    args = arguments()
    (generate if args.command == "generate" else evaluate)(args)


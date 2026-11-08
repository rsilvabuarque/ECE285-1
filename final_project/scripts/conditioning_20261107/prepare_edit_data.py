"""Freeze optional editor examples from verified DEVELOPMENT mutation lineages.

The observed CHILD energy class is the label, including worse/unchanged outcomes.
Core-final candidates and protected H0 VAL/TEST identities are excluded. Joint
seed-child symmetry is used only for deduplication; stored learning occupancies
retain their original shared orientation. Connected identity components, rather
than individual rows, are assigned to editor train/validation/heldout roles.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


STAGES = ("policy", "latent", "spatial")
BINS = ("low", "middle", "high")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b""): h.update(block)
    return h.hexdigest()


def read_json(path): return json.loads(Path(path).read_text())


def load_npz(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def write_rows(path, rows):
    if not rows:
        Path(path).write_text(""); return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def identity(key): return f"mask48_{int(key, 2):012x}"


def validate_edit(seed, child, removed, added, k):
    seed, child = np.asarray(seed), np.asarray(child)
    if seed.shape != child.shape or seed.ndim != 1 or not np.isin(seed, [0, 1]).all() or not np.isin(child, [0, 1]).all():
        raise ValueError("Edit requires two same-size binary occupancies")
    if k not in (1, 2, 3) or len(set(removed)) != k or len(set(added)) != k or set(removed) & set(added):
        raise ValueError("Invalid distinct removal/insertion sets or swap budget")
    if min([*removed, *added]) < 0 or max([*removed, *added]) >= len(seed):
        raise ValueError("Edit site lies outside the parent")
    expected = seed.copy()
    if not np.all(expected[removed] == 1) or not np.all(expected[added] == 0):
        raise ValueError("Removal/insertion sites disagree with original seed")
    expected[removed], expected[added] = 0, 1
    if not np.array_equal(child, expected) or child.sum() != seed.sum() or np.count_nonzero(child != seed) != 2 * k:
        raise ValueError("Raw child does not reproduce the recorded exact-count edit")


def joint_key(seed, child, permutations):
    """Minimize a concatenated pair under ONE common source->destination action."""
    inv = np.argsort(np.asarray(permutations, dtype=int), axis=1)
    joined = np.concatenate([(1 - np.asarray(seed))[inv], (1 - np.asarray(child))[inv]], axis=1)
    packed = np.packbits(joined, axis=1, bitorder="big")
    witness = min(range(len(packed)), key=lambda i: packed[i].tobytes())
    bits = np.unpackbits(packed[witness], bitorder="big")[:2 * len(seed)]
    return "".join(map(str, bits.tolist())), witness


class Components:
    def __init__(self): self.parent = {}
    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[max(a, b)] = min(a, b)


def assign_components(seed_keys, child_keys, split_seed):
    """Keep every seed/child identity family in a single nested editor role."""
    components = Components()
    for a, b in zip(seed_keys, child_keys): components.union(str(a), str(b))
    groups = defaultdict(list)
    for key in components.parent: groups[components.find(key)].append(key)
    labels = {}
    for group in groups.values():
        anchor = min(group)
        code = hashlib.sha256(f"editor-component-v1|{split_seed}|{anchor}".encode()).digest()
        value = int.from_bytes(code[:8], "big") / 2 ** 64
        labels[components.find(anchor)] = (anchor, "train" if value < .8 else "val" if value < .9 else "heldout")
    identities = [labels[components.find(str(k))] for k in seed_keys]
    return np.asarray([v[0] for v in identities]), np.asarray([v[1] for v in identities])


def score_lookup(scores):
    ids = scores["structure_ids"].astype(str)
    if len(set(ids)) != len(ids): raise ValueError("Oracle union has duplicate scalar score identities")
    energies, atoms = np.asarray(scores["energy_eV"], float), np.asarray(scores["n_atoms"], int)
    status = scores.get("status", np.full(len(ids), "success")).astype(str)
    return {sid: dict(energy_eV_atom=float(e / a), n_atoms=int(a), status=str(st))
            for sid, e, a, st in zip(ids, energies, atoms, status)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20261107)
    parser.add_argument("--minimum-assessment-seeds-per-cell", type=int, default=8,
                        help="Declared feasibility gate, not a statistical adequacy claim")
    args = parser.parse_args(argv)
    from cathode_cvae.runtime import require_compute_allocation
    from cathode_cvae.acquisition import canonical_keys
    from cathode_cvae.conditions import classify, load_schema
    require_compute_allocation()
    root, out = args.root.resolve(), args.output_dir.resolve()
    run = root / "runs/conditioning_20261107"
    if out.exists(): raise FileExistsError("Preserve the existing edit-data preparation")
    paths = dict(parent=root / "artifacts/current/core/data/parents/O3_N48_pilot.npz",
        H0=root / "artifacts/current/improvement_v1/training_round0.npz",
        schema=root / "artifacts/overnight_20261107/data/condition_schema_uma.json",
        oracle=run / "collected/final/uma.npz", oracle_receipt=run / "collected/final/receipt.json",
        final_evaluation_receipt=run / "evaluation/final/receipt.json",
        final_candidates=run / "final/candidates.npz", required_gate=root / "artifacts/conditioning_20261107/required_gate.json")
    hashes = {name: digest(path) for name, path in paths.items()}
    gate = read_json(paths["required_gate"])
    if gate.get("status") != "COMPLETE" or gate.get("core_full_training_trajectories") != 28:
        raise ValueError("Optional edit preparation requires the completed required-report gate")
    report = gate["report_pdf"]; pdf = root / report["path"]
    if digest(pdf) != report["sha256"]: raise ValueError("Required frozen report changed")
    final_receipt_name = str(paths["final_evaluation_receipt"].relative_to(root))
    if gate["bindings"].get(final_receipt_name) != hashes["final_evaluation_receipt"]:
        raise ValueError("Frozen gate does not bind the final scalar-score evaluation")
    if read_json(paths["final_evaluation_receipt"])["source_hashes"]["scores_uma"] != hashes["oracle"]:
        raise ValueError("Oracle union differs from the gate-verified UMA score file")
    parent, data = load_npz(paths["parent"]), load_npz(paths["H0"])
    if data["occupancy"].shape[1] != 48 or parent["permutations"].shape[1] != 48:
        raise ValueError("This optional editor pilot is explicitly the 48-site benchmark")
    schema = load_schema(paths["schema"])
    if schema["provenance"]["data_sha256"] != hashes["H0"] or schema["provenance"]["parent_sha256"] != hashes["parent"]:
        raise ValueError("H0 energy schema is bound to different data/parent")
    keys = data["canonical_masks"].astype(str)
    h0roles = {str(k): str(role) for k, role in zip(keys, data["split"])}
    if len(h0roles) != len(keys): raise ValueError("H0 roles are not unique by canonical identity")
    protected = {k for k, role in h0roles.items() if role not in {"train"}}
    final = load_npz(paths["final_candidates"])
    final_ids = set(final["structure_ids"].astype(str))
    scores = score_lookup(load_npz(paths["oracle"]))
    original_labels = classify(schema, data["n_Li"], data["energy_eV_atom"])
    pairs, provenance, exclusions, missing_hash_repairs = [], [], [], Counter()
    orbit_to_index = {}
    for stage in STAGES:
        folder = run / "development" / stage
        receipt = read_json(folder / "receipt.json")
        for filename in ("attempts_all.csv", "raw_attempts.npz", "candidates.npz"):
            path = folder / filename; checksum = digest(path)
            if receipt["outputs"][filename] != checksum: raise ValueError(f"Saved DEVELOPMENT source changed: {stage}/{filename}")
            paths[f"{stage}/{filename}"] = path; hashes[f"{stage}/{filename}"] = checksum
        paths[f"{stage}/receipt.json"] = folder / "receipt.json"; hashes[f"{stage}/receipt.json"] = digest(folder / "receipt.json")
        raw = load_npz(folder / "raw_attempts.npz")
        candidate = load_npz(folder / "candidates.npz")
        if str(candidate["parent_sha256"].item()) != hashes["parent"]: raise ValueError("DEV candidate parent changed")
        candidate_map = {str(sid): x for sid, x in zip(candidate["structure_ids"], candidate["occupancy"])}
        with (folder / "attempts_all.csv").open() as f: rows = list(csv.DictReader(f))
        mutation = [r for r in rows if r.get("seed_source_row", "") != "" and r.get("diagnostic") == "primary"]
        seeds = np.asarray([raw["seed_occupancy"][int(r["raw_row"])] for r in mutation], dtype=np.uint8)
        children = np.asarray([raw["occupancy"][int(r["raw_row"])] for r in mutation], dtype=np.uint8)
        seed_keys = canonical_keys(seeds, parent["permutations"])
        child_keys = canonical_keys(children, parent["permutations"])
        for row, seed, child, skey, ckey in zip(mutation, seeds, children, seed_keys, child_keys):
            source_row = int(row["seed_source_row"]); rawrow = int(row["raw_row"])
            sid, cid = str(row["seed_structure_id"]), str(row["scoring_structure_id"])
            base = dict(source_stage=stage, source_attempt_id=row["attempt_id"], source_raw_row=rawrow,
                        seed_id=sid, child_id=cid, H0_source_row=source_row)
            if row.get("status") != "generated":
                exclusions.append(dict(base, reason="failed_generation", detail=row.get("error", ""))); continue
            if source_row < 0 or source_row >= len(data["occupancy"]): raise ValueError("Seed source row outside frozen H0")
            if sid != str(data["structure_ids"][source_row]) or not np.array_equal(seed, data["occupancy"][source_row]):
                raise ValueError("Raw seed does not match its exact H0 row and source identity")
            if row.get("seed_source_sha256") and row["seed_source_sha256"] != hashes["H0"]:
                raise ValueError("Recorded seed source hash disagrees with frozen H0")
            if not row.get("seed_source_sha256"): missing_hash_repairs[stage] += 1
            if str(skey) != str(keys[source_row]) or row["canonical_mask"] != str(ckey) or cid != identity(str(ckey)):
                raise ValueError("Seed/child canonical lineage identity disagrees with the recorded raw masks")
            if row["parent_sha256"] != hashes["parent"] or row["condition_schema_id"] != schema["identity"]:
                raise ValueError("Mutation lineage uses a different parent or energy schema")
            if cid not in candidate_map or not np.array_equal(candidate_map[cid], np.fromiter((1-int(c) for c in ckey), np.uint8)):
                raise ValueError("Recorded candidate does not encode the exact canonical child")
            removed = list(map(int, row["removed_sites"].split("|")))
            added = list(map(int, row["added_sites"].split("|")))
            k = int(row["swap_count"]); validate_edit(seed, child, removed, added, k)
            n = int(seed.sum())
            if n != int(row["n_Li"]) or n not in (12, 24, 36): raise ValueError("Unexpected edit composition")
            if str(skey) in protected or str(ckey) in protected:
                exclusions.append(dict(base, reason="protected_H0_VAL_TEST_identity", detail="seed or child")); continue
            if str(data["split"][source_row]) != "train": raise ValueError("Mutation seed is outside protected TRAIN")
            if identity(str(skey)) in final_ids or cid in final_ids:
                exclusions.append(dict(base, reason="core_FINAL_candidate_identity", detail="seed or child")); continue
            score = scores.get(cid)
            if not score or score["status"].lower() not in {"success", "complete", "completed", "ok", "fresh", "cached"} or not np.isfinite(score["energy_eV_atom"]):
                exclusions.append(dict(base, reason="missing_or_failed_child_energy", detail="kept visible, never imputed")); continue
            if score["n_atoms"] != 144+n: raise ValueError("Child scalar score has the wrong actual atom count")
            child_bin = str(classify(schema, np.asarray([n]), np.asarray([score["energy_eV_atom"]]))[0])
            seed_bin = str(original_labels[source_row])
            if seed_bin != row["requested_bin"]: raise ValueError("Mutation seed does not belong to its recorded requested bin")
            joint, witness = joint_key(seed, child, parent["permutations"])
            pair_id = "edit_" + hashlib.sha256(joint.encode()).hexdigest()[:24]
            if joint not in orbit_to_index:
                orbit_to_index[joint] = len(pairs)
                pairs.append(dict(seed=seed, child=child, n=n, k=k, seed_bin=seed_bin, child_bin=child_bin,
                    seed_id=sid, child_id=cid, seed_key=str(skey), child_key=str(ckey), joint=joint,
                    witness=witness, pair_id=pair_id, source_row=source_row, seed_energy=float(data["energy_eV_atom"][source_row]),
                    child_energy=score["energy_eV_atom"], base=base))
            provenance.append(dict(base, pair_id=pair_id, joint_witness_permutation=witness,
                duplicate_joint_pair=orbit_to_index[joint] != len(pairs)-1 or pairs[orbit_to_index[joint]]["base"] != base,
                recorded_seed_hash_present=bool(row.get("seed_source_sha256")), bound_H0_sha256=hashes["H0"],
                requested_seed_bin=seed_bin, actual_child_bin=child_bin, swap_count=k))
    if not pairs: raise ValueError("No role-safe verified DEVELOPMENT edits remain")
    component_ids, split = assign_components([p["seed_key"] for p in pairs], [p["child_key"] for p in pairs], args.split_seed)
    for p, component, role in zip(pairs, component_ids, split): p.update(component_id=component, split=role)
    # A canonical identity can be a child in one row and a seed in another.
    # Verify the joint component split, not separate seed/child-only split sets.
    roles = defaultdict(set)
    for p in pairs:
        roles[p["seed_key"]].add(p["split"]); roles[p["child_key"]].add(p["split"])
    if any(len(v) != 1 for v in roles.values()): raise ValueError("Editor identity bridge crosses nested roles")
    assessment = {}
    for p in pairs:
        if p["split"] == "heldout": assessment.setdefault(p["seed_key"], p)
    support = Counter((p["n"], p["seed_bin"]) for p in assessment.values())
    cells = [(n, b) for n in (12, 24, 36) for b in BINS]
    feasible = all(support[cell] >= args.minimum_assessment_seeds_per_cell for cell in cells)
    out.mkdir(parents=True, exist_ok=False)
    arrays = dict(seeds=np.asarray([p["seed"] for p in pairs], np.uint8), children=np.asarray([p["child"] for p in pairs], np.uint8),
        n_Li=np.asarray([p["n"] for p in pairs]), swap_count=np.asarray([p["k"] for p in pairs]),
        seed_bin=np.asarray([p["seed_bin"] for p in pairs]), child_bin=np.asarray([p["child_bin"] for p in pairs]), split=split,
        seed_ids=np.asarray([p["seed_id"] for p in pairs]), child_ids=np.asarray([p["child_id"] for p in pairs]),
        canonical_seed_keys=np.asarray([p["seed_key"] for p in pairs]), canonical_child_keys=np.asarray([p["child_key"] for p in pairs]),
        component_ids=component_ids, pair_ids=np.asarray([p["pair_id"] for p in pairs]),
        source_stage=np.asarray([p["base"]["source_stage"] for p in pairs]), source_attempt_id=np.asarray([p["base"]["source_attempt_id"] for p in pairs]),
        source_raw_row=np.asarray([p["base"]["source_raw_row"] for p in pairs]), H0_source_rows=np.asarray([p["source_row"] for p in pairs]),
        joint_orbit_key=np.asarray([p["joint"] for p in pairs]), joint_witness_permutation=np.asarray([p["witness"] for p in pairs]),
        seed_energy_eV_atom=np.asarray([p["seed_energy"] for p in pairs]), child_energy_eV_atom=np.asarray([p["child_energy"] for p in pairs]),
        parent_sha256=np.asarray(hashes["parent"]), H0_sha256=np.asarray(hashes["H0"]), condition_schema_id=np.asarray(schema["identity"]))
    np.savez_compressed(out / "edit_pairs.npz", **arrays)
    aa = list(assessment.values())
    np.savez_compressed(out / "assessment_seeds.npz", occupancy=np.asarray([p["seed"] for p in aa], np.uint8),
        seed_ids=np.asarray([p["seed_id"] for p in aa]), seed_bin=np.asarray([p["seed_bin"] for p in aa]),
        n_Li=np.asarray([p["n"] for p in aa]), canonical_seed_keys=np.asarray([p["seed_key"] for p in aa]),
        component_ids=np.asarray([p["component_id"] for p in aa]), H0_source_rows=np.asarray([p["source_row"] for p in aa]),
        seed_energy_eV_atom=np.asarray([p["seed_energy"] for p in aa]), split=np.full(len(aa), "heldout"),
        parent_sha256=np.asarray(hashes["parent"]), H0_sha256=np.asarray(hashes["H0"]), condition_schema_id=np.asarray(schema["identity"]))
    counts = Counter((p["split"], p["n"], p["seed_bin"], p["child_bin"], p["k"]) for p in pairs)
    write_rows(out / "support.csv", [dict(split=s, n_Li=n, seed_bin=b, child_bin=c, swap_count=k, pairs=count)
        for (s,n,b,c,k),count in sorted(counts.items())])
    write_rows(out / "assessment_support.csv", [dict(n_Li=n, seed_bin=b, unique_seed_families=support[(n,b)],
        minimum_required=args.minimum_assessment_seeds_per_cell) for n,b in cells])
    write_rows(out / "pair_provenance.csv", provenance)
    write_rows(out / "excluded_attempts.csv", exclusions)
    write_rows(out / "components.csv", [dict(pair_id=p["pair_id"], seed_key=p["seed_key"], child_key=p["child_key"],
        component_id=p["component_id"], split=p["split"]) for p in pairs])
    changed = [name for name,path in paths.items() if digest(path) != hashes[name]]
    if changed: raise ValueError(f"Preparation source changed during the read: {changed}")
    receipt = dict(status="READY_FOR_BOUNDED_EDITOR" if feasible else "INSUFFICIENT_HELDOUT_SEED_SUPPORT",
        created_UTC=datetime.now(timezone.utc).isoformat(), experiment_phase="optional_editor", data_rows=len(pairs),
        observed_verified_mutation_attempts=len(provenance), duplicate_joint_pairs=len(provenance)-len(pairs),
        exclusions=dict(Counter(p["reason"] for p in exclusions)), split_rows=dict(Counter(split.tolist())),
        components=len(set(component_ids)), identity_role_bridges=0,
        source_stages=list(STAGES), all_child_outcomes_retained=True, core_FINAL_identity_exclusion_count=len(final_ids),
        protected_H0_identity_count=len(protected), condition_schema_id=schema["identity"],
        representation="Original raw seed and child occupancies in ONE shared parent orientation; no independent canonicalization for learning",
        joint_deduplication="Minimum concatenated (vacancy seed, vacancy child) under one common parent permutation; witness recorded; duplicate source attempts retained in CSV",
        nested_split=dict(seed=args.split_seed, rule="SHA256(component minimum canonical identity): 80% train, 10% val, 10% heldout; no energy used in splitting"),
        heldout_seed_support={f"{n}:{b}":support[(n,b)] for n,b in cells}, assessment_support_feasible=feasible,
        assessment_proposal_budget="128 attempts per composition/requested CHILD bin/fit replicate; k drawn uniformly from {1,2,3}; each compared method receives exactly the same seed and k; heldout seeds sampled with replacement; all attempts retained",
        original_missing_seed_hash_fields=dict(missing_hash_repairs),
        seed_hash_binding_repair="Original missing fields preserved as absent; exact H0 row, ID, binary raw seed, and H0 file SHA verified and recorded separately",
        required_report_gate=dict(path=str(paths["required_gate"]), sha256=hashes["required_gate"], completed_UTC=gate["completed_UTC"]),
        source_hashes={name:dict(path=str(paths[name]),sha256=value) for name,value in hashes.items()},
        script_sha256=digest(__file__), outputs={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()},
        new_MLIP_calls=0, new_training_updates=0, limitations=["This is a small, proposal-biased DEVELOPMENT edit corpus, not a general chemical optimizer.",
            "H0 TRAIN seeds were known to core models; nested heldout roles test the new editor, not globally unseen H0 identities.",
            "Data feasibility is separate from learned-edit effectiveness; improvement must be evaluated against same-seed, same-budget mutation."])
    (out / "preparation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if feasible else 3


if __name__ == "__main__": raise SystemExit(main())

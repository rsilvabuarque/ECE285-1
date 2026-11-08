"""Freeze a uniform-raw, fresh-to-intake symmetry-deduplicated paired cohort.

This command reads masks/roles only, never their outcome energy columns. Run in
an analysis allocation. No calculator or scheduler is imported or invoked.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np

from paired_common import (atomic_json, atomic_npz, canonical, identity, load_npz,
                           mobile_indices, sha256, structure_id, task_for_row)


def prepare(args):
    from cathode_cvae.runtime import require_compute_allocation
    from cathode_cvae.mlip.registry import MODELS
    require_compute_allocation()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    parent = load_npz(args.parent)
    nsites = len(mobile_indices(parent))
    history_record = json.loads(args.history.read_text())
    history = set(history_record if isinstance(history_record, list) else history_record["canonical_masks"])
    if any(len(k) != nsites or set(k) - {"0", "1"} for k in history):
        raise ValueError("History must contain exact parent vacancy keys")
    # Protect old roles by excluding every already-exposed orbit. No old final
    # candidate is implicitly retired into this new acquisition's training set.
    with np.load(args.protected_data, allow_pickle=False) as archive:
        old_keys = set(archive["canonical_masks"].astype(str))
        old_roles = set(archive["split"].astype(str))
    if not old_keys <= history or not old_roles <= {"train", "val", "test"}:
        raise ValueError("Intake history must cover every original protected orbit")
    if any(c <= 0 or c >= nsites for c in args.counts):
        raise ValueError("Primary counts must lie strictly inside parent occupancy range")
    per_count = args.train_per_count + args.val_per_count + args.test_per_count
    if min(args.train_per_count, args.val_per_count, args.test_per_count) < 1:
        raise ValueError("Every independent role needs positive support")
    if args.pilot_per_count > args.train_per_count:
        raise ValueError("Pilot is a subset of independently assigned TRAIN masks")
    rng = np.random.default_rng(args.seed)
    raw_rows, raw_n, accepted_rows, rejection = [], [], [], []
    masks, keys, counts, roles, draw_indices, witnesses = [], [], [], [], [], []
    seen = set(history)
    pilot = []
    for count in args.counts:
        accepted = []
        for _ in range(args.max_draws_per_count):
            raw = np.zeros(nsites, np.uint8)
            raw[rng.choice(nsites, count, replace=False)] = 1
            key, witness = canonical(raw, parent["permutations"])
            reason = "intake_known" if key in history else "cohort_repeat" if key in seen else "accepted"
            draw = len(raw_rows)
            raw_rows.append(raw)
            raw_n.append(count)
            accepted_rows.append(reason == "accepted")
            rejection.append(reason)
            if reason != "accepted":
                continue
            seen.add(key)
            accepted.append((raw[witness], key, draw, witness))
            if len(accepted) == per_count:
                break
        if len(accepted) != per_count:
            raise RuntimeError("Frozen draw cap cannot supply requested independent cohort")
        role_array = np.array(["train"] * args.train_per_count + ["val"] * args.val_per_count + ["test"] * args.test_per_count)
        role_rng = np.random.default_rng(np.random.SeedSequence([args.seed, count, 71029]))
        role_array = role_array[role_rng.permutation(per_count)]
        train_pilot = 0
        for item, role in zip(accepted, role_array):
            row = len(masks)
            occupancy, key, draw, witness = item
            masks.append(occupancy); keys.append(key); counts.append(count); roles.append(role)
            draw_indices.append(draw); witnesses.append(witness)
            if role == "train" and train_pilot < args.pilot_per_count:
                pilot.append(row); train_pilot += 1
    if len(set(keys)) != len(keys) or set(keys) & history:
        raise AssertionError("New cohort overlaps itself or protected intake")
    shutil.copyfile(args.parent, out / "parent.npz")
    expected_teacher = MODELS["uma"]
    context = dict(schema="paired-fixed-endpoint-v1", parent_sha256=sha256(args.parent),
        teacher=dict(model="uma", checkpoint_hashes=expected_teacher["files"], head=expected_teacher["head"],
                     package_version=expected_teacher["version"], dtype="float32", tf32=False),
        geometry=dict(initial="canonical pristine-parent deletion", cell="fixed", all_atoms_move=True,
                      atom_identity="original full-parent index; never independent endpoint canonicalization"),
        optimizer=dict(name="FIRE", ase_version="3.26.0", fmax_eV_A=.05, maxstep_A=.15,
                       max_steps=500, wall_cap_seconds=120., snapshot_stride=5,
                       force_definition="maximum per-atom Euclidean norm; strict less-than stopping",
                       other_parameters="ASE 3.26 FIRE defaults"),
        units=dict(energy="eV/cell", forces="eV/Angstrom", stress="eV/Angstrom^3; ASE tensile-positive Voigt xx yy zz yz xz xy"),
        scientific_scope="Operational local fixed-cell endpoint labels; no global minimum or DFT claim")
    context["context_hash"] = identity(context)
    atomic_json(out / "context.json", context)
    atomic_npz(out / "cohort.npz", occupancy=np.asarray(masks, np.uint8),
               canonical_masks=np.asarray(keys), structure_ids=np.asarray([structure_id(k) for k in keys]),
               n_target=np.asarray(counts), n_Li=np.asarray(counts), split=np.asarray(roles),
               raw_draw_index=np.asarray(draw_indices), canonical_gather=np.asarray(witnesses),
               pilot_indices=np.asarray(pilot), parent_sha256=np.array(sha256(args.parent)))
    atomic_npz(out / "raw_draws.npz", occupancy=np.asarray(raw_rows, np.uint8), n_target=np.asarray(raw_n),
               accepted=np.asarray(accepted_rows), rejection_reason=np.asarray(rejection))
    cohort_sha = sha256(out / "cohort.npz")
    tasks = [task_for_row(out, i, key, role, out / "cohort.npz", cohort_sha)
             for i, (key, role) in enumerate(zip(keys, roles))]
    for name, rows in [("tasks_all.json", tasks), ("tasks_pilot.json", [tasks[i] for i in pilot])]:
        atomic_json(out / name, dict(schema="paired-relaxation-task-list-v1", context_hash=context["context_hash"], tasks=rows))
    outputs = {name: sha256(out / name) for name in ["parent.npz", "context.json", "cohort.npz", "raw_draws.npz", "tasks_all.json", "tasks_pilot.json"]}
    receipt = dict(status="FROZEN_BEFORE_ANY_LABELS", prepared_UTC=datetime.now(timezone.utc).isoformat(),
        frozen_unix=time.time(), source_sha256=sha256(__file__), seed=args.seed, n_sites=nsites,
        counts=args.counts, attempted_raw_draws=len(raw_rows), independent_masks=len(keys),
        roles_per_count={str(n): dict(Counter(role for count, role in zip(counts, roles) if count == n)) for n in args.counts},
        pilot_per_count=args.pilot_per_count, pilot_role="train", labels_read=False,
        sampling="uniform raw fixed-count masks, conditioned on fresh-to-intake, then symmetry deduplicated; not uniform over symmetry orbits",
        original_protected_roles_unchanged=True, original_protected_orbits=len(old_keys), intake_history_orbits=len(history),
        source_inputs={name: dict(path=str(path.absolute()), sha256=sha256(path)) for name, path in
                       [("parent", args.parent), ("history", args.history), ("protected_data", args.protected_data), ("goal_protocol", args.goal_protocol)]},
        endpoint_budget=dict(acquisition=6000, mandatory_current_low_assessment=2304,
            paired_target_assessment=3456, shared_baseline_reserve=1728, tighter_controls=96,
            conservative_first_wave_total=13584, maximum_new_primary_endpoint_attempts=18000,
            expansion="No acquisition expansion until measured throughput and the complete assessment reserve are reviewed"), outputs=outputs)
    atomic_json(out / "preparation.json", receipt)
    print(json.dumps({k: receipt[k] for k in ["status", "independent_masks", "attempted_raw_draws", "roles_per_count", "pilot_per_count"]}, indent=2))
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--protected-data", type=Path, required=True)
    p.add_argument("--goal-protocol", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--counts", type=int, nargs="+", default=[12, 24, 36])
    p.add_argument("--train-per-count", type=int, default=1200)
    p.add_argument("--val-per-count", type=int, default=400)
    p.add_argument("--test-per-count", type=int, default=400)
    p.add_argument("--pilot-per-count", type=int, default=32)
    p.add_argument("--max-draws-per-count", type=int, default=1000000)
    p.add_argument("--seed", type=int, default=2026110807)
    return prepare(p.parse_args(argv))


if __name__ == "__main__":
    main()

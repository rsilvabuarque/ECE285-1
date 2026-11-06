"""Bounded exact nearest-training-orbit distances for a sampled proposal subset.

This measures vacancy-mask locality, not relaxed-phase or geometric diversity.
Every retained parent permutation is applied to TRAINING masks only. Distances
are exact for completed queries; only a deterministic subset of proposals is
queried. Run substantial reference expansion/scans in a compute allocation.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


class BudgetReached(Exception):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--attempts", type=Path, default=HERE / "data/final_attempts.csv")
    ap.add_argument("--training", type=Path, default=HERE / "training_round1.npz")
    ap.add_argument("--original", type=Path, default=HERE / "training_original.npz")
    ap.add_argument("--known-keys", type=Path, required=True)
    ap.add_argument("--output-prefix", type=Path, default=HERE / "reports/final_diversity_subset")
    ap.add_argument("--sample-per-group", type=int, default=12)
    ap.add_argument("--max-seconds", type=float, default=300)
    ap.add_argument("--block", type=int, default=262144)
    args = ap.parse_args()
    assert os.environ.get("SLURMD_NODENAME", "").startswith("nid"), "Use a compute allocation"
    assert args.sample_per_group > 0 and args.block > 0 and args.max_seconds > 0
    assert hasattr(np, "bitwise_count"), "NumPy >=2 with vectorized bitwise_count is required"
    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {k: Path(str(prefix) + suffix) for k, suffix in
             [("queries", "_queries.csv"), ("groups", "_groups.csv"),
              ("methods", "_methods.csv"), ("receipt", "_receipt.json")]}
    assert not any(path.exists() for path in paths.values()), "Preserve earlier subset audits"
    started = time.perf_counter()

    def check_budget():
        if time.perf_counter() - started >= args.max_seconds:
            raise BudgetReached

    known_payload = json.loads(args.known_keys.read_text())
    known = set(known_payload["canonical_masks"] if isinstance(known_payload, dict) else known_payload)
    groups = defaultdict(set)
    attempts = defaultdict(int)
    fresh_attempts = defaultdict(int)
    for row in csv.DictReader(args.attempts.open()):
        key = (row["method"], int(row["seed"]), int(row["n_Li"]))
        mask = row["canonical_mask"]
        assert len(mask) == 48 and set(mask) <= {"0", "1"}
        assert mask.count("0") == key[2]
        assert row["scoring_structure_id"] == f"mask48_{int(mask, 2):012x}"
        is_known = mask in known
        assert (row["known_before"] == "True") == is_known
        attempts[key] += 1
        if not is_known:
            groups[key].add(mask)
            fresh_attempts[key] += 1
        else:
            groups[key]  # Retain groups with zero novel output.
    selected = {group: sorted(masks, key=lambda m: hashlib.sha256(
        ("locality_subset_v1:" + m).encode()).digest())[:args.sample_per_group]
                for group, masks in groups.items()}
    parent_path = HERE.parent / "data/parents/O3_N48_pilot.npz"
    with np.load(parent_path, allow_pickle=False) as z:
        inverse = np.argsort(z["permutations"], axis=1)
    shifts = np.arange(47, -1, -1, dtype=np.int64).astype(np.uint64)
    powers = np.left_shift(np.uint64(1), shifts)
    refs = {}
    for name, path in [("original", args.original), ("round1", args.training)]:
        with np.load(path, allow_pickle=False) as z:
            keep = z["split"].astype(str) == "train"
            assert all(str(key) in known for key in z["canonical_masks"][keep]), "Known boundary omits training masks"
            codes = np.array([int(str(key), 2) for key in z["canonical_masks"][keep]], dtype=np.uint64)
            counts = z["n_Li"][keep].astype(int)
            assert len(codes) == len(np.unique(codes))
            assert np.array_equal(48 - np.bitwise_count(codes).astype(int), counts)
            refs[name] = (codes, counts)
    cache, reference_counts = {}, {}
    partial = False
    try:
        for n in sorted({group[2] for group in selected}):
            expanded = {}
            reference_counts[str(n)] = {}
            for name, (codes, counts) in refs.items():
                train = codes[counts == n]
                assert len(train), (name, n)
                pieces = []
                for start in range(0, len(train), 256):
                    check_budget()
                    masks = ((train[start:start + 256, None] >> shifts) & 1).astype(np.uint8)
                    pieces.append((masks[:, inverse] @ powers).reshape(-1))
                expanded[name] = np.unique(np.concatenate(pieces))
                reference_counts[str(n)][name] = {
                    "train_orbits": len(train), "distinct_orientations": len(expanded[name])}
            queries = sorted({mask for group, masks in selected.items() if group[2] == n for mask in masks})
            for mask in queries:
                distances = {}
                code = np.uint64(int(mask, 2))
                for name, reference in expanded.items():
                    best = 48
                    for start in range(0, len(reference), args.block):
                        check_budget()
                        hamming = int(np.bitwise_count(code ^ reference[start:start + args.block]).min())
                        best = min(best, hamming)
                        if best == 0:
                            break
                    assert best % 2 == 0
                    assert best > 0, "Fresh query matches a training symmetry orbit"
                    distances[name] = best // 2
                cache[mask] = distances  # Store only complete reference pairs.
            print("LOCALITY_COUNT_DONE", n, len(queries), flush=True)
    except BudgetReached:
        partial = True
    query_rows, group_rows = [], []
    for group in sorted(selected):
        method, seed, n = group
        completed = [mask for mask in selected[group] if mask in cache]
        for mask in completed:
            query_rows.append(dict(method=method, seed=seed, n_Li=n,
                                   scoring_structure_id=f"mask48_{int(mask, 2):012x}",
                                   canonical_mask=mask, nearest_original_train_swaps=cache[mask]["original"],
                                   nearest_round1_train_swaps=cache[mask]["round1"]))
        row = dict(method=method, seed=seed, n_Li=n, attempts=attempts[group],
                   fresh_attempts=fresh_attempts[group], distinct_fresh_orbits=len(groups[group]),
                   subset_requested=min(args.sample_per_group, len(groups[group])),
                   subset_completed=len(completed))
        for name in refs:
            values = [cache[mask][name] for mask in completed]
            row[f"sample_mean_nearest_{name}_swaps"] = float(np.mean(values)) if values else ""
            row[f"sample_median_nearest_{name}_swaps"] = float(np.median(values)) if values else ""
        group_rows.append(row)
    method_rows = []
    for method in sorted({row["method"] for row in group_rows}):
        rows = [row for row in group_rows if row["method"] == method]
        row = dict(method=method, total_groups=len(rows),
                   groups_with_completed_samples=sum(r["subset_completed"] > 0 for r in rows),
                   sampled_group_rows=sum(r["subset_completed"] for r in rows),
                   requested_group_rows=sum(r["subset_requested"] for r in rows))
        for name in refs:
            values = [r[f"sample_mean_nearest_{name}_swaps"] for r in rows if r["subset_completed"]]
            row[f"macro_mean_sample_nearest_{name}_swaps"] = float(np.mean(values)) if values else ""
        method_rows.append(row)
    qfields = ["method", "seed", "n_Li", "scoring_structure_id", "canonical_mask",
               "nearest_original_train_swaps", "nearest_round1_train_swaps"]
    write_csv(paths["queries"], query_rows, qfields)
    write_csv(paths["groups"], group_rows, list(group_rows[0]))
    write_csv(paths["methods"], method_rows, list(method_rows[0]))
    receipt = dict(status="PARTIAL_TIME_CAP" if partial else "PASS", seconds=time.perf_counter() - started,
                   maximum_seconds=args.max_seconds, sample_per_group=args.sample_per_group,
                   selected_distinct_queries=len({m for masks in selected.values() for m in masks}),
                   completed_distinct_queries=len(cache), completed_group_query_rows=len(query_rows),
                   reference_counts=reference_counts, parent_permutations=len(inverse),
                   inputs={str(path): sha(path) for path in
                           [args.attempts, args.training, args.original, args.known_keys, parent_path]},
                   script_sha256=sha(__file__), outputs={k: sha(v) for k, v in paths.items() if k != "receipt"},
                   selection="Up to 12 distinct fresh orbits/group by stable SHA256 order; reused query cache across groups",
                   scope="Exact fixed-composition minimum Hamming/2 over all retained periodic parent permutations. Sampled proposal subset only; no all-candidate, phase or population guarantee. Partial runs must not be reported as complete samples.")
    paths["receipt"].write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()

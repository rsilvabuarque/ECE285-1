"""Assess fresh proposals against frozen, original training-energy thresholds.

Energy is measured on the same pristine-derived deletion geometry. Novelty is
relative to a key set frozen before generation, including the adaptation pool.
Duplicate attempts remain in the denominator. Run campaign-scale analysis in
an allocation on Perlmutter; this script is also portable to a local machine.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_keys(path):
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict):
        for name in ["canonical_masks", "canonical_keys", "keys", "known_keys"]:
            if name in value:
                value = value[name]
                break
        else:
            raise ValueError("Known-key JSON must contain canonical_masks/keys or be a list")
    keys = set(map(str, value))
    assert all(len(k) == 48 and set(k) <= {"0", "1"} for k in keys)
    return keys


def score_arrays(path, candidate_n):
    with np.load(path, allow_pickle=False) as z:
        ids = z["structure_ids"].astype(str)
        assert ids.ndim == 1 and len(set(ids)) == len(ids)
        if "energy_eV_atom" in z:
            energy = z["energy_eV_atom"].astype(float)
        else:
            total = z["energy_eV"].astype(float)
            counts = z["n_atoms"] if "n_atoms" in z else np.array([144 + candidate_n[s] for s in ids])
            energy = total / counts
        assert energy.shape == ids.shape and np.isfinite(energy).all()
        if "n_atoms" in z:
            for s, n in zip(ids, z["n_atoms"]):
                if s in candidate_n:
                    assert int(n) == 144 + candidate_n[s], (s, n)
    return dict(zip(ids, energy.tolist()))


def average_ranks(x):
    order = np.argsort(x, kind="stable")
    ranked = np.empty(len(x), float)
    start = 0
    while start < len(x):
        stop = start + 1
        while stop < len(x) and x[order[stop]] == x[order[start]]:
            stop += 1
        ranked[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranked


def evaluate(proposal_path, score_paths, known_path, threshold_path, catalog_path, output, pool):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    proposals = read_csv(proposal_path)
    assert proposals
    required = {"method", "seed", "n_Li", "scoring_structure_id", "canonical_mask"}
    assert required <= set(proposals[0])
    known = load_keys(known_path)
    original = {r["canonical_mask"] for r in read_csv(catalog_path)
                if r.get("parent") == "O3_N48_pilot"}
    assert original <= known, "Frozen known keys must include the original main-parent catalog"
    candidate_n, candidate_key, key_id = {}, {}, {}
    groups = defaultdict(list)
    for r in proposals:
        sid, key, n = r["scoring_structure_id"], r["canonical_mask"], int(r["n_Li"])
        assert len(key) == 48 and set(key) <= {"0", "1"} and key.count("0") == n
        assert 0 <= n <= 48, n
        assert candidate_key.get(sid, key) == key and key_id.get(key, sid) == sid
        candidate_n[sid], candidate_key[sid], key_id[key] = n, key, sid
        groups[r["method"], str(r["seed"]), n].append(r)
    scores = {m: score_arrays(path, candidate_n) for m, path in score_paths.items()}
    for m, data in scores.items():
        missing = set(candidate_n) - set(data)
        assert not missing, (m, len(missing), sorted(missing)[:3])
    thresholds = {}
    for r in read_csv(threshold_path):
        if r["source"] == "main_train":
            thresholds[r["teacher"], int(r["n_Li"])] = float(r["q20_eV_atom"])
    rows, candidates = [], []
    for m, data in scores.items():
        for sid, n in sorted(candidate_n.items()):
            key, e = candidate_key[sid], data[sid]
            q = thresholds[m, n]
            candidates.append(dict(pool=pool, teacher=m, scoring_structure_id=sid,
                canonical_mask=key, n_Li=n, energy_eV_atom=e, original_q20_eV_atom=q,
                delta_meV_atom=1000 * (e - q), original_catalog_novel=key not in original,
                frozen_history_novel=key not in known, passes_original_low_tail=e <= q))
        for (method, seed, n), rr in sorted(groups.items()):
            ids = [r["scoring_structure_id"] for r in rr]
            unique = sorted(set(ids))
            novel = [s for s in unique if candidate_key[s] not in known]
            e = np.array([data[s] for s in ids])
            q = thresholds[m, n]
            low = [s for s in novel if data[s] <= q]
            if "attempt" in rr[0]:
                assert len({r["attempt"] for r in rr}) == len(rr), (method, seed, n)
            rows.append(dict(pool=pool, teacher=m, method=method, seed=seed, n_Li=n,
                attempts=len(ids), unique_orbits=len(unique), duplicate_fraction=1 - len(unique) / len(ids),
                distinct_novel_orbits=len(novel), novel_orbit_yield_per_attempt=len(novel) / len(ids),
                original_catalog_novel_orbits=sum(candidate_key[s] not in original for s in unique),
                known_history_orbits=len(unique) - len(novel), distinct_novel_low_orbits=len(low),
                novel_low_yield_per_attempt=len(low) / len(ids),
                novel_unique_low_fraction=len(low) / len(novel) if novel else 0.,
                original_q20_eV_atom=q, attempt_median_delta_meV_atom=float(1000 * (np.median(e) - q)),
                attempt_p10_delta_meV_atom=float(1000 * (np.quantile(e, .1) - q)),
                minimum_delta_meV_atom=float(1000 * (e.min() - q)),
                threshold_scope="original_main_train", novelty_scope="frozen_original_and_adaptation_history"))
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "candidate_scores.csv", candidates)
    seed_groups = defaultdict(list)
    for r in rows:
        seed_groups[r["teacher"], r["method"], r["seed"]].append(r)
    seed_rows = []
    for (m, method, seed), rr in sorted(seed_groups.items()):
        assert len({r["n_Li"] for r in rr}) == len(rr)
        seed_rows.append(dict(pool=pool, teacher=m, method=method, seed=seed,
            compositions=len(rr), macro_low_yield=float(np.mean([r["novel_low_yield_per_attempt"] for r in rr])),
            attempts=sum(r["attempts"] for r in rr),
            distinct_novel_low_orbits=sum(r["distinct_novel_low_orbits"] for r in rr)))
    write_csv(output / "seed_metrics.csv", seed_rows)
    summary = []
    for m, method in sorted({(r["teacher"], r["method"]) for r in rows}):
        rr = [r for r in rows if r["teacher"] == m and r["method"] == method]
        ss = [r["macro_low_yield"] for r in seed_rows if r["teacher"] == m and r["method"] == method]
        summary.append(dict(pool=pool, teacher=m, method=method, seed_count=len(ss),
            composition_seed_groups=len(rr), mean_low_yield=float(np.mean(ss)),
            minimum_seed_macro_yield=float(min(ss)), maximum_seed_macro_yield=float(max(ss)),
            seed_macro_standard_deviation=float(np.std(ss, ddof=1)) if len(ss) > 1 else "",
            mean_duplicate_fraction=float(np.mean([r["duplicate_fraction"] for r in rr])),
            mean_novel_orbit_yield=float(np.mean([r["novel_orbit_yield_per_attempt"] for r in rr]))))
    write_csv(output / "summary.csv", summary)
    agreements = []
    names = list(scores)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for n in sorted(set(candidate_n.values())):
                ids = [s for s in candidate_n if candidate_n[s] == n]
                x, y = [average_ranks(np.array([scores[m][s] for s in ids])) for m in [a, b]]
                rho = float(np.corrcoef(x, y)[0, 1]) if len(ids) > 2 and x.std() > 0 and y.std() > 0 else ""
                agreements.append(dict(pool=pool, teacher_a=a, teacher_b=b, n_Li=n,
                    unique_orbits=len(ids), spearman=rho))
    write_csv(output / "teacher_agreement.csv", agreements)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for m in names:
            methods = sorted({r["method"] for r in rows if r["teacher"] == m})
            counts = sorted(set(candidate_n.values()))
            fig, axs = plt.subplots(1, len(counts), figsize=(max(12, len(methods) * .75), 4.8), sharey=True, squeeze=False)
            axs = axs[0]
            for ax, n in zip(axs, counts):
                for i, method in enumerate(methods):
                    yy = [r["novel_low_yield_per_attempt"] for r in rows
                          if r["teacher"] == m and r["method"] == method and r["n_Li"] == n]
                    ax.bar(i, np.mean(yy), color="steelblue", alpha=.55)
                    ax.scatter([i] * len(yy), yy, color="black", s=15)
                ax.set_xticks(range(len(methods)), methods, rotation=55, ha="right", fontsize=7)
                ax.set_title(f"Li{n}Co48O96")
                ax.set_ylim(0, 1)
                ax.grid(axis="y", alpha=.2)
            axs[0].set_ylabel("Novel low-tail orbits / proposal attempts")
            fig.suptitle(f"{pool}: {m} scoring; original training q20; dots are seed results", fontsize=11)
            fig.tight_layout()
            fig.savefig(output / f"yield_{m}.pdf")
            fig.savefig(output / f"yield_{m}.png", dpi=160)
            plt.close(fig)
    except ImportError:
        pass
    receipt = dict(status="PASS", pool=pool, proposal_attempts=len(proposals), unique_orbits=len(candidate_n),
        teachers=names, group_rows=len(rows), source_hashes={"proposals": sha(proposal_path),
            "known_keys": sha(known_path), "thresholds": sha(threshold_path), "catalog": sha(catalog_path),
            **{f"scores_{m}": sha(path) for m, path in score_paths.items()}},
        frozen_history_keys=len(known), original_catalog_keys=len(original),
        novel_orbits=sum(k not in known for k in candidate_key.values()),
        interpretation="Fresh MLIP consistency under fixed deletion geometry; not DFT truth, ground states, or relaxed-energy optimization.",
        limitations=["Seed ranges are descriptive, not a confidence interval.",
            "Different compositions and MLIP energy zero points are never pooled as raw energies.",
            "This receipt verifies identities/coverage/metrics; separate oracle provenance and freeze timestamps must be audited.",
            "Unique orbit yield penalizes repeated/known proposals; it does not establish broad structural diversity."])
    (output / "evaluation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser(description="Recompute the current 48-site sampler benchmark from cached labels; no MLIP is loaded")
    ap.add_argument("--snapshot", type=Path, help="Frozen current artifact directory; supplies all benchmark inputs")
    ap.add_argument("--proposals", type=Path)
    ap.add_argument("--scores", nargs="+", help="teacher=path.npz, or one path with --teacher")
    ap.add_argument("--teacher", default="uma")
    ap.add_argument("--known-keys", type=Path)
    ap.add_argument("--thresholds", type=Path)
    ap.add_argument("--catalog", type=Path)
    ap.add_argument("--pool", choices=["development", "final", "round1_development"], default="final")
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.snapshot:
        base = a.snapshot / "improvement_v1"
        a.proposals = a.proposals or base / "data" / f"{a.pool}_attempts.csv"
        a.known_keys = a.known_keys or base / ("known_keys_final.json" if a.pool == "final" else "known_keys_round0.json")
        a.thresholds = a.thresholds or a.snapshot / "reports/generation_thresholds.csv"
        a.catalog = a.catalog or a.snapshot / "data/mask_manifest.csv"
        a.scores = a.scores or [f"{m}={base / 'data' / (a.pool + '_scores_' + m + '.npz')}" for m in ["uma", "mace"]]
    if not all([a.proposals, a.scores, a.known_keys, a.thresholds, a.catalog]):
        ap.error("Provide --snapshot or all five explicit proposal/score/history/threshold/catalog inputs")
    if all("=" in s for s in a.scores):
        score_paths = dict(s.split("=", 1) for s in a.scores)
    else:
        if len(a.scores) != 1:
            ap.error("Multiple scores must be teacher=path pairs")
        score_paths = {a.teacher: a.scores[0]}
    return evaluate(a.proposals, score_paths, a.known_keys, a.thresholds, a.catalog, a.output_dir, a.pool)


if __name__ == "__main__":
    main()

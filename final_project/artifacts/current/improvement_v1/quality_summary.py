"""Summarize energy quality of distinct, history-novel final proposals only.

All energy differences are within one teacher and Li count. Macro averages use
those differences, never raw energies across compositions. This is post-freeze
assessment, not model selection. Run campaign-scale analysis in an allocation.
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


def mean_present(rows, key):
    values = [float(r[key]) for r in rows if r[key] != ""]
    return float(np.mean(values)) if values else ""


def analyze(proposals, scores, known_keys, training, identity, collection, output, thresholds, pool="final"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rr = read_csv(proposals)
    assert rr
    payload = json.loads(Path(known_keys).read_text())
    known = set(payload["canonical_masks"] if isinstance(payload, dict) else payload)
    assert all(len(k) == 48 and set(k) <= {"0", "1"} for k in known)
    ident = json.loads(Path(identity).read_text())
    collected = json.loads(Path(collection).read_text())
    assert ident["status"] == "PASS" and ident["pool"] == pool
    assert collected["status"] == "PASS" and collected["pool"] == pool
    assert ident["known_sha256"] == sha(known_keys)
    groups, keys, counts = defaultdict(list), {}, {}
    key_ids = {}
    for r in rr:
        sid, key, n = r["scoring_structure_id"], r["canonical_mask"], int(r["n_Li"])
        assert len(key) == 48 and set(key) <= {"0", "1"} and key.count("0") == n
        assert n in [12, 24, 36]
        assert sid == f"mask48_{int(key, 2):012x}"
        assert keys.get(sid, key) == key and key_ids.get(key, sid) == sid
        keys[sid], counts[sid], key_ids[key] = key, n, sid
        groups[r["method"], str(r["seed"]), n].append(sid)
    novel_union = {s for s, k in keys.items() if k not in known}
    assert ident["attempts"] == len(rr) and ident["all_orbits"] == len(keys)
    assert ident["fresh_orbits"] == len(novel_union) == collected["fresh_unique_orbits"]
    references = {}
    for r in read_csv(thresholds):
        if r["source"] == "main_train":
            references[r["teacher"], int(r["n_Li"])] = {
                "q20": float(r["q20_eV_atom"]), "minimum": float(r["minimum_eV_atom"])}
    with np.load(training, allow_pickle=False) as z:
        train = z["split"].astype(str) == "train"
        n_li, train_e = z["n_Li"], z["energy_eV_atom"]
        assert np.isfinite(train_e).all()
        if "canonical_masks" in z:
            assert set(z["canonical_masks"].astype(str).tolist()) <= known, "Training keys absent from frozen history"
        round1_min = {}
        for n in [12, 24, 36]:
            selected = train & (n_li == n)
            assert selected.any()
            round1_min[n] = float(train_e[selected].min())
    result = {}
    for teacher, path in scores.items():
        assert collected["scores"][teacher]["sha256"] == sha(path)
        with np.load(path, allow_pickle=False) as z:
            ids = z["structure_ids"].astype(str)
            assert ids.ndim == 1 and len(set(ids)) == len(ids)
            energy, atoms = z["energy_eV"].astype(float), z["n_atoms"]
            provenance = z["provenance"].astype(str)
            assert energy.shape == atoms.shape == ids.shape and np.isfinite(energy).all()
            assert provenance.shape == ids.shape
            assert set(ids) == set(keys), "Quality summary expects exactly the frozen proposal union"
            lookup = {}
            for i, sid in enumerate(ids):
                assert int(atoms[i]) == 144 + counts[sid]
                if sid in novel_union:
                    assert provenance[i] == "fresh", (teacher, sid)
                lookup[sid] = float(energy[i] / atoms[i])
            result[teacher] = lookup

    rows, best = [], []
    method_ids = defaultdict(set)
    for teacher, energies in result.items():
        for (method, seed, n), proposed_ids in sorted(groups.items()):
            fresh = sorted({sid for sid in proposed_ids if sid in novel_union})
            method_ids[teacher, method].update(fresh)
            q, original_min = (references[teacher, n][k] for k in ["q20", "minimum"])
            emin = round1_min[n] if teacher == "uma" else None
            e = np.array([energies[sid] for sid in fresh])
            row = dict(pool=pool, teacher=teacher, method=method, seed=seed, n_Li=n,
                attempts=len(proposed_ids), distinct_proposals=len(set(proposed_ids)),
                fresh_distinct_orbits=len(fresh), fresh_low_orbits=int((e <= q).sum()),
                fresh_below_original_train_minimum=int((e < original_min).sum()),
                fresh_below_round1_UMA_train_minimum=int((e < emin).sum()) if emin is not None else "",
                original_q20_eV_atom=q, original_train_minimum_eV_atom=original_min,
                round1_UMA_train_minimum_eV_atom=emin if emin is not None else "",
                fresh_minimum_eV_atom=float(e.min()) if len(e) else "",
                fresh_p10_eV_atom=float(np.quantile(e, .1, method="linear")) if len(e) else "",
                fresh_median_eV_atom=float(np.median(e)) if len(e) else "",
                fresh_minimum_delta_q20_meV_atom=float(1000 * (e.min() - q)) if len(e) else "",
                fresh_p10_delta_q20_meV_atom=float(1000 * (np.quantile(e, .1, method="linear") - q)) if len(e) else "",
                fresh_median_delta_q20_meV_atom=float(1000 * (np.median(e) - q)) if len(e) else "",
                fresh_minimum_minus_original_train_min_meV_atom=float(1000 * (e.min() - original_min)) if len(e) else "",
                fresh_minimum_minus_round1_UMA_train_min_meV_atom=float(1000 * (e.min() - emin)) if len(e) and emin is not None else "")
            rows.append(row)
            if fresh:
                sid = min(fresh, key=lambda s: (energies[s], s))
                best.append(dict(pool=pool, teacher=teacher, method=method, seed=seed, n_Li=n,
                    scoring_structure_id=sid, canonical_mask=keys[sid], energy_eV_atom=energies[sid],
                    delta_q20_meV_atom=1000 * (energies[sid] - q),
                    minus_original_train_min_meV_atom=1000 * (energies[sid] - original_min),
                    minus_round1_UMA_train_min_meV_atom=1000 * (energies[sid] - emin) if emin is not None else "",
                    source_geometry="Fresh canonical pristine-derived deletion geometry; no relaxation"))
    write_csv(output / "fresh_group_quality.csv", rows)
    write_csv(output / "fresh_best_candidates.csv", best)
    delta_fields = ["fresh_minimum_delta_q20_meV_atom", "fresh_p10_delta_q20_meV_atom",
                    "fresh_median_delta_q20_meV_atom", "fresh_minimum_minus_original_train_min_meV_atom",
                    "fresh_minimum_minus_round1_UMA_train_min_meV_atom"]
    seeds = []
    for teacher, method, seed in sorted({(r["teacher"], r["method"], r["seed"]) for r in rows}):
        r = [v for v in rows if (v["teacher"], v["method"], v["seed"]) == (teacher, method, seed)]
        assert len({v["n_Li"] for v in r}) == len(r)
        row = dict(pool=pool, teacher=teacher, method=method, seed=seed, composition_groups=len(r),
            nonempty_composition_groups=sum(v["fresh_distinct_orbits"] > 0 for v in r),
            attempts=sum(v["attempts"] for v in r), fresh_distinct_orbits=sum(v["fresh_distinct_orbits"] for v in r),
            fresh_low_orbits=sum(v["fresh_low_orbits"] for v in r),
            fresh_below_original_train_minimum=sum(v["fresh_below_original_train_minimum"] for v in r),
            fresh_below_round1_UMA_train_minimum=sum(v["fresh_below_round1_UMA_train_minimum"] for v in r) if teacher == "uma" else "")
        row.update({"macro_" + k: mean_present(r, k) for k in delta_fields})
        seeds.append(row)
    write_csv(output / "fresh_seed_quality.csv", seeds)
    summaries = []
    for teacher, method in sorted(method_ids):
        ss = [r for r in seeds if r["teacher"] == teacher and r["method"] == method]
        ids = method_ids[teacher, method]
        row = dict(pool=pool, teacher=teacher, method=method, seed_count=len(ss),
            attempts=sum(r["attempts"] for r in ss),
            global_distinct_fresh_orbits=len(ids),
            global_distinct_fresh_low_orbits=sum(result[teacher][sid] <= references[teacher, counts[sid]]["q20"] for sid in ids),
            global_distinct_fresh_below_original_train_minimum=sum(result[teacher][sid] < references[teacher, counts[sid]]["minimum"] for sid in ids),
            global_distinct_fresh_below_round1_UMA_train_minimum=sum(result[teacher][sid] < round1_min[counts[sid]] for sid in ids) if teacher == "uma" else "",
            nonempty_composition_seed_groups=sum(r["nonempty_composition_groups"] for r in ss))
        for field in delta_fields:
            key = "macro_" + field
            values = [r[key] for r in ss if r[key] != ""]
            row["mean_seed_" + key] = float(np.mean(values)) if values else ""
            row["minimum_seed_" + key] = float(min(values)) if values else ""
            row["maximum_seed_" + key] = float(max(values)) if values else ""
        summaries.append(row)
    write_csv(output / "fresh_method_quality.csv", summaries)
    receipt = dict(status="PASS", pool=pool, attempts=len(rr), proposal_union_orbits=len(keys),
        fresh_union_orbits=len(novel_union), teachers=list(scores), group_rows=len(rows),
        best_rows=len(best), seed_rows=len(seeds), method_teacher_rows=len(summaries),
        original_training_references="minimum_eV_atom and q20_eV_atom from original main_train threshold rows",
        adaptive_training_reference="UMA training_round1.npz split=train only, separately within Li count",
        round1_UMA_train_minima_eV_atom=round1_min,
        quantile_rule="NumPy quantile(method=linear), each group's distinct fresh orbits only",
        aggregation="Compute per-composition energy DIFFERENCES, macro-average across compositions within a seed, then average seeds equally. Global counts deduplicate across seeds within each method.",
        no_final_tuning="This script reports frozen assessment only; it does not select or change a model.",
        sources={"proposals": sha(proposals), "known_keys": sha(known_keys), "round1_training": sha(training),
                 "thresholds": sha(thresholds), "identity_audit": sha(identity), "collection": sha(collection),
                 **{f"scores_{m}": sha(path) for m, path in scores.items()}},
        limitations=["MLIP energy quality, not DFT correctness or a global minimum guarantee.",
            "Extreme energies depend on candidate count and are paired with p10/median and count columns.",
            "Small negative reference gaps are not certified significant against FP32/repeat/model uncertainty.",
            "MACE has its own original training reference; the UMA adaptive minimum is never compared to MACE energies.",
            "Missing fresh candidates produce blank quality entries, not fabricated zero energy gaps.",
            "Seed macro ranges are descriptive and do not constitute confidence intervals."])
    (output / "quality_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", choices=["final"], default="final")
    ap.add_argument("--proposals", type=Path)
    ap.add_argument("--scores", nargs="+", help="teacher=path.npz; defaults to pool UMA/MACE score files")
    ap.add_argument("--known-keys", type=Path)
    ap.add_argument("--training", type=Path, default=HERE / "training_round1.npz")
    ap.add_argument("--identity-audit", type=Path)
    ap.add_argument("--collection", type=Path)
    ap.add_argument("--thresholds", type=Path, default=ROOT / "reports/generation_thresholds.csv")
    ap.add_argument("--output-dir", type=Path)
    a = ap.parse_args()
    pool = a.pool
    scores = dict(s.split("=", 1) for s in a.scores) if a.scores else {
        m: HERE / f"data/{pool}_scores_{m}.npz" for m in ["uma", "mace"]}
    analyze(a.proposals or HERE / f"data/{pool}_attempts.csv", scores,
            a.known_keys or HERE / f"known_keys_{'final' if pool == 'final' else 'round0'}.json",
            a.training, a.identity_audit or HERE / f"data/{pool}_identity_audit.json",
            a.collection or HERE / f"data/{pool}_collection.json",
            a.output_dir or HERE / f"reports/{pool}/quality", a.thresholds, pool)


if __name__ == "__main__":
    main()

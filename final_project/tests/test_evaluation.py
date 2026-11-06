import csv
import json
import numpy as np

from cathode_cvae.evaluation import evaluate


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_repeats_and_known_low_masks_consume_attempts(tmp_path):
    # Synthetic example: 4 attempts, 1 known low, 1 fresh low repeated, 1 high.
    keys = ["".join(str(int(i not in {(j + k) % 48 for k in range(12)})) for i in range(48)) for j in range(3)]
    prop = tmp_path / "proposals.csv"
    rows = [dict(method="test", seed=1, n_Li=12, scoring_structure_id=f"s{i}", canonical_mask=keys[i], attempt=a)
            for a, i in enumerate([0, 1, 1, 2])]
    write_csv(prop, rows)
    known = tmp_path / "known.json"
    known.write_text(json.dumps([keys[0]]))
    catalog = tmp_path / "catalog.csv"
    write_csv(catalog, [dict(parent="O3_N48_pilot", canonical_mask=keys[0])])
    thresholds = tmp_path / "q.csv"
    write_csv(thresholds, [dict(source="main_train", teacher="synthetic", n_Li=12, q20_eV_atom=0)])
    scores = tmp_path / "scores.npz"
    np.savez(scores, structure_ids=np.array(["s0", "s1", "s2"]),
             energy_eV=np.array([-156., -156., 156.]), n_atoms=np.full(3, 156))
    out = tmp_path / "result"
    evaluate(prop, {"synthetic": scores}, known, thresholds, catalog, out, "final")
    r = next(csv.DictReader((out / "metrics.csv").open()))
    assert int(r["attempts"]) == 4
    assert int(r["distinct_novel_low_orbits"]) == 1
    assert float(r["novel_low_yield_per_attempt"]) == 0.25
    assert float(r["duplicate_fraction"]) == 0.25

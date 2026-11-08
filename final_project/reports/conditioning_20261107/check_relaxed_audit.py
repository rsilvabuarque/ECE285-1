"""Independent stored-force and historical-summary arithmetic checks.

The only tolerance covers double-precision operation ordering in meV/atom
conversion. It is not an acceptance tolerance on physical model predictions.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from cathode_cvae.runtime import require_compute_allocation


require_compute_allocation()
out = Path("reports/conditioning_20261107/relaxed_target_audit")
z = np.load("artifacts/current/core/data/geometry_uma_fixed.npz")
force_max = np.linalg.norm(z["forces_parent_eV_A"], axis=2).max(axis=1)
assert np.array_equal(force_max, z["force_max_eV_A"])
endpoint = z["cohort"] == "fixed_endpoint"
assert (force_max[endpoint] < .05).all()
old = list(csv.DictReader(Path("artifacts/current/reports/deletion_vs_relaxed_ranking.csv").open()))
new = list(csv.DictReader((out / "composition_ranking.csv").open()))
fields = {"spearman": "spearman", "kendall": "kendall",
          "initial_low20_recall_of_final_low20": "deletion_low20_recall"}
exact = 0
roundtrip = []
for row in new:
    matches = [r for r in old if r["model"] == "uma" and r["cohort"] == "fixed" and r["n_Li"] == row["n_Li"]]
    assert len(matches) == 1
    r = matches[0]
    for key, value in fields.items():
        assert float(r[key]) == float(row[value])
        exact += 1
    before = float(r["median_relaxation_gain_meV_atom"])
    after = float(row["median_relaxation_gain_meV_atom"])
    # The old analysis divides by atom count then multiplies by1000; the new
    # one reverses these two operations. Eight float64 eps relative is a
    # conservative arithmetic roundtrip bound, with no energetic gate changed.
    assert abs(before-after) <= 8*np.finfo(np.float64).eps*max(1.,abs(before))
    roundtrip.append(abs(before-after))
failure = dict(status="FAILED", attempt="independent-check-v1",
    failure="Expected bitwise equality of all historical gain values",
    diagnosis="Energy gain converts the same differences with multiplication/division in opposite order; max7.11e-15meVatom difference. Rank fields and force arrays match exactly.",
    correction="Replace inappropriate bitwise-energy-arithmetic criterion with documented8float64epsrelative bound. Physical convergence criterion remainsstrict0.05eVA.")
(out / "independent_check_attempt1_failure.json").write_text(json.dumps(failure,indent=2)+"\n")
receipt = dict(status="PASS", force_norm_array_bitwise_equal=True,
    force_norm_maximum_absolute_roundtrip_error_eV_A=0.,
    endpoints_strictly_below_recorded_criterion=int((force_max[endpoint]<.05).sum()),
    old_fixed_UMA_rank_fields_exactly_reproduced=exact,
    energy_gain_fields_compared=len(roundtrip),
    maximum_energy_gain_roundtrip_difference_meV_atom=max(roundtrip),
    energy_arithmetic_relative_tolerance=8*np.finfo(np.float64).eps,
    tolerance_provenance="Onlyfloat64operation-order roundoff in1000*deltaE/N versus(deltaE/N)*1000; no physicalthreshold relaxed",
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope="Independent arithmetic from storedforcearrays and existingrankCSV; no new inference.")
(out / "independent_checks.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))

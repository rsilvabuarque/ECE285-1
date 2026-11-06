import csv
from cathode_cvae.geometry_evaluation import evaluate


def write(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def test_direct_force_success_is_separate_from_optimizer_steps(tmp_path):
    scores, relax = tmp_path / "scores.csv", tmp_path / "relax.csv"
    rows = []
    for method, energy, rms, maximum in [("deletion", -1000., 1., 2.), ("regressor", -1001.56, .5, .04), ("cvae_draw0", -1001.56, .5, .06)]:
        rows.append(dict(teacher="synthetic", source_mask_id="a", method=method, valid="True", n_Li=12,
            energy_eV=energy, energy_minus_deletion_meV_atom=1000*(energy+1000)/156,
            force_component_rms_eV_A=rms, force_max_eV_A=maximum, force_rms_over_deletion=rms))
    write(scores, rows)
    write(relax, [dict(source_mask_id="a", method=m, converged="True", steps=s, wall_seconds=1., endpoint_energy_difference_meV_atom=0.)
                  for m, s in [("deletion", 10), ("regressor", 5), ("cvae_draw0", 5)]])
    out = tmp_path / "output"
    evaluate(scores, relax, out)
    summary = {r["method"]: r for r in csv.DictReader((out / "direct_summary.csv").open())}
    assert float(summary["regressor"]["fraction_fmax_below_criterion"]) == 1
    assert float(summary["cvae_draw0"]["fraction_fmax_below_criterion"]) == 0
    assert float(summary["cvae_draw0"]["median_force_rms_ratio"]) == .5
    paired = {r["method"]: r for r in csv.DictReader((out / "paired_relaxation_summary.csv").open())}
    assert float(paired["cvae_draw0"]["median_step_difference_vs_deletion"]) == -5

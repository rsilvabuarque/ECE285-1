"""Read-only independent check of policy selection against saved raw diagnostics."""
import csv
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


core = ROOT / "runs/conditioning_20261107/evaluation/policy"
selection_path = ROOT / "artifacts/conditioning_20261107/training_policy_selection.json"
rule_path = ROOT / "artifacts/conditioning_20261107/selection_rule.json"
selection = json.loads(selection_path.read_text())
assert digest(core/"summary.csv") == selection["source_summary_sha256"]
assert digest(rule_path) == selection["decision_rule_sha256"]
scopes = read(HERE/"distinct_metric_scopes.csv")
checks = []
for source, scope, field in [("summary.csv", "balanced_low_middle_high", "mean_novel_inbin_yield"),
                             ("fixed_q20_low_request.csv", "requested_low_only_original_q20", "mean_novel_fixed_q20_yield")]:
    for row in read(core/source):
        values = [float(r["macro_useful_yield"]) for r in scopes if r["method"] == row["method"] and r["scope"] == scope]
        assert len(values) == 3
        independent = statistics.mean(values)
        difference = abs(independent-float(row[field]))
        assert difference < 1e-14
        checks.append(dict(method=row["method"], scope=scope, independently_reduced=independent,
                           archived_core_value=float(row[field]), absolute_difference=difference))
candidates = [r for r in checks if r["scope"] == "balanced_low_middle_high" and r["method"] not in {"uniform", "mutation"}]
winner = max(candidates, key=lambda r:r["independently_reduced"])["method"]
assert winner == selection["method"]
method = selection["method"]
response = read(HERE/"condition_summary.csv")
latent = read(HERE/"latent_summary.csv")
posterior = read(HERE/"posterior_prior_summary.csv")
selected = read(HERE/"selected_latent_diagnostics.csv")
components = read(HERE/"latent_components.csv")
per_composition = []
for n in [12, 24, 36]:
    for comparison in ["low_to_middle", "low_to_high"]:
        rows = [r for r in response if r["method"] == method and int(r["n_Li"]) == n and r["comparison"] == comparison]
        assert len(rows) == 3
        per_composition.append(dict(n_Li=n, comparison=comparison,
            mean_delta_meV_atom=1000*statistics.mean(float(r["mean_delta_E_eV_atom"]) for r in rows),
            mean_positive_pair_fraction=statistics.mean(float(r["fraction_positive_delta"]) for r in rows),
            mean_raw_Hamming=statistics.mean(float(r["mean_raw_Hamming"]) for r in rows)))
activity = []
for row in selected:
    if row["fit"].startswith("balance_1_d8_") and float(row["selection_beta"]) == .1:
        for scope in ["pooled", "12", "24", "36"]:
            rr = [r for r in components if r["fit"] == row["fit"] and float(r["selection_beta"]) == .1 and r["scope"] == scope]
            assert len(rr) == 8
            activity.append(dict(fit=row["fit"], scope=scope,
                total_KL_nats=sum(float(r["KL_nats"]) for r in rr),
                active_by_cutoff={str(t):sum(float(r["posterior_mean_variance"]) > t for r in rr) for t in [.001,.01,.1]}))
ll = [r for r in latent if r["method"] == method]
pp = [r for r in posterior if r["method"] == method]
posterior_pairs = [r for r in read(HERE/"posterior_prior_pairs.csv") if r["method"] == method]
result = dict(status="PASS", selected_method=method, policy_checks=checks,
    selection_independently_matches_core=True, protected_test_used_for_selection=False,
    verification_tolerance=dict(quantity="absolute difference between deterministic probability reductions",
                                value=1e-14, scope="Numerical serialization/reduction check, not a scientific effect threshold"),
    selected_condition_response_by_composition=per_composition,
    selected_latent_response=dict(mean_raw_Hamming=statistics.mean(float(r["mean_raw_Hamming"]) for r in ll),
                                 mean_absolute_energy_shift_meV_atom=1000*statistics.mean(float(r["mean_absolute_delta_E_eV_atom"]) for r in ll)),
    selected_posterior_vs_prior=dict(posterior_target_Hamming=statistics.mean(float(r["posterior_mean_target_Hamming"]) for r in pp),
        prior_target_Hamming=statistics.mean(float(r["prior_mean_target_Hamming"]) for r in pp),
        posterior_target_energy_MAE_meV_atom=1000*statistics.mean(abs(float(r["posterior_minus_target_eV_atom"])) for r in posterior_pairs),
        prior_target_energy_MAE_meV_atom=1000*statistics.mean(abs(float(r["prior_minus_target_eV_atom"])) for r in posterior_pairs)),
    selected_latent_activity=activity,
    interpretation="Weak condition-directed energy shifts coexist with substantial latent sensitivity and better posterior reconstruction; these diagnostics do not establish complete posterior collapse or physical accuracy.",
    source_hashes={str(p.relative_to(ROOT)):digest(p) for p in [selection_path,rule_path,core/"summary.csv",core/"fixed_q20_low_request.csv",HERE/"distinct_metric_scopes.csv",HERE/"condition_summary.csv",HERE/"latent_summary.csv",HERE/"posterior_prior_summary.csv",HERE/"selected_latent_diagnostics.csv",HERE/"latent_components.csv"]},
    audit_source_sha256=digest(__file__))
(HERE/"selection_audit.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))

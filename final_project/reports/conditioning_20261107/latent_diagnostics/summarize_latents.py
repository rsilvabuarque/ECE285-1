"""Audit latent-size selection and summarize measured development diagnostics."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CORE = ROOT/"runs/conditioning_20261107/evaluation/latent"


def read(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


scopes = read(HERE/"distinct_metric_scopes.csv")
checks = []
for source, scope, field in [("summary.csv", "balanced_low_middle_high", "mean_novel_inbin_yield"),
                             ("fixed_q20_low_request.csv", "requested_low_only_original_q20", "mean_novel_fixed_q20_yield")]:
    for row in read(CORE/source):
        values = [float(r["macro_useful_yield"]) for r in scopes if r["method"] == row["method"] and r["scope"] == scope]
        assert len(values) == 3
        value = statistics.mean(values)
        assert abs(value-float(row[field])) < 1e-14
        checks.append(dict(method=row["method"],scope=scope,value=value,archived=float(row[field])))
selected = [r for r in read(HERE/"selected_latent_diagnostics.csv") if float(r["selection_beta"]) == .1]
assert len(selected) == 15
fit_rows = []
for r in selected:
    fit_rows.append(dict(method=r["methods"], fit_seed=r["fit_seed"], latent_dim=int(r["latent_dim"]),
        model_kind=r["model_kind"], latent_role=r["latent_role"], parameter_count=int(r["parameter_count"]),
        reconstruction_nats=float(r["val_reconstruction"]), total_KL_nats=float(r["total_KL_nats"]),
        mean_KL_nats_per_dimension=float(r["mean_KL_nats_per_dimension"]),
        active_at_0p001=int(r["active_at_0.001"]), active_at_0p01=int(r["active_at_0.01"]),
        active_at_0p1=int(r["active_at_0.1"]), actual_updates=int(r["actual_optimizer_updates"]),
        train_seconds=float(r["train_seconds"])))
assert all(r["actual_updates"] == 4000 for r in fit_rows)
latent = read(HERE/"latent_summary.csv")
posterior = read(HERE/"posterior_prior_pairs.csv")
null = [r for r in latent if r["method"] == "ar_baseline"]
null_posterior = [r for r in posterior if r["method"] == "ar_baseline"]
assert len(null) == 9 and len(null_posterior) == 144
assert all(float(r["mean_raw_Hamming"]) == 0 and float(r["mean_absolute_delta_E_eV_atom"]) == 0 for r in null)
assert all(int(r["raw_Hamming"]) == 0 and float(r["delta_E_eV_atom"]) == 0 for r in null_posterior)
summary = []
for dimension in [0,4,8,16,32]:
    rr = [r for r in fit_rows if r["latent_dim"] == dimension]
    assert len(rr) == 3
    method = rr[0]["method"]
    def metric(scope):return next(r["value"] for r in checks if r["method"] == method and r["scope"] == scope)
    summary.append(dict(latent_dim=dimension, method=method,
        model_role="Autoregressive baseline, no latent variables" if dimension == 0 else "CVAE",
        balanced_useful_yield=metric("balanced_low_middle_high"), low_original_q20_yield=metric("requested_low_only_original_q20"),
        parameter_count=rr[0]["parameter_count"],
        total_KL_nats_by_seed=[r["total_KL_nats"] for r in rr],
        mean_KL_per_dimension_by_seed=[r["mean_KL_nats_per_dimension"] for r in rr],
        pooled_active_at_0p01_by_seed=[r["active_at_0p01"] for r in rr],
        validation_reconstruction_by_seed=[r["reconstruction_nats"] for r in rr]))
selection_path = ROOT/"artifacts/conditioning_20261107/selected_policy.json"
rule_path = ROOT/"artifacts/conditioning_20261107/selection_rule.json"
selection = json.loads(selection_path.read_text()); rule = json.loads(rule_path.read_text())
assert selection["latent_summary_sha256"] == digest(CORE/"summary.csv")
assert selection["latent_rule_sha256"] == digest(rule_path)
candidate = [r for r in summary if r["latent_dim"] > 0]
best_balanced = max(r["balanced_useful_yield"] for r in candidate)
best_low = max(r["low_original_q20_yield"] for r in candidate)
margins = rule["competitive_margins"]
eligible = [r for r in candidate if r["balanced_useful_yield"] >= best_balanced-margins["balanced"]
            and r["low_original_q20_yield"] >= best_low-margins["low"]]
winner = min(eligible,key=lambda r:r["latent_dim"])
assert winner["latent_dim"] == selection["latent_dim"] == 4
fig, axes = plt.subplots(2,2,figsize=(10,7))
dimensions=[0,4,8,16,32]
for j, dimension in enumerate(dimensions):
    rr = [r for r in fit_rows if r["latent_dim"] == dimension]
    method = rr[0]["method"]
    for ax, scope in [(axes[0,0],"balanced_low_middle_high"),(axes[0,1],"requested_low_only_original_q20")]:
        values=[float(r["macro_useful_yield"])*100 for r in scopes if r["method"] == method and r["scope"] == scope]
        ax.scatter(j+np.linspace(-.09,.09,3),values,color="0.35" if dimension == 0 else "tab:blue",s=28)
        ax.plot([j-.16,j+.16],[np.mean(values)]*2,color="black",lw=1)
    axes[1,0].scatter(j+np.linspace(-.09,.09,3),[r["total_KL_nats"] for r in rr],s=28)
    axes[1,1].scatter(j+np.linspace(-.09,.09,3),[r["active_at_0p01"] for r in rr],s=28)
for ax, scope in [(axes[0,0],"balanced_low_middle_high"),(axes[0,1],"requested_low_only_original_q20")]:
    for method, color in [("uniform","tab:orange"),("mutation","tab:green")]:
        value=next(r["value"] for r in checks if r["method"] == method and r["scope"] == scope)
        ax.axhline(100*value,color=color,ls="--",lw=1,label=method+" mean")
    ax.legend(fontsize=8)
axes[0,0].set_ylabel("Novel requested-bin yield (%)")
axes[0,1].set_ylabel("Requested-low/original-q20 yield (%)")
axes[1,0].set_ylabel("Total validation KL (nats)")
axes[1,1].set_ylabel("Active components at Var(μ)>0.01")
for ax in axes.flat:
    ax.set_xticks(range(5),["AR\nno latent","4","8","16","32"])
    ax.set_xlabel("Model / CVAE latent dimension")
    ax.tick_params(labelsize=9)
fig.suptitle("Matched policy and 4,000 updates; each point is one fit seed",fontsize=11)
fig.tight_layout()
for ext in ["pdf","png"]:fig.savefig(HERE/("latent_dimension_summary."+ext),dpi=170,bbox_inches="tight")
plt.close(fig)
with (HERE/"latent_fit_diagnostic_summary.csv").open("w",newline="") as stream:
    writer=csv.DictWriter(stream,fieldnames=list(fit_rows[0]));writer.writeheader();writer.writerows(fit_rows)
result=dict(status="PASS", method_summaries=summary, core_metric_checks=checks,
    independent_selected_dimension=winner["latent_dim"], eligible_dimensions=[r["latent_dim"] for r in eligible],
    no_latent_control=dict(model_role="AR baseline, not a CVAE",latent_resampling_zero_in_all_9_groups=True,
                           posterior_prior_identical_in_all_144_pairs=True),
    activity_scope="Pooled validation Var(mu)>.01 counts; per-composition and .001/.1 sensitivity remain in raw component tables. No physical-convergence interpretation.",
    interpretation="D4 is the retained CVAE under the frozen rule. The no-latent AR control has higher observed balanced yield; mutation remains substantially stronger. Three-seed variation is descriptive, not a confidence interval.",
    source_hashes={str(p.relative_to(ROOT)):digest(p) for p in [selection_path,rule_path,CORE/"summary.csv",CORE/"fixed_q20_low_request.csv",HERE/"distinct_metric_scopes.csv",HERE/"selected_latent_diagnostics.csv",HERE/"latent_summary.csv",HERE/"posterior_prior_pairs.csv"]},
    audit_source_sha256=digest(__file__))
(HERE/"latent_selection_audit.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))

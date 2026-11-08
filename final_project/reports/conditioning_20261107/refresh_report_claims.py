"""Refresh the required report's source hashes without declaring the root gate.

Final-study additions have an explicit reviewed claim list below; the integrating
owner still records the required gate separately after compilation and audits.
"""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def source(path, details):
    p = ROOT / path
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),details=details)


def main():
    p = ROOT / "docs/midterm1/claim_sources.json"
    record = json.loads(p.read_text())
    keep = {"parent48", "half_orbit_capacity", "training_round0", "historical_yields",
            "ranking_context", "hu_five_counterexample", "geometry_secondary", "new_multibin_benchmark"}
    claims = [c for c in record["claims"] if c["id"] in keep]
    for c in claims:
        c["scope"] = "Retained historical/background evidence; not a result of new controlled models"
        for s in c["sources"]:
            s["sha256"] = hashlib.sha256((ROOT/s["path"]).read_bytes()).hexdigest()
    additions = [
        ("policy_development", "Six4000-updateD8fits; balancing/selection fourrows; selected32.12%balancedvs33.62%uniform58.97%mutation, not final", [
            ("runs/conditioning_20261107/evaluation/policy/summary.csv", "Mean/designseedSD balanced useful yield"),
            ("runs/conditioning_20261107/evaluation/policy/fixed_q20_low_request.csv", "Separate requestedlow originalq20 yield"),
            ("artifacts/conditioning_20261107/training_policy_selection.json", "Policy chosen on development only"),
            ("artifacts/conditioning_20261107/selection_rule.json", "Decision rule fixed before energy summaries")]),
        ("paired_condition_latent", "Sharedlatent/uniform diagnostics separate condition effect from nonzero latent use; no completecollapse inference", [
            ("reports/conditioning_20261107/policy_diagnostics_v2/condition_summary.csv", "Perseed/composition conditional energy shifts"),
            ("reports/conditioning_20261107/policy_diagnostics_v2/latent_summary.csv", "Sameuniform latent resampling changes output"),
            ("reports/conditioning_20261107/policy_diagnostics_v2/selected_latent_diagnostics.csv", "Policy-manifest-scoped KL and component diagnostics"),
            ("reports/conditioning_20261107/policy_diagnostics_v2/receipt.json", "Complete oracle coverage and target raw Hamming checks")]),
        ("physical_parent_graph", "Fullhost/mobile periodic graph with5Å provisionalradius and interlayeredges; descriptor invariant under192operations", [
            ("reports/conditioning_20261107/parent_graph/receipt.json", "Actual parent/symmetry/graph/descriptor audit"),
            ("reports/conditioning_20261107/parent_graph/descriptor_energy.csv", "Train/val/test descriptive association; use onlyTRAIN for selection"),
            ("reports/conditioning_20261107/parent_graph/descriptor_schema.json", "Train-only joint energy/descriptor support")]),
        ("paired_energy_target", "All1000matched fixedcellUMA lineages; ranks change without occupationorbit merges; sparseprimaryroles deferlabeltargetfit", [
            ("reports/conditioning_20261107/relaxed_target_audit/receipt.json", "All attempts, mapping/convergence and scope"),
            ("reports/conditioning_20261107/relaxed_target_audit/composition_ranking.csv", "Percomposition ranks/lowtailoverlap/gains"),
            ("reports/conditioning_20261107/relaxed_target_audit/label_target_support.csv", "Protected independent class support"),
            ("reports/conditioning_20261107/relaxed_target_audit/independent_checks.json", "Force array arithmetic and historical ranking checks")]),
        ("latent_development", "D4/8/16/32 plus no-latentAR; D4chosen by frozen practicalmean margins; individualseeds andparameters retained, not physicalconvergence", [
            ("runs/conditioning_20261107/evaluation/latent/summary.csv", "Balanced development yields and descriptive seedvariation"),
            ("runs/conditioning_20261107/evaluation/latent/fixed_q20_low_request.csv", "Separate requested-low originalq20 outcome"),
            ("reports/conditioning_20261107/latent_development/seed_values.csv", "All21 individual model/sampler replicate values and fullparameter counts"),
            ("reports/conditioning_20261107/latent_diagnostics/latent_fit_diagnostic_summary.csv", "Perfit reconstruction, summed/perdimension KL and activity-cutoff sensitivity"),
            ("reports/conditioning_20261107/latent_diagnostics/latent_selection_audit.json", "Independent latent-rule and scalar-metric reproduction"),
            ("artifacts/conditioning_20261107/selected_policy.json", "Development-only trainingpolicy andlatent choice"),
            ("artifacts/conditioning_20261107/selected_models.json", "Actualportable retainedcheckpoint identities")]),
        ("spatial_development", "Three-fit graphcontext DEV60.88%balanced/61.11%originalq20low vsmutation58.80/50.35; not final; encoderonly34.49/24.31", [
            ("runs/conditioning_20261107/evaluation/spatial_confirmed/summary.csv", "Balanced development yield and descriptive seed variation"),
            ("runs/conditioning_20261107/evaluation/spatial_confirmed/fixed_q20_low_request.csv", "Separate requested-low fixed originalq20 metric"),
            ("runs/conditioning_20261107/evaluation/spatial_confirmed/seed_metrics.csv", "All individual fit/sampler replicate values"),
            ("artifacts/conditioning_20261107/architecture_selection.json", "Development recommendation, not final result"),
            ("artifacts/conditioning_20261107/protocol_amendment_graph_context_replication.json", "Pre-final two-fit replication and28of30 corefit cap")]),
        ("spatial_condition_latent", "Graphcontext strongpairedconditionresponse butKL.007-.011nats/weakzresponse; supportscausalspatialAR not demonstratedlatentbenefit", [
            ("reports/conditioning_20261107/spatial_confirmed_diagnostics/family_diagnostic_summary.json", "Shared-latent/uniform condition shifts and separate latent/posterior diagnostics"),
            ("reports/conditioning_20261107/spatial_confirmed_diagnostics/paired_diagnostics_by_family_seed.csv", "All seed-level diagnostics"),
            ("reports/conditioning_20261107/spatial_confirmed_diagnostics/selected_latent_diagnostics.csv", "Actual selected-checkpoint KL/activity, parameters and reconstruction"),
            ("reports/conditioning_20261107/spatial_confirmed_diagnostics/receipt.json", "Raw proposal/score coverage and input hashes")]),
        ("trained_decoder_causality", "Three actualgraphcontextcheckpoint futuregradient/permutation testsPASS withfullencoderforbidden; notphysics/equivariancevalidation", [
            ("reports/conditioning_20261107/trained_causality_audit.json", "Actualtrainedweights, exactlyzero future gradients, unchanged future-permutation logits"),
            ("reports/conditioning_20261107/trained_causality_audit.py", "Executable independent audit and numerical convention")]),
        ("descriptor_negative_pilot", "Onefit middleenergy jointdescriptor yield28.13%vsplain28.52/mutation44.79; all2304attempts; no improvementclaim", [
            ("runs/conditioning_20261107/evaluation/descriptor/summary.csv", "Equal six-condition-cell joint useful yields"),
            ("runs/conditioning_20261107/evaluation/descriptor/receipt.json", "Attempt completeness, descriptor/energy schema and bounded distance-reference scope")]),
        ("frozen_final_assessment", "All34560UMAattempts; graphcontext62.04%balanced63.54%low vsmutation58.62/52.52; matched20736MACEattempts63.45/67.62vsmutation57.29/52.95", [
            ("runs/conditioning_20261107/evaluation/final/summary.csv", "Allten UMA families, seed SD, no post-final model selection"),
            ("runs/conditioning_20261107/evaluation/final/fixed_q20_low_request.csv", "Separate UMA original-q20 low-request outcome"),
            ("runs/conditioning_20261107/evaluation/final_matched/summary.csv", "Prespecified exactsixfamily paired UMA/MACE subset"),
            ("runs/conditioning_20261107/evaluation/final_matched/fixed_q20_low_request.csv", "Teacher-specific original-q20 low outcome"),
            ("runs/conditioning_20261107/evaluation/final/receipt.json", "Full attempt completeness and frozen-history novelty boundary"),
            ("runs/conditioning_20261107/evaluation/final_matched/receipt.json", "Matchedteacher coverage; not independent DFT validation"),
            ("runs/conditioning_20261107/final_protocol.json", "Methods/checkpoints/history278382 frozen before generation")]),
        ("independent_final_replay", "Raw192operation mask/bin/history recomputation matches16tables with2515436fieldchecks zero mismatches; not physicalvalidation", [
            ("reports/conditioning_20261107/independent_final_metrics/receipt.json", "Independent production-helper-free numerical check")]),
        ("final_structural_distance", "Exact192symmetry minimum tofixed1024TRAINorbits/count; upperboundfullTRAINdistance; context6.858uniform6.891mutation6.498sites", [
            ("reports/conditioning_20261107/final_structural_distances/receipt.json", "All34560attempts, all3456actualmutation2k edits; no labels read or selection change"),
            ("reports/conditioning_20261107/final_structural_distances/family_summary.csv", "Equalcount/bin/seed mean siteHamming distances"),
            ("reports/conditioning_20261107/final_structural_distances/METHOD.md", "Metric definition, fixedreference and upperbound proof")]),
        ("current_scope", "Requiredscientificevidencecomplete withrootreportgatepending; nonewDFT/MLIPfinetuning; no entropy/generalizationaccuracy implied", [
            ("artifacts/conditioning_20261107/protocol.json", "Frozen scientific and revised user resource contract"),
            ("reports/conditioning_20261107/SOURCE_REVIEW.md", "Actual baseline versus conversation suggestions")]),
    ]
    for identity, claim, sources in additions:
        claims.append(dict(id=identity,claim=claim,status="VERIFIED_WITH_STATED_SCOPE",
                           sources=[source(*s) for s in sources]))
    record["claims"] = claims
    record["status"] = "REQUIRED_FINAL_EVIDENCE_READY_REPORT_GATE_PENDING"
    record["assignment"]["discussion_pdf_sha256"] = json.loads((ROOT/"artifacts/conditioning_20261107/protocol.json").read_text())["sources"]["Evaluate Generative Model Project.pdf"]
    record["assignment"]["source"] = "User supplied course screenshots and current next-goal section8; updated89page discussionPDF is context"
    record["report_outputs"] = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT/"docs/midterm1"/name for name in ["main.tex","main.pdf","generated_conditioning_results.tex","generated_policy_table.tex","generated_latent_table.tex","generated_spatial_table.tex","generated_final_table.tex","generated_energy_audit.tex","references.bib"]]}
    record["report_verification"] = dict(body_pages=5,references_pages=1,
        scope="Required scientific evidence inserted; root-owned gate and later publication/optional study status remain separate",
        independent_teammate_checks="Remain assigned")
    record["primary_references"] = [r for r in record["primary_references"] if r["key"] != "omat2026"] + [
        dict(key="gilmer2017",url="https://proceedings.mlr.press/v70/gilmer17a.html",checked="Primary PMLR abstract/metadata",supports="Messagepassing framework; not evidence current graph wins"),
        dict(key="dieng2019",url="https://proceedings.mlr.press/v89/dieng19a.html",checked="Primary PMLR abstract/metadata",supports="Latentcollapse possible with expressive likelihood; not diagnosis of this network"),
        dict(key="ichikawa2024",url="https://proceedings.mlr.press/v238/ichikawa24a.html",checked="Primary PMLR abstract/metadata",supports="SimplifiedlinearVAE sensitivity only; no LCO convergence theorem"),
        dict(key="wakamiya2026",url="https://doi.org/10.1038/s41524-026-02075-0",checked="Primaryindexedmethods and archived accompanyingcode inpriorreview; SI numericalvalidation not independently retrieved",supports="Relatedfixedoccupation propertyconditionedCVAE+UMA loop; not noveltyclaim")]
    # Repeated interim regeneration should not duplicate bibliography records.
    record["primary_references"] = list({r["key"]:r for r in record["primary_references"]}.values())
    p.write_text(json.dumps(record,indent=2)+"\n")
    lines = ["# Report claim and source map", "", "Status: **required final evidence ready; integrating report gate pending**.",
             "This map does not certify the required delivery gate or teammates' independent contributions.", "",
             "| Claim family | Verified scope | Main source |", "| --- | --- | --- |"]
    for c in claims:
        s = c["sources"][0]
        lines.append(f"| {c['id']} | {c['claim']} | [{Path(s['path']).name}](../../{s['path']}) |")
    lines += ["", "The machine-readable map binds exact input hashes. Optional studies need explicit reviewed additions and retain a separate gate/status. Current final, development, historical and updated discussion evidence remain distinct."]
    (ROOT/"docs/midterm1/CLAIM_SOURCE_MAP.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(status=record["status"],claim_families=len(claims)),indent=2))


if __name__ == "__main__":
    main()

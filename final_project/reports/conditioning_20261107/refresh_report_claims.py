"""Refresh the live final report map, preserving the separate required checkpoint.

Final-study additions have an explicit reviewed claim list below; the integrating
owner records publication, allocation release and final cost separately. The
required snapshot, gate and original public_source_map are never edited here.
"""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def source(path, details):
    p = ROOT / path
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),details=details)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification",type=Path,help="Current report mechanical/render receipt")
    parser.add_argument("--visual-review",type=Path,help="Current all-page visual-review receipt")
    parser.add_argument("--neb-receipt",type=Path,action="append",default=[])
    parser.add_argument("--cost-receipt",type=Path)
    parser.add_argument("--final",action="store_true")
    args=parser.parse_args()
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
        ("optional_editor", "Same-seed editor balanced yield59.375% vs57.234%; low-q20 yield50.347/50.521%; all6912attempts and raw pair checks retained", [
            ("runs/conditioning_20261107/evaluation/editor/summary.csv", "Three learned fits and paired mutation streams; balanced outcome and separate low metric"),
            ("reports/conditioning_20261107/independent_editor_metrics/receipt.json", "Independent raw pairing,192operation mapping, bins, novelty and seed-label parity"),
            ("runs/conditioning_20261107/editor/assessment_v1/protocol.json", "History/seed/budget/uniforms frozen before scoring"),
            ("runs/conditioning_20261107/editor/data_v2/preparation_receipt.json", "All source outcomes and connected protected seed/child roles")]),
        ("editor_low_request_metrics", "LOW paired mean favors mutation by0.07656meV/atom; learned lower in548of1152pairs; distinct-quintile yield tied", [
            ("reports/conditioning_20261107/editor_low_request_v2/request_energy_summary.csv", "Per-request, per-fit and per-composition paired energies; avoid interpreting HIGH as minimization"),
            ("reports/conditioning_20261107/independent_editor_metrics/requested_bin_energy_posthoc.json", "Independent request-scoped reduction")]),
        ("editor_secondary_minima", "All9LOWfit/count minima at128attempts: learned lower novel child in7cases; mean advantages-.572/-.229/-.428meVatom; supplied-seed minimum is not globalhistory minimum", [
            ("reports/conditioning_20261107/editor_low_request_v2/terminal_best_novel_low_children.csv", "All9 exact terminal energies/IDs plus best identical supplied current-SCF seed"),
            ("reports/conditioning_20261107/editor_low_request_v2/terminal_minima_composition_means.csv", "Three composition means;7of9 learned and5of9mutation beat their supplied-seed minimum"),
            ("reports/conditioning_20261107/editor_low_request_v2/receipt.json", "Secondary posthoc scope and budget definitions"),
            ("reports/conditioning_20261107/editor_low_request_v2/visual_review.json", "All-seed running-best and paired-low figure review")]),
        ("editor_stage_cost", "7330new logicalUMA labels=6845child-only+485sharedseed-only;15alreadyavailable; counts verified against PRE-editor registry/cache and completedtask receipts", [
            ("reports/conditioning_20261107/editor_low_request_v2/cost_audit.json", "Actual stage-specific fresh labels; network forward calls not instrumented"),
            ("reports/conditioning_20261107/editor_low_request_v2/query_accounting.csv", "Per-method standalone queries vs deduplicated union; shared seed verification separate")]),
        ("editor_checkpoint_alias", "Different ZIP-root file hashes for best.pt and named selected checkpoint; normalized members and full loaded payload identical; assessment already used named hashes", [
            ("reports/conditioning_20261107/editor_checkpoint_alias_audit/receipt.json", "All3fits:20tensors/97000elements, fullconfiguration/epoch/update/latentmetadata match; no model or label edits"),
            ("reports/conditioning_20261107/execution_preclosure.json", "Preserved earlier cost-audit findings from an overly broad byte-hash alias convention; not evidence of changed assessment weights")]),
        ("posthoc_gaussian_prior", "TRAIN-only posterior-moment Gaussian, noNNupdates: balanced UMA34.259vs31.047/MACE34.028vs30.787; LOWyield regresses; graphcontext retained", [
            ("reports/conditioning_20261107/independent_aggregate_prior/receipt.json", "Independent tables/actual transformed-z checks; per-row encoder outputs unavailable for independent moment recomputation"),
            ("reports/conditioning_20261107/independent_aggregate_prior/summary.csv", "Separate UMA/MACE balanced yields in new assessment namespace"),
            ("reports/conditioning_20261107/independent_aggregate_prior/original_q20_LOW_request.csv", "LOW UMA23.958vs30.122/MACE26.042vs32.552%"),
            ("runs/conditioning_20261107/aggregate_prior/preparation/receipt.json", "Unchanged networks,27TRAIN-only groups, no gradients"),
            ("runs/conditioning_20261107/aggregate_prior/assessment/protocol.json", "New312715-orbit history; sameepsilon/U but condition-specific actualz")]),
        ("temperature_followup", "24schedules/768attempts/102geometries; identical integerallocations and actualpairedproposals across4T; lowreferenceESS remains", [
            ("reports/conditioning_20261107/temperature_followup/analysis/receipt.json", "All attempts and exact cross-temperature mask identity checks"),
            ("reports/conditioning_20261107/temperature_followup/analysis/schedule_summary.csv", "Measured distributions; pure-low fractions65.625/65.625/53.125%"),
            ("reports/conditioning_20261107/temperature_followup/reference_diagnostics.csv", "Static cell-energy weights/raw frequencies; ESS1.850-6.544"),
            ("reports/conditioning_20261107/temperature_followup/finite12_support.json", "Existing87orbit mathematical accounting only"),
            ("reports/conditioning_20261107/temperature_followup/protocol.json", "Required gate precedes study; registered editor geometries known before generation")]),
        ("current_scope", "Required gate complete and preserved; later extensions separate; no newDFT/MLIPfinetuning or entropy/generalization claim", [
            ("artifacts/conditioning_20261107/protocol.json", "Frozen scientific and revised user resource contract"),
            ("artifacts/conditioning_20261107/required_gate.json", "Immutable completed required checkpoint before extensions"),
            ("reports/conditioning_20261107/SOURCE_REVIEW.md", "Actual baseline versus conversation suggestions")]),
    ]
    if args.neb_receipt:
        if len(args.neb_receipt)!=2:raise ValueError("Record both terminal event receipts together")
        for path in args.neb_receipt:
            if not json.loads(path.read_text()).get("completed_UTC"):raise ValueError("NEB event is not terminal")
        neb_sources=[(str(path.relative_to(ROOT)) if path.is_absolute() else str(path),"Terminal event receipt; not a free-energy barrier or calibrated diffusivity") for path in args.neb_receipt]
        neb_sources += [("runs/conditioning_20261107/neb/mace_profile/receipt.json", "MACE single points on the FIVE-interior UMA band, not refined7 or optimizedMACEbarrier"),
            ("reports/conditioning_20261107/neb_pilot/prepared_v3/protocol.json", "ExplicitASE3.26 method, fixedcell/atomidentity/PBC, provisionalforcecriteria and boundedresources"),
            ("reports/conditioning_20261107/neb_pilot/results/receipt.json", "Profiles retain coarse and refined event1; no newmodelcalls in summary"),
            ("reports/conditioning_20261107/neb_pilot/independent/receipt.json", "Independent saved-band endpoint/barrier/force/MACE coordinate/call-count checks; no physical validation")]
        additions.append(("defined_hop_NEB", "Two numerically converged UMA event paths; event1 forward0.638→0.189eV with coupled7image/tightertarget change; event2 coarse0.192eV; MACEcoarseSP is not transport validation",neb_sources))
    if args.cost_receipt:
        path=args.cost_receipt
        additions.append(("resource_cost_receipt", "Resource and logical-label accounting use the supplied receipt's own as-of/final scope; not inferred from attempts",
            [(str(path.relative_to(ROOT)) if path.is_absolute() else str(path),"Integrating cost audit; read terminal/as-of state before treating as complete allocation usage")]))
    for identity, claim, sources in additions:
        claims.append(dict(id=identity,claim=claim,status="VERIFIED_WITH_STATED_SCOPE",
                           sources=[source(*s) for s in sources]))
    record["claims"] = claims
    record["status"] = "FINAL_REPORT_REVIEWED_WITH_SEPARATE_IMMUTABLE_REQUIRED_GATE" if args.final else "LIVE_REPORT_SOURCES_UPDATED_FINAL_REVIEW_PENDING"
    record["assignment"]["discussion_pdf_sha256"] = json.loads((ROOT/"artifacts/conditioning_20261107/protocol.json").read_text())["sources"]["Evaluate Generative Model Project.pdf"]
    record["assignment"]["source"] = "User supplied course screenshots and current next-goal section8; updated89page discussionPDF is context"
    record["report_outputs"] = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT/"docs/midterm1"/name for name in ["main.tex","main.pdf","generated_conditioning_results.tex","generated_policy_table.tex","generated_latent_table.tex","generated_spatial_table.tex","generated_final_table.tex","generated_energy_audit.tex","references.bib"]]}
    record["report_outputs"]["docs/OPTIONAL_EXTENSIONS_20261107.md"]=hashlib.sha256((ROOT/"docs/OPTIONAL_EXTENSIONS_20261107.md").read_bytes()).hexdigest()
    verification=None;visual=None
    if args.verification:
        verification=json.loads(args.verification.read_text())
        if (verification["status"]!="PASS" or verification["pdf_sha256"]!=record["report_outputs"]["docs/midterm1/main.pdf"]
                or verification["tex_sha256"]!=record["report_outputs"]["docs/midterm1/main.tex"]):
            raise ValueError("Report verification does not bind the current PDF")
    if args.visual_review:
        visual=json.loads(args.visual_review.read_text())
        if visual["status"]!="PASS_VISUAL_REVIEW" or visual["pdf_sha256"]!=record["report_outputs"]["docs/midterm1/main.pdf"]:
            raise ValueError("Visual review does not bind the current PDF")
    if args.final and (verification is None or visual is None or not args.neb_receipt):
        raise ValueError("Final map needs current build/visual and both terminal NEB receipts")
    record["report_verification"] = dict(body_pages=verification["body_pages"] if verification else None,
        references_pages=verification["reference_pages"] if verification else None,
        mechanical=source(str(args.verification.relative_to(ROOT)) if args.verification.is_absolute() else str(args.verification),"Current report rendering/build checks") if verification else None,
        visual=source(str(args.visual_review.relative_to(ROOT)) if args.visual_review.is_absolute() else str(args.visual_review),"Current all-page visual review") if visual else None,
        scope="Live report includes later optional evidence; required snapshot/gate/public_source_map are immutable; publication/allocation release are root-owned",
        independent_teammate_checks="Remain assigned")
    record["primary_references"] = [r for r in record["primary_references"] if r["key"] != "omat2026"] + [
        dict(key="gilmer2017",url="https://proceedings.mlr.press/v70/gilmer17a.html",checked="Primary PMLR abstract/metadata",supports="Messagepassing framework; not evidence current graph wins"),
        dict(key="dieng2019",url="https://proceedings.mlr.press/v89/dieng19a.html",checked="Primary PMLR abstract/metadata",supports="Latentcollapse possible with expressive likelihood; not diagnosis of this network"),
        dict(key="ichikawa2024",url="https://proceedings.mlr.press/v238/ichikawa24a.html",checked="Primary PMLR abstract/metadata",supports="SimplifiedlinearVAE sensitivity only; no LCO convergence theorem"),
        dict(key="wakamiya2026",url="https://doi.org/10.1038/s41524-026-02075-0",checked="Primaryindexedmethods and archived accompanyingcode inpriorreview; SI numericalvalidation not independently retrieved",supports="Relatedfixedoccupation propertyconditionedCVAE+UMA loop; not noveltyclaim")]
    # Repeated interim regeneration should not duplicate bibliography records.
    record["primary_references"] = list({r["key"]:r for r in record["primary_references"]}.values())
    p.write_text(json.dumps(record,indent=2)+"\n")
    lines = ["# Report claim and source map", "", f"Status: **{record['status']}**.",
             "The required gate and its snapshot remain immutable. This live map does not certify teammates' independent contributions or final publication/resource release.", "",
             "| Claim family | Verified scope | Main source |", "| --- | --- | --- |"]
    for c in claims:
        s = c["sources"][0]
        lines.append(f"| {c['id']} | {c['claim']} | [{Path(s['path']).name}](../../{s['path']}) |")
    lines += ["", "The machine-readable map binds exact input hashes. Optional studies need explicit reviewed additions and retain a separate gate/status. Current final, development, historical and updated discussion evidence remain distinct."]
    (ROOT/"docs/midterm1/CLAIM_SOURCE_MAP.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(status=record["status"],claim_families=len(claims)),indent=2))


if __name__ == "__main__":
    main()

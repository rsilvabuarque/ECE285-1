# Midterm evidence map — November 8

The November 7 packet, `docs/midterm1`, and the immutable November 8 interim
snapshot remain historical evidence. This report uses the new frozen experiments.
FINAL static and endpoint scoring are complete, with separately authored
arithmetic recounts. No final outcome is used to retune or select a model.

All paths below are relative to the integrating project root unless marked
as research-worktree output. Exact absolute paths and SHA256 bindings are in
`claim_sources.json`; report/PDF hashes are attached by the final snapshot.

| Claim | Evidence | Scope |
|---|---|---|
| 37 full fits, 148,000 updates | Fit summaries, `reports/overnight_20261108/confirmed_model_selection.json` | Different methods/screens, not 37 replicate fits of one model |
| Frozen selected methods and FINAL-only replicates | `runs/overnight_20261108/final/static_manifest.json`, confirmed selection | A/B and shell A/B five fits; C/D/remaining B three; 0804/05 excluded from development selection |
| Complete known-history novelty | `runs/overnight_20261108/paired/final_history/{history,receipt}.json` | 344,001 orbits; includes prior assessed/acquired/observed research structures |
| Condition control differs from latent use | Initial development learning curves, paired probes and `sensitivity_v2` | A weak latent use; C active posterior but poor prior utility; signed iid-prior energy shift is not a collapse test |
| Shell selection and negative screens | Development screen/confirmation summaries and frozen selection | B-shell improvement consistent across development fits; A-shell improvement unresolved; screen LOW tradeoffs retained |
| FINAL raw proposals valid and frozen | `runs/overnight_20261108/final/independent_raw_audit/receipt.json` | 44,928 static primary + 4,992 latent probes; 10,944 endpoint-arm attempts |
| FINAL static comparison and negative latent findings | `final/static_evaluation/{summary,seed_metrics,paired_attempts}.csv`, `final/static_independent_metrics/{receipt.json,contrast_summary.csv}` | 99,840 two-teacher outcome rows independently checked; 5-fit base/shell A/B contrasts and 3-fit C/D/remaining controls |
| Actual condition/latent energy responses | `final/static_sensitivity_v2/{receipt.json,summary.csv}` | Paired randomness; absolute/RMS latent response separate from signed shifts and conditional utility |
| Exact bounded structural distance | `final/static_distances_v2/receipt.json`, `metrics_by_seed.csv` | All 44,928 primary attempts; min over 192 operations and deterministic 1,024 TRAIN references/count is an upper bound on full-TRAIN nearest distance; first aborted implementation preserved |
| Paired acquisition/support complete | `runs/overnight_20261108/paired/full_collection/{receipt.json,development_class_support.csv}` | 5,992/6,000 converged; shared 3,594 TRAIN/1,200 VAL; thresholds TRAIN-only |
| Deletion energy useful but imperfect endpoint proxy | `paired/trainval_label_audit`; final-only `paired/TEST_final_audit` | TEST opened only after authorized FINAL scoring gate, 1,198 known pairs/1,200 attempts; no threshold refit |
| Complete FINAL endpoint cross-evaluation | `runs/overnight_20261108/final/paired_evaluation/{current_low,paired_deletion,paired_endpoint,shared_uniform}` and `final/paired_independent_recount/receipt.json` | 10,944 attempts; 7,616 physical endpoints converged; 1,233 independently recounted class/count fields; not independent physical inference |
| Latent-specific current LOW gain unresolved; endpoint labels no gain | `reports/overnight_20261108/paired_final_evaluation/contrasts/{summary_contrasts,per_replicate_contrasts}.csv` | Three-fit paired differences; original H0 q20 is separately named and never applied to endpoint energies |
| Endpoint numerical and teacher dependence | `paired/controls_comparison_with_mace`, `paired/mace_sp_hash96_audit` | TRAIN-only controls; same-geometry scores distinct from independently quenched MACE endpoints |
| Primary relaxation call counts | `reports/overnight_20261108/paired_call_receipt.json` | 13,712 unique optimization attempts, 558,458 logical E/F/stress bundle attempts; not independent training examples or allocated node-hours |
| Exact finite raw-state reference with approximate symmetry | `reports/overnight_20261108/smallcell`, independent-v2 research receipt | Literal Hu parent; all 1,364 raw states scored, not exact energy degeneracy assumed |
| Research physical limits | `docs/research_supplement_20261108` and its source map | Endpoint/path sensitivity; nonconverged route retained; saddle index does not establish full hop; harmonic null is not MD thermodynamics |

Fit-level SD and paired fit contrasts describe the observed fitted replicates;
proposal counts do not increase that independent fit count. Uniform shared-bin
coupling makes balanced yield one third only before repeat/novelty losses.
FINAL uniform is 33.316% with 0.039 percentage-point descriptive fit SD;
the development exact-third identity does not imply zero bin uncertainty.
Neither an MLIP teacher nor a numerical force target
establishes DFT or experimental validity.

Beihao independently verifies latent/no-latent behavior and controls. Varad
independently replays attempts, novelty, endpoint pairing and costs. Ricardo
reviews physical geometry and integrates/manual-submits after all authors
review. Automated execution is not their completed independent contribution.

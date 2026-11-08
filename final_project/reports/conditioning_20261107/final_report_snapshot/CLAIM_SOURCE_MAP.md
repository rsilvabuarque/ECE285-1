# Report claim and source map

Status: **FINAL_REPORT_REVIEWED_WITH_SEPARATE_IMMUTABLE_REQUIRED_GATE**.
The required gate and its snapshot remain immutable. This live map does not certify teammates' independent contributions or final publication/resource release.

| Claim family | Verified scope | Main source |
| --- | --- | --- |
| parent48 | Li48Co48O96, Li rows48..95, 192 retained site permutations, fixed cell | [O3_N48_pilot.npz](../../artifacts/current/core/data/parents/O3_N48_pilot.npz) |
| half_orbit_capacity | At nLi24: 32247603683100 raw masks and168025313938 orbits under this parent action | [make_examples.py](../../reports/ranking_audit_20261107/make_examples.py) |
| training_round0 | 50247 rows; train40157/val4967/test5123 | [preparation.json](../../artifacts/current/improvement_v1/data/preparation.json) |
| historical_yields | Old UMA/MACE AR35.57/40.06%; uniform18.99/19.75%; strongestmutation52.65/54.08%; adaptive33.81/38.91% | [summary.csv](../../artifacts/current/improvement_v1/reports/final/summary.csv) |
| ranking_context | 34368 matched rows/11456geometries; cohort-specific rho/topk/regret/gap metrics | [receipt.json](../../reports/ranking_audit_20261107/receipt.json) |
| hu_five_counterexample | CompleteHuoptimizedO3Li10poolN5; UMAstructure4159rank1versusDFTrank5; regret7.9710587meV/atom | [Hu_O3_Li10_optimized_counterexample.csv](../../reports/ranking_audit_20261107/Hu_O3_Li10_optimized_counterexample.csv) |
| geometry_secondary | Old32maskUMAmedianoptimizersteps40deletion,24regressor,24CVAE; noCVAE-specificadvantage | [decoded_matched_relaxations.csv](../../artifacts/current/reports/decoded_matched_relaxations.csv) |
| new_multibin_benchmark | Balanced 41,472-attempt benchmark, 40,662 unique geometries; complete current UMA/MACE scoring; three-bin yield does not beat controls | [receipt.json](../../artifacts/overnight_20261107/evaluation/benchmark/receipt.json) |
| policy_development | Six4000-updateD8fits; balancing/selection fourrows; selected32.12%balancedvs33.62%uniform58.97%mutation, not final | [summary.csv](../../runs/conditioning_20261107/evaluation/policy/summary.csv) |
| paired_condition_latent | Sharedlatent/uniform diagnostics separate condition effect from nonzero latent use; no completecollapse inference | [condition_summary.csv](../../reports/conditioning_20261107/policy_diagnostics_v2/condition_summary.csv) |
| physical_parent_graph | Fullhost/mobile periodic graph with5Å provisionalradius and interlayeredges; descriptor invariant under192operations | [receipt.json](../../reports/conditioning_20261107/parent_graph/receipt.json) |
| paired_energy_target | All1000matched fixedcellUMA lineages; ranks change without occupationorbit merges; sparseprimaryroles deferlabeltargetfit | [receipt.json](../../reports/conditioning_20261107/relaxed_target_audit/receipt.json) |
| latent_development | D4/8/16/32 plus no-latentAR; D4chosen by frozen practicalmean margins; individualseeds andparameters retained, not physicalconvergence | [summary.csv](../../runs/conditioning_20261107/evaluation/latent/summary.csv) |
| spatial_development | Three-fit graphcontext DEV60.88%balanced/61.11%originalq20low vsmutation58.80/50.35; not final; encoderonly34.49/24.31 | [summary.csv](../../runs/conditioning_20261107/evaluation/spatial_confirmed/summary.csv) |
| spatial_condition_latent | Graphcontext strongpairedconditionresponse butKL.007-.011nats/weakzresponse; supportscausalspatialAR not demonstratedlatentbenefit | [family_diagnostic_summary.json](../../reports/conditioning_20261107/spatial_confirmed_diagnostics/family_diagnostic_summary.json) |
| trained_decoder_causality | Three actualgraphcontextcheckpoint futuregradient/permutation testsPASS withfullencoderforbidden; notphysics/equivariancevalidation | [trained_causality_audit.json](../../reports/conditioning_20261107/trained_causality_audit.json) |
| descriptor_negative_pilot | Onefit middleenergy jointdescriptor yield28.13%vsplain28.52/mutation44.79; all2304attempts; no improvementclaim | [summary.csv](../../runs/conditioning_20261107/evaluation/descriptor/summary.csv) |
| frozen_final_assessment | All34560UMAattempts; graphcontext62.04%balanced63.54%low vsmutation58.62/52.52; matched20736MACEattempts63.45/67.62vsmutation57.29/52.95 | [summary.csv](../../runs/conditioning_20261107/evaluation/final/summary.csv) |
| independent_final_replay | Raw192operation mask/bin/history recomputation matches16tables with2515436fieldchecks zero mismatches; not physicalvalidation | [receipt.json](../../reports/conditioning_20261107/independent_final_metrics/receipt.json) |
| final_structural_distance | Exact192symmetry minimum tofixed1024TRAINorbits/count; upperboundfullTRAINdistance; context6.858uniform6.891mutation6.498sites | [receipt.json](../../reports/conditioning_20261107/final_structural_distances/receipt.json) |
| optional_editor | Same-seed editor balanced yield59.375% vs57.234%; low-q20 yield50.347/50.521%; all6912attempts and raw pair checks retained | [summary.csv](../../runs/conditioning_20261107/evaluation/editor/summary.csv) |
| editor_low_request_metrics | LOW paired mean favors mutation by0.07656meV/atom; learned lower in548of1152pairs; distinct-quintile yield tied | [request_energy_summary.csv](../../reports/conditioning_20261107/editor_low_request_v2/request_energy_summary.csv) |
| editor_secondary_minima | All9LOWfit/count minima at128attempts: learned lower novel child in7cases; mean advantages-.572/-.229/-.428meVatom; supplied-seed minimum is not globalhistory minimum | [terminal_best_novel_low_children.csv](../../reports/conditioning_20261107/editor_low_request_v2/terminal_best_novel_low_children.csv) |
| editor_stage_cost | 7330new logicalUMA labels=6845child-only+485sharedseed-only;15alreadyavailable; counts verified against PRE-editor registry/cache and completedtask receipts | [cost_audit.json](../../reports/conditioning_20261107/editor_low_request_v2/cost_audit.json) |
| editor_checkpoint_alias | Different ZIP-root file hashes for best.pt and named selected checkpoint; normalized members and full loaded payload identical; assessment already used named hashes | [receipt.json](../../reports/conditioning_20261107/editor_checkpoint_alias_audit/receipt.json) |
| posthoc_gaussian_prior | TRAIN-only posterior-moment Gaussian, noNNupdates: balanced UMA34.259vs31.047/MACE34.028vs30.787; LOWyield regresses; graphcontext retained | [receipt.json](../../reports/conditioning_20261107/independent_aggregate_prior/receipt.json) |
| temperature_followup | 24schedules/768attempts/102geometries; identical integerallocations and actualpairedproposals across4T; lowreferenceESS remains | [receipt.json](../../reports/conditioning_20261107/temperature_followup/analysis/receipt.json) |
| current_scope | Required gate complete and preserved; later extensions separate; no newDFT/MLIPfinetuning or entropy/generalization claim | [protocol.json](../../artifacts/conditioning_20261107/protocol.json) |
| defined_hop_NEB | Two numerically converged UMA event paths; event1 forward0.638→0.189eV with coupled7image/tightertarget change; event2 coarse0.192eV; MACEcoarseSP is not transport validation | [receipt.json](../../runs/conditioning_20261107/neb/basin_1_repeat2/receipt.json) |

The machine-readable map binds exact input hashes. Optional studies need explicit reviewed additions and retain a separate gate/status. Current final, development, historical and updated discussion evidence remain distinct.

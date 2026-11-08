# Report claim and source map

Status: **required final evidence ready; integrating report gate pending**.
This map does not certify the required delivery gate or teammates' independent contributions.

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
| current_scope | Requiredscientificevidencecomplete withrootreportgatepending; nonewDFT/MLIPfinetuning; no entropy/generalizationaccuracy implied | [protocol.json](../../artifacts/conditioning_20261107/protocol.json) |

The machine-readable map binds exact input hashes. Optional studies need explicit reviewed additions and retain a separate gate/status. Current final, development, historical and updated discussion evidence remain distinct.

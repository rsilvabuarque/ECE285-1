# Midterm evidence map — November 8 interim

This source preserves the prior submitted-ready draft in `docs/midterm1`. Its new results are development comparisons, not a newly completed FINAL assessment.

| Claim | Evidence inspected | Scope |
|---|---|---|
| Nine A/B/C fits each completed4,000updates | Root `runs/overnight_20261108/fits/*/summary.json`; `development/initial_controls_v1/learning_curves_summary.csv` | TRAIN/VAL only; TEST labels not evaluated |
| A and spatial AR have similar useful prior generation; C's active posterior gives no balanced-yield gain | Root `reports/overnight_20261108/initial_controls/{summary,seed_metrics,metrics}.csv` and `receipt.json` | New frozen8640-attempt DEVELOPMENT pool; both teachers complete |
| Condition response and latent response differ | Model-agent `reports/overnight_20261108/initial_controls_v1/{diagnostics_by_fit.csv,interpretation_receipt.json}` | Shared-randomness paired probes; not posterior reconstruction alone |
| Uniform balanced33.333% with SD0 is a coupling identity | Same64uniform draws repeated across3requested bins; raw `attempt_outcomes.csv` | Individual requested-bin yields vary; no zero-uncertainty claim |
| Paired pilot96/96 converged; not enough TRAIN/VAL support for target-model fitting | Root `runs/overnight_20261108/paired/pilot_collection/receipt.json` and `development_class_support.csv` | First32TRAIN masks/count; no VAL or TEST label use |
| Full paired acquisition, target fitting, endpoint assessment and FINAL still pending | Root central protocol/experiment ledger | Report is explicitly interim; replace only when actual receipts exist |
| Complete raw12-site finite reference, approximate-orbit spreads and teacher disagreement | `reports/overnight_20261108/smallcell/{receipt.json,thermodynamics.csv,orbit_spreads.csv,repeatability.csv}` | Literal parent; separate composition/teacher normalizations; no physical entropy or size-transfer claim |
| Independent finite arithmetic/geometry check | `reports/overnight_20261108/smallcell_independent_v2/{receipt.json,rank_scope.csv}` | 1,364 geometry checks and 11,104 arithmetic field checks; failed cancellation-scale checker preserved separately |
| NEB route nonconvergence and harmonic-null scope | Separate research supplement; raw NEB receipts and27-case harmonic outputs | Numerical MLIP/estimator diagnostics only; no DFT/MD campaign |

`claim_sources.json` binds exact files and hashes used by this build. It distinguishes prepared, executing and completed evidence. Final resource counts and publication receipts are owned by the integrating root session and must be attached after resource closure.

Beihao independently verifies the spatial no-latent control and interpretation. Varad independently replays raw attempts, novelty, pairing, failures and costs. Ricardo reviews physical geometry/teacher interpretation and manually submits the final course report. Machine execution is not their completed independent contribution.

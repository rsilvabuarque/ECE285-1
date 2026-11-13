# Angular, latent-dimension, and iteration evidence

This packet contains compact summaries and provenance copied from the
November 10–11 study. `manifest.json` records the original relative path,
size and SHA-256 of each copied file. It is intended to make the report's
figures and numerical comparisons reviewable in a normal GitHub clone.

- `development_aggregated/`: model-by-teacher yields, paired feature controls,
  latent interventions, replicate results, and aggregation receipt.
- `angular_dimension_figure/`: dimension-sweep plotted points and summaries.
- `development_teacher_agreement/`: within-composition teacher-rank summaries.
- `iteration_round{1,2}/assessment/comparison_*`: paired effects, per-fit
  metrics, coverage and verification receipts, separately for UMA and MACE.
- `iteration_paired_figure/`: the seed-level values behind the two-round plot.
- Root-level protocols, threshold and disclosure files: acquisition rules,
  fixed-threshold provenance, assessment-boundary disclosures, and checkpoint
  sidecar consistency audit.

The [report](../../reports/angular_20261110/report.pdf) explains the metric
definitions and negative results. All comparisons here are development or
previously frozen assessment results, not fresh validation sets for selecting
new hyperparameters. Preserve the existing TRAIN/VAL/TEST roles and freeze a
new assessment before choosing or evaluating further interventions.

This is **not** the complete replay archive: raw generated masks, scorer
registries, large model checkpoints, and historical inference environments
are intentionally not copied here. Some receipts contain absolute Perlmutter
paths and source hashes identifying their historical inputs; those paths need
not exist on a collaborator's computer. The post-freeze cached verifier is
identified in each comparison receipt. The compressed NPZ loader previously
materialized TEST values and an earlier trainer recorded a coarse TEST-support
aggregate; see `test_split_support_disclosure.json`. A separate accidental
historical-result display is documented in
`protected_result_access_disclosure.json`.

The earlier frozen dataset and checkpoints are available in the [November 8
release](https://github.com/rsilvabuarque/ECE285-1/releases/tag/ece285-overnight-20261108-v1).
The newer raw angular and iteration outputs remain in the original local
`runs/angular_20261110` archive; their absence from this compact packet means
the full latest audit cannot be replayed from this Git checkout alone.

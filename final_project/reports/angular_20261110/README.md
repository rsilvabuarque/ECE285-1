# Angular features, latent space, and iterative sampling

`report.tex` is the source for the scientific summary. Build it on a NERSC compute allocation with `module load texlive/2024` and `latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex`. The figure copies in `assets/` make the report source portable; the originating plotted values and receipts remain in the run directory.

Compact copies of the results and receipts are available in
[`../../artifacts/angular_20261110/`](../../artifacts/angular_20261110/README.md).
The full run directory was `../../runs/angular_20261110/` relative to the
original execution worktree and is not included in a normal clone. Its raw
outputs remain on the original shared filesystem. Important starting points:

* `fresh_development_plan.json`, `development_aggregated/{yield_summary,paired_control_summary,sensitivity_summary}.csv`, and `development_aggregated/receipt.json`: matched fresh-pool comparisons and inputs.
* `angular_dimension_figure/`, `development_summary_figure/`, `development_teacher_agreement/`, and `checkpoint_latent_sidecar_audit.json`: plotted data, teacher agreement, latent audits, and hashes.
* `iteration_protocol.json`, `iteration_round2_protocol.json`, `iteration_r2_assessment_anchors.json`, and `iteration_r2_lineage_controls_protocol.json`: acquisition, assessment, and control definitions.
* `iteration_round1/assessment/comparison_{uma,mace}_cached/`: first-round per-fit metrics, paired effects, coverage, and detailed verification receipts.
* `iteration_round2/assessment/comparison_{uma,mace}_cached_r2/`: second-round per-fit comparisons and verification receipts. The 87 frozen task bundles and separate score registries remain only in the full local run archive.
* `iteration_paired_figure/`: plotted seed values, input hashes, and PASS receipt. The paired figure itself is `assets/iteration_paired_effects.pdf`.
* `final_compute_receipt.json` in the full local run archive: the final Slurm snapshot, including all 50 identified jobs. It records 20.164 estimated GPU node-hours and 24.023 allocated job-node-hours; shared-node allocations can overlap, and the estimate is not an official posted Iris charge. The account-level receipt is not included in the public compact packet.
* `train_threshold_population_audit.json`, `protected_result_access_disclosure.json`, and `test_split_support_disclosure.json`: threshold provenance and limitations on protected-data handling.

The original research code, the second-round source frozen in the assessment, and the later optimized verifier live in separate worktrees. The verifier caches only exact symmetry-set calculations and score-array loading; its post-freeze status and source binding are explicit in the comparison receipts. Historical unoptimized audits that exceeded wall time are retained and must not be described as completed.

The original shell entry points and exact options are retained in `run_development_analysis.sh`, `analyze_iteration_assessment.py`, `run_r1_comparison_cached.py`, `iteration_*_gpu_config.json`, and the scheduler/assessment logs in the run directory. The final assessment was invoked on a `shared_interactive` allocation with the class-core Python and the audit-worktree driver:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/pscratch/sd/r/rsb/ece285_conditioning_20261107/venvs/class_core/learning/bin/python \
  /global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_angular_audit_goal_20261111/final_project/scripts/overnight_20261108/analyze_iteration_r2_assessment.py \
  --workers 12
```

The driver deliberately refuses to overwrite existing evaluations. Reproduction should use an isolated copy and a new output location, and must preserve the frozen protocol and its original assessment history. A later run on a different history is a different experiment, even if it uses the same checkpoints.

The final figure was built using the same Python with the audit-worktree `scripts/angular_20261110/plot_iteration_pairs.py` and `--r1 "$R/iteration_round1/assessment" --r2 "$R/iteration_round2/assessment" --output-dir "$R/iteration_paired_figure"`, where `R` is the run directory. The cost snapshot is reproducible from the audit-worktree `scripts/angular_20261110/cost_receipt.py --output NEW_PATH --job-id 61130675`; its receipt records the exact `sacct` command and all job rows. Both scripts refuse to overwrite their outputs.

For the scientific interpretation, see the report. A PASS receipt establishes the stated computational checks; it does not establish DFT agreement, unbiased estimates over all possible crystals, a thermodynamic ground state, or transfer to another cell or phase.

## Suggested follow-up responsibilities

* **Beihao — generative models:** reproduce one radial CVAE/AR pair from its saved configuration and checkpoint, inspect posterior KL and latent-resampling diagnostics, and propose a single controlled latent or descriptor change. Keep the same count and proposal budgets for a comparison.
* **Varad — assessment:** independently reproduce the per-cell, seed-paired yield calculations and the symmetry/novelty accounting; check both teacher-specific cutoffs and report negative effects as well as gains. Freeze any new assessment before evaluating proposals.
* **Ricardo — materials interpretation:** review representative candidate structures, fixed-parent assumptions, phase/size limits, and the planned independent physical validation. Any future geometry optimization or DFT calculation requires a separately approved computational plan; none was run in this analysis.

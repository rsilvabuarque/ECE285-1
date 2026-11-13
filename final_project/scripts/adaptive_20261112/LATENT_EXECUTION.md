# Reproducing the November 12 latent experiments

These campaign scripts preserve the executed analysis and orchestration logic while accepting an explicit `--run-root`. The immutable executed copies and exact task argv, source hashes, checkpoints, raw proposals, exclusions, registry inputs and cost receipts remain in `final_project/runs/adaptive_20261112/latent_diagnostics` and `latent_phase2`. The committed scripts have different hashes because they accept run directories rather than assuming they are copied inside them. They do not alter old results; analysis output directories must be new.

Run substantive commands inside a valid CPU/GPU allocation. Set `PYTHONPATH=final_project/src` using the checkout being evaluated. UMA workers require the validated UMA environment; MACE workers require the validated MACE environment. A successful import of both packages into one environment did not guarantee compatible e3nn checkpoint loading. Scheduler workers already bind one visible GPU each; no script here submits a new allocation.

For the initial diagnostic experiment:

```bash
python scripts/adaptive_20261112/latent_freeze_raw_union.py \
  --run-root "$INITIAL_DIAGNOSTICS" --task-file "$INITIAL_TASK_FILE"
python scripts/adaptive_20261112/latent_prepare_oracle.py \
  --run-root "$INITIAL_DIAGNOSTICS" --boundary-dir "$BASE_BOUNDARY"
python scripts/adaptive_20261112/latent_collect_monitor.py \
  --run-root "$INITIAL_DIAGNOSTICS"
python scripts/adaptive_20261112/latent_analysis_initial.py \
  --run-root "$INITIAL_DIAGNOSTICS" --stage oracle \
  --output-dir "$NEW_ANALYSIS_DIRECTORY"
```

`INITIAL_TASK_FILE` must contain exactly the 40 fully completed diagnostic tasks. The freeze reads no oracle energies. Eligibility is determined against an immutable identity-only boundary before the newly generated diagnostic union is added to future acquisition history. All raw valid orbits become known to later acquisition; only genuinely new eligible identities become newly protected, preserving original TRAIN roles. The oracle registers fresh canonical geometries separately for each teacher. Every raw proposal remains in its method/cell denominator even when known, protected or duplicated.

The strength follow-up has six model families: auxiliary weight 10 or 100 crossed with genuine latent supervision, condition-only control, or within-cell shuffled supervision. It includes original AR and D2 reference checkpoints sampled on the same fresh streams. The preparation scripts intentionally reproduce the frozen campaign layout: the new strength root and `latent_diagnostics` are siblings under the campaign root, which also contains `initial_fits`, `initial_fit_tasks.json` and `controls/mace.json`. They consume existing frozen source/task manifests; they are not a generic scheduler installation.

```bash
python scripts/adaptive_20261112/latent_strength_prepare_protocol.py \
  --diagnostic-root "$INITIAL_DIAGNOSTICS" --output-dir "$STRENGTH_ROOT"
python scripts/adaptive_20261112/latent_strength_fit_diagnostics.py \
  --run-root "$STRENGTH_ROOT"
python scripts/adaptive_20261112/latent_strength_prepare_scoring.py \
  --run-root "$STRENGTH_ROOT"
python scripts/adaptive_20261112/latent_condition_controls.py \
  --run-root "$STRENGTH_ROOT"
python scripts/adaptive_20261112/latent_gradient_audit.py \
  --run-root "$STRENGTH_ROOT" --output-dir "$NEW_GRADIENT_AUDIT"
python scripts/adaptive_20261112/latent_analysis_strength.py \
  --run-root "$STRENGTH_ROOT" --stage descriptors \
  --output-dir "$NEW_DESCRIPTOR_ANALYSIS"
```

The fit/diagnostic driver publishes immutable task files to the already allocated worker group only after the first oracle monitor reports complete. Its first job is fitting; it does not create workers or allocations. The scoring-preparation script registers inputs and writes a scoring task file but does not publish it: the integrating owner must first deliver the complete raw orbit union to the acquisition owner and freeze the common boundary. `latent_collect_monitor.py` collects complete registries, stops only completed teacher workers, and may publish nine additional UMA worker tasks after MACE finishes, using the same existing GPU group. Do not run competing publishers on `controls/mace.json`.

The terminal gradient audit uses a predeclared 512-row TRAIN minibatch, independent posterior noise, actual normalized targets and exact head/checkpoint states. It performs no optimizer update. It verifies manual nearest-neighbor vacancy-pair products, target normalization and shuffled permutation hashes, then reports auxiliary and reconstruction gradient norms separately. The condition-only model must have an exactly zero auxiliary gradient into the encoder. The audit uses an unweighted diagnostic mean; the actual training objective uses inherited/class-balanced weights. Model gradient norms above 10 identify possible clipping and are not proof of convergence.

All figures label SD across five independent fits. Paired mean-difference confidence intervals use Student-t with four degrees of freedom. Posterior probe prediction, prior prediction mismatch, measured descriptor response, energy class accuracy and novel correctly conditioned orbit yield are separate observables. An increased KL or reduced head loss alone does not establish useful physical control.

The posterior-mean follow-up uses `--latent-physical-input mean`; the default remains `sampled`. This changes only which posterior representation feeds the auxiliary loss. Reconstruction still samples z, and VAL reports both mean and sampled-posterior descriptor errors. `latent_mean_prepare_campaign.py`, `latent_mean_fit_diagnostics.py`, `latent_mean_prepare_scoring.py` and `latent_analysis_mean.py` preserve this campaign's preparation/execution/analysis. They accept `--run-root`; preparation also requires `--source-checkout` pointing at the `final_project` source. The scientific protocol must already exist in that root. The same ownership and immutable-task rules apply; these are not commands to repeat a completed campaign in place.

`latent_conditional_information_audit.py` is explicitly posthoc after the fixed-condition scatter. It fits ridge probes with TRAIN-only centering/scaling separately within each condition, evaluates all nine cells and five fits, and compares five seeded random graph initializations at 24 Li/middle. Deterministic posterior means can contain recoverable geometric fingerprints at extremely small scale; this does not show information in the noisy latent samples used by the decoder. The random-initialization control distinguishes learned information from architectural features. No extra supervised-only training was performed.

`latent_fixed_condition_embedding.py`, `latent_stable_structural_summary.py` and `latent_posterior_information_bound.py` produce the fixed-condition illustration, structural sensitivity, and finite-VAL sampled-information bound. Moment calculations use float64 accumulation even for FP32 forward outputs. Earlier executed illustrative `np.std(float32)` summaries exaggerated some tiny spreads through centering error; those originals remain preserved with an erratum and corrected figures. The conditional ridge fits already used float64 and their MSE results are unaffected. The information bound uses exact diagonal-Gaussian KL values averaged over the empirical fixed-condition VAL rows and convexity of KL; it is not a population or thermodynamic estimate.

`latent_task_cost_receipt.py` reports task-attributed GPU wall times and raw inference seconds separately. These quantities overlap and must not be added. The root SLURM ledger remains authoritative for allocation/node-hour accounting. `freeze_oracle_queue_snapshot.py` creates a separate full source tree with only a claim-directory enumeration optimization; its reversal proof and `test_oracle_queue.py` establish unchanged inference and exclusive atomic claims. It never reclaims completed, failed, or abandoned claims. The mean campaign finished with its original frozen queue code; subsequent R2 work adopted the optimized snapshot under an operational amendment.

# What each person should do before November 9

We already have enough calculated data for the midterm. The remaining work is
to reproduce the important comparisons, explain the actual models, investigate
one well-controlled improvement, and write a coherent approximately five-page
report. Use the existing frozen data. **Do not start a new DFT, MD, MLIP
fine-tuning, or broad inference campaign for these tasks.**

The midterm should contain the revised proposal, mathematical method/algorithm,
and discussion of novelty, advantages, and limitations. Results make the method
concrete, but a positive result against every baseline is not required. The
present scientific finding is that the AR CVAE beats random sampling while
strong mutation remains better, and that learned geometry starts help an
optimizer without yet showing a CVAE-specific advantage.

## Shared working rules

Use a separate branch and a new run/output directory. Preserve the supplied
lineage/orbit splits; never randomly split individual trajectory frames. Record
the input SHA256, command, seed, checkpoint selection rule, and environment.
Agree on one integrating editor. Training and substantial analysis belong in
the class's permitted GPU/compute allocation; the commands do not choose an
account or submit a job for you.

Cached labels are the default for all midterm analysis. If a generated mask is
not in the supplied cache, its energy is **unknown**. Save it as pending. Do not
assign it the nearest mask's energy, treat it as a failed low-energy sample, or
drop it and report a supposedly complete success rate. A score computed only
where the cache has coverage must be labeled as such. Scoring a new candidate
is a new oracle query and needs to be part of a separately agreed budget.

The completed final pool is a fixed historical comparison. It can be
independently recomputed and checked. If we use its results to select a model,
we cannot call that pool an untouched final evaluation of the selected model.
For the next model, use the supplied development/validation data for selection
and retain an assessment boundary that the training run has not seen.

## Beihao: model training, controlled ablations, and mathematical method

### Required for this midterm

1. **Reproduce one fit and verify reload/sampling.** Start with the original
   three-count original data for a two-epoch operability check. For the full
   reference reproduction, switch to `training_round0.npz` with `ar_cvae`,
   energy conditioning, canonical orientation, and β = 0.1, using the archived
   reference settings in `configs/learning/README.md`. Run in the class
   allocation. Check that generated occupancies are binary and
   have exactly the requested Li count. Compare the new history with the saved
   configuration; numerical agreement depends on software/hardware, so report
   deviations rather than forcing matching output.
2. **Write the model equations in your own words.** Describe encoder, latent
   dimension, GRU decoder, teacher forcing versus generation, exact count,
   training-only low-class condition, and reconstruction/KL objective. Explain
   what the AR baseline and composition-only model control for. State that the
   model does not optimize an energy function or guarantee a minimum.
3. **Run one interpretable adaptation ablation using existing data.** The
   hypothesis is that the last update changed the learned distribution or
   optimization tradeoff, rather than simply lacking more labels. Fix a common
   starting checkpoint, β, threshold definition, validation set, random seed,
   optimizer, and **number of optimizer updates**. Compare continued training
   on round 0 with training on round 1 under the same unbalanced weighting.
   If time permits, a second factor toggles class balancing on/off. Keep energy
   labels and all other settings fixed. Do not change architecture and weighting
   simultaneously and then attribute the difference to one cause.
4. **Report more than posterior reconstruction.** Plot train/validation
   reconstruction and KL with the selected epoch marked. On the protected
   existing validation masks, report held-out likelihood/reconstruction and
   conditional response. For prior samples, measure exact count, orbit
   uniqueness, distance to training, and lookup coverage. Quality claims require
   an actual cached energy for that exact geometry; new uncached samples remain
   pending. Do not choose a new checkpoint on the historical final yield.

Use the existing run configuration as the reference for all parameters. A
comparison that changes the class labels requires an explicit new experiment
label. For the clean data-only ablation, freeze the round-0 training-derived
thresholds for both branches rather than silently recomputing them. If the
provided CLI lacks a setting needed for this control, add a small explicit
option and test the labels/weights rather than approximating the experiment.
After a warm start, `--beta-warmup 1` holds β constant from the first epoch;
match validation at common update intervals as well. The portable trainer
exposes `--max-updates` and `--validation-beta`, but its current warmup and
validation schedule are epoch-based. Equal update caps alone do not control
those schedules when training set sizes differ.

Deliver a new run configuration and history, a learning-curve figure, a compact
ablation table with its controls/limits, and roughly one page of model/method
text. Three independent fit seeds are preferable for the final report; one
reproduction plus the already completed three-seed results is sufficient for
the midterm if time is tight.

### Geometry contribution using the existing data

Reproduce **both** the geometry CVAE and deterministic regressor with the same
lineage split and training budget. Check that scalers, condition statistics,
displacement normalization, and requested endpoint tuples are training-only.
Keep absent atoms masked in the loss. Compare held-out coordinate errors by Li
count and cohort. A useful bounded ablation removes the four scalar property
conditions from both models while keeping occupancy/cell/cohort inputs, then
asks whether those properties add predictive information. A smaller alternative
is to compare endpoint-only training with the existing weighted trajectory
training, using the same held-out endpoint lineages and update budget. If
subsetting frames, renormalize the remaining frame weights to sum to one per
lineage. For a property-input ablation, zero the same four standardized features
in training, validation, and generation, rather than removing them only at test
time. Refit the models under that explicitly recorded condition policy.

The geometry task is displacement prediction at a supplied cell, not guaranteed
force minimization. Do not claim that a lower coordinate error establishes
better energy or fewer optimizer steps. Those claims must be checked against
the existing decoded-score and matched-relaxation tables, or remain proposed
tests if a new model has not been scored.

## Varad: baselines, validation, statistics, and figures

### Required for this midterm

1. **Independently recompute the existing primary result.** Run the package
   evaluation on the supplied snapshot. Check proposal counts, canonical
   keys, frozen history, teacher-specific original thresholds, and duplicates.
   Verify the numerator counts distinct novel low-tail orbits and the
   denominator includes every attempt. Reproduce the UMA and MACE table and
   retain seed-specific values.
2. **Make the comparison figure.** Include uniform fixed-count sampling,
   low-energy-mask mutation, the legacy dense CVAE, exact-count dense model,
   no-latent AR baseline, composition-only AR CVAE, energy-conditioned AR CVAE,
   and adaptive versions. Use the same attempt budget. Show individual fit
   seeds and label which methods use a larger seed/training pool. Keep UMA and
   MACE as separate teacher panels. A second teacher is a robustness check,
   not DFT validation.
3. **Separate quality, uniqueness, and local exploration.** Report orbit
   uniqueness, historical novelty, low-tail membership, their combined yield,
   and the supplied symmetry-minimized distance-to-training diagnostic. Plot
   energy quantiles and best energy versus attempt count from the existing
   ordered proposal/energy records. Lower single best energy and higher yield
   are different objectives. Mark any record for which cache reuse or missing
   labels limits the statistic.
4. **Explain uncertainty and cost.** Three training seeds are three fitted
   models; 512 draws from one model are correlated outputs of that model, not
   512 independent training experiments. Use descriptive seed variation and
   paired comparisons where appropriate. Count adaptive labels and model
   training in computational-cost statements. Do not claim a net savings from
   generation time alone.

Deliver a verified summary CSV plus receipt, one primary figure with seed
points, one quality/diversity or best-energy-budget figure, and the validation
and baseline paragraphs. Flag a discrepancy rather than editing a frozen
result to match an expected conclusion.

### Broader composition analysis with current labels

The wider main-parent dataset already covers all Li counts 0–48. First show a
histogram of available symmetry-unique masks per count and its split. The
current improved-model result covers only 12, 24, and 36. Wider coverage in the
report is useful if clearly labeled as dataset coverage or a separate test.

For a controlled model test, agree with Beihao on held-out interior Li counts
and a composition-balanced training set using the existing arrays. State
whether the test is interpolation between trained compositions or extrapolation
beyond them. The current 12/24/36-trained decoder can be queried at other
counts, but the cache may not contain all its proposals. Use held-out likelihood,
exact-count/novelty, and exact labeled-mask lookup coverage where valid. Do not
invent energy scores for missing geometries. Endpoint counts with only one
possible orbit are useful plumbing checks, not meaningful generative discovery
successes.

### Small-cell exact benchmark, if time permits

The prepared 12-site O3 parent has 87 symmetry orbits across all compositions,
including 17 at six Li. Its full supplied MLIP lookup can evaluate any occupancy
proposal in that finite parent without new calculations. This is a useful exact
test of novelty accounting and coverage. Train a **separate compatible 12-site
model**; do not apply the 48-site weights to a padded or truncated vector and
call it size transfer. Near-empty/full compositions contain too few orbits for
independent train/validation/test inference; state that limitation and begin
with the six-Li 17-orbit subproblem if necessary. Report the small sample size.

These 87 MLIP labels are not 87 newly computed DFT references. The historical Hu
DFT archive contains geometries and settings that must be matched explicitly
before assigning its labels to a benchmark.

### Geometry validation contribution

Recompute the existing decoded-coordinate and 32-mask paired-relaxation summary
with `cathode-cvae evaluate-geometry --snapshot artifacts/current --output-dir
runs/varad_geometry`, then independently check a few lineages against the source
CSV records.
Keep the same teacher, input mask, optimizer, cell mode, and stopping tolerance
within each comparison. Report direct decoded maximum force and component RMS,
energy change, minimum atomic separation, and subsequent iterations/calls where
available. Include the deterministic regressor. Show per-lineage paired
differences and final endpoint energies, since stopping faster in a different
high-energy basin is not automatically better. Cross-teacher direct scores
test robustness to the potential; they do not measure a new generative fit.

## Ricardo: chemistry, data, symmetry, and interpretation

Verify the parent, Li-site indexing, occupancy/vacancy convention, energy
normalization, and VESTA examples. Check that symmetry grouping applies to the
stated parent/cell and that no trajectory lineage crosses a split. Explain the
difference between pristine-derived deletion, decoded coordinates, and a
converged relaxation. Supply a concise recycling motivation and the physical
limits of the present MLIP-label benchmark. Check any comparison with Hu's DFT
archive against the actual coordinates, phase, system size, and DFT setup.

For a later DFT audit, propose a compact selection covering CVAE, mutation,
random sampling, and teacher disagreement at matched composition. Selection
is planning only; no additional DFT is needed for the present midterm tasks.

## Integration schedule

| Date | Beihao | Varad | Ricardo / shared |
| --- | --- | --- | --- |
| November 7 | Fit/reload check, model paragraph, controlled ablation specification | Recomputed table, discrepancy check, figure draft | Parent/provenance/symmetry check, motivation paragraph |
| November 8 | Learning curves and bounded ablation result, geometry-method check | Final seed figure, geometry paired summary, limits | Integrate report; freeze numerical tables and identify one editor |
| November 9 | Verify equations, implemented method, and configurations | Verify figure/table numbers and statistical wording | Check approximately five-page body, references, and submission rules |

Plan roughly 6–8 focused hours each before this deadline; prioritize the required
outputs before optional wider-composition or small-cell experiments. Presentation
dates are November 24 and December 1, and the final report is due December 12.

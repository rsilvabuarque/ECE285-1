# Paired policy diagnostics

The complete policy pool has 10,368 primary and 3,456 diagnostic attempts.
All 13,824 attempts have exact UMA energies; all masks are valid. The analysis
contains 6,912 low-to-middle/high pairs, 2,304 latent-resampling pairs and 576
matched posterior/prior pairs. All 1,152 posterior/prior raw-target distances
were independently checked against canonical protected validation masks.

Use `receipt.json` for numerical input/output hashes and `execution_command.txt`
for actual allocated commands. `analyze_diagnostics.executed.py` preserves the
numerical-analysis source. Fit diagnostics include only the six policy fits in
the frozen model manifest, not other fits that finished concurrently. An earlier
version in `../policy_diagnostics/` retains the initial broader fit scan; its
primary, paired and metric-scope tables are byte-identical to these tables.

`selection_audit.py` independently reproduces all twelve balanced/original-q20
policy summary values and checks the selected method and source hashes.
Its result and concise selected-model diagnostics are in `selection_audit.json`.
It reads the selector's records and does not alter the selected policy.

For the selected balanced-training, validation-beta-0.1 model, changing the
request from low to high gives a mean shift of only +0.0773 meV/atom. The
composition-specific shifts are +0.1099, +0.1299 and -0.00784 meV/atom for
12, 24 and 36 retained Li, respectively. Latent resampling at the same AR
uniforms changes a mean 12.257 physical-site occupations and produces a mean
absolute energy change of 2.097 meV/atom. These are different probes: condition
responsiveness is weak while the latent affects generation. Posterior target
Hamming distance averages 15.958 sites, compared with 18.181 for the matched
prior. Neither a KL value nor these finite probes alone proves a general
statement about latent collapse.

Active-component counts use variance of posterior means across validation
configurations, with dimensionless provisional cutoffs 0.001, 0.01 and 0.1.
Pooled and within-composition values are separate in `latent_components.csv`;
the selected model has 6–8 components above 0.01 within a composition, depending
on fit seed and composition. These cutoffs are descriptive, not physical
convergence criteria.

`paired_conditions.csv` records shared latent and AR uniforms for CVAE condition
pairs. Latent-resampling and posterior/prior pairs share AR uniforms but have
different latent vectors. Sampler controls use attempt-index pairs without the
same conditional random draws; mutation changes its requested-bin seed pool.
Raw Hamming distance and symmetry-orbit equality are reported separately.

Balanced requested-bin yield and requested-low/original-q20 yield remain
separate in `distinct_metric_scopes.csv`. All attempts remain denominators.
Individual fit/design replicates are retained; proposal draws are not treated
as independent fits or used to manufacture confidence intervals.

Final figures were visually checked after repairing clipped labels and an empty
legend entry. `plot_receipt.json` binds the final plots and verifies every CSV
remained unchanged; `rerender.py` and `analyze_diagnostics.renderer.py` preserve
that rendering command/source. Earlier plots remain in `original_plot_layout/`.
No new scoring, fitting, DFT, or MLIP fine-tuning was performed for this analysis.

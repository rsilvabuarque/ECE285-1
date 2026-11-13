# Adaptive sampling, latent control, and entropy: November 12–13

Read [overview.pdf](overview.pdf) first; [technical_report.pdf](technical_report.pdf)
contains the methods, diagnostic derivations, matched controls, and negative results.
These are research updates after the preserved November 9 midterm submission.

The completed campaign contains 166 fits, 520,000 optimizer updates, and 469,672
MLIP geometry–teacher evaluation events. No new DFT, MD, geometry relaxation, or
MLIP weight fitting was performed. Results concern fixed-parent, unrelaxed O3
lithium-vacancy masks. UMA-S-1.2.1/omat provides N48 training labels; MACE-MH-1/
omat_pbe audits the same candidates independently. The entropy reference is a
new idealized N12 parent, not Hu's DFT geometry.

## Rebuild the tables and reports

From `final_project`, with Python 3.11+ and TeX Live/latexmk installed:

```bash
python reports/adaptive_20261112/build_tables.py \
  --run-root artifacts/adaptive_20261112
cd reports/adaptive_20261112
latexmk -pdf -interaction=nonstopmode -halt-on-error overview.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error technical_report.tex
```

The compact packet preserves each relative table-input path. `table_sources.json`
binds the input used for the saved build; rebuilding it records your local input
paths. `asset_sources.json` binds each copied figure to its full raw-run source.
The artifact manifest binds all 94 copied summaries/receipts, with a separate
final resource receipt and report-build receipt. The compact packet reproduces
report tables, not raw-attempt scoring or full model training. Original attempts,
checkpoints, raw MLIP arrays, exclusions, and technical failures remain under
`final_project/runs/adaptive_20261112/` on the shared filesystem.

## Reuse the methods in a new development namespace

Install the package with the main [installation procedure](../../README.md#install-without-materials-specific-setup).
Commands respect visible resources and never select an HPC account automatically.
For the generic interfaces and exact argument documentation:

```bash
python -m cathode_cvae.learning.train_mask --help
python -m cathode_cvae.adaptive_iteration --help
python -m cathode_cvae.adaptive_oracle --help
python scripts/entropy_20261112/prepare.py --help
python scripts/entropy_20261112/boltzmann_followup.py --help
```

The executed campaign's frozen arguments and immutable source copies are in the
full run archive. [Adaptive protocol and CLI](../../scripts/adaptive_20261112/README.md),
[latent execution](../../scripts/adaptive_20261112/LATENT_EXECUTION.md), and
[collaborator tasks](../../docs/COLLABORATOR_TASKS.md) explain the controls and
specific follow-up work. Dated orchestration drivers preserve site-specific
execution history; they should not be launched unchanged on another machine.

## Conclusions to preserve

* The first observed-range gain is largely reproduced by relabeling/reweighting
  the original data. It mainly improves HIGH conditioning; it does not establish
  a comparable gain in low-energy discovery. The second range acquisition loses
  fixed-reference yield, while extending the observed upper-energy frontier.
* D=1,2,4 and the physical-head variants do not establish a latent-specific yield
  advantage. Conditional posterior means contain clustering information, partly
  already present in random graph features, at a scale overwhelmed by sampled
  posterior noise. A mean scatter is not evidence of useful sampled control.
* Boltzmann-weighted AR training helps importance sampling at its 1000 K target,
  but transfers poorly to 5000 K. Complete small-cell enumeration reveals a
  substantial UMA low-temperature Wang–Landau bin-width error despite schedule
  completion. These are estimator results, not experimental thermodynamics.
* Greedy oracle-guided swaps improve local tails against matched blind walks.
  Their gains are relative to common seed pools, not global ground states or
  a direct comparison against neural acquisition.

The independently audited causal contrasts and all 166 terminal metadata/role
checks pass. The inherited 1,254 H0 TRAIN/protected overlaps remain disclosed;
new training/protected overlap is zero. H0 TEST values are materialized by the
historical NPZ loader but are not indexed for new labels, training, or selection.

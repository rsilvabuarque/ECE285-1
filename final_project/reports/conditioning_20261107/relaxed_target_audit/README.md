# Same-mask deletion versus fixed-cell relaxed energy

This is a cached audit, not a new inference or relaxation campaign. Run from
`final_project/` in a permitted compute allocation:

```bash
python reports/conditioning_20261107/audit_relaxed_targets.py \
  --snapshot artifacts/current --output-dir runs/my_relaxed_target_audit
```

The original execution also supplied `--source-root` pointing to the historical
full campaign. That optional check compared all 1000 exported initial/final
energy scalars with their selected executed source JSONs and checked all 20 used
worker receipts for one teacher identity. The portable cached command needs no
MLIP environment or historical absolute paths. Install the core package first.

The homogeneous cohort has one fixed O3 Li48Co48O96 parent, UMA-S-1.2.1/`omat`,
FP32, TF32 off, and FIRE atomic relaxation in the unchanged parent cell. The
historical maximum-force stopping target was 0.05 eV/Å with 500 steps allowed. This
is an operational numerical setting, not a universal physical standard.
`historical_source_receipts.json` records checkpoint hash, executed package
versions, inference settings, source hashes, and the first full attempt record.

All 1000 attempts are included. All converged, and the largest stored endpoint
maximum force is 0.0499899611 eV/Å. Six saved frames per lineage produce 6000 rows;
these are 1000 independent masks, not 6000 training examples with independent
geometries. Frozen lineage roles are 787 train/107 validation/106 test. Selection
was approximately 50% random, 25% low-UMA, 25% cross-teacher-disagreement, balanced
across Li counts 2–46. These distributions are not population estimates.

`paired_lineages.csv` retains every paired energy, endpoint force, step,
convergence, split, nearest-site mapping and initial/final occupation-orbit key.
`composition_ranking.csv` supplies Spearman and Kendall ordering agreement,
nearest-rank low-quintile support/overlap and median relaxation energy gain.
At one composition, per-atom normalization cannot change energy order. Each
low-quintile edge is `sorted_energy[ceil(.2*N)-1]`, with ties kept. Audit-cohort
quintiles describe this paired comparison; they do not replace training-fitted
class thresholds used in the generation benchmark.

Endpoint coordinates are reconstructed from parent positions plus exported
translation-removed minimum-image displacements. For every atom, assignment to
the nearest same-species parent site is checked under periodic boundaries;
second-nearest margins, collisions and assignment changes remain explicit.
There are no Li or host assignment changes/collisions, and all 1000 initial
occupation orbits remain distinct at the endpoints. The smallest same-species
nearest-versus-second-nearest margin is 2.21618 Å. This distinguishes parent-site
occupation classes; it does not certify unique PES minima or prove that two
endpoints with the same occupation cannot be different geometric basins.

For the main nLi 12/24/36 counts, there are 23 lineages each. Rank correlations are
0.90217/0.97233/0.91403; four of five audit-low-quintile members agree at each
count. Median relaxation gains are −29.91/−19.86/−11.03 meV/atom. Across 45 counts,
median Spearman is 0.93789, minimum 0.63834, and minimum low-quintile recall 0.40.
Both broad agreement and local reordering therefore matter.

The matched label-target training ablation is **deferred**, with exact support
in `label_target_support.csv`. Primary counts contain 49 train/11 validation/9 test
lineages. Training-low endpoint classes contain 4/4/3 masks; the corresponding
validation classes contain 0/1/0. Neither repeating trajectory rows nor randomly
resplitting lineages repairs the missing independent validation support.

A future comparison would retain the same mask inputs/roles and model policy,
fit separate training-only deletion/endpoint thresholds, train both targets with
matched seeds/update budgets, and evaluate both generators under both energy
definitions. Additional paired development masks and genuinely held-out endpoint
labels should be acquired under one consistent geometry protocol; any fresh
proposal evaluated for relaxed quality requires its own relaxation. Endpoints
and intermediate descendants must remain in their ancestor's role. The current
audit does not claim that a mask CVAE outputs relaxed coordinates.

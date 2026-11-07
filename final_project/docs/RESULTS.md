# Evidence, measured results, and limits

The frozen binary-CVAE comparison, the matched DFT audit, and the new multibin
campaign are separate experiments. Numerical claims must name the experiment,
teacher, geometry protocol, reference pool, and novelty boundary. New campaign
results below use completed score/evaluation receipts and independent raw-input audits.

## Reproduced historical occupancy result

The completed November 6 assessment used 512 attempts per method, fit seed and
Li count (n=12,24,36), with three independent CVAE fit seeds and three
random-seed replicates for sampler controls. Its yield was

$$
Y=\frac{\#\{\text{distinct proposed orbits absent from frozen history and with}
\ e\le q_{20}^{\rm original}(n)\}}{\#\{\text{all attempts}\}}.
$$

The original training q20 is fixed separately by teacher and composition.
Repeated or previously known proposals remain attempts. Uniform fixed-count
sampling is uniform over raw masks, not over symmetry representatives. These
means are historical and must not be labeled outcomes of the new multibin model.

| Historical method | UMA yield (%) | MACE yield (%) |
| --- | ---: | ---: |
| Uniform fixed-count | 18.99 | 19.75 |
| Legacy dense CVAE, top-count decoding | 17.14 | 18.66 |
| Legacy dense CVAE, β=0.1 | 20.29 | 23.57 |
| Dense original-data, Bernoulli | 22.37 | 23.81 |
| Dense original-data, exact count | 22.61 | 24.57 |
| Dense expanded-data, exact count | 27.86 | 29.58 |
| AR expanded-data, no latent | 25.17 | 27.08 |
| AR composition-only CVAE | 31.88 | 35.76 |
| AR energy-conditioned CVAE | **35.57** | **40.06** |
| Extra adaptive AR, β=0.1 | 33.81 | 38.91 |
| Extra adaptive AR, β=1 | 27.84 | 30.43 |
| Mutation, original low-energy training pool | 45.40 | 46.96 |
| Mutation, expanded low-energy training pool | 50.91 | 51.84 |
| Mutation, round-1 low-energy training pool | **52.65** | **54.08** |

Source: `artifacts/current/improvement_v1/reports/final/{metrics,seed_metrics,summary}.csv`.
The portable evaluator reproduced their fields exactly in the November 6
validation. The new execution rechecks this historical result before interpreting
new results. Three fit seeds define descriptive model variation; hundreds of
latent draws are not hundreds of independent fits.

![Historical comparison](../artifacts/current/improvement_v1/reports/final/primary_comparison.png)

The AR CVAE improved on random sampling but did not beat low-energy-mask mutation.
The strongest historical mutation pool also contains more labeled history than
the round-0 training set. Comparisons that change data, orientation, initialization
or schedule are complete-method comparisons, not a causal isolation of one change.

### Why more adaptation did not automatically improve yield

The extra round coupled 25,000 additional development masks, warm starts,
recomputed training classes, and class balancing. Original assessment thresholds
stayed fixed. Its lower yield is therefore not mechanically caused by tightening
the final success gate, nor does it prove extra data are intrinsically harmful.
At selected β=1 adaptive checkpoints, validation KL was about 0.19–0.35 nats,
versus 17.1–17.7 for β=0.1. This is consistent with a changed reconstruction/prior
tradeoff, not proof of complete posterior collapse. The AR decoder itself is
stochastic. A larger KL likewise does not guarantee useful prior samples.

All these fits selected checkpoints by validation reconstruction plus unit-weight
KL, including β=0.1 training. That rule can disagree with downstream discovery.
The β=0.1 update found lower individual energy extremes at some compositions
while producing fewer distinct novel low-tail structures per attempt. Report
both objectives. The present task extends the energy condition; it does not
assert that adaptation alone was repaired.

## Matched DFT ranking context, independently recomputed

`reports/ranking_audit_20261107/` recovers 34,368 saved model–geometry rows:
11,456 exact geometries evaluated by UMA, MACE and M3GNet. Each pool fixes source,
phase, composition, atom count and cohort. The historical and Hu DFT protocols
are kept separate. The source audit used FP64/general settings; the new deletion
benchmark has its own explicit FP32 teacher configuration.

The table below uses descriptive, unweighted medians across informative
composition pools. Top-quintile requests k=ceil(0.2 Npool) and includes all exact
boundary ties, with resulting set sizes recorded. Pair-gap RMSE compares energy
differences within each pool, so constant offsets cancel.

| Source / cohort | Model | Pools / frames | Median Spearman ρ | Median top-quintile precision | Median gap RMSE (meV/atom) |
| --- | --- | ---: | ---: | ---: | ---: |
| Historical O3 / rattled | UMA | 7 / 524 | 0.97366 | 96.30% | 2.047 |
| Historical O3 / rattled | MACE | 7 / 524 | 0.95808 | 88.89% | 2.755 |
| Historical O3 / rattled | M3GNet | 7 / 524 | 0.91537 | 83.33% | 3.767 |
| Hu O3 / optimized | UMA | 10 / 104 | 0.73177 | 41.67% | 4.077 |
| Hu O3 / optimized | MACE | 10 / 104 | 0.48865 | 0.00% | 5.227 |
| Hu O3 / optimized | M3GNet | 10 / 104 | 0.43317 | 0.00% | 6.576 |

High rank agreement in strongly rattled frames does not automatically imply
accurate ordering of nearly degenerate optimized vacancy configurations.
Small optimized pools often request only one structure at the low-quintile
boundary, so zero precision can coexist with moderately positive ρ. Consult
`pool_metrics.csv` and `topk_metrics.csv` for sizes, ties and DFT energy spans.
The recomputation matched 432 individual archived diagnostic fields.

![Cohort-specific ranking context](../reports/ranking_audit_20261107/ranking_context.png)

### Verified five-structure counterexample

Hu's optimized O3 Li10Co12O24 pool contains exactly five geometries. UMA selects
`structure_4159` first, while DFT ranks it fifth. UMA's ρ is −0.1 and its selected
minimum has a DFT penalty of 7.9711 meV/atom relative to the best member of this
pool. MACE's ρ is 0.1 and its selected minimum's penalty is 1.7435 meV/atom. The
same five identities necessarily have perfect top-five set overlap in a
five-member pool; that is not recall from a larger unseen population.

![Complete five-member pool](../reports/ranking_audit_20261107/five_member_counterexample.png)

This evidence supports an explicitly MLIP-defined class benchmark and later
matched DFT validation. It does not support the universal claim that delithiated
LCO energies differ from DFT only by a constant. Gap errors can change static
weights even with perfect ranks. Rank agreement does not validate forces,
phonons, or thermal free energies. Archived SCF/spin convergence is not recertified
from these scalar exports, and trajectory frames are correlated.

## November 7 multibin and acquisition measurements

Three multibin fits completed on the protected round-0 splits, without class
balancing, using canonical masks, latent dimension 8, GRU hidden dimension 64,
β warmup over 20 epochs to 0.1, and validation reconstruction plus unit-weight KL checkpoint
selection. They ended by validation early stopping, not the 600 s wall limit:

| Fit seed | Epochs / optimizer updates | Best epoch | Fit wall seconds |
| --- | ---: | ---: | ---: |
| 20261106 | 53 / 4,187 | 38 | 30.806 |
| 20261107 | 44 / 3,476 | 29 | 27.761 |
| 20261108 | 49 / 3,871 | 34 | 27.707 |

Source: `artifacts/overnight_20261107/fits/multibin_v2_SEED/{configuration,summary}.json`.
These are measured fit receipts; generation quality still requires the complete
proposal/score/evaluation outputs below. A100 timing does not predict class GPU
wall time, and it excludes label acquisition and assessment.

The new result set is separate from the historical table. Its frozen balanced
benchmark has **41,472 attempts and 40,662 distinct geometries**: 256 attempts
per method, composition, requested bin and design replicate. All masks are
valid and exact-count; all have current UMA and MACE labels; no requests failed.
The novelty boundary contains 40,157 training keys and 188,919 frozen historical
keys. Success requires absence from both.

| Method | UMA useful yield (%) | MACE useful yield (%) |
| --- | ---: | ---: |
| Uniform raw masks | 33.00 ± 0.65 | 33.38 ± 0.26 |
| AR, composition only | 31.73 ± 0.77 | 31.94 ± 0.85 |
| AR, binary energy | 31.92 ± 1.01 | 31.03 ± 1.11 |
| AR, three energy bins | 31.00 ± 0.81 | 31.24 ± 1.22 |
| Requested-bin seed mutation | 58.39 ± 0.61 | 57.54 ± 0.16 |
| Low-bin seed mutation | 33.93 ± 1.05 | 34.40 ± 0.82 |

Useful yield is distinct novel in-requested-bin orbits divided by all attempts,
macro-averaged equally over three compositions and requested bins, then three
design seeds. ± is descriptive standard deviation across those three seeds, not
a confidence interval. CVAEs have three fit seeds; sampler controls have three
random replicates and no trained parameters. MACE uses its own training-derived
schema. Both new mutation controls select seeds from protected round-0
training rows: the requested bin for `mutation_bin`, always low for `mutation_low`.
They do not use the historical round-1 larger seed pool.

The multibin model does **not** improve balanced controllability or useful yield.
Its UMA low/middle/high hit rates are 35.59%/54.43%/8.42%, compared with
16.02%/63.32%/19.70% for uniform sampling and 49.18%/71.48%/54.60% for bin-seeded
mutation. It favors low energies while failing to distinguish all three requested
ranges reliably. This observation does not identify the cause; training imbalance,
prior/decoder behavior and model selection require controlled checks.

On requested-low attempts, the original-q20 UMA yield is 32.99% multibin,
33.94% binary, 31.25% composition, 17.49% uniform, 51.65% bin mutation and
52.00% low-bin mutation. This retains the old teacher-specific threshold
but uses a new history, pool and 256-attempt group budget; it is not the historical
35.57% experiment.

Source: `artifacts/overnight_20261107/evaluation/benchmark/{summary,summary_by_request,
fixed_q20_low_request,metrics,seed_metrics,confusion,budget_curves}.csv` and
`receipt.json`. Complete per-attempt outcomes and canonical candidate IDs remain
available. Per-method query curves are stand-alone deduplicated costs; summing
them across methods double-counts shared labels. Actual campaign calls belong
to the cost receipt. Layer diversity here is only the standard deviation of six
Li-plane filling fractions in the fixed parent frame, not phase/basin diversity.

[UMA condition quality](../artifacts/overnight_20261107/evaluation/benchmark/figures/condition_quality_uma_fixed-balanced-bin-budget_fixed_6c7d5ec1_6c7d5ec1.pdf) and
[MACE condition quality](../artifacts/overnight_20261107/evaluation/benchmark/figures/condition_quality_mace_fixed-balanced-bin-budget_fixed_6c7d5ec1_bd5c067d.pdf)
show individual design replicates. The packet also supplies low-request proposal-
and query-budget curves.

### Executed acquisition-guided generation

All three fitted models executed the complete prespecified schedule file: 90
schedules, 11,520 attempts, 11,080 distinct geometries, zero failed requests and
complete exact-count UMA/MACE scoring. This includes pure and exploration plans
at each of four temperatures, plus both equal-temperature-mixture plans, for
each of three compositions and three fits. Each schedule has a 128-attempt
budget; mixture schedules share that total across temperatures.

| Policy scope | Attempts | Requested low (%) | Measured low UMA (%) | Measured low MACE (%) |
| --- | ---: | ---: | ---: | ---: |
| Per-temperature static | 4,608 | 100 | 36.48 | 40.26 |
| Per-temperature exploration | 4,608 | 93.75 | 35.79 | 39.24 |
| Joint-temperature static | 1,152 | 100 | 36.02 | 38.98 |
| Joint-temperature exploration | 1,152 | 93.75 | 37.15 | 40.63 |

These are attempt-weighted fractions pooled over the declared scope, not
independent physical populations. Requested floor fractions are 93.75%/3.125%/
3.125% after rounding 120/4/4; the underlying continuous weights are approximately
93.333%/3.333%/3.333%. Each teacher measures its own bins. The large gap between
requested and realized fractions is the practical consequence of weak conditional
control; an exact request budget does not fix it. Temperature changes the
allocation calculation, not the decoder or lattice geometry. The coarse requests
are the same across these temperatures and do not establish temperature-specific
generation behavior.

Source: `artifacts/overnight_20261107/demo/generation_receipt.json` and
`evaluation/demo/{receipt.json,allocation_realization.csv,allocation_joint_realization.csv,allocation_joint_summary.csv}`.
The latter deduplicates useful orbit identities across temperatures within a
joint schedule. The [requested-versus-measured figure](../reports/overnight_20261107/evaluation_audit/demo_requested_vs_measured.pdf)
and independent `benchmark_audit.json` / `demo_audit.json` checks accompany the
results.

### Measured new campaign cost

`artifacts/overnight_20261107/cost_receipt.json` is a small Git-tracked receipt,
separate from the large release ZIP. It records **127,481 new logical single-point
energy/force/stress calls**, including 24 interface-pilot calls. The primary
reference requires 24,575 UMA calls; the balanced benchmark uses 81,324
geometry–teacher pairs; the demo adds 21,558 after reusing 602 pairs already
scored in the benchmark. A logical call is an ASE energy/force/stress evaluation
for one exact geometry and teacher, not an instrumented neural-backbone forward.

The released allocations consumed **12.401111 GPU-device hours**, equivalent to
**3.100278 four-GPU-node hours**, or **3.636111 physical node-hours**. The latter
counts the shared node's elapsed time as a physical node even though only two
GPUs were reserved. All startup, failures, analysis and idle allocation time are
included. Both jobs completed and were released. Historical label-generation
costs predate this goal and remain additional costs for any end-to-end comparison;
the short fit timings do not make the project a 30-second data-generation method.
No new DFT, MLIP fine-tuning or geometry-CVAE campaign was run in this extension.

### Complete independent-reference acquisition finding

The new independent reference contains 24,576 uniform raw fixed-count draws,
8,192 at each of n=12,24,36. Collapsing a repeated orbit leaves 24,575 exact
geometries; every geometry has a finite current UMA/omat FP32 energy. Frequencies
and original draw order are retained. No orbit multiplicity is multiplied into
these already raw-mask-weighted observations.

At 433.15 K the full-pool particle ESS is 4.629, 2.271 and 5.480 for n=12,24,36;
the largest representative masses are 0.361, 0.654 and 0.299. Across all four
temperatures and three compositions ESS ranges from 1.850 to 6.544. Low-bin
mass is 99.98624%–99.99999982%; a budget of 128 therefore rounds to 128/0/0
low/middle/high requests under pure static weights, or 120/4/4 under the 10%
exploration policy. These rounded allocations are identical across all tested
temperatures and compositions; the energies change weights and ESS without
producing resolved temperature-specific bin requests at this granularity. Coarse bin mass stability does not establish adequate tail
coverage: the minimum cell energy in the 512-draw prefix is above the 8,192-draw
minimum by 0.0101, 0.1758 and 0.2459 eV for the three respective compositions.
The nested prefixes are correlated sensitivity checks, not independent repeats.

Source: `reports/overnight_20261107/reference_diagnostics/receipt.json`,
`prefix_metrics.csv`, `allocation_fractions.csv`, and `reference_population.csv`.
The receipt links exact parent/schema/model hashes and the complete score
collector. The available plots are
[requested bin fractions](../reports/overnight_20261107/reference_diagnostics/requested_bin_fractions.pdf)
and [prefix sensitivity](../reports/overnight_20261107/reference_diagnostics/prefix_sensitivity.pdf).
This measures acquisition from a finite empirical reference under static
unrelaxed UMA energies; it is not a converged physical Boltzmann population.

## Secondary geometry result retained

One displacement CVAE and one deterministic regressor were fitted on the same
fixed-cell UMA relaxation corpus. Their 128 fresh-mask direct predictions lowered
energy relative to deletion and reduced force-component RMS. A CVAE draw gave
about −19.1 meV/atom median energy change and about 24% of the deletion force RMS.
However, none of the 128 direct outputs per group met maximum force <0.05 eV/Å.
The operational maximum-force criterion is distinct from RMS and is not a
universal physical standard.

On the matched 32-mask UMA optimizer comparison, all starts converged under the
same fixed-cell protocol:

| Start | Median steps | Median wall time in that run |
| --- | ---: | ---: |
| Deletion | 40 | 3.71 s |
| Regressor | 24 | 2.19 s |
| Geometry CVAE draw 0 | 24 | 2.19 s |

The learned start helped; a CVAE-specific advantage over regression was not
established. Three latent draws from one geometry CVAE are not three model fits.
Source tables are the frozen `reports/decoded_geometry_summary.csv` and
`reports/decoded_matched_relaxations.csv`. This remains a secondary result; a
new geometry-model campaign does not replace the current multibin objective.

## Conclusions that remain outside the evidence

No present result establishes global minimum discovery, DFT accuracy of new
48-site candidates, a full density of states, physical configurational entropy,
calibrated hydrothermal populations, equilibrium reactor conditions, or downstream
MLIP fine-tuning improvement. A finite uniform-raw reference can miss rare
low-energy states. Exploration floors are chosen request policies. Correct bin
allocation does not imply correct within-bin sampling. Thermal corrections from
a separate Phase 2 pilot must not be inserted into these class results as already
validated physical free energies.

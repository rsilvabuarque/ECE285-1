# Optional experiments after the required report checkpoint

The [required report gate](../artifacts/conditioning_20261107/required_gate.json)
completed before these experiments. Its
[frozen PDF and sources](../reports/conditioning_20261107/required_report_snapshot/manifest.json)
remain preserved. The editor and temperature studies below use new assessment
histories and do not replace the core final comparison or the collaborators'
assigned independent checks.

## Learning edits from known seeds

The editor receives an existing occupation pattern, a requested child-energy
class, composition, and an exact swap count. One swap removes one occupied Li
and fills one site that was vacant in the original seed. A one/two/three-swap
budget therefore changes exactly two/four/six occupancy bits. This is an edit
size, not a distance in Å, a physical diffusion event, or a geometry relaxation.
The removal and insertion sites have no imposed spatial-distance restriction.

The training encoder observes seed, child and conditions and predicts a Gaussian
latent distribution. The first decoder chooses removal sites from the seed's
occupied sites; the second chooses insertion sites from its originally vacant
sites, also observing the chosen removals. Exact-count subset sampling enforces
the budget without duplicate selections or reinserting a removed atom at its
original site. Generation uses the seed, conditions and a prior latent draw;
the target child and training encoder are absent. The condition is
`[nLi/48, onehot(child energy class), swap_count/3]`. The model has a four-component
latent, two-layer dense networks of width 128, and 97,000 parameters. This is a
separate variational edit model, not the physical-graph de novo decoder.

The [pair preparation audit](../runs/conditioning_20261107/editor/data_v2/preparation_receipt.json)
verified the original mutation seeds and edited sites against immutable records,
retained unfavorable as well as favorable children, and deduplicated pairs using
one common symmetry operation on both masks. Connected seed/child identity
components define the editor's train/validation/heldout roles: 4,141/505/538 pairs
out of 5,184. The heldout bank contains 492 distinct seed orbits. These are
known H0-training structures with heldout *editor pair families*, not globally
unseen physical structures. Protected H0 validation/test identities and core
final proposal identities were excluded from editor training.

Three editor fits completed 4,000 updates each. For each fit, count and requested
class, 128 trials selected a heldout seed uniformly from the matching seed class
and a swap count uniformly from one/two/three. Learned editing and uniform
mutation received the **same actual seed, exact budget, removal uniforms and
insertion uniforms** in every trial. The baseline has no energy acceptance rule
or adaptive population update. The history and all 6,912 attempts were frozen
before scoring. Seed geometries were included in the query pool so that
child–seed differences use the same current canonical UMA protocol. Full raw
alignment, actual random arrays and failed-request handling are retained in the
[generation receipt](../runs/conditioning_20261107/editor/assessment_v1/receipt.json).

### Conditional yield, average energy and extreme discoveries differ

| Outcome | Learned editor | Matched mutation |
| --- | ---: | ---: |
| Balanced distinct-novel requested-bin yield | 59.375% ±1.434 points | 57.234% ±1.092 points |
| Original-q20 distinct-novel yield on low requests | 50.347% | 50.521% |
| Mean child minus its seed on low requests | +1.17335 meV/atom | +1.09679 meV/atom |
| Lowest novel child versus its paired competitor | Lower in 7 of 9 fit×count cases | Lower in 2 of 9 cases |

The first outcome averages counts/classes equally and then the three fits or
paired sampler replicates. Its ± values are descriptive seed standard deviations,
not confidence intervals from thousands of generated draws. The observed balanced
gain is 2.141 percentage points. The separate original-q20 low-request yield is
essentially tied. In 1,152 low-request pairs, the learned child has lower UMA
energy in 548 cases (47.569%), mutation in 594 (51.563%), with ten exact stored
energy ties. Learned-minus-mutation energy averages **+0.07656 meV/atom** on low
requests; all three fit-level means are positive. These are strict comparisons
under this teacher/precision, not a claim that every tiny gap is physically
resolvable. Middle/high requests are property-targeting tasks; increasing energy
can be correct for a high request. Their outcomes must not be interpreted as an
unconstrained energy-minimization test.

The minimum discovered within a fixed budget is a different observable. The next
table includes **every** fit×count case at 128 low-request attempts per method.
Each entry is in meV/atom; negative is lower energy. “Best seed” means the lowest
current-SCF energy among that case's identical supplied starting seeds, not the
minimum over all H0 training structures or all previously known configurations.
Each case uses 41–54 distinct starting orbits, sampled with replacement.

| nLi | Fit seed | Best novel learned − best novel mutation | Best novel learned − best supplied seed | Best novel mutation − best supplied seed |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 20261106 | −0.952 | −1.344 | −0.391 |
| 12 | 20261107 | +0.913 | −0.137 | −1.050 |
| 12 | 20261108 | −1.677 | −2.244 | −0.567 |
| 24 | 20261106 | +1.295 | +0.631 | −0.664 |
| 24 | 20261107 | −0.420 | −0.198 | +0.222 |
| 24 | 20261108 | −1.562 | −1.398 | +0.163 |
| 36 | 20261106 | −0.529 | −1.203 | −0.673 |
| 36 | 20261107 | −0.321 | +0.008 | +0.329 |
| 36 | 20261108 | −0.433 | −0.378 | +0.056 |

The mean best-novel-child advantage of learned editing is −0.572/−0.229/−0.428
meV/atom for nLi 12/24/36. Mean best-child minus best-supplied-seed differences
are −1.241/−0.322/−0.524 for the editor and −0.670/−0.093/−0.096 for mutation.
The learned novel minimum beats the best supplied seed in seven of nine cases;
mutation does so in five. All exact energies and structure IDs appear in
[the nine-case table](../reports/conditioning_20261107/editor_low_request_v2/terminal_best_novel_low_children.csv)
and [composition means](../reports/conditioning_20261107/editor_low_request_v2/terminal_minima_composition_means.csv).
This is a secondary, posthoc best-of-budget comparison, with substantial fit
variation. It is evidence worth testing on another prespecified assessment,
alongside the tied low-quintile yield and less favorable paired mean; it is not
a blanket editor win or a global-minimum certificate.

The [running-best curves](../reports/conditioning_20261107/editor_low_request_v2/low_request_best_energy.pdf)
show all three fits faintly and their equal-fit step mean. They use novel children
of **low requests only**, on attempt and standalone unique-child-query axes.
Repeated or known masks remain attempts. A mean is displayed only where all
three fits have a finite novel minimum, over their common budget range. The
[paired low-energy distributions](../reports/conditioning_20261107/editor_low_request_v2/paired_low_energy.pdf)
retain the same-seed comparison; negative differences favor learned editing.
The pairing uses shared random streams, so this win fraction describes that
specified coupling, not independent draws from arbitrary two samplers.

### Exact labels and cost attribution

The [independent editor audit](../reports/conditioning_20261107/independent_editor_metrics/receipt.json)
recomputed all 6,912 attempts, 3,456 pairings and 192-operation canonical mappings.
Every one of the 492 seeds retains its original H0 energy class when re-evaluated
with current canonical UMA SCFs. Median/p95/maximum absolute old–current label
differences are 0.000291/0.016842/0.030040 meV/atom, versus class widths of
3.694–5.263 meV/atom. This observed class stability supports the present matched
comparison; no universal energy-parity tolerance or DFT accuracy follows.

The query pool contains 7,345 geometries including seed checks. Matching its hash
against the pre-editor required oracle snapshot, immutable certified cache, and
completed task receipts gives **7,330 new logical UMA EFS labels**: 6,845 child-only
labels and 485 seed-only verification labels; fifteen pool geometries were
already available. The two methods share some queried children, so adding their
individual costs would double count them.

| Low-request cost component | Unique geometries | Fresh labels after the required checkpoint if run alone |
| --- | ---: | ---: |
| Learned children, 1,152 attempts | 1,150 | 1,147 |
| Mutation children, 1,152 attempts | 1,151 | 1,150 |
| Union of both child sets | 2,286 | 2,282 |
| Shared seed verification | 155 | 155 |

The standalone curve counts unique child lookups within each fit/composition
run and excludes seed verification. Actual global fresh labels are deduplicated
once across methods, fits and classes. Old global `new_scoring` flags are not
mistaken for editor-specific calls. The
[cost audit](../reports/conditioning_20261107/editor_low_request_v2/cost_audit.json)
and [query table](../reports/conditioning_20261107/editor_low_request_v2/query_accounting.csv)
bind these distinctions. Allocation time, fitting, startup and idle resources
belong to the final campaign resource receipt, not to a count of unique labels.

## Reusing the static temperature allocator

The improved measured condition response triggered the small allocation follow-up.
It uses the development-selected graph-context checkpoint for fit seed 20261106,
not the optional editor. At each of nLi 12/24/36 and four temperatures
393.15/433.15/473.15/493.15 K, a pure-static schedule and a 10% uniform-over-bins
exploration mixture allocate 32 attempts. This gives 24 schedules and 768 attempt
records from one fitted model. There are 102 distinct query geometries because
identical allocations across temperatures deliberately share actual random
streams; these repetitions are paired controls, not independent evidence.

The reference remains the earlier 24,576 uniform raw fixed-count draws, collapsed
to 24,575 symmetry orbits while retaining raw draw frequencies. Weights use

\[
w_i=f_i\exp[-(E_i-E_{\min})/(k_B T)],
\]

where `E` is **total eV per cell**, `f` is the number of original raw draws, and
`k_B = 8.617333262145e-5 eV/K`. No additional orbit multiplicity is multiplied into
already raw-mask-weighted frequencies. The separate 12-site support audit again
verifies 87 exhaustive orbits and the correct binomial multiplicity totals,
including 924 raw half-filled masks. It checks accounting, not size transfer.

Across all four temperatures, integer allocation is unchanged: **32/0/0**
low/middle/high for pure static requests and **30/1/1** with exploration. The
unrounded static reference mass is overwhelmingly in the low bin; the full
reference's particle ESS is only 1.850–6.544 out of 8,192 raw draws per composition.
Thus only a few sampled configurations dominate the weighted reference. New
temperature metadata or more generated attempts do not repair missing tail
coverage or make this reference thermodynamically converged.

The [requested-versus-realized plot](../reports/conditioning_20261107/temperature_followup/analysis/requested_vs_realized.pdf)
shows the remaining class errors. Pure low requests produce 65.625%, 65.625% and
53.125% measured-low children at nLi 12/24/36. Every temperature produces exactly
the same paired masks and measured distributions at a given composition/policy.
No resolved temperature-dependent generation is observed at this coarse-bin
allocation. Temperature is an outer scheduling parameter; it never changes
the latent prior, decoder probabilities, cell or atom dynamics.

Coarse-bin requests also do not enforce the detailed reference weights *within*
the low class. At 393.15 K the generated pure-static mean is above the static
weighted-reference mean by 0.507/0.691/0.675 eV per cell for nLi 12/24/36. The
[energy distributions](../reports/conditioning_20261107/temperature_followup/analysis/energy_distributions.pdf)
make that distinction visible. These are acquisition weights for unrelaxed
deletion energies, not free energies, calibrated equilibrium populations,
configurational entropy, or a validated hydrothermal reactor response.

The [frozen protocol](../reports/conditioning_20261107/temperature_followup/protocol.json)
includes all registered editor seed/child identities in its history without
reading their energy outcomes. Its
[analysis receipt](../reports/conditioning_20261107/temperature_followup/analysis/receipt.json)
retains every attempt and the exact cross-temperature identity checks. No new
reference pool, MD or DFT was needed.

## A posthoc proposal from the encoded training distribution

The retained dense D4 CVAE uses its latent pathway but controls energy poorly.
This prompted a different question from increasing its latent dimension: does
generation suffer because the standard normal sampling distribution differs
from where the encoder places training examples? The test keeps the trained
networks fixed and estimates one Gaussian proposal for each composition/energy
class from **training rows only**. It changes a sampling law, not the decoder,
training objective, model weights or the physical teacher.

For a group of `N` training masks, the encoder returns means `mu_i` and diagonal
variances `sigma_i^2`. The empirical mixture of their posterior Gaussians has
mean and covariance

\[
\bar\mu=\frac1N\sum_i\mu_i,\qquad
C=\frac1N\sum_i\operatorname{diag}(\sigma_i^2)
 +\frac1N\sum_i(\mu_i-\bar\mu)(\mu_i-\bar\mu)^T.
\]

The first covariance term measures each encoded example's posterior spread;
the second measures variation of the encoded means between examples. This is
the covariance identity for a finite equally weighted mixture, hence denominator
`N`, rather than the `N−1` estimator used for an unknown population. The study
uses a **single Gaussian with those moments**; it does not preserve possible
multiple peaks of the actual mixture. Sampling transforms a standard normal
base vector by `z = mu_bar + L epsilon`, with `L L^T = C` under the recorded
numerical factorization convention. The standard-prior control uses `z=epsilon`.
The executed calculation checks finite positive posterior variances and a
positive-definite full covariance, without shrinkage, added jitter or tuning.

The same base `epsilon` and AR uniforms are paired, but the **actual z differs**
between the two laws and across requested classes. This comparison therefore
does not isolate decoder condition sensitivity at fixed z, and it is not a
newly trained conditional-prior CVAE. The moment calculation performs no neural
optimizer updates. There are 27 groups: three existing fit seeds × three counts
× three energy classes. The preparation encodes 40,157 protected TRAIN rows per
fit; its [receipt](../runs/conditioning_20261107/aggregate_prior/preparation/receipt.json)
records unchanged weights and input hashes.

| Matched assessment | Standard normal prior | Posthoc Gaussian proposal |
| --- | ---: | ---: |
| UMA balanced useful yield | 31.047% | 34.259% |
| MACE balanced useful yield | 30.787% | 34.028% |
| UMA original-q20 low-request yield | 30.122% | 23.958% |
| MACE original-q20 low-request yield | 32.552% | 26.042% |

The new assessment has a separately frozen 312,715-orbit history and 6,912
attempts per teacher. Better balanced class targeting coexists with substantially
worse low-tail yield. The selected graph-context method is retained; this
posthoc change is not recommended as a repair of low-energy discovery.

The [independent audit](../reports/conditioning_20261107/independent_aggregate_prior/receipt.json)
recomputes mask/bin/novelty metrics, verifies actual transformed latents, TRAIN
group membership hashes and covariance/Cholesky identities. It does **not**
independently reproduce every encoded mean/variance from model inference because
those per-row values were not saved. That limitation is narrower than the
verified frozen-network execution/source audit and remains explicit.

## Defined-hop NEB feasibility and a large refinement sensitivity

Two Li hops were recovered from the existing Li36Co48O96 thermal archive, using
the original atom order and continuous host-aligned coordinates:

| Event | Parent Li sites, zero based | Periodic hop image | Ideal site-to-site length |
| --- | --- | --- | ---: |
| basin1/repeat2 | 18 → 31 | (0,0,0) | 2.81126 Å |
| basin2/repeat1 | 11 → 23 | (1,0,0) | 2.81126 Å |

The marked atom and intended occupied-to-vacant event survive fixed-cell endpoint
quenching in both cases. All atoms can move; this is not a rigid one-atom scan.
Endpoint species/order and one common cell are preserved. Independently wrapping
or canonicalizing endpoint structures could erase the marked periodic event,
so those operations are not used to construct the path.

The [prepared protocol](../reports/conditioning_20261107/neb_pilot/prepared_v3/protocol.json)
pins ASE 3.26.0, UMA-S-1.2.1/omat, FP32 with TF32 off, FIRE optimization and the
explicit improved-tangent NEB method. Endpoints target maximum physical atomic
force 0.03 eV/Å. The initial band has five interior images, with IDPP initialization
followed by nonclimbing and climbing-image stages; the climbing projected-force
target is 0.05 eV/Å. One bounded refinement uses seven interior images and a
0.03 eV/Å projected-force target. These numerical choices come from the stated
pilot design, not from a universal physical convergence standard.

For a saved band, the forward span is `E_peak − E_initial`, and the reverse span
is `E_peak − E_final`. They are **eV per marked hop**, not eV per atom. Their
difference equals `E_final − E_initial` for that same chosen peak. A discrete
climbing-image path only approximates a saddle; satisfying a force target does
not by itself establish that the correct minimum-energy path was found.

| UMA path | Forward span (eV/hop) | Reverse span (eV/hop) | Final projected maximum force (eV/Å) |
| --- | ---: | ---: | ---: |
| Event1, five interior images | 0.638221 | 0.682169 | 0.043432 |
| Event1, seven images and tighter target | 0.188711 | 0.232689 | 0.029966 |
| Event2, five interior images | 0.191544 | 0.194652 | 0.048522 |

Both original paths and the event1 refinement report numerical convergence.
Nevertheless, event1's forward span changes by **−0.449510 eV**. Image count and
force target changed together, so this is a combined path/numerical-sensitivity
result, not an attribution to image count alone. Neither event1 value should be
treated as an established transport barrier; event2 has no image-count check.
The [profiles](../reports/conditioning_20261107/neb_pilot/results/profiles.pdf)
show both event1 bands rather than hiding the coarse result. The horizontal
arclength is distance through the full atomic configuration, including host
motion; it is not the marked Li site's 2.81126 Å hop distance.

Physical and NEB force maxima have different meanings. The endpoint criterion
uses each atom's physical force. The band criterion uses projected NEB forces,
which remove or transform the path-tangent component and include the band
constraint. At a climbing image the complete 3N-dimensional force is reflected
along the normalized tangent: `F_CI = F − 2(F·t)t`. Reflection preserves the norm
of that **whole vector**, but not the largest three-component force norm of an
individual atom. Thus the refined event1 physical climbing maximum of 0.03423
eV/Å and projected maximum of 0.02997 eV/Å are not contradictory criteria. Neither
should be described as all physical image forces being below 0.03 eV/Å.

The [MACE disagreement check](../runs/conditioning_20261107/neb/mace_profile/receipt.json)
evaluates **the event1 five-interior UMA band**, not the refined seven-image path.
It performs seven single points and zero optimization steps. Its forward/reverse
profile spans are 0.639771/0.726074 eV per hop. The close coarse forward span does
not validate the much lower refined UMA path. MACE endpoint physical maxima
are about 0.309/0.306 eV/Å, so those UMA endpoints are not MACE-relaxed minima.
This is not an independently optimized MACE barrier.

The [event1 receipt](../runs/conditioning_20261107/neb/basin_1_repeat2/receipt.json)
and [event2 receipt](../runs/conditioning_20261107/neb/basin_2_repeat1/receipt.json)
retain endpoint mappings, raw bands, residuals and any failed stages. There were 3,749
successful counted UMA calculator calls for endpoint/band work and seven MACE
profile calls; those counts are not optimizer steps or instrumented neural
forward passes. Final allocation cost also includes model loading, startup,
cleanup and idle allocated GPUs, and is reported separately.
The [independent saved-band audit](../reports/conditioning_20261107/neb_pilot/independent/receipt.json)
checks unchanged endpoint geometry/IDs between the five/seven-image paths,
recomputes physical/projected forces and energy spans, verifies the exact MACE
source coordinates, and reconciles all 3,756 calculator calls with zero failures.
That numerical audit does not remove the large path sensitivity or establish
physical barrier accuracy.

These are unvalidated MLIP potential-energy results, not finite-temperature
activation free energies, DFT barriers or measured diffusivities. No Hessian or
DFT saddle validation was performed. Geometry candidates for later reference
checks are saved, with no new DFT run or scheduled. Two events cannot establish
a connected transport network or train a barrier-conditioned CVAE. A future
barrier label must include parent, composition, occupation mask, marked occupied
source, empty destination, periodic hop image and geometry/teacher protocol;
an entire mask does not have one unspecified “barrier.”

## Recompute these supplementary results

Run inside a permitted allocation on NERSC, with the learning environment and
current package installed. These analysis commands perform no MLIP inference or
training; give fresh output directories to preserve existing evidence.
The cost audit uses the full shared campaign archive, including its pre-editor
registry, certified cache and completed task files. The small published result
tables alone do not replace those raw provenance inputs.

```bash
python scripts/conditioning_20261107/evaluate_edit_comparison.py \
  --generated runs/conditioning_20261107/editor/assessment_v1 \
  --scores runs/conditioning_20261107/collected/editor/uma.npz \
  --output-dir runs/editor_metric_recompute

python scripts/conditioning_20261107/summarize_editor.py \
  --output-dir runs/editor_low_request_recompute

mkdir runs/temperature_analysis_recompute
cp reports/conditioning_20261107/temperature_followup/protocol.json \
  reports/conditioning_20261107/temperature_followup/history.json \
  runs/temperature_analysis_recompute/
cp -R reports/conditioning_20261107/temperature_followup/generated \
  runs/temperature_analysis_recompute/generated
python scripts/conditioning_20261107/temperature_followup.py analyze \
  --scores runs/conditioning_20261107/collected/editor/uma.npz \
  --output-dir runs/temperature_analysis_recompute
```

The temperature analyzer uses copies of the frozen protocol/history/generated
files; it creates a new analysis rather than overwriting the old one. The main campaign archive and manifests bind
the original source paths and hashes. They are provenance, not portable default
directories for another machine. These outcomes remain MLIP-defined, and a
separate matched no-latent edit control and later physical validation would be
needed to attribute benefits to the variational channel or to real materials.

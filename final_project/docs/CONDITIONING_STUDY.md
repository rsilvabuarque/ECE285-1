# Conditioning, latent variables, and physical-neighbor study

The scientific contract is
[`protocol.json`](../artifacts/conditioning_20261107/protocol.json); completed
versus proposed experiments are distinguished by the integrating
[`experiment_ledger.json`](../reports/conditioning_20261107/experiment_ledger.json).
Historical tables in `RESULTS.md` keep their original thresholds and assessment
histories. This document explains the new implementation and what its comparisons
can establish; it is not a claim that every planned fit has finished.

## Hard constraints and learned requests

The occupancy vector has one binary value per candidate mobile-ion site. One
means the ion is present. The requested count is a hard constraint: during
sequential decoding, the remaining occupied-site count forces a zero or one
when no other valid choice is possible. All sampled masks must pass binary and
count checks independently of their energy.

Energy conditions are statistical requests. The encoder and every AR decoder
step receive composition and an energy-class vector. The reconstruction loss
rewards training-mask likelihood; it does not query the MLIP or directly
minimize its energy. A requested class therefore differs from a measured class.
Composition-only generation can already prefer low energies because the training
distribution is proposal biased; it is not a uniform baseline.

The three supported class encodings remain distinct:

| Family | Condition | Limitation |
| --- | --- | --- |
| Composition CVAE | Li fraction | No energy request |
| Binary energy CVAE | Li fraction, low-energy bit | Middle and high both map to the rest class |
| Multibin CVAE | Li fraction, one-hot low/middle/high | Must demonstrate three distinct measured responses |

For three exhaustive bins, a generator that ignores the requested bin has
average hit rate `(P(low)+P(middle)+P(high))/3 = 1/3`, regardless of its overall
energy bias. Novelty and duplicate filtering can reduce useful yield further.
Thus low-energy enrichment and balanced three-bin control are different claims.

Per-composition q20/q80 edges use training energies only. Ties remain together,
and absent joint energy/descriptor support is explicit. Changing an evaluation
threshold after seeing candidates would change the question; fixed historical
q20 and new requested-bin metrics are kept separate.

## What the latent pathway contributes

An encoder maps a complete observed mask and condition to Gaussian means and
variances. The decoder receives a latent draw plus the requested condition.
Prior generation samples a standard normal without observing a target mask.
Posterior reconstruction does observe the target through the encoder, so its
quality is not a direct measure of useful novel generation.

The objective is summed count-constrained negative log likelihood plus beta times
the Gaussian KL divergence. The KL sum is over latent dimensions; changing the
dimension does not silently divide that penalty by dimension. The encoder
predicts total and per-component KL and the variance of its means across
validation configurations. A component is descriptively called active when that
mean variance exceeds a declared cutoff; the implementation reports 0.001,0.01,
and 0.1 together. These dimensionless heuristics have no physical convergence
meaning. Small KL, weak conditional response and weak latent response are
different observations, not interchangeable proofs of posterior collapse.

The supported sizes 4, 8, 16, 32 form a sensitivity comparison, not a sequence
converging toward exact physics. The matched no-latent AR control has no Gaussian
encoder or KL and is accurately named an AR baseline. The smallest tested
dimension with competitive development performance is preferred over choosing
the largest size by default.

Training policy is also controlled. Class balancing changes train-row weights
within each composition and nonempty energy class, preserving its total base
weight. Validation uses its fixed inherited weights. Both reconstruction and KL
are reported consistently. Each full trajectory saves checkpoints selected by
validation reconstruction plus 1 or 0.1 times KL; this is one trajectory with two
selection rules, not two independently trained models. Fixed validation noise is
independent of training RNG. The full update cap and stopping reason are saved.

### Measured latent-size development result

The retained policy is balancing plus validation selection weight 0.1. At that
policy, the D4/D8/D16/D32 CVAE balanced yields are 32.64/29.98/28.82/32.47%, while
original-q20 low-request yields are 33.16/29.34/27.43/31.42%. The dimension 8 fit
is reused from policy selection and evaluated under a new development namespace;
its changed Monte Carlo result does not indicate another training update.
The common known-history boundary remains 239,867 orbits.

D4 has 30,793 trainable parameters, D8 has 32,337, D16 has 35,425 and D32 has 41,601.
The no-latent AR control has 14,209 parameters and mean balanced/low yield 34.32/22.05%.
Uniform gives 33.04/19.62% and frozen mutation 59.72/52.08%. Thus the no-latent
control's balanced mean is higher than D4's, while the genuine CVAE is better
on low-request yield; mutation remains stronger on both. Report both objectives.

The predeclared practical selection rule chooses the smallest dimension within
one percentage point of the best balanced mean and two points of the best
low-request mean. These are provisional decision heuristics, not confidence
intervals or physical tolerances. D4 is selected; no monotonic size trend
justifies a dimension 64 extension. Individual fit/sampler replicate values are
saved in `reports/conditioning_20261107/latent_development/seed_values.csv`.
This is development selection, not a fresh final assessment.

Across the selected validation checkpoints, D4 reconstruction spans 16.77–17.34
nats versus 14.81–15.00 for D32; total KL spans 11.58–11.95 versus 17.64–18.20.
At encoded-mean variance cutoff 0.01, active counts are 3–4 of 4 for D4 and 26–31
of 32 for D32. At cutoff 0.1 they are 3 and 9–12, respectively; at 0.001 all components
are counted active. Thus activity depends on the declared numerical diagnostic,
and improved reconstruction is not monotonic improvement in discovery yield.
Full per-dimension values, all seeds, curves and
[latent-size summary](../reports/conditioning_20261107/latent_diagnostics/latent_dimension_summary.pdf)
are supplied with the
[independent selection audit](../reports/conditioning_20261107/latent_diagnostics/latent_selection_audit.json).

## Physical graph and information available during generation

`ParentSpec` stores species, parent coordinates/cell in Å, periodic axes,
candidate mobile indices, occupancy convention and validated parent symmetries.
The graph retains both host atoms and all candidate mobile sites, including
vacancies. It stores directed periodic neighbor edges with source/destination
indices, integer image shifts and distances. A nonzero-shift self-image is valid
when the graph rule finds it; an ordinary diagonal self-edge is not substituted.

The current LCO construction uses a provisional 5 Å cutoff. It gives 192 nodes,
14,784 directed full-parent image edges and 864 mobile-site image edges. Both the
host-inclusive graph and mobile subgraph are connected. Mobile distances are
2.81126, 4.86924, 4.91237 Å; the last shell provides 288 directed cross-plane edges,
so the model is not confined to isolated Li planes. These are geometric edges,
not verified low-barrier diffusion pathways. All settings and shell grouping
tolerances are in the parent-graph receipt.

The learned graph encoder uses elemental identity, host/mobile flags,
occupation and radial-neighbor information. Two distance-weighted message-passing
steps combine features across actual periodic neighbors. Separate mobile/host
pooling supplies Gaussian latent parameters. Reconstruction and KL gradients
propagate back through this computation, training feature combinations useful
for the CVAE objective. This is not force-field training or an energy gradient.
The encoder sees the complete training mask, but its occupation-dependent
features reach generation only through the latent variable.

The generation-context comparison keeps the learned graph encoder and adds
causal decoder summaries. Its matched control is graph-encoder-only with the same
selected dimension/policy and three fit seeds. The first positive development
pilot was followed by two identically configured fits under a documented
pre-final amendment, bringing the required core to 28 fits. This is not a
dense-encoder-plus-features comparison.

The cheaper generation-context variant appends six fixed graph-derived features
to the GRU: occupied, vacant and unassigned neighbor fractions under a nearest-
shell weighting and a distance-decay weighting. At site j, only sites i<j may be
assigned. Every future site remains unassigned even in teacher-forced training.
This prevents a target-mask shortcut unavailable during real generation.
These engineered decoder summaries are not a learned graph decoder. A graph
encoder plus an indexed GRU is not automatically fully equivariant; the parent
index frame still associates outputs with sites.

### Measured spatial development result and its latent limitation

The three-fit context model gives 60.88% balanced UMA useful yield and 61.11%
original-q20 low-request yield, versus graph encoder only 34.49/24.31%, dense
31.83/29.69%, symmetry augmentation 32.12/29.34%, uniform 32.75/19.79% and frozen
mutation 58.80/50.35%. All are development-only, with their individual seeds and
denominators in [the result tables](RESULTS.md). The separately frozen final
assessment gives context 62.04% UMA/63.45% MACE balanced yield, versus mutation
58.62/57.29%. Its final known-history boundary is 278,382 orbits, rather than the
development history. There are 34,560 UMA final attempts and a matched 20,736
MACE subset; the MACE audit measures the same proposals against its own energy
boundaries. It is not DFT validation or MACE-specific retraining.
Encoder-only/context parameter counts are 36,669/37,821.

Shared-latent/shared-uniform request pairs give high-minus-low energy shifts of
+5.263 meV/atom for context, positive in 97.40% of pairs, versus +0.026 meV/atom for
dense D4. This is evidence of learned conditional response. However, the context
model's validation KL is only 0.007–0.011 nats. Changing the latent at fixed
uniforms changes only 0.795 sites, versus 12.72 for dense D4; 68.23% of context
pairs remain in the same orbit. Posterior/prior target distances are nearly equal.
No encoded-mean component crosses even the 0.001 descriptive variance cutoff.
Taken together, these checks indicate a nearly unused latent pathway. The model
can still generate diverse samples through its sequential random uniforms.
Better sampling therefore supports the causal spatial AR mechanism, rather
than proving that the variational component helps. An otherwise matched
no-latent spatial decoder would isolate that remaining question.

The [actual-checkpoint causality audit](../reports/conditioning_20261107/trained_causality_audit.json)
passes for all three context fits: future-target gradients are exactly zero;
permuting future bits leaves current features/logits unchanged; full-mask encoder
calls are forbidden. Teacher-forced and one-step feature arithmetic agree within
recorded FP32 rounding bounds. This is an implementation check against future
information leakage, not a physical validation or proof of full equivariance.

Canonical versus augmented training changes orientation policy separately from
graph structure. Symmetry augmentation is applied only after orbit-based splits.
An augmentation transformation must preserve the fixed host chemistry and act
consistently on the physical features. No assessment identity becomes a new
training example merely because it is reoriented.

## Structural condition and energy target

The nearest-shell vacancy-pair fraction is

\[
d(o)=\frac{\sum_{(i,j)\in\mathcal P}(1-o_i)(1-o_j)}{|\mathcal P|},
\]

where the set contains each eligible undirected periodic-image pair once. For
this parent there are 144 pairs at 2.81126 Å. The descriptor is dimensionless; its
pair kernel was checked under all 192 retained parent permutations. Training
descriptor boundaries and energy/descriptor support are saved. The descriptor
is strongly associated with energy in the present corpus, so its useful role
may be control of structural variety within a middle-energy class, rather than
an independent second physical objective. It must not be presented as a known
rule that more or fewer vacancy pairs always improves a real cathode.

The one-fit middle-energy pilot gives 28.13% novel joint-request yield, versus
28.52% for plain CVAE and 44.79% matched mutation. Its 2304 attempts are all kept.
This negative pilot does not establish an advantage for the extra condition,
and one fit cannot estimate fit-to-fit variation. Joint requests are defined
from training-only thresholds; no rare cell is silently filled by relabeling.

The separate [paired energy audit](../reports/conditioning_20261107/relaxed_target_audit/README.md)
uses exactly matched starting-mask and fixed-cell endpoint labels. All 1000
attempts converged in the recorded historical protocol, but rankings are not
identical. The primary counts have only 49 train/11 validation/9 test independent
lineages, with endpoint-low validation support 0/1/0. A label-target model fit is
therefore deferred rather than inflating frame count or changing protected roles.
Even a future model conditioned on endpoint energy would generate an initial
mask, not automatically a relaxed coordinate minimum.

## Frozen mutation, learned edits, and interpretation

The baseline chooses a training seed uniformly from the requested bin, a swap
count uniformly from 1, 2, 3, then distinct occupied removals and vacant insertions
uniformly. It has no distance filter, energy acceptance rule or online population
update. Its proposals are frozen before scoring. Any learned editor must use
the same seeds, swap counts, attempts and score budget, and record actual edited
sites. Exactly k swaps change 2k bits in the common seed frame. Independently
canonicalizing seed and child would destroy that alignment; transformations
must be joint. A copied or symmetry-equivalent known mask is not a new discovery.

A descriptor request does not entitle mutation to a different seed pool than
the CVAE/edit comparison without declaring the changed information. Rejected
draws, duplicates, failed scores, cache hits and new score calls remain visible.
The final history and method list must be frozen before final generation. Report
individual fit seeds and descriptive variation; latent draws are not independent
training replications.

The new parent/graph interfaces are not hard-coded to 48 mobile sites. The current occupancy trainer still canonicalizes integer vacancy codes only for at most 62 sites; larger-parent training needs a separate scalable canonicalization change before claiming that support.

The study tests a conditional proposal distribution under declared teachers.
It does not measure configurational entropy, prove ground states, validate DFT
accuracy, or establish transfer to another size/material. The reusable parent
schema is software infrastructure; another chemistry still needs its own
physical data and validation. A later NEB label must refer to a defined hop,
not a single undefined barrier attached to an entire occupation mask.

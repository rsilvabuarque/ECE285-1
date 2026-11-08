The new controlled study is explained in [CONDITIONING_STUDY.md](CONDITIONING_STUDY.md).
Its protocol and actual ledger distinguish completed experiments from planned
extensions. The material below preserves the November 7 method/data definitions;
its reported fit policy is a historical reference, not a silent default for the
new latent/graph comparisons.

# Current methods and data

The November 7 extension keeps the existing energy-conditioned autoregressive
CVAE and adds three requested energy ranges plus a separate temperature-dependent
acquisition layer. The frozen November 6 binary comparison remains historical
evidence. New CVAE training and bounded MLIP evaluation are authorized for this
extension; new DFT and a new MLIP fine-tuning campaign are not.

The core question is whether generation can be controlled at a requested
composition and teacher-defined energy range while producing useful distinct
structures per evaluation budget. MD, thermal free energies, reactor equilibrium,
and downstream potential improvement are separate research questions.

## Physical tools and what one row means

DFT computes an electronic-structure reference for specified species,
coordinates and cell under a numerical/magnetic protocol. An MLIP predicts an
energy and its force/cell derivatives much more cheaply. MD propagates atoms
using those forces and can estimate thermal observables after suitable sampling
checks. A single-point MLIP call keeps the coordinates fixed; relaxation
iteratively moves them toward a local minimum.

The main pristine parent is Li48Co48O96, with 48 removable sites and 144 host
atoms. Li rows are zero-based indices 48–95 in the full parent, not its first
48 rows. Its SHA256 is
`ae8f8bc8d7e767165b0172b396e3c723b15e09f4b79429ac01b7c78fd4741e95`.
Read `li_indices` (or generic `target_indices`) from the parent NPZ.
For n retained Li, x = n/48 and the number of present atoms is Nₐ = 144+n.
Occupancy oᵢ=1 means present Li; vacancy vᵢ=1−oᵢ means removed.

A mask row describes **pristine-derived deletion coordinates**. It does not move
the other atoms and is not a relaxed minimum. Do not substitute the energy of a
relaxed or thermally displaced structure with the same vacancy mask. Teacher,
head, precision, parent, composition, geometry protocol, and coordinates define
the label identity. The main new comparison fixes n=12,24,36 and uses pinned
UMA-S-1.2.1/`omat`, with MACE-MH-1/`omat_pbe` for robustness. Both use the campaign's
explicit FP32 settings and TF32 off. Their agreement is not independent DFT
validation.

## Symmetry and independent units

The supplied parent has 192 retained site permutations. An orbit is the set of
physical occupation masks related by this group. Its canonical key is the
lexicographically smallest transformed **vacancy** string. Orbit grouping occurs
before splitting and before any training augmentation. A lineage and all its
optimizer frames/teacher labels stay together; different frames are not
independent configurations. Identical mask keys in different supercells are not
automatically equivalent.

An orbit multiplicity gᵢ is the number of different raw occupancy masks in orbit i.
For a finite group G it equals |G| divided by the number of group operations
that leave the mask fixed. It is not a permutation count for atom labels. The
48-site half-lithiation parent has 32,247,603,683,100 raw masks and
168,025,313,938 symmetry orbits under this group. The saved capacity was checked
independently with Burnside cycle polynomials. The smaller 12-site reference
contains 87 orbits across all counts, including 17 at n=6 covering 924 raw masks.
It is an exact numerical accounting check, not evidence for size transfer.

Canonical mask identities do not by themselves rotate coordinates or vector
labels. Physical scoring uses canonical deletion geometries where justified;
forces belong to the evaluated orientation. The older geometry archive retains
a fixed parent atom-index frame and its input orientations, rather than rotating
whole geometries into canonical-mask orientations.

## Frozen data and adaptation history

| Learning array | Masks | Train / validation / test | Meaning |
| --- | ---: | --- | --- |
| `training_original.npz` | 1,536 | 1,227 / 153 / 156 | Original n=12/24/36 subset |
| `training_round0.npz` | 50,247 | 40,157 / 4,967 / 5,123 | Original plus earlier evaluated proposals |
| `training_round1.npz` | 75,247 | 60,152 / 7,493 / 7,602 | Round 0 plus 25,000 development masks |

These files and checkpoints remain immutable in `artifacts/current/`. New fits
use new output directories. Original splits are protected; added orbit keys
received deterministic hash-based splits. A previous final pool promoted into
training becomes development evidence. The new assessment must freeze its
history before sampling and keep allocation-reference masks distinct from its
assessment pool. Novelty to the training set and novelty to all prior history
are different quantities and must both be reported.

The expanded datasets were produced by earlier random/generative/mutation
proposals. Their energy histograms are proposal-biased and cannot be used as a
density of states. The new acquisition reference is separate.

## Shared energy-range condition schema

`cathode-energy-bins-v1` is shared by training, checkpoint loading, generation,
acquisition and evaluation. Let eᵢ=Eᵢ/Nₐ be a teacher's energy in eV/atom. At each
composition, sort only the m training energies. The nearest-rank boundary at
fraction p is the element with one-based index ceil(pm). The default boundaries
are q20 at p=0.2 and q80 at p=0.8:

$$
 b(e)=\begin{cases}
 \mathrm{low},&e\le q_{20},\\
 \mathrm{middle},&q_{20}<e\le q_{80},\\
 \mathrm{high},&e>q_{80}.
 \end{cases}
$$

Equal energies are never split to make a desired class size. The schema saves
actual counts, supported bins, sparse compositions, and `merged_bins` (empty when
no merge occurred). An empty bin is an unsupported request, not an excuse to
silently redirect it or repeatedly draw until it succeeds. One-structure
compositions need not have three classes.

The multibin condition is c=[x,onehot(low,middle,high)], dimension four. The
encoder and decoder both receive it. The historical binary control still uses
[x,is_low], dimension two, while the composition-only control uses [x], dimension
one. Binary middle/high requests map explicitly to its rest class; the control
cannot claim to distinguish them. Existing checkpoints remain loadable; a binary
checkpoint is not silently interpreted as a four-component model.

A saved schema records energy units, quantiles/edges, tie policy, empirical
support, teacher/head/checkpoint identity, parent and geometry protocol, source
and protected-split hashes, and a content identity. `--condition-schema-in`
freezes a partition across runs; changing it is an explicit experimental change.
Class balancing, when used, changes training weights only and must be recorded.
The new controlled comparison preserves training data and optimization choices;
a difference between complete methods is not automatically a causal isolation
of every architecture or schedule effect.

## CVAE and exact-count autoregressive decoder

The encoder qφ(z|o,c) predicts an eight-dimensional Gaussian's mean and diagonal
variance from an observed occupancy and condition. A random latent vector z and
condition c enter a GRU decoder (hidden size 64). It predicts each occupancy from
the preceding bits through

$$
p_\theta(o\mid z,c)=\prod_{j=1}^{48}p_\theta(o_j\mid o_{<j},z,c).
$$

The subscript j indexes a parent site; this sequence is not physical time. During
training, observed earlier bits are provided (teacher forcing). During sampling,
the model feeds back its own draws. When the requested remaining count forces a
zero or one, that choice has probability one and contributes zero negative log
likelihood. Every valid draw therefore has exactly n Li. A fixed 48-site network
is not a 12-site model, equivariant graph model, or cross-cathode model.

Training minimizes

$$
\mathcal L_\beta=\mathbb E_{q_\phi}[-\log p_\theta(o\mid z,c)]
+\beta D_{\rm KL}\bigl(q_\phi(z\mid o,c)\Vert\mathcal N(0,I)\bigr).
$$

The first term penalizes poor reconstruction probability; KL compares the encoded
Gaussian with the standard-normal prior used for new samples. At β=1 this is a
negative conditional evidence lower bound for the chosen likelihood; β=0.1 is a
different regularized objective. Neither a better posterior reconstruction nor a
larger KL alone establishes better prior generation. See
[Kingma and Welling, §2](https://arxiv.org/abs/1312.6114),
[Sohn et al., §4](https://proceedings.neurips.cc/paper_files/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf),
and [Wu et al., Eq. 4](https://doi.org/10.1103/PhysRevLett.122.080602) for the
related chain-rule construction.

Reference optimizer settings and actual campaign deviations belong in each run
configuration. Keep fit seeds distinct from sampling seeds and latent draws.
Checkpoint selection uses development/validation evidence; the assessment pool
cannot select a new model and remain untouched evidence for it.

## Temperature-dependent acquisition

Physical temperature T in kelvin controls an outer request policy, not the
latent normal prior or a decoder softmax-temperature parameter. The project grid
is 393.15, 433.15, 473.15, 493.15 K, confirmed against reactor target manifests.
No pressure term or calibrated hydrothermal population is assumed.

For a complete list of symmetry representatives at one parent/composition,

$$
a_i(T)=\exp[-(E_i-E_{\rm ref})/(k_B T)],\quad
w_i=g_i a_i,\quad W_b(T)=\frac{\sum_{i:b_i=b}w_i}{\sum_i w_i}.
$$

Here i indexes an orbit representative; b is low, middle or high; gᵢ counts raw
occupancy masks in its orbit; Eᵢ and the common reference Eref are total eV per
cell; kᴮ=8.617333262145×10⁻⁵ eV/K; aᵢ and wᵢ are dimensionless; Wᵦ is the bin's
allocation fraction. The implementation uses stable logarithmic normalization.
Stored eV/atom values are multiplied by the actual Nₐ before weighting. For a
168-atom cell, 1 meV/atom is 0.168 eV/cell, not 0.001 eV/cell.

For a prespecified IID **uniform raw-mask sample**, collapse repeated orbit
identities for scoring but retain their raw draw frequencies fᵢ. Use wᵢ=fᵢaᵢ;
do not multiply by gᵢ again. Every reference geometry must be scored or covered
by a justified exact-geometry cache. Dropping cache misses or failed scores
would bias the allocation measure. The exhaustive 12-site audit verifies each
composition's multiplicities sum to binomial(12,n), including all 924 masks at
n=6. It does not validate large-cell coverage.

Report representative ESS=1/Σᵢpᵢ² for collapsed pᵢ=wᵢ/Z, where Z=Σᵢwᵢ. Also
report particle/microstate ESS=Z²/Σᵢ(mᵢaᵢ²), with mᵢ=fᵢ for sampled raw draws or
gᵢ for complete support. These answer different concentration questions; neither
is a confidence interval. Include largest masses and nested-prefix pool-size
sensitivity. A concentrated finite sample can miss rare favorable states.

Save both pure Wᵦ and Pᵦ=(1−ε)Wᵦ+ε/K, where K is the number of supported bins
and ε=0.1 is the chosen exploration floor. Temperature-mixture weights αₜ are
explicit algorithmic priorities. Largest-remainder integer allocation preserves
exact attempt totals. A per-temperature budget and a joint mixture budget are
different plans. The generator executes every schedule in its allocation file;
account for all of their attempts. Correct request proportions do not
prove the CVAE actually generates those energy bins.

## Evaluation and limits

Retain requested bin, effective condition, physical T, allocation policy, schema,
teacher, fit seed, sampling seed, checkpoint and status for every attempt. Keep
duplicates and failed requests. Score one actual geometry/protocol/teacher identity
once and join scalar labels only where justified; do not rotate force vectors by
relabeling a key. New evaluations remain separated by these conditions rather
than being pooled into the historical method/seed/count groups.

Measure exact count/structure validity, orbit uniqueness, novelty to training and
to frozen history, requested-versus-measured bin confusion, in-bin hits and energy
quantiles, unique novel in-bin yield per **all** attempts, fixed-original-q20 yield on the
requested-low stratum,
best/useful energies versus attempt/oracle budget, layer-filling descriptors, reference
concentration, cache coverage, failures, and total label/training/evaluation costs.
Uniform fixed-count sampling and bin-seeded composition-preserving mutation use
matched attempt budgets. Both new mutation controls use the same protected
round-0 training information as the CVAEs; one selects seeds from the requested
bin and the other always from the low bin. Both choose a training seed uniformly
and apply one to three uniformly selected occupied/vacant-site swaps.
Filtered random methods must count rejected draws and
all score calls. In the fully enumerated 12-site reference, novelty to all 87
known orbits is impossible; any novelty must refer to a protected training subset.

The [matched DFT ranking context](../reports/ranking_audit_20261107/README.md) shows
both useful rank agreement and counterexamples. A constant offset cancels in
relative weights; gap distortions can preserve ranks while changing weights.
Accurate ranks do not establish accurate forces, phonons or thermal free energy.
The current result is **temperature-dependent acquisition under a static MLIP
surrogate**, not a basin free-energy calculation or measured configurational
entropy. DFT accuracy of new proposals, equilibrium populations, reactor optima,
and downstream MLIP improvements remain unestablished.

## Secondary geometry result

The existing fixed-cell UMA displacement dataset contains 6,000 frames from
1,000 mask lineages split 787/107/106. The separate CVAE predicts padded atomic
displacements from occupancy, supplied cell/cohort, and training-standardized
energy/force/stress scalars; its latent dimension is 16. Missing atoms are masked.
The desired endpoint tuple comes from attained training endpoints, not held-out
oracle properties. A deterministic regressor uses the same context without a
latent variable. Direct outputs are unminimized coordinate predictions. The
historical matched optimizer saving was similar for both models; no new geometry
campaign is part of the core extension. Full evidence remains in
`artifacts/current/` and [RESULTS.md](RESULTS.md).

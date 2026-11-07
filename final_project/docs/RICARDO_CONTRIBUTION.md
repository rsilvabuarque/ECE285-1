# Ricardo's prepared midterm contribution

This text supplies the physical motivation, data provenance, symmetry definition,
and limits for the revised report. It is integrated into `midterm1/main.tex`.
Beihao's and Varad's independent checks remain explicitly assigned in
`COLLABORATOR_TASKS.md`; preparing their commands does not complete their work.

## Physical question and computational tools

Direct recycling seeks to recover a cathode's useful composition and structure.
For example, Shi, Chen and Chen demonstrated hydrothermal treatment followed by
annealing of degraded LiCoO₂ ([Green Chemistry 20, 851 (2018), DOI
10.1039/C7GC02831H](https://doi.org/10.1039/C7GC02831H), publisher abstract).
Our longer-term motivation is to supply informative structures for reference
calculations and more reliable models of lithium replenishment. The present
class experiment does not model the liquid, reaction pathway, or reactor.

Three computational operations play different roles. Density functional theory
(DFT) solves an approximate electronic-structure problem for specified atoms and
cell. It supplies reference energy, forces, and stress under a stated functional,
magnetic treatment, and numerical protocol. A machine-learned interatomic
potential (MLIP) approximates the energy function and supplies its position and
cell derivatives much more cheaply. Molecular dynamics (MD) uses a force model
to propagate atomic motion; after appropriate equilibration and uncertainty
checks, trajectories can estimate thermal observables. A static MLIP call leaves
coordinates fixed; a geometry relaxation repeatedly moves them toward a local
minimum. None of these operations is performed by drawing an occupancy mask.

In LiₓCoO₂, the dimensionless composition x is the number of Li atoms per Co.
The O3 host is a particular layered stacking family with possible Li positions
between Co–O slabs. Removing Li creates a discrete choice of which sites remain
occupied. These arrangements can have different energies at the same x; ordering
and competing host structures have long been studied in first-principles LCO
work ([Van der Ven et al., Physical Review B 58, 2975 (1998), §§I–III](https://doi.org/10.1103/PhysRevB.58.2975)).
Our generator keeps one O3 parent and cannot establish the most stable phase
among O3, O1, spinel, or reconstructed hosts.

## Revised project objective

Develop and evaluate a conditional variational autoencoder (CVAE) that proposes
novel, valid Li/vacancy occupation patterns at requested Li counts and
composition-specific energy ranges. Then demonstrate how a separate
**temperature-dependent acquisition policy** allocates generation requests among
those ranges under static MLIP energy weights. The classifier condition changes
which distribution the model is asked to learn; it does not enforce an energy
inequality, minimize an energy gradient, or compute entropy.

The teacher benchmark is useful even when physical reference accuracy is
incomplete. Large foundation datasets provide broad equilibrium and displaced
structures, but do not guarantee coverage of the vacancy orderings relevant to
our system. OMat24 explicitly discusses defect/nonstoichiometry and magnetic
reference limitations ([Barroso-Luque et al., arXiv:2410.12771v2,
§2.1.2](https://arxiv.org/pdf/2410.12771)). This motivates a specific coverage
question; it does not establish that every foundation potential saw only
pristine LCO or fails monotonically as Li is removed.

## Parent, structures, and independent units

The main pristine parent is Li₄₈Co₄₈O₉₆. Its full atom order contains the 48 Li
rows at indices 48–95 (zero based), rather than at the start. Its cell vectors,
in Å, are

```text
( 5.6225139536, 0,            0)
(-5.6225139536, 9.7384798339, 0)
( 0,            0,           27.8189128651)
```

The parent file is `artifacts/current/core/data/parents/O3_N48_pilot.npz`,
SHA256 `ae8f8bc8d7e767165b0172b396e3c723b15e09f4b79429ac01b7c78fd4741e95`.
Occupancy oᵢ = 1 means Li site i is occupied; vacancy vᵢ = 1 − oᵢ means removed.
Retaining n Li gives LiₙCo₄₈O₉₆ with Nₐ = 144 + n present atoms. The controlled
main comparison retains n = 12, 24, 36, corresponding to x = 0.25, 0.50, 0.75.
The wider existing corpus covers every integer count 0–48, although its endpoint
and pilot-holdout rows do not all support model fitting.

A symmetry operation maps the periodic parent to itself and permutes its
possible Li sites. The 192 retained site permutations define a finite action.
The *orbit* of a mask is the set of different physical occupancy patterns reached
by those permutations. We identify an orbit by the lexicographically smallest
transformed vacancy string. This key groups aliases before splitting and counts
repeated proposals once in a uniqueness numerator. Every draw remains an
attempt in the denominator. Its multiplicity g is the number of different raw
occupancy patterns in that orbit, not a factorial count of identical atom labels.

At half lithiation there are exactly 32,247,603,683,100 raw 48-site masks and
168,025,313,938 orbits under this declared group. The latter was independently
recomputed by Burnside cycle polynomials and matched the stored capacity. The
count belongs to this parent and group, not every 48-site supercell shape.
The complete 12-site reference has 87 orbits across all counts, including 17
orbits representing 924 raw masks at n = 6. It checks finite-support accounting,
not large-data learning performance or size transfer.

Two supplied [Li₂₄ examples](../reports/ranking_audit_20261107/examples/manifest.json)
have the same parent and composition but different occupied sites and teacher
energies. They are existing training deletion structures, with POSCAR/CIF files;
they are neither new CVAE discoveries nor relaxed minima. The
[Li-layer diagram](../reports/ranking_audit_20261107/occupation_examples.pdf)
uses indexed sites within each plane, not an x–y geometric projection.

## What the existing DFT ranking evidence establishes

The new audit recomputes selection diagnostics from 34,368 saved matched-geometry
rows: 11,456 exact geometries evaluated by UMA, MACE, and M3GNet. The historical
source contains 524 O3 rattled geometries; Hu contributes 10,932 unique frames
across phases and cohorts. Each comparison pool fixes source, phase, composition,
atom count, and cohort. The archived DFT setups differ, so absolute energies are
not merged. This older audit used FP64/general inference, whereas the current
48-site deletion benchmark uses the separately pinned FP32 teachers.

For historical O3 rattled pools, UMA's median Spearman rank correlation is
0.97366 and median low-quintile precision is 96.30%. For Hu O3 optimized pools,
those values are 0.73177 and 41.67%, respectively, across ten informative
composition pools containing 104 frames. The optimized subset is a much more
relevant warning than an aggregate dominated by strongly displaced frames.
At Li₁₀Co₁₂O₂₄, the complete optimized pool contains five structures. UMA ranks
`structure_4159` first while DFT ranks it fifth; its DFT selection regret is
7.9711 meV/atom. MACE's selected minimum has regret 1.7435 meV/atom. Matching the
same five identities in a five-member pool is not evidence of top-five recall
from a larger population.

Spearman correlation measures agreement of ordering. Top-k precision asks how
many of the MLIP-selected members also belong to the DFT-selected set; exact
boundary ties expand both sets and their sizes are recorded. Selection regret
measures the DFT energy penalty relative to the best DFT member of that complete
evaluated pool. Pair-gap RMSE measures errors in energy differences and cancels
any common offset. Their definitions, every pool, top-k sizes, source hashes,
and 432 independent matches to the archived metrics are in
`reports/ranking_audit_20261107/`. Frames from a trajectory remain correlated;
no independent-frame p-values or ground-state claims are made.

## Temperature weighting and physical limits

At a fixed composition, a constant teacher offset cancels from relative weights.
Gap distortion does not: even perfect ranks can give wrong relative weights.
At 433.15 K, kᴮT ≈ 0.03733 eV. In a 168-atom Li₂₄ cell, an illustrative
1 meV/atom energy-gap change is 0.168 eV per cell and changes an equal-multiplicity
weight ratio to approximately 0.0111. Using 0.001 eV directly would incorrectly
give approximately 0.974. Whole-cell weights must use whole-cell energies.

For complete symmetry representatives, wᵢ = gᵢ exp[−(Eᵢ−Eref)/(kᴮT)] and the
bin allocation is Wᵦ = Σᵢ∈ᵦ wᵢ / Σᵢ wᵢ. Eᵢ is total eV per cell, Eref a common
reference, T is kelvin, and kᴮ = 8.617333262145×10⁻⁵ eV/K. For a uniformly sampled
raw-mask reference, its retained sampling frequency replaces gᵢ; multiplying by
both would count degeneracy twice. The proposal-biased training histogram is
not a density of states. A finite independently sampled reference can miss rare
favorable states, so concentration, effective sample size, and pool-size
sensitivity must accompany its weights.

The confirmed reactor target temperatures are 393.15, 433.15, 473.15, and
493.15 K. They motivate the acquisition grid; they do not calibrate hydrothermal
populations. The exploration version assigns 10% uniformly over supported bins
and 90% by the static weights. That floor is an algorithmic choice, not a
thermodynamic constant. The actual generated energies must still be evaluated:
correct request totals do not prove successful conditional generation.

Deletion energies omit relaxed-basin and thermal free energies. The current
results cannot establish equilibrium occupation probabilities, configurational
entropy, reactor optima, DFT accuracy of new proposals, or downstream MLIP
improvement. The older geometry CVAE is secondary: its learned starts helped an
optimizer, but the deterministic regressor gave similar savings. New DFT is not
authorized in this run; a later matched reference set is required before claiming
physical accuracy beyond the archived audit.

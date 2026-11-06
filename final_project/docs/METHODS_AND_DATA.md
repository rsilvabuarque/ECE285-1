# What the models learn

## The two prediction problems

The vacancy model chooses **which sites contain Li** in a fixed parent crystal.
The geometry model chooses **where the present atoms move**, conditional on a
given vacancy pattern and cell. These are separate trained networks in the
current experiment. A useful factorization is

$$
p(o,u\mid c)=p(o\mid c)\,p(u\mid o,c),
$$

where $o$ is the occupancy vector, $u$ is the displacement array, and
$c$ contains requested properties and other supplied context. This expression
describes a possible two-stage workflow; it does not mean the current networks
were trained end to end or that either stage guarantees a physical optimum.

For the main parent, pristine Li₄₈Co₄₈O₉₆ has 192 atoms and 48 possible Li
sites. A sample with $n$ Li atoms contains $144+n$ atoms. Occupancy
$o_i=1$ means site $i$ contains Li; $o_i=0$ means it is vacant. Raw
vacancy masks use the opposite convention, $v_i=1-o_i$. Li atoms occupy parent
rows 48–95; always read `li_indices` rather than assuming the first rows are Li.

## Symmetry and independent data

Many binary vectors describe the same periodic configuration viewed through a
translation or point-symmetry operation of the parent. Their set is a symmetry
**orbit**. The main parent supplies 192 retained site permutations. Applying
each permutation to a vacancy mask and taking the smallest binary string gives
its canonical key. One key identifies one orbit within this parent and cell.

Canonicalization precedes split and novelty decisions. All trajectory frames,
teacher labels, and geometric views of one mask lineage stay in the same split.
Symmetry augmentation, when used, happens only inside training after that
grouping. The improved mask fits use the canonical orientation. Do not count
augmented views as independent configurations or rotate force vectors merely
by changing an orbit key: vectors require the actual atom mapping and spatial
rotation. Canonical keys do not by themselves remove aliases between different
supercell sizes or parents.

## Energy conditions

An MLIP is a neural model of the potential energy $E(R,h)$, where $R$
contains atomic positions and $h$ the cell vectors. It also provides forces
and stress related to energy derivatives. Here an MLIP evaluates a deletion
geometry: removing Li leaves the other atoms at pristine positions. The mask
model does not train on forces or stresses and does not learn their derivatives.

For a fixed Li count, define energy per atom

$$
e=\frac{E}{144+n}\quad\text{(eV/atom)}.
$$

Only **training rows** determine the low-energy class. Sort the $m$ training
energies at that composition. The threshold is the element with one-based
index $\lceil0.2m\rceil$, called $q_{20}(n)$. Set

$$
b=\mathbb 1[e\le q_{20}(n)],\qquad c=(n/48,b).
$$

Thus $b=1$ requests the training distribution's lowest-energy 20%, with ties
retained; $b=0$ means the remaining class. Both classes are used in fitting.
The composition-only CVAE receives $c=(n/48)$. Requesting $b=1$ changes
the sampled distribution; it is not a hard energy constraint and does not send
the model down an energy gradient. Every generated geometry needs a measured
energy to establish whether it meets a low-tail criterion.

Thresholds and raw energies are teacher- and composition-specific. A change
in the number of Li atoms changes stoichiometry, so raw total energies across
Li counts are not a common ranking target. The current result concerns ranks
and low-tail membership **within** a composition and parent.

## CVAE objective and latent variables

The encoder $q_\phi(z\mid o,c)$ reads a training occupancy and condition,
then predicts the mean and diagonal variance of an eight-dimensional Gaussian
latent variable $z$. The decoder $p_\theta(o\mid z,c)$ predicts the
occupancy distribution. At generation time, draw $z\sim\mathcal N(0,I)$,
supply the desired $c$, and sample from the decoder.

Training minimizes

$$
\mathcal L_\beta=
\mathbb E_{q_\phi(z\mid o,c)}[-\log p_\theta(o\mid z,c)]
+\beta\,D_{\rm KL}\!\left(q_\phi(z\mid o,c)\,\|\,\mathcal N(0,I)\right).
$$

The first term penalizes assigning low probability to the observed training
mask. The second measures how far its encoded Gaussian moves from the prior
used when drawing new samples. $\beta$ sets their tradeoff. At $\beta=1$
this is the negative conditional evidence lower bound for the specified model.
At $\beta=0.1$ it is a different regularized objective. A smaller validation
reconstruction error alone does not establish better new samples. A nearly
zero KL can indicate weak latent use, while a large KL alone does not prove
useful latent information or good prior sampling. See [Kingma and Welling,
§2](https://arxiv.org/abs/1312.6114) and [Sohn et al., §4](https://proceedings.neurips.cc/paper_files/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf).

## Dense and exact-count decoders

The legacy dense decoder outputs 48 logits $\ell_i$, each describing the
preference for occupancy at one site conditional on $z,c$. Independent
Bernoulli reconstruction does not enforce the requested total count during
fitting. Taking the largest $n$ logits at generation enforces the count but
changes the sampling rule.

The new dense exact-count model uses

$$
p(o\mid z,c,\sum_i o_i=n)=
\frac{\exp(\sum_i\ell_i o_i)}{Z_n(\ell)},\qquad
Z_n(\ell)=\sum_{u:\sum_i u_i=n}\exp\!\left(\sum_i\ell_i u_i\right).
$$

Here $u$ runs over every mask with $n$ occupied sites. A dynamic program
computes this normalization without enumerating all masks, and samples exactly
from the resulting conditional distribution. Count is correct in both the
likelihood and the sampling rule. This repair does not itself establish that
the model captures the energetic interactions among vacancies.

## Autoregressive (AR) decoder

An AR decoder makes the 48 decisions sequentially in a fixed **site order**:

$$
p(o\mid z,c)=\prod_{i=1}^{48}p(o_i\mid o_1,\ldots,o_{i-1},z,c).
$$

The current decoder uses a GRU with hidden size 64 to remember earlier occupancy
choices. At site $i$, it receives the previous bit, site index, condition,
and latent vector. During fitting, previous bits come from the observed mask
(teacher forcing); during sampling, they are the model's own previous draws.
If no Li remain to place, every remaining bit is forced to zero. If every
remaining site must be occupied to meet the target count, each is forced to
one. Such forced decisions have probability one and add zero reconstruction
loss.

For example, with four sites and two Li, the decoder may choose `1`, then `0`,
then `1`. The last bit is necessarily `0`. Unlike independent logits, its
third-site preference can depend on the first two choices. The order is not
physical time, MD, or an atomic relaxation. The implemented network is not
rotation-equivariant; symmetry is handled by the discrete parent mappings.

An AR baseline removes the encoder and latent variable but retains sequential
stochastic sampling. Its comparison tests whether a latent CVAE helps beyond
the AR decoder itself. A composition-only AR CVAE tests the added value of the
energy-class condition. The chain-rule construction is related to
[Wu, Wang and Zhang, Eq. 4](https://doi.org/10.1103/PhysRevLett.122.080602);
their physical training objective is not the objective used in this project.

## Data rounds and adaptation

| Dataset | Total masks | Train / validation / test | Contents |
| --- | ---: | --- | --- |
| `training_original.npz` | 1,536 | 1,227 / 153 / 156 | Original Li₁₂/Li₂₄/Li₃₆ subset |
| `training_round0.npz` | 50,247 | 40,157 / 4,967 / 5,123 | Original plus previously evaluated proposals |
| `training_round1.npz` | 75,247 | 60,152 / 7,493 / 7,602 | Round 0 plus 25,000 fresh development masks |

Original held-out masks stay protected. New canonical keys receive deterministic
hash-based splits. Reusing a previous assessment pool for adaptation changes
its status to development/training data; it is no longer an untouched final
assessment. All new energy conditions are calculated from that round's training
rows only.

The extra adaptive fine-tuning warmed from the round-0 AR CVAE, added round-1
data, recomputed training low-class thresholds, and used class-balanced training
weights. It compared $\beta=0.1$ and $\beta=1$. This is a coupled
exploratory update, not a clean experiment isolating sample count. The original
AR family was selected on development results before the final pool was scored.
Current best-average checkpoints are the round-0 `ar_adapt_cvae` family, not the
later adaptive checkpoints.

## Geometry CVAE and regressor

`core/data/geometry_uma_fixed.npz` contains 6,000 frames from 1,000 independent
mask lineages: the deletion geometry, four intermediate optimization frames,
and one fixed-cell endpoint per lineage. The lineage split is 787 training,
107 validation, and 106 test masks (4,722 / 642 / 636 frames). A lineage's frame
weights total one, so long trajectories do not automatically dominate fitting.

Displacement $u_{a\alpha}$, in Å, is present atom $a$'s motion from its
matched pristine parent site in direction $\alpha$. The parent fractional
site is mapped through the supplied frame cell. A global mass-weighted
translation is removed and periodic differences use the minimum-image
convention. Absent Li have zero padding and a false presence mask; those zeros
are not physical displacement targets. Arrays are `(frames,192,3)`.

The decoder context has 65 components: 48 occupancy bits, Li fraction, a
three-class cohort flag (deletion/intermediate/endpoint), nine supplied cell
deformation components, and four standardized scalar properties:

- energy per present atom;
- `log1p(force component RMS / (1 eV/Å))`;
- configurational pressure in GPa;
- `log1p(deviatoric stress norm / (1 GPa))`.

All standardization statistics use training frames only, separately by
composition and cohort. A requested endpoint tuple is an attained training
endpoint: among endpoints with force RMS below that training group's median,
choose the lowest-energy one. It is not a held-out structure's measured answer.

The CVAE encoder reads the 576 normalized displacement coordinates and context,
then produces a 16-dimensional Gaussian latent distribution. Its decoder
predicts displacements. A presence-masked squared-error reconstruction term and
Gaussian KL are trained with lineage weights. The deterministic regressor uses
the same context without a latent variable and is the essential control for
whether stochastic generation adds value. Both models currently have one fit.

The cell is **supplied, not predicted**. The geometry decoder does not integrate
forces or minimize an MLIP energy; direct output therefore need not meet a force
or stress tolerance. Coordinate reconstruction on held-out frames, fresh
prior-decoded energy/forces/stress, and the iterations needed by a subsequent
optimizer answer different questions. Report all three separately.

## File conventions and safe comparisons

Forces are eV/Å; stress is tensile-positive ASE Voigt order
`xx, yy, zz, yz, xz, xy`, in eV/Å³. Force component RMS is
$\sqrt{\sum_{a\alpha}F_{a\alpha}^2/(3N_{\rm atoms})}$; maximum force is
$\max_a\|F_a\|_2$. They are different observables. An RMS force below a
number does not imply every atom meets that maximum-force threshold.

Some proposal score rows reuse a known scalar energy. They are labeled
`provenance=cached` and intentionally have NaN in extra force/stress summaries.
Fresh scores contain finite energies, forces, and stress. Cached scalar reuse
does not license copying vector labels between differently oriented structures.

The wider original main-parent corpus contains 21,294 masks covering all 49 Li
counts. Auxiliary parents have other sizes, including an exactly enumerated
12-site orbit benchmark. These support further analyses, but a 48-input
checkpoint is not a size-transfer model. Check primitive-cell aliases before
claiming independent cross-size generalization. The 87 possible 12-site masks
are MLIP-labeled deletion geometries; they are not 87 new DFT calculations.

The geometry archive uses a fixed parent atom-index frame. Its whole geometries
were not rotated into canonical vacancy orientations. Canonical mask keys identify
lineages for splitting; a coordinate or force vector still belongs to its actual
orientation. Any future symmetry augmentation must transform the full structure,
cell and vector labels consistently, rather than permuting occupancy alone.

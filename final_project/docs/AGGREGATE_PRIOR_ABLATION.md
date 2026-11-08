# Posthoc latent sampling for the frozen dense CVAE

The dense D4 model uses its latent variables, but its generated energies respond
weakly to energy conditions. This bounded test asks whether sampling latent
vectors closer to those encountered during training helps, without updating the
network. It compares the original standard-normal proposal with one Gaussian
proposal per composition and energy bin, using all three frozen D4 checkpoints.
It changes the sampling rule and does not train a conditional prior.

For one fitted checkpoint and one composition/bin group, let the encoder assign
TRAIN example `i` the posterior

\[
q_i(z)=\mathcal N\!\left(\mu_i,\operatorname{diag}(\sigma_i^2)\right).
\]

Here `z` has four dimensionless coordinates, `mu_i` is the encoded mean, and
`sigma_i^2 = exp(logvar_i)` is the encoded variance. If an index `I` is selected
uniformly among the `N` training examples in this group, then selecting `I` and
drawing from `q_I` defines the empirical aggregate mixture

\[
q_{\mathrm{agg}}(z\mid c)=\frac1N\sum_{i=1}^N q_i(z).
\]

The mean of a conditional mean gives

\[
m_c=\mathbb E[z]=\frac1N\sum_i\mu_i.
\]

The law of total covariance separates uncertainty inside each posterior from
variation among its means:

\[
S_c=\mathbb E[\operatorname{Cov}(z\mid I)]
 +\operatorname{Cov}(\mathbb E[z\mid I])
=\frac1N\sum_i\operatorname{diag}(\sigma_i^2)
 +\frac1N\sum_i(\mu_i-m_c)(\mu_i-m_c)^T.
\]

These are exact moments of the finite mixture, apart from numerical arithmetic.
The denominator is `N`, since the target is that explicitly defined empirical
distribution. An `N-1` covariance correction would estimate a different target.
The proposed `N(m_c,S_c)` matches these two moments but generally does not
reproduce the mixture's modes or tails. It may fill regions between encoded
clusters and reduce usefulness or diversity; this is a hypothesis to measure.

For a synthetic one-dimensional example, two equally weighted posteriors with
means -1 and +1 and individual variance 0.25 have aggregate mean 0 and variance
0.25 + 1 = 1.25. The Gaussian approximation is therefore `N(0,1.25)`. It has one
central peak, while the original mixture can retain two peaks. Matching its
mean and variance does not prove that the decoder behaves well between them.

For a Cholesky factor `L_c` satisfying `L_c L_c^T = S_c` and standard-normal
noise `epsilon`, the new proposal is

\[
z_{\mathrm{aggregate}}=m_c+L_c\epsilon,
\qquad z_{\mathrm{standard}}=\epsilon.
\]

Both arms use the same checkpoint, requested condition, underlying `epsilon`
and autoregressive uniforms. Their actual latent vectors differ. The aggregate
transform also changes with the requested energy bin, so a low/high comparison
in this experiment is **not** a same-`z` test of the decoder's condition input.
Saved arrays and SHA256 fingerprints distinguish base noise, actual latent
vectors and decoder uniforms.

Moments use only the 40,157 protected TRAIN masks and their frozen UMA energy
bins. No validation, test or final masks/labels enter estimation. Each original
network remains unchanged, with its original standard-normal training prior.
Means and full 4-by-4 covariances are calculated in float64 from FP32 CPU encoder
outputs. Positive posterior variances, finite matrices and positive-definite
Cholesky factors are required. No shrinkage, fitted jitter or alternative
covariance strength is tried. The FP32 decoder receives an explicitly recorded
FP32 transform.

The protocol fixes 128 attempts per count, requested bin, fitted seed and method:
3 counts × 3 bins × 3 fits × 2 methods × 128 = 6912 attempts. Every prespecified
attempt is assessed by both UMA and MACE, using each teacher's own frozen bins.
Repeated or known masks remain attempts. A new history is frozen before
generation from the core history and every already registered core/editor/
temperature mask identity. Only identifiers enter that history; final energies
are not used to choose moments or tune the proposal.

The main outcomes remain distinct novel requested-bin orbits per all attempts,
the separate original-q20 low-request yield, measured energy distributions,
uniqueness, training/history novelty, query cost and individual fitted seeds.
This empirical sampling rule is not an equilibrium measure, entropy estimate,
physical temperature response, energy minimizer or DFT validation.

The integrating owner runs generation only after the editor and temperature
comparison have completed. Commands below run inside an appropriate allocation;
the script itself submits no jobs:

```bash
python scripts/conditioning_20261107/aggregate_prior_ablation.py prepare \
  --root . --output-dir runs/conditioning_20261107/aggregate_prior/preparation
python scripts/conditioning_20261107/aggregate_prior_ablation.py generate \
  --root . --prepared runs/conditioning_20261107/aggregate_prior/preparation \
  --output-dir runs/conditioning_20261107/aggregate_prior/assessment --device cuda
python -m cathode_cvae.pool_scoring register \
  --root runs/conditioning_20261107/oracle \
  --input runs/conditioning_20261107/aggregate_prior/assessment/candidates.npz \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --models uma mace --batch-size 64
python -m cathode_cvae.pool_scoring collect \
  --root runs/conditioning_20261107/oracle \
  --output runs/conditioning_20261107/collected/aggregate_prior
python scripts/conditioning_20261107/aggregate_prior_ablation.py evaluate \
  --root . --generated runs/conditioning_20261107/aggregate_prior/assessment \
  --scores-uma runs/conditioning_20261107/collected/aggregate_prior/uma.npz \
  --scores-mace runs/conditioning_20261107/collected/aggregate_prior/mace.npz \
  --output-dir runs/conditioning_20261107/evaluation/aggregate_prior
```

The registration command does not produce labels by itself; resident workers
must complete the registered tasks before collection/evaluation. All output
directories are write-once. The moment receipt reports actual posterior calls,
encoded rows, load/inference time, unchanged checkpoint hashes and zero optimizer
updates. This ablation adds no full training trajectories.

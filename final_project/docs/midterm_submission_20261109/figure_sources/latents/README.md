# Submission latent figures: saved-evidence analysis only

All plots use saved CSV/JSON outputs; no model was loaded and no fitting,
encoder/generator forward, MLIP/DFT/MD call or allocation submission occurred.
The baseline repository was read-only. Every plot is 5.5 inches wide, uses
7–8.4 pt fonts, and is supplied as vector PDF and 300 dpi PNG.

Re-render from the included local evidence files:

```bash
python make_figures.py
```

The extraction script contains the exact original source paths and can rebuild
these evidence CSVs when those archived worktrees are available. Its source
hash registry is `source_hashes.json`; figure/data hashes and text-boundary
checks are in `figure_receipt.json`. `VISUAL_REVIEW.json` binds the inspected
renderings. No claim of independent physical validation follows from rendering.

## Main figure 1: latent_dimension.pdf

Suggested caption: **Latent-dimension sensitivity of the older dense-posterior
CVAE without spatial context (development assessment).** (a) Independent
prior-generation useful yield and (b) weighted validation reconstruction
negative log likelihood. Three fits per dimension, each trained for 4,000
optimizer updates; gray points show fits and bars descriptive fit SD. The
horizontal guide is the uniform-mask reference from the same assessment.
Increasing dimension improves reconstruction without monotonic improvement in
useful generation. This is a finite sensitivity study, not demonstrated
convergence of the selected spatial CVAE.

The 12 evidence rows are independently checked against each original
configuration, named validation-beta0.1 latent JSON, summary selection epoch,
and original history row. The D8 fits were reused from the prior policy study.
The selected dimensions were 4/8/16/32 with parameter counts
30,793/32,337/35,425/41,601; this is not a constant-parameter-count comparison.
Do not pool this older DEVELOPMENT namespace/history with the newer FINAL
assessment or claim its absolute yields are a controlled feature comparison.

## Main figure 2: latent_mechanism.pdf

Suggested caption: **What the latent encodes and what the decoder uses.**
For selected A-shell (five fits) and C-dense (three fits), (a) the weighted-VAL
variance of each encoded mean and (b) the mean posterior variance distinguish
near-prior, almost input-independent graph codes from informative narrow dense
codes. Faint points denote components; diamonds denote fit means. Components
within a fit are not independent fits, and component axes are not aligned as
physical coordinates across fits. The activity cutoff 0.01 is a dimensionless
working diagnostic, not a physical criterion. (c) Separate UMA-rescored
interventions compare signed LOW-to-HIGH energy response with fixed actual
latent/AR uniforms against mean absolute energy response from resampling only
the latent at fixed LOW condition/uniforms. Points are fit means and bars
mean ± descriptive fit SD; these are distinct observables.

**Causal caveat:** A-shell uses radial-shell decoder context; C-dense uses the
legacy context. This requested figure is a diagnostic comparison of the
selected model with an active-posterior control, not a pure encoder-only
ablation. The clean same-context comparison elsewhere is base
`A_graph_cvae` versus `C_dense_cvae`. Base A has similarly tiny KL and weak
posterior-mean variance, so the latent-use conclusion is not inferred solely
from changing the shell kernel.

Posterior collapse here means near-input-independence, not a point distribution.
A-shell's posterior variance remains broad (0.854–0.943); its means vary very
little across masks. The prior N(0,I) is fixed. The exact decomposition

    E_(m,c) KL[q(z|m,c)||p(z)]
      = I_q(m;z|c) + E_c KL[q(z|c)||p(z)]

bounds encoded conditional information for the specified class-weighted
empirical VAL distribution. Selected A-shell mean/range of this upper bound
are 0.017705 / 0.010422–0.029121 bits per mask. These are **upper bounds**, not
actual MI estimates or physical configurational entropies. High KL alone does
not establish aggregate-prior mismatch or useful generation.

## Optional optimization_traces.pdf

One prespecified seed (2026110801) for A-shell/C-dense illustrates saved weighted
validation reconstruction and KL versus optimizer updates. The dashed line
marks full KL weight from update 1502, and stars the actual named-beta0.1
selected checkpoints. The fixed 4,000-update budget is not a convergence proof.
Only two representative curves are shown, not an uncertainty ensemble.

# ECE285 submission mid-term report — 9 November 2026

`main.pdf` is the submission copy: seven main pages and one page of references.
It uses the unmodified `neurips_2025.sty`, preprint option, title and author order
from the original proposal. It includes three vector figures and nine cited
references. There is no appendix in the submission PDF.

The report covers the class sampling study: composition/energy conditioning,
exact-count autoregression, symmetry-aware novelty, dense and graph posterior
encoders, physical decoder features, baseline comparisons, completed parameter
studies, and a paired deletion/relaxed-energy target test. It retains negative
results and distinguishes MLIP teacher labels from DFT validation.

The revised exposition follows the user's preferred November 8 report: it
introduces DFT/MLIPs and relaxation before use, explains periodic masks and
symmetry, and describes each method through its effect on useful sampling yield.
Proposal comparisons, execution-history narration and collaborator task lists
are omitted from the manuscript. The prior five-page PDF remains available in
`for_chat_gpt/ECE285_midterm_submission_20261109.pdf`; the revised deliverable
is `for_chat_gpt/ECE285_midterm_submission_20261109_v2.pdf`.

## Build

With a normal TeX Live installation containing pdfLaTeX, BibTeX, natbib,
amsmath, amssymb, booktabs, tabularx, graphicx, microtype and hyperref:

```bash
cd final_project/docs/midterm_submission_20261109
bash build.sh
```

The build is pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX. All figure PDFs and the
compiled bibliography `main.bbl` are included. Rebuilding the report needs no
GPU, trained checkpoint, external dataset or MLIP installation.

For the NERSC environment used for the verified build, from the worktree root:

```bash
PATH=/global/common/software/nersc9/texlive/2024/bin/x86_64-linux:/usr/bin:/bin \
  bash final_project/docs/midterm_submission_20261109/build.sh
```

## Evidence and figures

`CLAIM_SOURCE_MAP.json` links claims to archived inputs and copied evidence.
`BUILD_RECEIPT.json` records hashes, pagination and build checks.
`figure_sources/` includes plotting scripts, small local CSV/JSON evidence and
original source hash registries. NumPy and Matplotlib suffice to re-render:

```bash
(cd figure_sources/features && python make_feature_figure.py)
(cd figure_sources/latents && python make_figures.py)
```

The feature renderer writes `featurization_comparison.pdf`; the report uses its
identical copy `assets/featurization.pdf`. The latent renderer writes
`latent_dimension.pdf` and `latent_mechanism.pdf`, copied into `assets/`.
`optimization_traces.pdf` is an additional saved-evidence plot provided with the
sources, outside the seven-page submission. It illustrates two representative
training traces; it does not establish convergence.

Scientific interpretation matters: the D=4/8/16/32 sweep used the older dense
CVAE without spatial context, not the selected graph/shell CVAE. Learning rate,
batch size, cutoff and training duration were fixed rather than converged.
The conditional-information calculation is an upper bound under the saved
weighted empirical validation distribution, not a measured mutual information
or a population guarantee. Small encoder KL does not make decoder sampling
deterministic. The paper explains the separate roles of conditions, latent
samples and stochastic autoregressive draws.

All new work for this submission was document preparation, small saved-table
analysis, plotting and compilation. No new training, structures, MLIP/DFT/MD
calls or scheduler allocations were used. Raw evidence and protected assessment
boundaries were preserved. This local submission branch has not been pushed,
and the coursework has not been submitted to Gradescope.

# Midterm 1 report and insertion map

The editable report is [midterm1/main.tex](midterm1/main.tex), compiled as
[midterm1/main.pdf](midterm1/main.pdf), with
[midterm1/references.bib](midterm1/references.bib). It targets five body pages,
including the revised proposal, with references afterward. The provided
screenshots specify NeurIPS format for the proposal; no additional midterm
layout was found in those sources. Verify any newer course instruction before
manual submission.

Build from `final_project/` with TeX Live or an equivalent LaTeX installation:

```bash
bash docs/midterm1/build.sh
```

The report centers on the new composition-and-energy-range CVAE and the
static-energy temperature-acquisition demonstration. The old binary benchmark,
rank-audit context, and geometry/regressor comparison are clearly historical.
Ricardo's completed physical contribution is also available as editable text in
[RICARDO_CONTRIBUTION.md](RICARDO_CONTRIBUTION.md).

| Report location | Content | Owner's independent follow-up |
| --- | --- | --- |
| Sections 1–2 | Motivation, computational tools, parent, geometry and symmetry | Ricardo reviews physical interpretation and representative structures |
| Section 3 | Shared condition schema, encoder/AR decoder, loss and assessment | Beihao reproduces a fit, checks equations, supplies learning curves/ablation |
| Section 4 | Reference measure, static weights, ESS, temperature mixtures and floor | Varad independently checks multiplicities, units, allocation and concentration |
| Section 5 / `generated_results.tex` | New measured generation/acquisition results | Varad checks source metrics and prepares seed/condition figures |
| Section 6 | Preserved historical evidence, DFT rank limitations and integration | All authors approve limits and their actual contributions |

The numerical claim map is [midterm1/claim_sources.json](midterm1/claim_sources.json),
with a [readable source table](midterm1/CLAIM_SOURCE_MAP.md). The measured
`generated_results.tex` includes completed benchmark, acquisition demonstration
and allocated-resource accounting. Its negative result is retained. Passing code
tests alone did not establish that result: complete teacher scores, exact-geometry
joins and independent raw-input audits support it.

Beihao's and Varad's exact inputs, commands, expected outputs and acceptance
checks are in [COLLABORATOR_TASKS.md](COLLABORATOR_TASKS.md). Preparing those
commands does not complete their independent checks. Use
[INTEGRATION_CHECKLIST.md](INTEGRATION_CHECKLIST.md) for November 7/8/9.

No class result in this draft establishes DFT accuracy of new structures,
physical configurational entropy, phase stability, reactor equilibrium or
post-fine-tuning improvement. Any subsequent MLIP-only thermal-method pilot is
kept outside the class report except as future work. No DFT or Gradescope
submission is automated.

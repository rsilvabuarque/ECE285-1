# Midterm 1 editable draft

The report source is [midterm1/main.tex](midterm1/main.tex), with references in
[midterm1/references.bib](midterm1/references.bib). The draft reserves five body
pages, including the revised proposal, and places references afterward. Confirm
the final class formatting before submission; the supplied `article` layout is
an editable working format.

Build from `final_project/`:

```bash
bash docs/midterm1/build.sh
```

The source has owner notes that must be replaced with completed contributions:

- Ricardo checks materials motivation, parent structure, symmetry, and label
  provenance. Add materials references matched to the claims actually retained.
- Beihao checks the implemented equations, supplies a reproduced fit and
  learning curves, and reports the controlled adaptation study if completed.
- Varad independently recomputes the table, supplies a seed-resolved comparison
  figure, and checks geometry and uncertainty statements.

The numerical comparison in the draft is the frozen completed experiment, not
an evaluation of future collaborator fits. If an ablation has only held-out
likelihood or coordinate errors so far, report those observables and leave
fresh physical sample quality as unmeasured.

The expected storyline is already supported by the data: AR conditioning helps
relative to uniform sampling; low-energy mutation remains stronger; further
adaptation does not improve average yield; learned geometry starts reduce
optimizer iterations, but the regressor matches the geometry CVAE. The draft
keeps size/material transfer, DFT accuracy, and equilibrium claims outside the
established result.

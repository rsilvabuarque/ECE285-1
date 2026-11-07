# Matched DFT/MLIP ranking context

This directory recomputes existing matched-geometry evidence; it contains no new
DFT, inference, relaxation, or training results. `receipt.json` identifies the
source CSV by SHA256 and reports the independent field checks.

- `paired_energies.csv` contains the minimal energy/identity columns needed for
  portable recomputation. Historical and Hu source labels remain separate.
- `pool_metrics.csv` fixes source, phase, composition and atom count, with a named
  cohort-specific and a separately named all-cohort view.
- `topk_metrics.csv` records requested k, actual tie-expanded set sizes, overlap,
  precision/recall, and selection regret. Low-quintile uses k = ceil(0.2 Npool).
- `cohort_summary.csv` gives descriptive medians across informative composition
  pools, not confidence intervals from independent frames.
- `five_member_counterexample.pdf` shows the complete five-member optimized Hu
  O3 Li10 pool. Its top-five identity overlap is necessarily one; its rank
  ordering and selected minimum are not necessarily correct.
- `occupation_examples.pdf` and `examples/` provide two existing Li24 training
  deletion structures and their manifest. They are illustrative examples, not
  new discoveries or relaxed structures.

For a portable recomputation, `recompute.py` also accepts `paired_energies.csv`
as `--input`; the original larger `per_structure_EFS.csv` is needed only to
independently compare the saved older cohort metrics. Use a new output directory.
On an HPC system, run the analysis in a suitable compute allocation.

```bash
python reports/ranking_audit_20261107/recompute.py \
  --input reports/ranking_audit_20261107/paired_energies.csv \
  --output runs/ranking_reproduction
```

This portable command was executed during the November 7 handoff check. Its
`pool_metrics.csv`, `topk_metrics.csv`, `cohort_summary.csv` and complete
five-member counterexample CSV were byte-identical to the original-source
recomputation.

The paired archive includes relaxed, displaced and migration frames. These are
not the 48-site pristine-deletion training pool. Hu SCF and spin convergence are
not independently recertified from these exports. The old DFT audit uses FP64
inference; the new deletion benchmark uses its own explicit FP32 configuration.

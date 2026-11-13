# Structured causal-range review

PASS_STRUCTURED_CAUSAL_REVIEW_WITH_LIMITS

All paired signs use policy minus reference, in percentage points. The interaction is (augmented range − H0 with the same R1 range edges) − (augmented fixed − H0 fixed).

| Family | Teacher | Relabel H0 | Add rows, fixed | Add rows, identical range edges | Interaction |
|---|---|---:|---:|---:|---:|
| ar | uma | +6.48148 ± 1.01356 | -1.56250 ± 0.88098 | -1.35995 ± 0.39143 | +0.20255 ± 1.06078 |
| ar | mace | +6.10532 ± 0.97568 | -1.09954 ± 1.32882 | -2.05440 ± 0.84310 | -0.95486 ± 1.69438 |
| d2 | uma | +7.06019 ± 1.08536 | +0.40509 ± 0.77803 | +1.85185 ± 1.32598 | +1.44676 ± 1.99904 |
| d2 | mace | +6.79977 ± 0.91180 | +0.08681 ± 0.45933 | +1.99653 ± 1.38889 | +1.90972 ± 1.82911 |

± denotes sample SD across three paired fit/draw replicates.

- Post-hoc mechanistic study prompted by previous range-policy gains; not blind confirmation.
- Relabeling recomputes class-balanced TRAIN weights, so it changes both conditions and effective row weights.
- Matched-edge added-row effects also include changed minibatches/class weights; they do not isolate information content alone.
- H0 with R1 edges uses acquired labels through the edge summary; only H0-own-range needs no acquired labels.
- Three paired fit/draw seeds give descriptive sample SD, not adjusted significance or proof of equivalence.
- Balanced gains are mainly high-bin success, not a demonstrated low-energy-search improvement.
- No NPZ or PT file was opened or rehashed in this audit. Dataset preservation, checkpoint warmstarts, and fresh scoring rely on structured writer/worker/verifier contracts.
- Schema source decompresses the complete NPZ energy member but indexes TRAIN only; training encodes TRAIN/VAL while TEST conditions remain inert.

CSV arithmetic checks: max pair difference 0 pp; max summary difference 3.55e-15 pp. The audit inspected structured sources, fit configurations, summaries, task/worker receipts, stage protocols and strict oracle verifiers. It did not repeat collectors, load checkpoints or inspect TEST energies.

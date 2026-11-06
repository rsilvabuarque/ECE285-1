# Validation of the portable package

The [machine-readable receipt](validation/receipt.json) records the checks run on
November 6. All five model SDKs were installed in fresh, isolated environments,
with dependencies resolved and `pip check` passing. The CPU package lists are
stored beside the receipt. They document this validation; use the installer and
a compatible PyTorch wheel channel for a GPU installation, rather than copying
the CPU package lists into it.

The final numerical suite passed **28 tests**. It checks exact-count probability
normalization and gradients against exhaustive toy enumeration, AR support,
symmetry identities and closure, train-only statistics, protected lineage splits,
parent binding, target-site mapping including Na, displacement/presence handling,
stress/force units, environment dispatch, and failure accounting. A fresh
PyTorch 2.13 CPU environment also passed the preceding 27-test suite; the last
test adds the actual spglib parent-preparation path. These are software and
numerical checks, not evidence that generated structures are physically optimal.

The cached occupancy evaluation exactly reproduced every field in the archived
`metrics.csv`, `seed_metrics.csv`, `summary.csv`, and `teacher_agreement.csv`.
The geometry evaluation independently recomputed baseline energy differences
and force ratios from 3,200 stored score rows and the 32-mask paired relaxation
summary. No MLIP was called for those results.

A short two-epoch mask fit, checkpoint reload, exact-count generation, and
decoding with both frozen geometry checkpoints passed. The fit is an operability
check and has no claimed new sampling performance. The broader-data adapter
successfully preserved the existing main-parent and 12-site protected splits.
All files listed by the snapshot manifest passed their size and SHA256 checks.

Finally, **all five pinned MLIP checkpoints loaded successfully on CPU** in their
new backend environments: hashes, package versions, constructors, and selected
heads were checked. This reused the project's existing authorized weights.
It did not test a fresh gated download, lazy graph construction, GPU execution,
energy/force/stress predictions, or numerical parity on the class GPU. Run the
README's small `doctor --load` interface check on that platform before starting
an inference campaign. UMA access must belong to the actual model user.

The editable midterm draft compiles to five body pages plus one reference page.
It was rendered and checked for spillover, undefined references, and overfull
text. Owner notes deliberately remain for the collaborators to complete.

Two setup attempts are retained in the receipt: one failed before the validation
script was present, and one was stopped after confirmed DVS filesystem I/O wait.
The successful installation used native scratch storage. The tests consumed
about 0.34 CPU-node hours and zero GPU-node hours; all owned allocations ended.
No new DFT, MD, MLIP labels, relaxation dataset, or scientific training campaign
was produced.

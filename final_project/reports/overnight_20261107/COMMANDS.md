# November 7 execution and reproduction

Run from `final_project/`. The portable commands and required package installation
are in the README and `docs/COLLABORATOR_TASKS.md`. Cluster-specific launchers in
`scripts/overnight_20261107/` preserve the actual experiment; their absolute
interpreter paths are execution provenance, not portable defaults.

The actual sequence was preparation (`prepare.py`, `mace_schema.py`), three
`train_launcher.py` fits, frozen benchmark generation (`generate_launcher.py` /
`generate_benchmark.py`), dynamic single-GPU scoring (`worker_launcher.py` /
`cathode_cvae.pool_scoring`), allocation (`cathode_cvae.acquisition`), allocation
generation (`cathode_cvae.acquisition_generate`), and condition-aware evaluation
(`evaluate_benchmark.sh`, `evaluate_demo.sh`). The output receipts bind parents,
schemas, teachers, checkpoints, source hashes and attempted generation rows.

Raw receipts and preserved failed attempts remain at the HPC location recorded
by `artifacts/overnight_20261107/manifest.json`. The downloadable packet contains
the exact inputs needed for cached evaluation and the three newly fitted CVAE
checkpoints. Foundation-model weights are not redistributed.

Final checks actually executed on allocation 60954645:

```bash
python -m pytest -q                       # 67 passed, 8.08 s
python -m pip check                       # No broken requirements found
```

The advertised condition-aware benchmark command and 48-site allocation command
in the collaborator tasks were also run against the **public packet paths**, in
`runs/overnight_20261107/public_replay/{benchmark,allocation}`; both returned PASS.
The independent benchmark/demo audits recompute the relevant quantities from
raw energies and attempt records rather than calling the evaluator under test.
Their receipts are under `evaluation_audit/` beside this file.

`validation_receipt.json`, `final_integrity.json`, and the cost receipt record
software checks, historical preservation, downloadable payload integrity, and
allocation accounting separately. A passing software check does not establish
DFT accuracy, global optimality, equilibrium sampling, or successful physical
conditioning. See the negative measured results in `docs/RESULTS.md`.

The completion gate records the actual published scientific commit and remote
artifact verification; it is written only after those steps finish.

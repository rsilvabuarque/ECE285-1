# November 8 execution and replay

Run commands from `final_project/` on the review branch
`overnight/latent-relaxed-neb-20261108`. Do not overwrite existing output
directories. The release's `replay_manifest.json` binds the portable inputs by
SHA256. Original absolute paths in execution receipts remain provenance; the
replay helper supplies relocated inputs explicitly.

## Required cached analysis

Install the learning environment with the README procedure, activate it, and
fetch the archive using the URL and SHA256 in
`artifacts/overnight_20261108/download.json`. Then:

```bash
python scripts/overnight_20261108/replay_packet.py \
  --packet artifacts/overnight_20261108 \
  --output-dir runs/varad_overnight_replay --plots
```

This recomputes the static comparison and all four paired assessment scopes.
It fits no model, generates no new masks, and makes no MLIP calls. Both label
definitions are evaluated on the same endpoint proposals. The original fixed
deletion q20 is a separate metric. Cached replay is a numerical verification,
not an independent training replicate or physical validation.

On NERSC, analysis must run in a permitted compute allocation. The software does
not submit jobs or select an account. On another GPU server/workstation, the
ordinary CPU replay needs no Slurm installation.

## Exact executed commands

The packet's `execution/` contains the actual argv arrays, source-manifest
hashes and seeds, without shell-quoting ambiguity:

- `control_fit_tasks.json`: nine A/B/C fits.
- `screen_fit_tasks.json`: three D controls and five single-factor screens.
- `confirmation_fit_tasks.json`: six independent confirmation fits.
- `extra_replicate_fit_tasks.json`: eight additional FINAL-only A/B fits.
- `paired_target_fit_tasks_v2.json`: six deletion/endpoint-target fits.
- `*_gpu_config.json`: pinned interpreters, source snapshots, entrypoints,
  visible GPU binding and complete worker arguments.
- `final_static_generation.log`, `final_paired_generation.log`, and each
  proposal `protocol.json`: proposal freezes and execution outcomes.
- `sources/`: immutable executed source versions. In particular, `final_v8`
  produced the frozen final proposals, while `initial_v1` supplied the endpoint
  workers' physical implementation. Later analysis-only fixes do not replace
  these snapshots.

All 37 fits completed 4,000 optimizer updates: 148,000 total, with no adaptive
continuation. The selected payload is `best_validation_beta_0p1.pt`, not the
default `best.pt` or a `summary.json` epoch selected with a different beta.
The reference-update warm-up is
`beta(u)=0.1*min(1,(floor(u/79)+1)/20)` for zero-based update `u`.

For independent training, keep all recorded flags and replace only the
machine-specific data/schema/receipt/output paths and the permitted device.
Use the packet's matching data and schema for the chosen task. The present
mask decoder generates pristine-derived deletions even when the conditioning
energy was obtained after relaxation. It does not emit optimized coordinates.

## Collection and authentication commands

These reproduce the analysis interfaces, rather than submit new inference:

```bash
python scripts/overnight_20261108/evaluate_controls.py --help
python scripts/overnight_20261108/paired_facts.py --help
python scripts/overnight_20261108/paired_evaluate.py --help
```

The first accepts repeated `--score uma=... --score mace=...` and matching
`--schema` flags. `paired_evaluate.py --facts ... --facts-sha256 ...` reads
authenticated compact facts instead of any original HPC queue. The replay
manifest supplies the literal paths and SHA256 values for each stage.

The raw companion archives preserve trajectory coordinates, forces, stresses,
optimizer traces, all interruption records and claim times. They are not
required to recompute the released summary tables. The full arrays and
compact facts are different views of the same physical task identities;
shared candidates do not enlarge the attempt budgets or fit counts.

## Report builds and checks

```bash
bash docs/midterm_20261108/build.sh
bash docs/research_supplement_20261108/build.sh
python -m pytest -q
python -m pip check
```

The report builds require an installed TeX distribution. On Perlmutter the
executed TeX binaries were under
`/global/common/software/nersc9/texlive/2024/bin/x86_64-linux`.
The final integration suite passed 261 tests; six targeted portable-analysis
checks also passed. These software tests establish implementation behavior, not
scientific correctness of a potential.

For a new, optional inference check after the class review, use the existing
`cathode-cvae mlip doctor` and `cathode-cvae mlip score` interfaces documented
in the README. No new DFT, MD or MLIP fine-tuning was performed in this goal.

## Resource receipt

The resource collector reads only top-level allocations, excluding child
steps to avoid double counting:

```bash
python scripts/overnight_20261108/resource_receipt.py \
  --jobs 60998325 60998550 60999239 60999380 61001747 61003452 \
  --output reports/overnight_20261108/resources_final.json
```

The final receipt separates charged GPU node-hours, physical node allocation
hours and allocated GPU-device-hours. These are capacity/accounting quantities,
not measurements of GPU utilization. Startup, idle time and failed allocation
work remain charged. Private account balances and foundation-model weights
are excluded from the public packet.

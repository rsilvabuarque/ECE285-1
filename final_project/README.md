# Cathode CVAEs: midterm working package

**Beihao and Varad: start here, then read [your tasks](docs/COLLABORATOR_TASKS.md).**
The immediate question is whether a conditional generative model can produce many
**new, symmetry-distinct, low-energy vacancy configurations** more efficiently
than simple samplers. A second prototype predicts atomic displacements to give a
geometry optimizer a better starting point. The current data and checkpoints
are sufficient for the November 9 midterm; the tasks below use those data and do
not require new DFT, MD, or MLIP training.

The current occupancy experiment uses one O3 LiCoO₂ parent with **48 removable
Li sites**. For a requested Li count, a sampler decides which sites remain
occupied. It does not move atoms. A fast learned potential (an MLIP) evaluates
the resulting structure's energy, forces, and stress. These MLIP values are our
current *labels*; they are not newly computed DFT reference values. The longer
term motivation is to choose useful structures for reference calculations and
potential improvement, ultimately supporting simulations of cathode recycling.

Read the documents in this order:

1. [Method and data guide](docs/METHODS_AND_DATA.md): what each model learns,
   what the conditions mean, and what one data row represents.
2. [Measured results and limits](docs/RESULTS.md): the current comparison,
   adaptation diagnostics, and geometry results.
3. [Collaborator tasks](docs/COLLABORATOR_TASKS.md): concrete outputs, controls,
   and division of work before November 9.
4. [Midterm text draft](docs/MIDTERM_DRAFT.md) and [editable LaTeX source](docs/midterm1/main.tex)
   ([compiled PDF](docs/midterm1/main.pdf)): five body pages plus references, with
   owner notes to resolve before submission. Check the course's final formatting
   requirements when integrating the report.

The [validation record](docs/VALIDATION.md) covers fresh dependency installation,
all five checkpoint constructors, 28 numerical tests, and exact reproduction of
the cached comparison tables. New GPU prediction parity remains a check for the
class platform; no new physical data were generated while preparing this branch.

## Install

Use Python 3.12 (3.11–3.13 supported) and a compatible GPU driver. The package itself can train and
evaluate on CPU or CUDA. On a cluster, obtain a permitted compute allocation
before training or invoking the MLIPs; the package does not request allocations
or select an account for you. It respects the GPUs made visible by your platform.

To obtain this midterm branch in a new checkout:

```bash
git clone --branch midterm/portable-cvae-20261106 \
  https://github.com/rsilvabuarque/ECE285-1.git ECE285-midterm
cd ECE285-midterm/final_project
```

Install from `final_project`:

```bash
bash scripts/install.sh --help
bash scripts/install.sh --python python3 --prefix .venvs
source .venvs/learning/bin/activate
```

The installer creates a learning environment and backend-specific environments for
all five MLIP models. It installs their declared dependencies and checks package
consistency. Separate environments prevent one potential's dependency versions
from silently changing another's. Model weights are downloaded separately from
the Python packages, with the model/head and checkpoint identity recorded.
UMA requires legitimate access to its gated model repository; follow the
installer's message to authenticate once. Do not put tokens or downloaded model
weights in Git.

For the midterm's stored-data analysis and learning tasks, the core-only
installation is enough:

```bash
bash scripts/install.sh --python python3 --prefix .venvs --core-only
source .venvs/learning/bin/activate
```

`--no-weights` skips model downloads while still installing the selected Python
dependencies. `--torch-index-url URL` selects a compatible official PyTorch
wheel channel when your platform requires it. For CPU-only checks use
`--torch-index-url https://download.pytorch.org/whl/cpu`. Package installation
cannot upgrade the host GPU driver. The scientific campaign used NVIDIA A100s;
other hardware should pass the interface check before a new benchmark.
On NERSC, put environments and the pip cache on native scratch with `--prefix`
and `PIP_CACHE_DIR` rather than installing thousands of files on the project
filesystem. On a workstation, the default local `.venvs` is appropriate.
No MLIP needs to be loaded to
recompute the existing figures. Record the environment receipt with any new run.

## Existing data and results

The frozen snapshot is under `artifacts/current/`. Use the packaged data, not
machine-specific absolute paths from historical run configurations. The
snapshot preserves original files and hashes; the portable package code lives
under `src/cathode_cvae/`. Scripts inside the frozen snapshot preserve historical
provenance and can contain old machine paths; use the package commands below.

```text
artifacts/current/
  core/data/                         Original learning arrays and geometry frames
  core/data/parents/                 Parent coordinates and site permutations
  core/checkpoints/                  Original mask and geometry checkpoints
  improvement_v1/training_*.npz       Three-count original/adaptation datasets
  improvement_v1/runs/               New frozen fits and learning curves
  improvement_v1/data/               Proposal attempts and cached MLIP scores
  improvement_v1/reports/final/       Final measured sampler comparison
  reports/                           Original report and supporting tables
  extras/learning_all_parents/        Existing 12/24/36/alternate-48 labels, all 5 models
  extras/geometry/                    Existing UMA/MACE additional geometry exports
```

Verify the snapshot and reproduce the stored-energy comparison:

```bash
cathode-cvae data verify --snapshot artifacts/current
cathode-cvae evaluate --snapshot artifacts/current --output-dir runs/varad_recompute
cathode-cvae evaluate-geometry --snapshot artifacts/current --output-dir runs/varad_geometry
```

The evaluation command uses existing proposal IDs and energies. It does not
generate structures or start MLIP calculations. Use a new output directory for
each run, preserve the supplied splits, and retain the command/configuration
with the outputs.

## Train and sample

The command help documents all options and defaults:

```bash
cathode-cvae train-mask --help
cathode-cvae generate-mask --help
cathode-cvae train-geometry --help
cathode-cvae generate-geometry --help
```

A small installation check, followed by generation from its checkpoint:

```bash
cathode-cvae train-mask \
  --data artifacts/current/improvement_v1/training_original.npz \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --kind ar_cvae --condition-mode energy --orientation canonical \
  --beta-final .1 --validation-beta 1 --device auto --epochs 2 \
  --output-dir runs/beihao_ar_smoke

cathode-cvae generate-mask \
  --checkpoint runs/beihao_ar_smoke/best.pt \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --n-li 24 --count 32 --condition 1 --device auto --seed 20261106 \
  --output runs/beihao_ar_smoke/proposals.npz
```

For a full reference fit, use the supplied run configuration rather than the
two-epoch smoke setting; the current new fits used up to 100 epochs and a
600-second fit cap. Hardware-dependent time caps can give different update
counts, so match optimizer updates for a controlled ablation.

The geometry model and its deterministic control use the same data:

```bash
cathode-cvae train-geometry \
  --data artifacts/current/core/data/geometry_uma_fixed.npz \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --kind cvae --device auto --epochs 150 --output-dir runs/beihao_geometry_cvae

cathode-cvae train-geometry \
  --data artifacts/current/core/data/geometry_uma_fixed.npz \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --kind regressor --device auto --epochs 150 --output-dir runs/beihao_geometry_regressor

cathode-cvae generate-geometry \
  --checkpoint artifacts/current/core/checkpoints/cvae_20261400/best.pt \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --input runs/beihao_ar_smoke/proposals.unique.npz --draws 3 --device auto \
  --output runs/beihao_ar_smoke/decoded.npz
```

The recommended current mask family is `ar_cvae`, trained on
`improvement_v1/training_round0.npz`, using the energy condition and canonical
orientation. This is the model **before** the extra adaptive fine-tuning round;
the largest dataset did not produce the best average result. The exact frozen
configurations and checkpoints remain in `improvement_v1/runs/`.

A short CPU fit is an installation check. A scientific comparison needs the
planned training budget and independent fit seeds. Sampled latent draws from
one checkpoint do not count as independent fitted models. Follow
[the controlled experiments](docs/COLLABORATOR_TASKS.md) before changing data,
loss weights, thresholds, or validation rules.

## Get energy, forces, and stress without operating an MLIP directly

The MLIP commands load the configured model and convert the results to the
common project units. The same commands work on a workstation, notebook GPU,
or allocated cluster GPU when the correct environment and driver are present.
They do not assume a Slurm hostname or a particular computing account.

```bash
cathode-cvae mlip doctor --help
cathode-cvae mlip fetch-models --help
cathode-cvae mlip score --help
cathode-cvae mlip relax --help
```

For example, after installing the backends and obtaining checkpoint access:

```bash
cathode-cvae mlip fetch-models --models uma mace sevennet chgnet m3gnet
cathode-cvae mlip doctor --model uma --device cuda --load
cathode-cvae mlip score --model uma --device cuda \
  --input runs/beihao_ar_smoke/proposals.unique.npz \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --output runs/uma_interface_check
```

This last command makes new inference calls. It is an example of the interface,
not a required step in the current-data-only midterm tasks. To evaluate supplied
POSCAR/CIF files, replace the NPZ input with the structure paths and omit
`--parent`. The corresponding `relax` command accepts `--cell fixed|variable`,
`--fmax`, `--stress-gpa`, and `--steps`; a successful process exit is not itself
physical convergence, so inspect the recorded final criteria.

`score` is a **single-point evaluation** of the input coordinates: it returns
energy in eV per cell, per-atom forces in eV/Å, and stress in eV/Å³. It does not
perform the electronic self-consistent iterations of DFT. `relax` moves atoms
(and optionally the cell) iteratively; it is a different operation. Requesting
a low-energy CVAE sample does not replace either measurement.

Generation saves every attempt in `proposals.npz` and `proposals.attempts.csv`,
and also saves a canonical, unique-row `proposals.unique.npz` for physical
scoring. Keep the attempt table for the denominator; score each unique geometry
once. Force vectors apply to the evaluated orientation, not any arbitrary
symmetry-related representation. Trajectory archives contain repeated lineage
IDs: export frame-specific IDs before scoring their individual frames.

Available teachers are UMA, MACE, SevenNet, CHGNet, and M3GNet. The current
occupancy comparison uses **UMA-S-1.2.1 / `omat`** and **MACE-MH-1 /
`omat_pbe`**, with FP32 and TF32 disabled. Changing a model head or precision
changes the experiment and must be recorded. Failed structures remain visible
in the output; do not silently remove them from a benchmark's denominator.

For this midterm, use cached scores first. New inference is useful for checking
the package or a separately agreed candidate set, but no additional oracle
campaign is needed to reproduce the reported comparison.

## Current finding

Across three independently trained seeds and Li counts 12, 24, and 36, the
energy-conditioned AR CVAE produced **35.57% / 40.06%** novel low-tail orbits
under UMA / MACE, versus **18.99% / 19.75%** for uniform fixed-count sampling.
The strongest low-energy training-mask mutation baseline reached **52.65% /
54.08%**. The CVAE improves on random sampling but does not yet beat this strong
local search baseline. The geometry CVAE improves starting coordinates, but a
deterministic regressor gives similar savings; a CVAE-specific geometry advantage
has not been established. [Read the definitions and controls](docs/RESULTS.md)
before interpreting these percentages.

## Use the wider existing label sets

The all-count and smaller-cell label files use the original scalar-label schema.
Convert them to trainer input without changing the protected splits:

```bash
cathode-cvae data prepare-learning \
  --labels artifacts/current/core/data/learning_O3_N48_pilot_uma.npz \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --splits artifacts/current/core/splits/lineage_splits.csv \
  --weighting equal-composition --output runs/main_all_counts_uma.npz
```

This command measures no new energies. It validates parent identity and canonical
keys, preserves the original split table, and excludes diagnostic/pilot-holdout
rows. For the main parent, trainable counts are 2–46; counts 0, 1, 47 and 48
remain available for coverage and deterministic endpoint checks. For the 12-site
parent, use `extras/learning_all_parents/learning_Hu_O3_12_uma.npz` and its
`parents/Hu_O3_12.npz`; the existing non-diagnostic split covers counts 5–7.
All 87 label rows remain available in the source for exact finite-support lookup.
Change the teacher suffix to use the other supplied MLIP labels. Train separate
compatible models for different parents; this adapter does not create size transfer.

## Other cathodes and structure sizes

Parent preparation and MLIP scoring take a user-supplied pristine structure and
target species, so the surrounding workflow can use Na instead of Li or a
different cathode. The supplied dense and recurrent networks/checkpoints use a
fixed site list and coordinate frame. The current fast training/export canonicalizer supports 1–62 removable sites;
parent preparation and MLIP scoring can handle larger structures. The networks
must be trained for a compatible parent; the 48-site checkpoint cannot be applied to a 12-site structure by
changing a command-line count. Cross-size and cross-material transfer are
future model-design and validation questions.

## Collaboration

Beihao owns generative training/methods; Varad owns baselines, evaluation, and
figures; Ricardo owns chemistry, data provenance, symmetry, and physical
interpretation. Work on separate branches and use new run directories. Commit
source, small configuration files, and checked summary tables; keep personal
tokens, environments, large raw calculations, and new unreviewed checkpoints
out of Git. Freeze the report tables on November 8 and check the approximately
five-page midterm body plus references before submission on November 9.

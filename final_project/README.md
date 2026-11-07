# Cathode vacancy CVAEs: midterm package

**Beihao and Varad: read [your tasks](docs/COLLABORATOR_TASKS.md), then use the
commands below.** The class question is whether a conditional generator can
produce new Li/vacancy arrangements at requested Li counts and low, middle, or
high teacher-energy ranges. A separate temperature-dependent acquisition policy
decides how often to request each range. The current parent is O3 LiCoO₂ with
48 removable Li sites.

This extends the existing binary energy-conditioned AR CVAE; energy conditioning
itself was already present. The benchmark uses MLIP energies of pristine-derived,
unrelaxed deletion structures. A low-energy request is not a certificate of a
minimum or a geometry relaxation. The temperature policy is **acquisition under
a static MLIP surrogate**, not measured configurational entropy or calibrated
reactor equilibrium.

The completed balanced benchmark is a negative result for the extension:
UMA unique-novel in-bin yield is 31.00% for multibin, 33.00% for uniform raw
sampling and 58.39% for requested-bin mutation. MACE gives the same ordering.
The useful course result is a reproducible comparison and a diagnosis of limited
conditional control, not an assumed CVAE win. See the exact denominator, seeds,
new versus historical results and figures in [results](docs/RESULTS.md).

Read these in order:

1. [Methods and data](docs/METHODS_AND_DATA.md): variables, bins, symmetry,
   autoregressive model, acquisition measure and units.
2. [Measured results](docs/RESULTS.md): historical versus new experiments,
   baselines, DFT ranking context and limitations.
3. [Collaborator tasks](docs/COLLABORATOR_TASKS.md): exact inputs, commands,
   expected outputs, acceptance checks and report insertion points.
4. [Report PDF](docs/midterm1/main.pdf), [LaTeX source](docs/midterm1/main.tex),
   [Ricardo's prepared contribution](docs/RICARDO_CONTRIBUTION.md), and
   [integration checklist](docs/INTEGRATION_CHECKLIST.md).

## Obtain the review branch

```bash
git clone --branch overnight/multibin-acquisition-20261107 \
  https://github.com/rsilvabuarque/ECE285-1.git ECE285-midterm
cd ECE285-midterm/final_project
```

Use a personal branch and new run directories. The historical
`artifacts/current/` snapshot, original splits and completed results remain
immutable. The new extension permits bounded CVAE training and MLIP checks;
**no new DFT or new MLIP fine-tuning campaign is authorized by this package.**

## Install without materials-specific setup

Python 3.11–3.13 is supported (3.12 was used in the historical GPU campaign).
Training works on CPU or a visible CUDA GPU. Use a PyTorch wheel compatible with
your platform's GPU driver; Python installation cannot upgrade that driver.
On a cluster, obtain the class's permitted compute allocation first. Commands
respect `CUDA_VISIBLE_DEVICES` and do not select an account or submit jobs.

For the required fits and cached analyses:

```bash
bash scripts/install.sh --python python3 --prefix .venvs --core-only
source .venvs/learning/bin/activate
python -m pip check
```

Existing users can activate their existing learning environment and update only
the project code:

```bash
python -m pip install -e '.[dev]'
python -m pip check
```

The new schema/allocation code needs no new MLIP backend installation. For
optional UMA/MACE inference on a fresh machine, install those providers:

```bash
bash scripts/install.sh --python python3 --prefix .venvs --backends uma,mace
source .venvs/learning/bin/activate
```

The default installer without `--core-only` or `--backends` installs all five
models in backend-specific environments (M3GNet and CHGNet share MatGL).
It installs all declared/transitive dependencies, runs `pip check`, records
resolved packages, and downloads hash-checked weights unless `--no-weights` is
supplied. Separate environments avoid incompatible backend dependencies.
UMA requires each model user's own approved Hugging Face access; authenticate
as instructed, and never put tokens or foundation weights in Git.

Use `--torch-index-url` for an appropriate official wheel channel. CPU-only
checks can use `https://download.pytorch.org/whl/cpu`. On NERSC, put environments
and the pip cache on native scratch via `--prefix` and `PIP_CACHE_DIR`; installing
thousands of files on the project filesystem can be slow. On a workstation the
local `.venvs` default is suitable. See `bash scripts/install.sh --help`.

## Retrieve and verify the measured packet

The frozen historical data/checkpoints are already under `artifacts/current/`.
The new packet keeps small tables/figures and its download manifest in Git;
new CVAE weights and larger attempt/scalar arrays are in a checksum-protected
release asset. The small `cost_receipt.json` is tracked separately in Git. Foundation-model weights and raw trajectories are not in it.

```bash
cathode-cvae data verify --snapshot artifacts/current
python - <<'PY'
import json, subprocess
from pathlib import Path
record = json.loads(Path('artifacts/overnight_20261107/download.json').read_text())
subprocess.run(['cathode-cvae', 'fetch-artifact', '--url', record['url'],
                '--sha256', record['sha256'], '--output-dir', record['extract_to']],
               check=True)
PY
```

The fetch command verifies the archive before extraction, rejects unsafe paths,
and preserves differing existing files. An unavailable asset or missing label is
not a successful experiment. Its manifest and completion receipt identify the
actual run, hashes, models, counts, and evaluation boundary.

```text
artifacts/current/                    Frozen November6 evidence and original data
artifacts/overnight_20261107/
  data/                              Schemas, independent reference, frozen keys
  fits/                              New multibin checkpoints/configurations/curves
  benchmark/                         Every benchmark attempt and unique scalar scores
  allocation/                        Temperature policies and request budgets
  demo/                              Executed acquisition-guided proposals/scores
  evaluation/benchmark/               Condition-aware benchmark tables/figures
  evaluation/demo/                    Requested-versus-realized acquisition results
  cost_receipt.json                   Label acquisition, training and evaluation cost
reports/ranking_audit_20261107/        Recomputed matched DFT context and examples
```

Raw force shards, foundation environments and later research trajectories stay
on HPC storage with retrieval manifests. Use the portable paths above; absolute
paths inside historical receipts are provenance, not runtime defaults.

## Reproduce the cached results

No MLIP is loaded by these evaluations:

```bash
cathode-cvae evaluate --snapshot artifacts/current \
  --output-dir runs/historical_recompute
cathode-cvae evaluate-geometry --snapshot artifacts/current \
  --output-dir runs/geometry_recompute
```

The new evaluator separates requested bin, teacher, temperature/policy, fit seed
and sampling seed. The complete benchmark command and its demo variation are in
[Varad's tasks](docs/COLLABORATOR_TASKS.md). Its entry point is:

```bash
cathode-cvae evaluate-conditions --help
```

Retain all attempts, including duplicates and failures. Novelty to the training
set differs from novelty to all prior history. MACE has its own training-derived
energy-bin boundaries; UMA absolute energy thresholds cannot classify MACE values.
Compare historical fixed-q20 yield on requested-low attempts, not a pooled mix of
low/middle/high requests. Fit seeds are not interchangeable with random baseline
replicates or latent draws.

## Train, reload, and request an energy range

The full reference reproduction is in [Beihao's tasks](docs/COLLABORATOR_TASKS.md).
The common inputs are protected round-0 data, the 48-site parent, and the frozen
UMA condition schema. Preserve its protocol string exactly:

```bash
cathode-cvae train-mask \
  --data artifacts/current/improvement_v1/training_round0.npz \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --condition-schema-in artifacts/overnight_20261107/data/condition_schema_uma.json \
  --condition-mode multibin --kind ar_cvae --counts 12 24 36 \
  --teacher uma --teacher-head omat --orientation canonical \
  --geometry-protocol "pristine-derived deletion; unrelaxed; exact canonical parent coordinates" \
  --beta-final .1 --beta-warmup 20 --validation-beta 1 \
  --epochs 100 --minimum-epochs 20 --patience 15 --max-seconds 600 \
  --seed 20261106 --output-dir runs/new_multibin_fit

cathode-cvae generate-mask \
  --checkpoint runs/new_multibin_fit/best.pt \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --n-li 24 --requested-bin middle --count 128 --seed 20261117 \
  --temperature-K 433.15 --allocation-policy independent-bin-check \
  --output runs/new_multibin_fit/middle_attempts.npz
```

Each valid row has exactly the requested Li count. `.attempts.csv` retains every
request with condition, schema, temperature/policy, seeds, checkpoint and status.
The main NPZ retains attempts, and `.unique.npz` contains unique canonical
geometries for physical scoring. Unsupported bins fail explicitly instead of
being silently redirected. Historical binary/composition checkpoints remain
loadable; their conditions have dimensions two/one, versus four for multibin.

A requested low/middle/high class does not prove the measured energy is in that
class. Use exact existing labels or a declared complete new scoring batch.
Uncached proposals remain unknown; do not substitute a nearby mask's energy or
drop misses from a supposedly complete yield. Temperature annotates the
acquisition request; it is not a decoder softmax temperature.

## Allocate a finite request budget by temperature

The completed packet used budget 128. This fresh example doubles it to 256:

```bash
cathode-cvae allocate \
  --reference artifacts/overnight_20261107/data/reference.npz \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --schema artifacts/overnight_20261107/data/condition_schema_uma.json \
  --reference-kind uniform_raw_sample --counts 12 24 36 \
  --temperatures 393.15 433.15 473.15 493.15 \
  --mixture-weights 1 1 1 1 --exploration .1 --budget 256 \
  --output-dir runs/new_allocation
cathode-cvae generate-allocation \
  --allocation runs/new_allocation/allocation.json \
  --checkpoints \
    20261106=artifacts/overnight_20261107/fits/multibin_v2_20261106/best.pt \
    20261107=artifacts/overnight_20261107/fits/multibin_v2_20261107/best.pt \
    20261108=artifacts/overnight_20261107/fits/multibin_v2_20261108/best.pt \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --device auto --sampling-seed 20261107 --output-dir runs/new_acquisition
```

`allocation.json`/`requests.csv` save pure static-weight and exploration plans.
A per-temperature plan independently totals its budget; a mixture plan has one
joint temperature/bin budget. `generate-allocation` executes **all** schedules
in the file, for every supplied fit: pure and exploration per-temperature
schedules plus both temperature-mixture schedules. With three compositions,
four temperatures, three fits and the budget 256 above, that is
3 × (2 × 4 + 2) × 3 × 256 = **23,040 attempts**, not 256. Use a smaller
allocation budget for a quick check. The receipt records each schedule and its
integer request counts. See [the acquisition guide](configs/acquisition/README.md)
and the completed demo receipt for its executed schedules.

Weights use total eV per cell at fixed composition. In an exhaustive orbit list,
multiplicity counts raw occupation patterns. In a uniformly sampled raw-mask
reference, retained raw frequencies already carry that measure: never multiply
by multiplicity again. The entire prespecified reference must have energies;
cache-hit-only selection would bias it. Report ESS, largest weights and
pool-size sensitivity. The 10% exploration floor is a chosen algorithmic setting,
not a thermodynamic constant. Actual generated energy-bin proportions can differ
from the requested allocation.

The 12-site exact reference and its multiplicities can be checked without fitting
a model; the command and acceptance criteria are in Varad's tasks. Its 87 known
orbits are a numerical reference, not a large-data generation test or 87 new DFT
calculations.

## Get energy, force, and stress through the common interface

The wrapper selects the pinned model/head and backend environment. Default
precision matches the campaign: FP32, TF32 off, and no implicit dispersion or
accelerator substitution. On an allocated/class GPU:

```bash
cathode-cvae mlip doctor --model uma --device cuda --load
cathode-cvae mlip score --model uma --device cuda \
  --input runs/new_multibin_fit/middle_attempts.unique.npz \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --output runs/uma_interface_check
```

This makes new inference calls. Required cached midterm analyses do not need it.
The same `score` accepts POSCAR/CIF/extxyz paths without `--parent`. Energies are
eV per cell, forces eV/Å, and stress eV/Å³ in ASE tensile-positive
`xx yy zz yz xz xy` order. Force vectors belong to the evaluated orientation.
`relax` is a separate operation with explicit fixed/variable-cell mode, force/
stress criteria and step cap; successful process exit alone is not physical
convergence. Frame archives need frame-specific IDs before individual scoring.

The default teachers available through the infrastructure are UMA, MACE, SevenNet,
CHGNet and M3GNet. The current class extension uses UMA-S-1.2.1/`omat` and
MACE-MH-1/`omat_pbe`; changing a checkpoint/head/protocol changes the experiment.
No new MLIP fine-tuning or DFT is performed by these commands.

## Wider counts, other parents, and the secondary geometry model

The original main-parent labels cover counts0–48. `data prepare-learning` joins
those existing labels to protected splits; diagnostics/pilot holdouts remain
excluded. Separate 12/24/36/alternate-48 parent labels are available in the frozen
snapshot. Parent preparation accepts a user-provided pristine structure and
removable element (for example Na), and scoring uses the actual site map.
The current networks use a fixed site list; new material/cell requires a
compatible fit. The fast integer trainer supports1–62 removable sites. Padding
or truncating 48-site weights does not establish size transfer.

The displacement CVAE is a separate fixed-cell learned-start model. Its old
regressor control and data remain supplied. It predicts unminimized coordinates,
not guaranteed force-minimized structures. Its reproduction is optional for this
midterm; the new core is multibin controllability and acquisition. See
[methods](docs/METHODS_AND_DATA.md) and [results](docs/RESULTS.md).

## Verification and boundaries

The preserved [November6 validation](docs/VALIDATION.md) covers the portable base.
The new completion receipt, tests and claim-source map identify the actual
November7 campaign. Reports distinguish old evidence, new completed measurements,
prepared infrastructure and independent teammate work. Good teacher ranks or
correct request totals do not establish DFT accuracy, global minima, physical
entropy, reactor equilibrium or downstream MLIP improvement. The later MLIP-only
thermal pilot and prioritized DFT-required-next package are separate from the
class result. Nothing submits or schedules DFT or sends messages to teammates.

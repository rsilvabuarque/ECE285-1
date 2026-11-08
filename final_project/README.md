# Cathode vacancy generation: current midterm packet

**Beihao and Varad: start with [your independent tasks](docs/COLLABORATOR_TASKS.md).**

The November 8 study tests whether energy-conditioned vacancy generation benefits
from a variational latent, or mainly from a spatial autoregressive decoder. It
also compares training labels from pristine-derived deletions with labels from
fixed-cell relaxed endpoints. The working example has 48 Li sites in O3 LCO;
requested counts are 12, 24 and 36. The parent/mobile-ion interface is general,
while the present physical evidence and neural checkpoints are LCO-specific.

**Current status:** 37 training trajectories, each with 4,000 optimizer updates,
are complete. Models and proposals are frozen; `H_FINAL` contains 344,001 known
occupation orbits. Final UMA/MACE scoring and the frozen endpoint-assessment
queue are running. Final yields and completion/cost receipts are pending. These
37 trajectories span different methods and screens; they are not 37 independent
replicates of one selected model.

The current packet belongs under `artifacts/overnight_20261108/`. Its verified
download and portable fetch/replay entrypoint are pending assembly and validation.
The planned `scripts/overnight_20261108/replay_packet.py` is not present at this
revision. The sections below identify the analysis interfaces already available.
Use the published manifest and actual helper once they are supplied.

Read the [new midterm source](docs/midterm_20261108/main.tex),
[research supplement](docs/research_supplement_20261108/main.tex), and
[collaborator handoff](docs/COLLABORATOR_TASKS.md). Local PDF builds are
`docs/midterm_20261108/main.pdf` and
`docs/research_supplement_20261108/main.pdf`; the verified packet will carry the
reviewed PDFs and their build inputs. The main report targets five body pages
plus references. The separate research supplement retains the finite-cell,
NEB and harmonic-estimator results and their physical limits.

Development evidence supports the spatial decoder: a matched no-latent AR model
has similar observed yield to the graph-posterior CVAE, while a dense posterior can reconstruct well
without useful prior-generation improvement. The final comparison remains
separate. A generated mask has exact composition; its energy class must be
measured, and an endpoint-energy condition still requires an actual relaxation
to obtain relaxed coordinates. No new DFT, MD campaign or MLIP fine-tuning is
needed for the midterm.

The [prior published packet](https://github.com/rsilvabuarque/ECE285-1/releases/tag/ece285-conditioning-20261108)
uses `artifacts/conditioning_20261107/`; the older balanced benchmark uses
`artifacts/overnight_20261107/`. Their results, protected splits, installation
procedures and download links are preserved in the historical reference section
below. They have distinct assessment histories and thresholds.

## Current packet analysis and collaboration

Use a personal branch and fresh output directories. On the current review
branch, the following interfaces inspect frozen data and make no MLIP calls:

```bash
python scripts/overnight_20261108/evaluate_controls.py --help
python scripts/overnight_20261108/paired_evaluate.py --help
```

`evaluate_controls.py` consumes the frozen proposal directory, parent, history,
training data, repeated `--score TEACHER=scalarNPZ` inputs and a separate
`--schema TEACHER=frozenJSON` for each teacher. `paired_evaluate.py` accepts
portable `--facts` and `--facts-sha256` instead of an HPC queue, along with the
original attempts, registration, history and separate deletion/endpoint schemas.
Exact input paths and the one-command replay will be added with the verified
packet manifest.

Cached analysis and independent model checks need only the core learning
environment below. Optional complete new energy/force/stress checks use the
[common MLIP interface](#get-energy-force-and-stress-through-the-common-interface).
It installs provider dependencies in isolated environments and respects the
visible CPU/GPU resources; students do not need to operate the providers' native
materials workflows or use Ricardo's personal allocation.

Beihao independently verifies the matched latent/no-latent comparison. Varad
recomputes raw-attempt, novelty, paired-label and cost results. Ricardo reviews
geometry, teacher interpretation and physical limits, integrates returned work,
and submits manually. Automated runs and prepared commands are evidence for
these checks, not their completed independent contributions.

The saved course deadline is **November 9, 2026, 11:59 p.m. Pacific**. The final
report is due **December 12**; presentations are **November 24 and December 1**,
remote during lecture time. Ricardo should check any newer course announcement
before manual submission.

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

## Historical reference: November 7 study and earlier packets

These are the preserved instructions and results from the prior study, published
early November 8. Its claims about work remaining apply to that historical
checkpoint. Current assignments and final status are described above and in
[the collaborator tasks](docs/COLLABORATOR_TASKS.md).

<details>
<summary>Open the historical results, branches, downloads and reproduction commands</summary>

### Prior conditioning, graph and edit study

**Beihao and Varad: start with [your tasks](docs/COLLABORATOR_TASKS.md).**
The current study asks which training choices, latent sizes, and physical-neighbor
features improve generation at requested composition and energy range. A
seed-conditioned edit model is compared with the same frozen mutation seeds and
swap budgets in a separate completed assessment after the required report gate. The O3 LiCoO₂ example
has 48 possible Li sites; new parent/graph interfaces also retain general host
chemistry, mobile-site indices, cell and periodic topology.

Read the [controlled-study explanation](docs/CONDITIONING_STUDY.md),
[frozen protocol](artifacts/conditioning_20261107/protocol.json), and
[actual experiment ledger](reports/conditioning_20261107/experiment_ledger.json).
The current compiled report is [main.pdf](docs/midterm1/main.pdf), with LaTeX
sources alongside it. The completion/report receipts distinguish an early interim
draft from the required completed packet. Implemented code or a planned run is
not a measured result.

The [reviewed PDF](https://github.com/rsilvabuarque/ECE285-1/releases/download/ece285-conditioning-20261108/ECE285_midterm_20261108.pdf)
and [both verified result archives](https://github.com/rsilvabuarque/ECE285-1/releases/tag/ece285-conditioning-20261108)
are published. The [completion receipt](reports/conditioning_20261107/completion_receipt.json)
links exact commands, final resource accounting, preserved failures and remote
verification. All seven owned allocations are released.

The [paired deletion/relaxation audit](reports/conditioning_20261107/relaxed_target_audit/README.md)
uses existing 1000 fixed-cell UMA lineages; it needs no new reference calculations.
The [parent/descriptor audit](reports/conditioning_20261107/parent_graph/receipt.json)
checks physical periodic edges, full host chemistry, and symmetry invariance.
Keep the original protected splits and historical artifacts immutable.

The frozen final comparison is complete: graph encoder plus causal decoder
context gives **62.04% UMA /63.45% MACE** balanced useful yield, versus frozen
mutation **58.62/57.29%**. UMA original-q20 low-request yields are 63.54% versus
52.52%. The final history was frozen before 34,560 UMA attempts and a matched
20,736-attempt MACE audit. Independent raw-mask replay reproduces all 16 result
tables. This is a teacher-defined conditional-sampling result. Context's tiny KL
and weak latent response indicate that its variational pathway is nearly unused;
a matched spatial no-latent control remains important. The main report and
[results](docs/RESULTS.md) retain that limitation and the negative comparisons.

The [optional supplement](docs/OPTIONAL_EXTENSIONS_20261107.md) keeps the later
experiments separate. Learned editing improves balanced requested-bin yield by
2.14 percentage points over matched mutation, with tied low-quintile yield and
metric-dependent low-energy behavior: lower best novel children in 7/9 cases,
but a less favorable paired average. Posthoc Gaussian latent proposals improve
balanced dense-CVAE yield while reducing low yield; they do not replace the graph
context recommendation. The small temperature follow-up repeats identical coarse
allocations and paired structures across all four temperatures. Defined-hop UMA
NEB paths reach numerical criteria, but one forward span changes from 0.638 to
0.189 eV/hop after coupled image/force refinement. That sensitivity and the MACE
single-point-only check remain explicit; later physical validation is required.
Extension archive availability is determined by the final download
manifest and verified packaging receipt, not by the existence of these scripts.

The sections below retain installation and November 7 reference-reproduction
commands. Exact new-study commands and selected policies are recorded with the
new packet and collaborator tasks.

The earlier multibin extension built on a binary energy-conditioned AR CVAE;
energy conditioning itself was already present. The benchmark uses MLIP energies of pristine-derived,
unrelaxed deletion structures. A low-energy request is not a certificate of a
minimum or a geometry relaxation. The temperature policy is **acquisition under
a static MLIP surrogate**, not measured configurational entropy or calibrated
reactor equilibrium.

The completed controlled **development-policy** comparison selected balancing
and validation selection weight 0.1 after six 4000-update fits. Its balanced UMA
yield is 32.12%, compared with 33.62% uniform and 58.97% frozen mutation. This is a
modest training-policy effect, not a final assessment or a CVAE win. Paired
requests show weak energy response despite measurable latent use. See
[measured scopes and diagnostics](docs/RESULTS.md) for exact definitions;
remaining core/final statuses are explicit in the ledger.

The separate latent-size development sweep retained D4: 32.64% balanced useful
yield and 33.16% original-q20 low-request yield, against 33.04/19.62% uniform and
59.72/52.08% mutation in that namespace. D8 reused its existing weights; its new
Monte Carlo outcome is not another training round. The predeclared size rule
and all individual seeds are saved. Larger latent size did not systematically
improve discovery, even though reconstruction improved.

The three-fit **spatial development** comparison is complete. Adding strict-past
neighbor summaries to the graph-encoder decoder gives 60.88% balanced and 61.11%
original-q20 low-request yield, versus 58.80/50.35% mutation. The graph encoder
alone gives 34.49/24.31%. These remain development comparisons; the separately
frozen final assessment is reported above. Paired requests show much stronger energy control, but the
context model's very small KL and weak response to latent changes indicate a
nearly unused latent pathway. The supported finding is a useful conditional
spatial AR mechanism; the benefit of its variational component remains unresolved.
The one-fit structural-descriptor pilot did not improve joint useful yield.
See [the full interpretation](docs/CONDITIONING_STUDY.md) and source-bound
[results](docs/RESULTS.md); final status remains explicit in the ledger.

The completed November 7 balanced benchmark is a negative reference result:
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
git clone --branch midterm/conditioning-graph-edit-20261107 \
  https://github.com/rsilvabuarque/ECE285-1.git ECE285-midterm
cd ECE285-midterm/final_project
```

Use a personal branch and new run directories. The historical
`artifacts/current/` snapshot, original splits and completed results remain
immutable. The new extension permits bounded CVAE training and MLIP checks;
**no new DFT or new MLIP fine-tuning campaign is authorized by this package.**

## Retrieve and verify the measured packet

The frozen historical data/checkpoints are already under `artifacts/current/`.

Fetch the required November8 packet before its cached replay. No MLIP weights
are needed. Use a permitted analysis allocation on a cluster:

```bash
python - <<'PY'
import json, subprocess
from pathlib import Path
record = json.loads(Path('artifacts/conditioning_20261107/download.json').read_text())
subprocess.run(['cathode-cvae', 'fetch-artifact', '--url', record['url'],
                '--sha256', record['sha256'], '--output-dir', record['extract_to']],
               check=True)
PY
python scripts/conditioning_20261107/replay_cached_results.py \
  --packet artifacts/conditioning_20261107 \
  --output-dir runs/conditioning_20261107/portable_replay --plots
```

Use `--print-training-command graph_context_20261106` with that script to print
the portable measured fit command without executing training. The payload
contains all 28 core fit folders and both saved selection rules; its manifest
binds exact data, history, masks, scalar labels and checkpoint hashes.

The optional companion has a separate manifest and depends on the same core ZIP.
The following command keeps both downloaded ZIPs, checks their SHA256 values,
and replays all seven optional analysis jobs in a fresh directory. It needs the
learning environment, with zero MLIP calls and zero neural training updates:

```bash
python - <<'PY'
import hashlib, json, shutil, subprocess, sys, urllib.request
from pathlib import Path
from urllib.parse import urlparse

manifests = [Path('artifacts/conditioning_20261107/download.json'),
             Path('artifacts/conditioning_20261107/extensions_v2/download_ext.json')]
folder = Path('downloads/conditioning_20261107')
folder.mkdir(parents=True, exist_ok=True)
archives = []
for manifest in manifests:
    record = json.loads(manifest.read_text())
    target = folder / Path(urlparse(record['url']).path).name
    if not target.exists():
        with urllib.request.urlopen(record['url']) as source, target.open('xb') as dest:
            shutil.copyfileobj(source, dest)
    with target.open('rb') as source:
        actual = hashlib.file_digest(source, 'sha256').hexdigest()
    if actual != record['sha256']:
        raise ValueError(f'Checksum mismatch; preserve and inspect {target}')
    archives.append(target)
subprocess.run([sys.executable, 'scripts/conditioning_20261107/replay_extensions.py',
                '--core-archive', str(archives[0]), '--extensions-archive', str(archives[1]),
                '--download-ext', str(manifests[1]),
                '--output-dir', 'runs/varad_optional_replay'], check=True)
PY
```

Small tables/figures, download manifests and final receipts remain in Git;
CVAE weights and larger attempt/scalar arrays are in the authenticated release
assets. Foundation MLIP weights remain separate. Global raw force/stress shards
stay on HPC with their hash inventories; the optional archive includes the
bounded NEB pilot's own bands and structural exports.

For the earlier November7 reproduction, fetch its separate historical packet:

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
The November7 core suite passed 67 tests, the public run bundle passed download
and hash verification, and its cached benchmark tables replayed byte for byte.
UMA and MACE produced the actual GPU campaign labels. The three optional
backends also passed [single-point CPU interface checks](reports/portable_backend_smoke_20261107/README.md)
in their installed environments: SevenNet, CHGNet and M3GNet returned finite
energy/force/stress arrays without changing the input geometry. Those one-case
checks establish operability; their startup-inclusive timings are not a model
speed or accuracy comparison. Fresh gated downloads and the class GPU remain
separate platform/access checks.
The [completion receipt](artifacts/overnight_20261107/phase1_gate.json), tests and claim-source map identify the actual
November7 campaign. Reports distinguish old evidence, new completed measurements,
prepared infrastructure and independent teammate work. Good teacher ranks or
correct request totals do not establish DFT accuracy, global minima, physical
entropy, reactor equilibrium or downstream MLIP improvement. The later MLIP-only
thermal pilot and prioritized DFT-required-next package are separate from the
class result. Nothing submits or schedules DFT or sends messages to teammates.

The separate [thermal method pilot](docs/thermal_pilot/RESEARCH_REPORT.md) and
[DFT-required-next package](docs/thermal_pilot/DFT_REQUIRED_NEXT.md) document
Ricardo's later research extension. They are not prerequisites for Beihao's or
Varad's required cached-data tasks. The extension keeps same-MLIP numerical
comparisons distinct from physical DFT validation and stores large raw results
outside Git with retrieval manifests.

</details>

# Independent midterm work for Beihao, Varad, and Ricardo

The current study diagnoses conditional energy control and tests training policy,
latent dimension, physical-neighbor information, symmetry and structural targets.
It retains a genuine CVAE and the strong frozen mutation baseline. Its scientific
contract is `artifacts/conditioning_20261107/protocol.json`; actual status and
outputs are in `reports/conditioning_20261107/experiment_ledger.json`.
Prepared infrastructure and results do not mean your independent checks are done.

The midterm is due **November 9, 2026, 11:59 p.m. Pacific**. The supplied
announcement requests approximately five body pages excluding references,
covering the revised proposal, mathematical/model/algorithm details, and
novelty/advantages/limitations. Verify any newer course formatting instructions.
The supplied overview explicitly requires NeurIPS for the proposal, without
stating an additional midterm template. No coursework submission or messages are
automated.

## Start from the review branch and install

```bash
git clone --branch midterm/conditioning-graph-edit-20261107 \
  https://github.com/rsilvabuarque/ECE285-1.git ECE285-midterm
cd ECE285-midterm/final_project
bash scripts/install.sh --python python3 --prefix .venvs --core-only
source .venvs/learning/bin/activate
python -m pip check
cathode-cvae data verify --snapshot artifacts/current
```

Use your own branch and a new run directory for every experiment. Python
3.11–3.13 is supported; select a PyTorch wheel compatible with the class GPU.
The core installs ordinary and transitive dependencies, including ASE, SciPy and
spglib for physical graphs. Training and cached analyses do not load an MLIP.
Obtain the class's permitted compute allocation first on a cluster. Commands
respect visible GPUs and do not choose an account, queue or scheduler.

The new larger packet is retrieved from the checksum-protected release described
by `artifacts/conditioning_20261107/download.json` once publication completes:

```bash
python - <<'PY_FETCH'
import json, subprocess
from pathlib import Path
r = json.loads(Path('artifacts/conditioning_20261107/download.json').read_text())
subprocess.run(['cathode-cvae', 'fetch-artifact', '--url', r['url'],
                '--sha256', r['sha256'], '--output-dir', r['extract_to']], check=True)
PY_FETCH
```

This installs measured project artifacts, not foundation-model weights.
A missing file or incomplete score batch is not a successful reproduction.
Historical absolute paths inside receipts are provenance, not portable defaults.
Keep protected train/validation/test orbit roles and old artifacts unchanged.

## Beihao: reproduce a fit and explain the model

Budget about 6–8 focused hours. Your deliverables are an independent fit receipt,
checkpoint/count/latent/graph checks, learning curves and one controlled ablation.

### 1. Reproduce the dense reference and inspect the retained spatial design

The completed development comparisons selected class balancing, validation
selection beta 0.1, and latent dimension 4. Inspect
`artifacts/conditioning_20261107/selected_policy.json` and `selected_models.json`:
they bind the actual retained checkpoint identities. The trajectory writer
retains primary validation beta 1 for its legacy `best.pt`; the selected checkpoint
is explicitly `best_validation_beta_0p1.pt`. The following reproduces the selected
dense reference. The separate spatial development comparison recommends graph
encoder plus causal decoder context; its exact checkpoint identities and
development-only status are in
`artifacts/conditioning_20261107/architecture_selection.json`. Final assessment
and any later editor retain separate roles.

```bash
cathode-cvae train-mask \
  --data artifacts/current/improvement_v1/training_round0.npz \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --condition-schema-in artifacts/overnight_20261107/data/condition_schema_uma.json \
  --kind ar_cvae --condition-mode multibin --latent-dim 4 --encoder dense \
  --orientation canonical --counts 12 24 36 --teacher uma --teacher-head omat \
  --geometry-protocol "pristine-derived deletion; unrelaxed; exact canonical parent coordinates" \
  --seed 20261106 --epochs 100 --minimum-epochs 100 --patience 100 \
  --max-updates 4000 --batch-size 512 --learning-rate .001 \
  --beta-final .1 --beta-warmup 20 --validation-beta 1 --selection-betas 1 .1 \
  --class-balance --max-seconds 1200 --device auto --threads 4 \
  --output-dir runs/beihao_selected_d4
```

Expected files are `configuration.json`, `history.csv`, `latent_history.jsonl`,
`summary.json`, `best_validation_beta_1.pt`, `best_validation_beta_0p1.pt`, and
`best.pt` for the primary validation rule. Inspect the summary: the trajectory
must actually reach 4000 updates, rather than merely exit successfully. A time-cap
stop on another GPU is a bounded partial reproduction; report it as such.

To reproduce the recommended spatial configuration on your class GPU, repeat
this exact command with `--encoder graph --graph-context` in place of
`--encoder dense`, and use `--output-dir runs/beihao_graph_context_d4`. Keep
`--latent-dim 4`, the canonical orientation, seed, data, 4000 updates and both
selection rules unchanged. This second run is a separately recorded reference,
not another seed in the already frozen integrating campaign. Record the expected
37,821 parameters, versus 30,793 for the dense reference. If available time only
permits one full fit, prioritize the spatial design and inspect the saved dense
checkpoint rather than representing a partial dense run as complete.

Acceptance checks:

- Parent/data/schema hashes and 40157/4967/5123 protected rows match the source.
- Condition dimension 4 means Li fraction plus low/middle/high one-hot. It happens
  to equal the retained latent dimension 4; those are independent concepts.
- Explicit latent dimension 4, hidden 64, optimizer settings, orientation and weighted
  reconstruction/KL conventions agree. KL remains summed across latent dimensions.
- Both checkpoint rules use the same trajectory and fixed validation noise.
  Record selected epochs and the checkpoint actually used for generation.
- No protected test labels select a checkpoint or a new policy.

### 2. Reload and check exact composition

```bash
for bin in low middle high; do
  cathode-cvae generate-mask \
    --checkpoint runs/beihao_selected_d4/best_validation_beta_0p1.pt \
    --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
    --n-li 24 --requested-bin "$bin" --count 64 --seed 20261117 \
    --output "runs/beihao_selected_d4/${bin}_attempts.npz"
done

python -m pytest -q tests/test_latent_policy.py tests/test_parent_graph.py \
  tests/test_descriptor_conditions.py
```

Every valid mask is binary and contains exactly 24 Li. Keep all 192 attempts,
including duplicates or known masks. The command records a requested condition,
not a measured energy. Uncached generated structures have unknown energy until
scored; never transfer a neighbor mask's label or report cache-only yield.

Explain the encoder Gaussian, prior sampling, teacher-forced GRU decoder, hard
count constraint and reconstruction/KL objective. Plot reconstruction and KL
against actual updates with both selected checkpoints marked. Explain the
per-component KL and encoded-mean variance, including cutoff sensitivity. A
larger latent dimension is not a physical convergence parameter.

For graphs, independently verify periodic image edges, host/mobile chemistry,
vacant-site nodes, nearest-shell descriptor invariance, and the no-future-feature
test. Explain why a graph encoder alone does not directly provide the generation
decoder with physical neighborhoods. An engineered causal neighbor summary is
not a learned graph decoder or a proof of full equivariance.

Explain the important negative diagnostic in the spatial result. Despite stronger
energy control, graph-context has KL only 0.007–0.011 nats, no pooled active mean
component even at variance cutoff 0.001, and little response to latent changes.
Its generation can remain random because the AR decoder samples uniform random
numbers at every free step. Distinguish useful conditional AR sampling from an
established benefit of the variational latent. Independently review the actual
trained-checkpoint future-gradient and future-mask tests under
`reports/conditioning_20261107/trained_causality_audit.json`; a synthetic test alone
does not establish this property of trained weights.

### 3. Priority controlled ablation: does the spatial sampler need a latent?

The strongest current model responds to energy requests but nearly ignores its
latent, so isolate that question before another balancing or latent-size sweep.
Compare your reproduced graph-context CVAE with a no-latent spatial AR control.
Use the same parent, data, split, conditions, canonical orientation, hidden size
64, fit seed, batch size, learning rate, class balancing and 4000-update budget.
Fit both from scratch; do not warm-start an incompatible zero-latent model from
a D4 checkpoint. Repeat the full command above, replacing its model/output flags
with these control arguments:

```bash
--kind ar_baseline --encoder dense --graph-context --latent-dim 0 \
--output-dir runs/beihao_spatial_ar_no_latent
```

Here `--encoder dense` is the required CLI default for an AR model with **no
encoder**; `--graph-context` still constructs the physical parent graph and the
same six causal decoder summaries. Do not pass `--encoder graph`: that requests
an encoder the no-latent model does not have, and is rejected. The control removes
the variational channel and its training encoder while retaining the generation
context/GRU/count mechanism. Save actual parameter counts; this is not an exact
equal-parameter comparison. Keep both checkpoint-selection settings recorded;
the no-latent KL is identically zero, so their objectives coincide.

First run a separately labeled two-update smoke test, reload it, check exact
count and the no-future-feature behavior, then launch the full bounded fit in
another directory. Compare protected validation reconstruction, sampled novelty,
and paired low/middle/high response under a new independent DEVELOPMENT namespace
with shared uniforms and equal attempt/scoring budgets. If energy checking uses
the class GPU, call the package's MLIP scorer; no new DFT is needed. Unknown
labels remain unknown, never a cache-only success denominator. This control has
not been performed in the integrating campaign; record negative outcomes too.

**Report return:** one model paragraph, one learning-curve figure, and a small
configuration/results table for Sections 2–4, with actual run commands and limits.
Your independent fit does not retrospectively create another seed in the
integrating campaign's frozen final assessment.

## Varad: recompute outcomes and validate the comparison

Budget about 6–8 focused hours. Your main outputs are an independent metric
receipt, seed-resolved plots and an explanation of the actual denominators and
costs. No fresh MLIP or DFT labels are required for these cached checks.

### 1. Check historical and new scopes separately

```bash
cathode-cvae evaluate --snapshot artifacts/current \
  --output-dir runs/varad_historical_recompute
cathode-cvae evaluate-geometry --snapshot artifacts/current \
  --output-dir runs/varad_geometry_recompute
```

Reproduce historical binary UMA/MACE yields 35.57/40.06%, uniform 18.99/19.75%,
and strongest mutation 52.65/54.08%. These use the original history and 512 attempts
per group. The November 7 balanced multibin figures 31.00/33.00/58.39%UMA and its
requested-low figures 32.99/17.49/51.65% are a separate assessment. They do not
become new training-policy results by being placed in a common table.

The new policy development table compares six 4000-update trajectories and two
checkpoint rules. Verify selected-policy mean balanced yield 32.118%, versus
uniform 33.623% and frozen mutation 58.970%. Its requested-low/original-q20 yield is
30.729%, versus 21.354% uniform and 51.736% mutation. Those are distinct metrics.
The public payload retains the attempts, scalar scores and frozen history needed
to reproduce the corresponding `evaluate-conditions` command. The exact
final-study replay is:

```bash
python scripts/conditioning_20261107/replay_cached_results.py \
  --packet artifacts/conditioning_20261107 \
  --output-dir runs/varad_final_replay --plots
```

It requires only the published checksum payload and learning environment, not
a GPU/MLIP call. Verify frozen-final graph-context balanced UMA/MACE yield
62.037/63.455%, versus mutation 58.623/57.292%. Separate original-q20 low results
are UMA 63.542/52.517% and MACE 67.622/52.951%. Check all ten UMA families and the
prespecified matched six-family MACE subset, rather than only the headline rows.

Also reproduce the confirmed spatial development table: graph-context balanced
60.880% and original-q20 low 61.111%, versus mutation 58.796/50.347%. The matched
graph encoder without decoder context gives 34.491/24.306%, and symmetry-only
32.118/29.340%. Use `evaluation/spatial_confirmed` source tables and the full
`spatial_confirmed_diagnostics` evidence, not the earlier one-fit pilot. Check the
three individual graph-context balanced yields 59.028/60.938/62.674%. Verify the
one-fit structural-condition pilot separately: joint yield 28.125%, plain CVAE
28.516%, mutation 44.792%. No result from these development comparisons is
silently renamed final performance.

### 2. Check paired response, latent use and novelty

Use the saved output tables/figures under
`reports/conditioning_20261107/policy_diagnostics_v2/` and the later retained-model
diagnostics. The paired analysis command is
`scripts/conditioning_20261107/analyze_diagnostics.py --help`; its receipt binds
the exact proposal arrays, manifest, schemas, score file and history.

Verify that low/middle/high pairs share actual latent and AR-uniform fingerprints.
Changing latent vectors must hold the uniforms fixed. Posterior diagnostics use
only the documented validation targets. Recompute target Hamming distances from
raw masks; a saved Hamming scalar alone is not an independent check.
Show individual fit seeds, mean energy shifts and complete distributions rather
than treating generated draws as independent model replications.

For every useful success, check binary/count validity, measured bin, canonical
orbit identity, uniqueness and novelty to both training and frozen history.
All attempts remain denominators. A condition-independent proposal distribution
has average three-bin hit rate 1/3 before novelty filtering, even if it favors low
energies; this explains why balanced and requested-low metrics differ.

### 3. Check physical/data audits without new calculations

```bash
python reports/conditioning_20261107/audit_relaxed_targets.py \
  --snapshot artifacts/current --output-dir runs/varad_relaxed_target_audit

python scripts/conditioning_20261107/audit_parent_graph.py \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --data artifacts/current/improvement_v1/training_round0.npz \
  --energy-schema artifacts/overnight_20261107/data/condition_schema_uma.json \
  --output-dir runs/varad_parent_graph_audit
```

The first audit must retain 1000 lineages, match fixed-cell convergence and support,
and distinguish occupation classes from true PES basins. The second must verify
full host chemistry, periodic edge images, connected interlayer communication and
symmetry-invariant vacancy-pair counts. Descriptor thresholds and joint support
must use training rows only. Do not treat the descriptor as an independent
thermodynamic property merely because it adds another condition component.

### 4. Check mutation, edit and cost accounting

Frozen mutation chooses 1/2/3 distinct occupied-to-vacant swaps uniformly from a
fixed training seed pool. It has no distance rule, energetic acceptance or online
population update. If the learned editor is evaluated, verify identical seed
identities, requested classes, exact swap counts and budgets. A seed and child
must share a coordinate frame; independently canonicalized masks do not prove
local edits. Every evaluated outcome, including unfavorable children, matters.

Count decoder attempts, unique physical labels, exact cache hits, fit time,
relaxation/NEB calls if any, and allocated resource time separately. Summing
per-method encountered labels double-counts shared oracle evaluations. Use the
campaign cost receipt for actual fresh calls; a generic evaluator without
fresh/cache flags cannot infer those calls. Preserve failed attempts and their
repairs. Do not infer physical accuracy from a successful job or checkpoint.

**Report return:** one verified table, one paired-response or seed-resolved figure,
one concise validation/cost paragraph for Sections 3–5, and your exact commands.

## Ricardo: physical interpretation and final integration

Review the definition of the parent/mobile sites, occupation symmetry, teacher
and geometry protocol. Explain why deletion and relaxed labels can reorder
structures; the current paired primary cohort has only 49 train/11 validation/9 test
lineages and low-endpoint validation support 0/1/0, so a new label-target CVAE was
not manufactured from repeated frames. The geometry CVAE/regressor comparison
remains separate from occupation generation.

Review any optional edit/NEB supplement independently of the class core. Defined
MLIP barriers are neither DFT-validated barriers nor a trained kinetic generator;
no new DFT is submitted or scheduled. Keep claims of material/size transfer,
physical entropy, reactor optima and downstream MLIP improvement outside the
current evidence. Integrate the collaborators' actual returned work, verify the
current rubric/format, and submit manually before the deadline.

Complete independent checks on November 8, freeze the integrated tables/figures
that day, and reserve November 9 for final verification. Prepared commands do not
substitute for the named owners' independent contributions.

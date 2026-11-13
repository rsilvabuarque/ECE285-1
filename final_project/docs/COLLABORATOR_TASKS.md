# Current independent tasks: November 13

Use `main` as the shared starting point. The latest [overview](../reports/adaptive_20261112/overview.pdf)
and [technical report](../reports/adaptive_20261112/technical_report.pdf) supersede
claims below that adaptive bins or entropy are unimplemented. Earlier datasets,
midterm documents, results, and assessment boundaries remain preserved.

```bash
git clone --branch main https://github.com/rsilvabuarque/ECE285-1.git
cd ECE285-1
# Choose your own descriptive branch name.
git switch -c collaborator/my-analysis
cd final_project
bash scripts/install.sh --python python3 --prefix .venvs --core-only
source .venvs/learning/bin/activate
python -m pip check
python reports/adaptive_20261112/build_tables.py --run-root artifacts/adaptive_20261112
```

The table replay uses only committed aggregates. Original data and checkpoints
for retraining require the separately verified release/raw archive. No new
training, energy calls, DFT, or job submission occurs in the block above. Generic
MLIP installation and scoring are explained in the main README; use the class's
resources and your own approved access, with one provider process per visible
GPU. Do not launch dated site-specific orchestration scripts unchanged.

## Beihao: distinguish useful conditioning from a useful latent

Reproduce the report tables first. Independently inspect the 5-fit D=1/2/4
comparison and its paired AR controls. Explain why a deterministic encoder mean
can carry a microscopic structural fingerprint while posterior noise erases it
in sampled z. Compare the within-condition TRAIN-fitted probe with random graph
initialization; the latter is essential before attributing pair-order information
to training. Review genuine, condition-only, and shuffled physical-head controls,
plus mean supervision. Keep primary terminal checkpoints separate from
validation-selected sensitivity analyses.

Return one independently generated paired-effect figure and a short account of
what these tests do and do not establish. For a subsequent experiment, propose
one hypothesis and matched controls before training: for example, explicit
ordering conditioning versus a comparable latent-control mechanism. Preserve
protected identities and use a new assessment namespace. Do not select a new
model on old assessment results and call those results an untouched test.

## Varad: audit denominator, iteration, and entropy statistics

Regenerate all seven report tables from the compact packet and check each against
its SHA-bound source CSV. Reproduce the causal control's paired gains, then
separate novelty from condition fidelity in the second-round drop. Contrast
fixed-reference yield with actual observed endpoint growth; neither is a
substitute for the other. Check that rebinning changes loss weights as well as
labels, and that moving UMA condition edges never classify MACE absolute energy.

Independently recompute the N12 entropy sum with orbit multiplicities and verify
its high-temperature limit. Separate importance-weight ESS, entropy estimator
bias, and Wang–Landau energy-bin bias. Preserve the positive 1000 K proposal
result and negative 5000 K transfer. Return a source-bound figure/table and a
short methods audit. Compact tables alone cannot replay raw proposal exclusions;
use the complete HPC archive or verified data packet for that deeper check.

## Ricardo: physical scope, structures, and report integration

Review fixed-parent geometry and symmetry definitions, teacher/head/precision
receipts, and the interpretation of low and high energies at fixed composition.
Explain why unrelaxed fixed-parent rankings do not establish phase stability,
DFT accuracy, or equilibrium recycling conditions. Review representative novel
samples with the existing structure export/scoring interface; retain the exact
mask/parent identity for every example. Compare the two teachers as a sensitivity
check, not as independent proof of DFT correctness.

Integrate the independent teammate analyses for the presentations and final
report. Any later DFT labels, relaxation/MD tests, force/stress training, other
phases, size transfer, or MLIP fitting require a separate physical validation
plan. They are not needed to reproduce these completed sampling results.

The campaign has already completed 166 specified-budget fits and 469,672 MLIP
evaluation events. These automated experiments are supplied evidence; teammates
should describe their own independent work accurately. All owned allocations
are closed in the final resource receipt. Raw failures, inherited split caveats,
and negative results remain available.

# Historical November 8 midterm handoff

The following instructions retain their original packet, branch names, and
submission dates. Use them to reproduce that frozen study, not as the latest
branch/status instructions.

# Independent midterm tasks: Beihao, Varad and Ricardo

The current study is the November 8 packet at `artifacts/overnight_20261108/`.
Its final models and proposals are frozen: 37 completed training trajectories,
148,000 optimizer updates, and a final known-orbit history of 344,001. Final
static scoring and endpoint assessment are complete, with independent arithmetic
recounts. Fresh extraction of the final core archive reproduced all 36 checked
tables exactly. Use the release's completion/publication/resource receipts for
the final transport verification and allocation closure.

The integrating campaign already performed the matched latent/no-latent
controls. Your contribution is an **independent reproduction, audit and scientific
interpretation**, rather than claiming those automated fits as work you completed.
No new DFT, MD campaign or MLIP fine-tuning is required for the midterm. Current
cached data and the class GPU allocation are sufficient for the tasks below.

The midterm deadline saved in the supplied guidelines is **November 9, 2026,
11:59 p.m. Pacific**. It requests approximately five body pages excluding
references, with the revised proposal, mathematical/model/algorithm details and
novelty/advantages/limitations. The final report is due **December 12**;
presentations are **November 24 and December 1**, remote during lecture time.
Ricardo checks any newer announcement and submits manually. No message to
collaborators or coursework submission is automated.

## Start with the current checkout and core installation

The current integrating branch is `overnight/latent-relaxed-neb-20261108`.
Create your own branch and keep every new run in a fresh directory. The packet's
download manifest binds its URL and SHA256. After fetching it, use:

```bash
python scripts/overnight_20261108/replay_packet.py \
  --packet artifacts/overnight_20261108 --output-dir runs/varad_replay --plots
```

This uses cached labels and verifies hashes without model inference. The
historical commands later in this file retain their original packet names.
The [current README fetch block](../README.md#cathode-vacancy-generation-current-midterm-packet)
reads the actual URL/SHA256 from `artifacts/overnight_20261108/download.json`.
Its selected release is `ece285-overnight-20261108-v1`. Record your own replay
receipt and interpretation; the automated integrating run is not your independent contribution.

From the current project checkout:

```bash
cd final_project
bash scripts/install.sh --python python3 --prefix .venvs --core-only
source .venvs/learning/bin/activate
python -m pip check
```

If already inside `final_project`, omit the first line. Existing users can update
the package in their learning environment with `python -m pip install -e '.[dev]'`.
The installer includes required/transitive dependencies and verifies them. Python
3.11–3.13 is supported; Python 3.12 was used for these runs. On a cluster, obtain
the class's permitted allocation and use a compatible PyTorch wheel. Commands
respect `CUDA_VISIBLE_DEVICES`; they do not select a personal account or queue.

Cached metrics and model checks need no foundation MLIP weights. If a complete
new scoring batch is needed for an independent fit, the existing installation
procedure supports UMA/MACE providers:

```bash
bash scripts/install.sh --python python3 --prefix .venvs --backends uma,mace
```

The default full installer and all five backend procedures remain in
[the README](../README.md#install-without-materials-specific-setup). Each UMA
user supplies their own approved model access. Use the project's common scorer
for energies in eV/cell, forces in eV/Angstrom and ASE stress in eV/Angstrom³;
its receipts bind model/head, geometry and inference precision. Full new scoring
is optional for this midterm handoff because measured outcomes are supplied.
For new mask batches, pass the current packet's parent with
`--parent artifacts/overnight_20261108/data/parent.npz`; the scorer also accepts
POSCAR/CIF/extxyz geometries directly without a parent-mask argument.

## Beihao: independently verify the generative-model mechanism

Plan roughly 4–6 focused hours for a reproduction and interpretation. Begin with
the published current checkpoints/configurations and exact recorded training
commands. Keep the same parent, TRAIN/VAL orbit roles, conditions, feature schema,
seed, weighted loss, checkpoint-selection rule and 4,000-update budget. A
hardware time cap is a partial reproduction, even when a command exits cleanly.
Do not copy a historical dense-reference command and label it the new retained
spatial model.

The following is the measured A-shell fit command with only portable paths and
a fresh output directory substituted. Run it inside `final_project` on your
allocated GPU after downloading the current packet:

```bash
python -m cathode_cvae.learning.train_mask \
  --data artifacts/overnight_20261108/data/H0.npz \
  --parent artifacts/overnight_20261108/data/parent.npz \
  --condition-mode multibin \
  --condition-schema-in artifacts/overnight_20261108/data/condition_schema_uma.json \
  --label-inference-receipt artifacts/overnight_20261108/reports/H0_label_inference_receipt.json \
  --orientation canonical --hidden-dim 64 --graph-context \
  --context-kernel-mode radial_shells --context-shell-tolerance-A 0.00001 \
  --kind ar_cvae --encoder graph --latent-dim 4 \
  --beta-schedule reference_updates --beta-reference-updates-per-epoch 79 \
  --beta-warmup 20 --beta-final 0.1 --validation-beta 1 \
  --selection-betas 1 0.1 --class-balance --max-updates 4000 \
  --epochs 1000 --batch-size 512 --learning-rate 0.001 \
  --geometry-protocol 'pristine-derived deletion; unrelaxed; exact canonical parent coordinates' \
  --proposal-geometry-protocol 'pristine-derived deletion; unrelaxed; exact canonical parent coordinates' \
  --energy-target deletion --seed 2026110802 --device cuda --threads 4 \
  --max-seconds 1800 --output-dir runs/beihao_A_shell_0802
```

For its matched B-shell control, keep every other flag and seed, replace
`--kind ar_cvae --encoder graph --latent-dim 4` with
`--kind ar_baseline --encoder dense --latent-dim 0`, and choose
`--output-dir runs/beihao_B_shell_0802`. Here `dense` selects the no-graph-encoder
configuration; the AR baseline constructs no encoder or latent pathway, while
retaining the same physical causal decoder features. Assess the explicitly
selected `best_validation_beta_0p1.pt`, not an ambiguously named alias. A fit
stopped by the wall-clock cap before 4,000 updates is partial; preserve it and
report the actual updates rather than claiming a completed reproduction.

1. Independently reproduce a matched spatial CVAE/no-latent AR pair on the class
   GPU, or prioritize one full fit and thoroughly reload/audit the supplied
   matched checkpoint if time is short. The retained radial-shell features must
   match across the pair. Record actual parameter counts; removing an encoder
   changes parameter count, so this is not an equal-parameter experiment.
2. Verify exact requested Li counts, checkpoint reload, parent/species/cell
   hashes and strict-past decoder features. In training the encoder may see the
   full target mask; generation cannot. Remaining-count features use the known
   condition and the prefix, not future target bits. Review the saved trained
   causality evidence as well as your own software check.
3. Plot reconstruction and KL against actual optimizer updates. Inspect
   component KL, encoded-mean variance and the saved activity-cutoff sensitivity.
   Compare changing the condition with fixed latent/uniform draws against
   changing the latent with fixed condition/uniform draws. Raw Hamming change,
   orbit equality and absolute/RMS energy response answer different questions.
4. Explain why a strong prior sampler can use its latent weakly, and why the
   dense posterior's better reconstruction/active latents did not give useful
   prior-generation improvement. The expected signed energy change between
   exchangeable prior draws is zero even for a strong latent effect; it is not
   evidence of collapse. Retain negative findings.
5. Keep deletion-target and endpoint-target training separate. Their initial
   input masks/splits are paired, but their TRAIN-fitted class thresholds differ.
   The endpoint-conditioned mask generator still produces a pristine-derived
   deletion geometry; it does not output optimized atomic coordinates.

Use existing complete scoring pools for the main comparisons. A new independent
fit produces new masks whose energies can be unknown; report that honestly or
score its entire prespecified batch through the MLIP wrapper. A cache-only yield
would omit unknown outcomes. Personal ablations belong to a new DEVELOPMENT
namespace; previously inspected FINAL outcomes cannot select another model.

**Return:** your own run/config/checkpoint receipt, one update-resolved learning
figure, one paired condition/latent figure, and a short model interpretation for
Sections 2–3. Include exact commands, hashes, update counts and limitations.
Your independent rerun does not add a fit seed to the integrating campaign's
already frozen FINAL assessment.

## Varad: recompute raw outcomes, physical labels and costs

Plan roughly 4–6 focused hours. Start with cached scalar facts and all raw
attempts; no fresh MLIP or DFT calls are needed. The existing portable interfaces
are:

```bash
python scripts/overnight_20261108/evaluate_controls.py --help
python scripts/overnight_20261108/paired_evaluate.py --help
```

The first consumes repeated `--score TEACHER=scalarNPZ` and
`--schema TEACHER=frozenJSON` arguments. The paired evaluator accepts
`--facts`/`--facts-sha256` instead of the HPC queue, with the frozen registration,
attempts/history and two assessment schemas. Use the current packet's manifest
for exact input paths/hashes. The one-command replay above has been validated
against the complete extracted packet and requires no original HPC queue.

1. Recompute distinct novel in-request orbits divided by **all attempted
   proposals**, including duplicates, invalid requests, score failures and
   nonconverged endpoints. Average composition/request cells equally, then show
   individual fit/sampler replicates and descriptive spread. Repeat proposals
   are not independent fitted models. The 37 trajectories span several methods.
2. Verify each canonical mask against all retained parent symmetries and the
   frozen `H_FINAL` history. Confirm the history was fixed before final proposals
   and includes saved acquisition/research exposures. Novelty to TRAIN and to
   all known history are different quantities. Inspect identity-only history
   receipts without changing protected roles or using TEST labels for selection.
   The structural-distance audit minimizes over all parent operations but only
   1,024 reference TRAIN orbits/count. Its distance is an upper bound on the
   true all-TRAIN nearest distance; it does not certify far exploration.
   Mutation separately has a known TRAIN seed at raw distance exactly `2*k`,
   so its all-TRAIN nearest distance is at most `2*k`. Keep those facts separate
   from exact full-history orbit identity/novelty.
3. Keep three metrics distinct: requested-bin yield, the original fixed-q20
   **LOW-request deletion** metric, and relaxed-endpoint yield under its separate
   TRAIN-fitted thresholds. Score both paired label-target generators under both
   deletion and endpoint definitions. UMA and MACE have separate energy
   namespaces; an absolute UMA threshold cannot classify MACE labels.
4. Recount raw versus unique physical candidates, compatible cached results,
   nonconvergence, time caps and unknown outcomes. Keep fixed-cell 0.05 and
   0.01 eV/Angstrom endpoint definitions separate. Exact initial masks and all
   trajectory frames belong to one lineage. A repeated endpoint occupation key
   does not prove the same continuous geometry or a unique PES minimum.
5. Explain the development pool's exact one-third uniform balanced mean:
   each unique, novel shared draw belongs to one of three classes. In FINAL,
   repeat/novelty losses reduce this to 33.316%, with descriptive fit SD 0.039
   percentage points. The coupling identity does not guarantee exact one-third
   after those losses or imply zero uncertainty in individual requested-bin
   yields. Preserve per-bin results and paired method contrasts.
6. Recompute resource costs from global task/query records. Distinguish decoder
   attempts, unique geometry/teacher/precision queries, fresh/cache status,
   actual logged E/F calls, optimizer updates and allocated resources. Do not
   sum per-method encountered labels as fresh calls or sum overlapping process
   times as node-hours. Separate GPU-device-hours from charged GPU node-hours;
   include startup, idle time, failures and interruptions. Scope any unlogged
   in-flight call uncertainty explicitly.
   Start with `artifacts/overnight_20261108/reports/paired_call_receipt.json`:
   its primary optimization queue has 13,712 unique attempts and 558,458 logical
   E/F/stress API bundle attempts. These are not 13,712 independent TRAIN masks
   or 558,458 independent training structures; verify the stage identities and
   retain secondary MACE/research costs separately.

**Return:** an independently written metric/cost receipt, one per-seed/per-bin
figure, the paired deletion/endpoint table and a concise paragraph for Sections
3–4. Record exact commands and any mismatch; preserve original outputs when
investigating it. Replaying saved FINAL scores verifies the result, not a new
untouched assessment.

## Ricardo: physical review, report integration and manual submission

Review the scientific interpretation while the independent ML/statistics checks
are returned. The current default is O3 LCO with a 48-site Li sublattice; the
software's general parent/mobile-ion interface does not establish transfer to
another material or system size.

- Check the actual initial/relaxed geometry, force/stress units, fixed-cell
  convention, convergence status and teacher/head/precision receipt. Explain
  useful but imperfect deletion-versus-endpoint rankings. Low teacher energy
  is neither a true DFT ground-state certificate nor a globally optimized
  structure. Same-geometry teacher checks differ from independently relaxed
  teacher endpoints.
- Review the exact 12-site static reference, approximate coordinate symmetry and
  teacher disagreement. Its exhaustive probabilities/entropy are finite-model
  quantities at fixed composition, not measured LCO entropy, a 48-to-12-site
  model transfer or calibrated hydrothermal equilibrium.
- Inspect the NEB candidate, original endpoints and both downhill structures in
  VESTA. The off-site low-force state has the same raw assigned vacancy mask as
  an end-like state. One stable negative-curvature direction does not verify the
  whole original elementary hop; the off-site state is not a certified minimum.
  Restricted image/tolerance convergence does not establish overall barrier
  stability after endpoint changes. Later DFT/basin/connection validation remains.
- Integrate only actual returned contributions and completed receipts into the
  approximately five-body-page [new report](midterm_20261108/main.tex), retaining
  the separate [research supplement](research_supplement_20261108/main.tex).
  Check any newer rubric, resolve comments and submit manually by November 9,
  11:59 p.m. Pacific. Reserve time for a final PDF/source/figure check.

There is no prerequisite new DFT, MD or MLIP fine-tuning campaign for this
midterm. Later physical validation and the possible downstream fine-tuning
extension should be planned after reviewing these results; the final report and
presentations have their own December/November deadlines above.

## Historical instructions and references

The section below preserves the prior `conditioning_20261107` handoff and its
commands/links. Its statements about unperformed controls or insufficient old
paired-cohort support describe that historical checkpoint. The November 8
campaign has since performed those controls and acquired a new paired cohort;
use the current packet's actual support/evaluation receipts.

<details>
<summary>Open the preserved prior-study handoff</summary>

### Prior-study independent handoff

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

The optional Gaussian-proposal result is posthoc TRAIN-moment matching, with no
new neural training; it is not the spatial no-latent ablation above. Keep those
questions separate. The earlier coordinate CVAE and deterministic-regressor
control remain a completed secondary result; do not restart that geometry
campaign for the midterm. Explain its saved comparison and what a predicted
unminimized coordinate field does and does not establish.

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

Review the optional outcomes using
[`OPTIONAL_EXTENSIONS_20261107.md`](OPTIONAL_EXTENSIONS_20261107.md) and the raw
tables in the extensions archive once its packaging receipt is verified. For
editing, reproduce the 59.375/57.234% balanced comparison and the separate
50.347/50.521% original-q20 low comparison. Then check all nine terminal novel
minima, their common current-SCF seed minimum and the 47.569% paired-low win
fraction. The running-best advantage and the less favorable average measure
different parts of the distribution; neither should be dropped.

For Gaussian proposals, shared base epsilon/uniforms do not mean shared actual z.
Reproduce the balanced gains and low-yield regressions under both teachers,
without calling the posthoc moments a trained conditional prior. For temperature
allocation, verify unchanged requests **and actual paired masks** across all four
temperatures, and retain the low-reference-ESS limitation. The raw scalar labels
are sufficient for these checks; no fresh DFT or MLIP training is needed.

### 2. Check paired response, latent use and novelty

Use the saved output tables/figures under
`reports/conditioning_20261107/policy_diagnostics_v2/` and the later retained-model
diagnostics. The paired analysis command is
`scripts/conditioning_20261107/analyze_diagnostics.py --help`; its receipt binds
the exact proposal arrays, manifest, schemas, score file and history.

For the core condition diagnostics, verify that low/middle/high pairs share actual
latent and AR-uniform fingerprints. The separate Gaussian-proposal comparison
shares base epsilon rather than actual z, as described above.
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

For the physical review, check the actual migrating atom and periodic hop image,
endpoint species/order/cell, residual forces, image-refinement evidence and the
MLIP energy reference for every reported barrier. A MACE single-point band check
would be a disagreement profile, not an independently optimized MACE barrier.
Convergence under an MLIP does not remove the need for later physical validation.

Complete independent checks on November 8, freeze the integrated tables/figures
that day, and reserve November 9 for final verification. Prepared commands do not
substitute for the named owners' independent contributions.

</details>

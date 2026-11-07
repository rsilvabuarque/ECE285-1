# Midterm work for Beihao, Varad, and Ricardo

The current scope is composition-and-energy-range vacancy generation plus a
measured temperature-dependent acquisition demonstration. The November 7
extension authorizes bounded new CVAE training and MLIP checks; it does not
authorize new DFT or a new MLIP fine-tuning campaign. Use the existing frozen
historical data and the new completed campaign packet for the required checks.
The separate geometry CVAE remains a secondary historical result.

The midterm is due **November 9, 2026, 11:59 p.m. Pacific**. The supplied announcement
asks for approximately five body pages excluding references: revised proposal,
mathematical/model/algorithm details, and novelty/advantages/limitations. The
provided project overview explicitly names NeurIPS format for the proposal; no
additional midterm template is specified in the supplied evidence. The editable
article draft should be checked against any newer course instructions before
submission. Nothing in this package submits to Gradescope or messages teammates.

## Start, inputs, and working rules

Work from the review branch named in the README and make a personal branch.
From the repository's `final_project` directory, activate the learning environment.
A fresh class machine only needs the core installation for the required cached
analyses and fits:

```bash
bash scripts/install.sh --python python3 --prefix .venvs --core-only
source .venvs/learning/bin/activate
cathode-cvae data verify --snapshot artifacts/current
```

Python 3.11–3.13 is supported; use a PyTorch wheel compatible with the class GPU.
The default full installer also prepares isolated MLIP environments and
hash-checked weights. UMA requires the actual user's approved model access.
Use a permitted compute allocation on a cluster; package commands select no
account or queue. Preserve all protected splits and use a new output directory
for every run. Save commands, input hashes, seed, environment and checkpoint rule.

Required inputs are:

| Purpose | Path |
| --- | --- |
| Common parent | `artifacts/current/core/data/parents/O3_N48_pilot.npz` |
| Training data | `artifacts/current/improvement_v1/training_round0.npz` |
| Historical assessment | `artifacts/current/improvement_v1/data/` and `reports/final/` |
| New schemas/reference/history | `artifacts/overnight_20261107/data/` |
| New proposal/score records | `artifacts/overnight_20261107/benchmark/` and `demo/` |
| Allocation requests and diagnostics | `artifacts/overnight_20261107/allocation/` |
| New completed assessment | `artifacts/overnight_20261107/evaluation/` |
| New fits and learning curves | `artifacts/overnight_20261107/fits/` |
| Label/training/evaluation costs | `artifacts/overnight_20261107/cost_receipt.json` |
| Paired DFT ranking context | `reports/ranking_audit_20261107/` |

The new packet's manifest and completion receipt identify actual contents and
checksums; pending files are not a completed experiment. Large raw force shards,
trajectories, model environments and foundation weights remain outside Git with
retrieval manifests. Follow the README if a packet must be downloaded first.
Do not copy absolute paths from historical configurations onto the class machine.

## Beihao: one reproduction and a controlled model check

Budget about 6–8 focused hours. Your deliverables are one reproduced multibin fit,
checkpoint/reload checks, a learning-curve figure, the method explanation, and one
bounded controlled ablation. These are your independent contributions; the
prepared campaign and draft do not mean you have completed them.

### 1. Reproduce a multibin reference fit

Run in the class compute environment. The actual campaign configurations are the
reference; record any hardware/software/time-cap deviation. A representative
full-fit command is:

```bash
cathode-cvae train-mask \
  --data artifacts/current/improvement_v1/training_round0.npz \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --condition-schema-in artifacts/overnight_20261107/data/condition_schema_uma.json \
  --kind ar_cvae --condition-mode multibin --counts 12 24 36 \
  --teacher uma --teacher-head omat --orientation canonical \
  --geometry-protocol "pristine-derived deletion; unrelaxed; exact canonical parent coordinates" \
  --beta-final .1 --beta-warmup 20 --validation-beta 1 \
  --epochs 100 --minimum-epochs 20 --patience 15 --max-seconds 600 \
  --seed 20261106 --output-dir runs/beihao_multibin_reference
```

Expected outputs: `best.pt`, `configuration.json`, `condition_schema.json`,
`history.csv`, and `summary.json`. Acceptance checks:

- Protected train/validation/test rows and parent hash match the input receipt.
- Schema identity, q20/q80 edges, actual class support and teacher/head agree.
- Edges use training energies only; ties are not split and empty bins fail.
- Model dimension is four: Li fraction plus three one-hot components.
- The selected checkpoint and completed optimizer updates/stop reason are saved.
- A short smoke fit is labeled as such, not reported as scientific reproduction.

### 2. Reload and make explicit requests

```bash
for bin in low middle high; do
  cathode-cvae generate-mask \
    --checkpoint runs/beihao_multibin_reference/best.pt \
    --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
    --n-li 24 --requested-bin "$bin" --count 64 --seed 20261117 \
    --temperature-K 433.15 --allocation-policy independent-bin-check \
    --output "runs/beihao_multibin_reference/${bin}_attempts.npz"
done
```

Every valid occupancy must be binary with exactly 24 Li. Inspect each
`.attempts.csv`: requested/effective bin, fit seed, sampling seed, schema identity,
physical temperature, policy, checkpoint and status must be recorded. Keep all
192 attempts, including repeats/failures. The `.unique.npz` files contain unique
canonical geometries for scoring. A requested class is not a measured energy.
Physical temperature here is provenance for request allocation, not a softmax
or latent-sampling temperature.

Use exact cached labels where present. Uncached generated structures have
**unknown energy**. Do not transfer a neighbor's label, drop misses, or report a
complete new energy-yield statistic without scoring the complete declared pool.
A small separately budgeted interface check can use the MLIP wrapper; a broad
new inference campaign is not required for this midterm reproduction.

### 3. Explain the model and supply a controlled ablation

Write the encoder, Gaussian latent prior, GRU/teacher-forcing decoder, exact-count
rule, reconstruction/KL objective, and train-only class schema in your own words.
Plot train/validation reconstruction and KL against updates or epochs, marking
checkpoint selection. Distinguish posterior reconstruction from prior generation
and independent fit seeds from repeated latent draws.

A bounded ablation can toggle **training-only class balancing**, with the same
round-0 data, schema, starting checkpoint, fit seed, beta, validation objective,
optimizer reset, schedule and update budget in both arms. For example, use `--warm-start runs/beihao_multibin_reference/best.pt` in both
arms, set `--beta-warmup 1 --beta-final .1`,
`--max-updates 500 --epochs 100 --minimum-epochs 1 --patience 100`, and a generous
common time cap; add `--class-balance` only to the second arm. Use distinct output
directories. Both optimizers start fresh. Pass the same frozen schema and exact
geometry-protocol string used above in both arms. Confirm both actually complete
500 updates. With the same dataset, validation/warmup exposure can be held common;
equal update caps alone are insufficient if dataset sizes differ.

Do not combine new data, label thresholds, beta schedules and balancing then
attribute the difference to one cause. Evaluate the same protected validation
masks and report the objective's actual weighting. New prior-sample physical
quality requires exact labels; do not select a new checkpoint on the historical
final yield and call that pool untouched. Three seeds are desirable for the final
report, but one independent reproduction plus the supplied three-seed experiment
is enough for the midterm if time is tight.

**Report insertion:** methods in Sections 3–4; new fit/ablation evidence in Section
5; independent-check status in Section 6. Supply one concise model paragraph,
one learning curve and a small configuration/result table.

## Varad: independent metrics, baselines, and allocation validation

Budget about 6–8 focused hours. Required deliverables are an independently checked
results CSV/receipt, seed-resolved comparison, conditioning/allocation figures,
and the validation/cost interpretation. No MLIP is required for these cached checks.

### 1. Recompute the historical benchmark

```bash
cathode-cvae evaluate --snapshot artifacts/current \
  --output-dir runs/varad_historical_recompute
```

Compare `metrics.csv`, `seed_metrics.csv`, `summary.csv` and
`teacher_agreement.csv` with the frozen source tables. Preserve original q20 and
its old novelty history. Check that each success is a distinct novel low-tail
orbit but every generation attempt is a denominator contribution. Reproduce the
historical 35.57%/40.06% AR binary, 18.99%/19.75% random, and 52.65%/54.08%
strongest-mutation means. They are not the new multibin results.

### 2. Recompute the new condition-aware benchmark

```bash
python -m cathode_cvae.condition_evaluation \
  --attempts artifacts/overnight_20261107/benchmark/attempts.csv \
  --candidates artifacts/overnight_20261107/benchmark/candidates.npz \
  --scores uma=artifacts/overnight_20261107/benchmark/scores_uma.npz \
           mace=artifacts/overnight_20261107/benchmark/scores_mace.npz \
  --schemas uma=artifacts/overnight_20261107/data/condition_schema_uma.json \
            mace=artifacts/overnight_20261107/data/condition_schema_mace.json \
  --training-keys artifacts/overnight_20261107/data/training_keys.json \
  --history artifacts/overnight_20261107/data/history_assessment.json \
  --historical-thresholds artifacts/current/reports/generation_thresholds.csv \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --cost-receipt artifacts/overnight_20261107/cost_receipt.json \
  --output-dir runs/varad_new_benchmark
```

Use the same command for `demo/` inputs, adding
`--reference-plans artifacts/overnight_20261107/allocation/allocation.json`, with a
new output directory. Compare with the packet's completed `evaluation/` tables.
Preserve requested bin, teacher, temperature/policy, fit seed and sampling seed
in grouping. MACE uses its own train-derived schema; applying UMA's absolute
energy boundaries to MACE would confuse energy zero points with controllability.

Check validity/count, orbit uniqueness, novelty to **training** and to **prior
history** separately, requested/measured confusion, in-bin hit rates, unique
novel in-bin yield, fixed-original-q20 yield on requested-low attempts and energy
quantiles. Check `fixed_q20_low_request.csv` for the fixed-threshold comparison
and `allocation_joint_summary.csv` for the deduplicated mixture result.
Do not compare a pooled low/middle/high q20 score with the old low-only
benchmark. Uniform/mutation replicates are random seeds, not fitted models. Missing or
failed labels must remain visible. Compare multibin, binary, composition-only,
uniform raw sampling, bin-seeded mutation and the low-energy mutation control at
matched attempt budgets and declared training information.

Plot individual fit seeds, best energy/useful unique structures against attempts
and scoring calls, and supplied structural diversity diagnostics. Count all
reference labeling, cache misses, rejected prefiltered proposals, adaptation and
training costs. Generation time alone is not an end-to-end saving. Correlated
frames/draws do not create independent fitted models or justify tiny error bars.

### 3. Check multiplicities and temperature allocation

The exact finite-support command is:

```bash
python -m cathode_cvae.acquisition \
  --reference artifacts/current/extras/learning_all_parents/learning_Hu_O3_12_uma.npz \
  --parent artifacts/current/extras/learning_all_parents/parents/Hu_O3_12.npz \
  --reference-kind exhaustive_orbits --validate-exhaustive \
  --output-dir runs/varad_hu_support
```

Expected: `finite_support.json` reports 87 orbits and 144 distinct parent
permutations; at n=6, 17 orbits have total multiplicity 924. Every count sums to
binomial(12,n). With no condition schema, zero allocation plans is expected:
this command audits support only. The focused test
`test_hu12_finite_support_multiplicities_and_exact_weights` additionally compares
actual teacher-weight sums against all 924 raw masks at 433.15 K.

Recompute the 48-site acquisition plans:

```bash
python -m cathode_cvae.acquisition \
  --reference artifacts/overnight_20261107/data/reference.npz \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --schema artifacts/overnight_20261107/data/condition_schema_uma.json \
  --reference-kind uniform_raw_sample --counts 12 24 36 \
  --temperatures 393.15 433.15 473.15 493.15 \
  --mixture-weights 1 1 1 1 --exploration .1 --budget 128 \
  --output-dir runs/varad_allocation
```

Verify total cell energies in eV, actual atom counts, complete reference scoring,
raw sampling frequencies and no extra degeneracy factor. Check pure and 10%
exploration policies, exact integer budget totals, temperature mixtures, ESS,
maximum masses, and nested-prefix sensitivity. Per-temperature plans each have
their own budget; a mixture budget is joint. The generator executes every
schedule in its input, including both pure/exploration per-temperature and
mixture schedules; its total is therefore larger than one `--budget`. Strong concentration is a finding,
not proof of thermal equilibrium. Compare requested proportions with measured
bins of generated masks; these can differ substantially.

### 4. Check the physical ranking context and interpretation

```bash
python reports/ranking_audit_20261107/recompute.py \
  --input reports/ranking_audit_20261107/paired_energies.csv \
  --output runs/varad_ranking_recompute
```

Keep source/phase/composition/cohort pools separate; inspect top-k set sizes/ties,
regret and energy-gap errors. In Hu's five-member optimized Li10 pool, matching
all identities is trivial top-five overlap, while UMA selects the DFT fifth-ranked
structure first. Avoid interpreting aggregate rattled-frame rank agreement as
universal optimized-configuration accuracy.

**Report insertion:** conditional/acquisition figures and new tables in Section
5; uncertainty, physical limitations and historical context in Section 6.
Provide one primary seed figure and one requested-versus-realized allocation or
confusion figure, with source CSV and plotting command.

## Ricardo: prepared material and final integration

`RICARDO_CONTRIBUTION.md` and the integrated LaTeX supply the motivation,
DFT/MLIP/MD definitions, parent/site map, geometry protocol, orbit/multiplicity
explanation, representative structures, recomputed ranking context, primary
materials references, and physical interpretation. The complete source/claim
manifest accompanies the report. Review the supplied POSCAR/CIFs in VESTA and
flag physical concerns. A new mask's low teacher energy does not establish a
DFT ground state or thermodynamically stable phase.

Ricardo integrates the independently checked team additions and chooses a compact
later DFT validation set from the prepared handoff. No new DFT is submitted or
scheduled in this overnight work. MD/3PT/phonon method-pilot results remain outside
the class report except a clearly identified future-work sentence.

## Optional work after the required checks

Broader Li-count learning can use the existing all-count labels and
`data prepare-learning` while preserving protected splits; endpoint compositions
with one orbit are not meaningful discovery tests. The exact 12-site reference
can support a separate compatible model, with novelty relative to a protected
training subset (not all 87 known orbits). Neither padding a 48-site vector nor
applying its weights establishes size transfer.

A geometry-CVAE reproduction, property-input ablation or endpoint-only comparison
is optional for this midterm. Keep the deterministic regressor control, full
lineage splits, train-only scalers, and frame weights summing to one per lineage.
Coordinate error alone does not establish low forces or optimizer savings. Do
not let that optional campaign displace the new multibin/acquisition checks.

## Integration schedule and acceptance

| Date | Beihao | Varad | Ricardo/shared |
| --- | --- | --- | --- |
| November 7 | Fit/reload, schema/method check, ablation configuration | Historical/new recomputation, support/units/budget checks | Parent/ranking/materials review; collect discrepancies |
| November 8 | Learning curves and bounded ablation result | Seed/conditioning/allocation figures and cost statement | Integrate, freeze tables and claim-source manifest |
| November 9 | Verify equations and implemented settings | Verify all numbers, denominators, grouping and uncertainty | Final five-page body/references/format check; manual submission before 11:59 p.m. Pacific |

Use the [integration checklist](INTEGRATION_CHECKLIST.md). Mark each contribution
complete only after its owner returns the stated evidence. Presentation dates
are November 24 and December 1; the final report is due December 12.

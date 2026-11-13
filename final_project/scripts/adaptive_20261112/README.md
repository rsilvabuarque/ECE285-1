# Adaptive energy-range experiment helpers

Run commands from `final_project` in an appropriate compute allocation. No
command starts a scheduler job, performs DFT, or interprets a plateau as proof
of the global extrema. The pinned UMA teacher supplies acquisition labels;
MACE can independently assess the same candidate union.

The module `python -m cathode_cvae.adaptive_iteration --help` exposes:

1. `boundary --boundary-dir OLD --h0 H0.npz --assessment-dir LAST_ASSESSMENT
   --output-dir NEW`: verify all old raw and eligible assessment identities,
   preserve inherited H0 roles, and produce canonical known/protected sets.
   It opens no old score or energy member. Raw generated identities become
   known; only actual eligible last-assessment candidates become protected.
2. `schema --data TRAINING.npz --parent PARENT.npz --reference-schema INITIAL.json
   --mode fixed|adaptive|range --output-dir SCHEMA`: freeze one round's labels.
   `adaptive` fits TRAIN empirical nearest-rank 20/80% cutoffs. `range` fits
   min+0.2(max−min) and min+0.8(max−min) from TRAIN. This range choice is an
   explicitly provisional outlier-sensitive comparator, not a universal
   physical setting. Quantile diagnostic fields remain actual quantiles.
3. `seed-bank --data H0.npz --parent PARENT.npz --schema INITIAL.json
   --boundary-dir BOUNDARY --output-dir BANK`: freeze original TRAIN-only,
   unprotected mutation seeds. Reuse the same bank across compared arms.
4. `propose --parent PARENT.npz --schema CURRENT.json --boundary-dir BOUNDARY
   --checkpoint MODEL.pt --namespace ROUND_STREAM --seed SEED --method NAME
   --attempts-per-cell 512 --queries-per-cell 256 --output-dir STAGE`: request
   every composition/bin cell. Replace `--checkpoint` by `--sampler uniform`
   or `--sampler mutation --seed-bank BANK` for baseline proposals. Defaults
   are Li counts 12,24,36 and LOW/MIDDLE/HIGH. Each arm receives its same
   prospective attempt/query caps; no energy-based rejection occurs before
   scoring. Same namespace/seed/count/bin means identical saved z/uniform
   arrays for neural arms. Mutation has its own seed/swap semantics and must
   not be described as having an identical edit stream to a neural sampler.
5. Register the union of `STAGE/candidates.npz` files with the normal
   `pool_scoring` queue, then run scoring and collect. All stages in a round
   must be frozen before the first scoring task starts. A fresh registry is
   required per acquisition or assessment phase.
6. `verify-oracle --stage STAGE [--stage OTHER_STAGE ...] --oracle-root ORACLE
   --collection-receipt COLLECTED/receipt.json --teacher uma --output VERIFIED.json`:
   check exact union coverage, raw model/checkpoint receipts, input masks,
   after-freeze task times, successful score rows, and collected-array equality.
7. `merge --base-data OLD.npz --parent PARENT.npz --schema CURRENT.json
   --stage STAGE --verifier VERIFIED.json --round R --output-dir MERGED`:
   retain every old row value and role, add selected successful UMA labels to
   TRAIN only, and classify them by measured rather than requested energy.
   The output is `MERGED/training.npz`; new rows have unit weight by default.
8. `summarize --stage STAGE --verifier VERIFIED.json --fixed-schema INITIAL.json
   --output-dir METRICS`: report both fixed-reference and moving-bin yields,
   discovery extrema and the full attempted denominator. Compare fixed metrics
   longitudinally; moving-bin success has different meaning after relabeling.
9. `extend-boundary --boundary-dir PREVIOUS --known-stage ACQUISITION
   --protected-stage ASSESSMENT --output-dir NEXT`: preserve all attempted
   identities as known and protect scored assessment identities before the next
   round. Repeat stage options for all arms, not only a selected method.

Refit a new schema from the merged TRAIN data before the next model update.
Changing schema semantics requires the trainer's explicitly declared re-binned
warm-start option; silently reinterpreting an old checkpoint is rejected.
Fixed-vs-adaptive-vs-range comparisons should first share an identical acquired
batch, then report divergent later rounds as policy comparisons. Include
equal-update no-new-data continuations; 1,000 updates do not establish optimizer
convergence. Freeze evaluation protocols/checkpoints before drawing samples.

The NPZ format decompresses the complete energy member when selected TRAIN
energies are read. TEST values are never classified, used for edge fitting,
or included in support/endpoint summaries here; their original values are
preserved in the output archive. Six tiny synthetic tests cover these
boundaries, exact counts, paired arrays, caps, measured labels, and raw oracle
provenance; they do not establish physical MLIP accuracy.

The dated campaign driver and its phase-2 boundary handoff are archived as
`orchestrate.py`, `resume_with_phase2.py`, and `analyze.py`.
These are exact campaign receipts with site-specific paths, not portable cluster
submission commands. Their executed copies remain under
`final_project/runs/adaptive_20261112/iteration/study_v2/`. That directory's
`design.json`, `phase2_boundary_amendment.json`, frozen taskfiles, raw worker
receipts, four strict oracle verifiers, and final boundary define the actual
run. The generic portable interfaces remain the module CLI documented above.

The initial driver was terminated before scoring after a clarification that
new latent diagnostic masks must not turn existing TRAIN structures into
protected assessment data. All twelve completed proposal bundles were retained
as unscored history. The replacement run uses a new namespace and a corrected
boundary with zero newly protected H0 TRAIN orbits. A later controlled CPU-only
handoff keeps the already frozen first-round assessment queue and adds the
independent phase-2 latent diagnostic identity union before round two. GPU tasks
continue through that handoff.

The analysis preserves raw tables and writes corrected derived tables: moving
condition fidelity is undefined for MACE because UMA supplied the training
conditions. Fixed MACE metrics use its own frozen TRAIN-only energy schema.
Paired-fit error bars are sample standard deviations across three fits, not
confidence intervals. Per-attempt acquisition yield is capped at one half by the
256-query/512-attempt design; fresh assessment uses 128/128.

The optional `frontier/frontier.py` is a separately budgeted, dated experiment
that runs only after the core and derived analyses complete. Greedy UMA-guided
single-swap search and an energy-blind random walk share original unprotected
TRAIN seeds, exactly identical first-step proposals and queried unions, and the
same eight one-swap neighbors per trajectory. A uniform fixed-count baseline
receives the same proposal/query cap without an edit-semantics claim. All raw
attempts are retained, new-orbit selection precedes scoring, and all teachers'
raw worker outputs receive exact-union verification. Four steps over 16 starts
per composition/endpoint cell and three seeds cap new work at 27,648 geometries
per teacher. Seed rescoring is recorded separately. The active 48-minute guard
reserves two minutes for cleanup within the approved 50-minute/4-node-hour cap.
This checks local endpoint search; it cannot establish global extrema or entropy.

Class balancing in `train_mask.py` rescales row loss weights; it does not
oversample minibatch draws. The derived support table computes actual TRAIN
class counts, extrema, edge shifts, and loss-weight ESS = (sum w)^2/sum(w^2)
from inherited row weights and the implemented within-composition class
rescaling. That ESS measures concentration of objective weight, not independent
physical observations. Frozen fixed-schema support counts describe the original
reference population; actual augmented-row support is reported separately.

The second CPU-only handoff (`handoff_to_optimized.py` and
`resume_r2_optimized.py`) keeps every generated structure and all scientific
code on the original snapshot. Only scoring workers use the tested queue
helper that reads existing claim names once instead of issuing one filesystem
existence query per task. Atomic claim creation and all raw-score checks stay
unchanged. `publish_lent_mace.py` records an explicit resource ownership handoff
and routes existing frozen tasks to the second group without adding evaluations.

`causal_range/` is an explicitly post-hoc mechanistic follow-up to the first
range-labeling gain. Its plan precedes TRAIN-only edge comparison. It compares
H0 and augmented rows under fixed and identical observed-range condition edges;
a separate H0-only range schema tests whether relabeling alone is sufficient.
The exact edge comparison triggered twelve new 2,000-update fits, including six
H0 fits using each lineage's R1 range edges. All classes have actual H0 TRAIN
support. The control starts only after core completion and derived analysis,
uses the extra 18-GPU group with a 40-minute/3-node-hour cap, and reserves every
fresh raw identity before any frontier proposal can start. The two follow-ups
then use disjoint groups. Neither changes the core experiment.

Tail tables in these campaign analyses mean **H0 fixed-low** and **H0
fixed-high**, using original teacher-specific H0 schema edges. They are distinct
from the pilot409 threshold diagnostic used in some latent summaries. Frontier
endpoint gains are measured within one Li count and require no tail threshold.
The frontier records the energy-blind move itself before scoring, checks exact
first-step proposal/query equality, and reports diversity as the minimum
Hamming distance over parent symmetries divided by two (Li/vacancy swaps).

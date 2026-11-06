ECE 285 collaborator data and code — 6 November 2026

Read TASKS.txt and improvement_v1/RESULTS.txt before starting work. The original portable learning package is
under core/ and is preserved with its own file manifest. New improvement-cycle
assets use their original relative paths under improvement_v1/. The integrating
owner's final_receipt.json records what actually ran and its validation scope.
No new DFT calculations are represented by these data.

START

Training uses NumPy and PyTorch; the original core requires no MLIP installation.
Fresh-candidate evaluation plots additionally use Matplotlib when available.
On Perlmutter, run training and substantial analysis in an allocation.

  cd core
  pip install -r requirements.txt
  python cvae/train.py --teacher uma --kind cvae --device cpu \
    --epochs 2 --output-dir runs/new_smoke

This is an operability test, not a convergence experiment. Output directories
must be new. For the improved trainer, from the archive's top-level directory:

  pip install -r improvement_v1/requirements-learning.txt
  python improvement_v1/train.py --help
  python improvement_v1/train.py --data improvement_v1/training_original.npz \
    --kind ar_cvae --condition-mode energy --orientation canonical \
    --device cpu --epochs 2 --output-dir runs/new_ar_smoke

Use the included configuration/receipt for the exact dataset, model kind,
condition, seed and checkpoint-selection rule; do not infer these from a label.
Preserve the frozen splits. The original full campaign report is included for
background; fresh improvement-cycle results should be read from the new tables.

FILE MAP

  core/                              Original MLIP-free package and checkpoints
  reports/preliminary_report.pdf     Original ten-page science/method background
  reports/generation_thresholds.csv   Original per-teacher/composition q20
  improvement_v1/training_original.npz  Original three-count learning subset
  improvement_v1/training_round*.npz  Expanded/adapted learning sets when present
  improvement_v1/runs/               New frozen fits, configurations and histories
  improvement_v1/data/               Proposal IDs, scalar labels and freeze receipts
  improvement_v1/reports/development/  Model-development evidence
  improvement_v1/reports/final/      Fresh final assessment tables and figures
  improvement_v1/final_receipt.json  Integrating owner's completed audit/results
  TASKS.txt                          Owners, concrete outputs and submission checks

The explicit source map and file manifest show the actual included files. Follow
the data SHA256 recorded in each run configuration: later rounds can use a larger
training set. An NPZ file can be inspected with numpy.load(..., allow_pickle=False).
The common learning fields are occupancy, n_Li, energy_eV_atom, split, weight and
structure_ids. Use the split field; never randomly resplit rows for convenience.

WHAT A DATA ROW MEANS

The primary pristine parent contains Li48Co48O96: 48 possible Li sites, 48 Co,
and 96 O atoms. A configuration containing n Li has 144+n present atoms.
Learning occupancy=1 means Li present. Vacancy masks use the opposite convention:
vacancy=1 means removed. Canonical mask keys identify symmetry orbits. The actual
evaluated orientation can differ from that key, so vectors cannot be reoriented
by relabeling the key alone. Read parent_slot/li_indices from the supplied arrays.

The mask experiment uses pristine-derived deletion geometries and scalar energy
labels. The new benchmark uses Li counts 12, 24 and 36 in the SAME 48-site parent,
corresponding to 25%, 50% and 75% lithiation. It is not a system-size transfer
experiment. Forces/stresses and geometry trajectories are not mask-trainer inputs.
Its low-tail score is not a relaxed-energy ranking. Do not pool raw
energies across compositions, parent cells or MLIP teachers. E is eV/cell;
energy per atom is E/(144+n), in eV/atom. Forces are eV/Angstrom. Stress is
eV/Angstrom^3, xx yy zz yz xz xy, tensile positive where supplied.

TRAINING, ADAPTATION AND ASSESSMENT

The original core has frozen training/validation/test mask lineages. The adaptive
corpus may add previously generated MLIP-labeled configurations to training.
Those old oracle evaluations are then development/training evidence, not an
untouched final test. Inspect the improvement training-data receipt for each
source's use, the grouped split and the canonical novelty cache.

An additional adaptation round can promote fresh DEVELOPMENT labels to training;
the corresponding round's receipt must identify them. The final pool remains
excluded from all such fitting and selection. Fresh development and final
proposal pools are separate. Choose model settings
using development evidence, freeze them, and assess fresh final proposals only
afterward. Inspect timestamps/checkpoint hashes in the final receipt. Do not
describe an examined proposal pool as an untouched final assessment for a later
training round. If final results influence another revision, use a new final pool.

Compare distinct novel low-energy symmetry orbits per ALL proposal attempts.
Duplicates and known configurations consume attempts. Thresholds in the main
comparison are the ORIGINAL training q20 within teacher and composition; these
remain fixed even when adaptation expands the training corpus. A low-condition
request is not a measured energy. New masks outside the scored pool need fresh
MLIP single-point evaluations before they can be labeled low energy.

Seed variation, energy quality and structural diversity must be reported together.
The mutation baseline explores closer to known configurations; novelty alone does
not establish broad exploration. Teacher agreement is not independent DFT truth.
An optimized exit code or force threshold does not establish phase stability.

REPRODUCIBILITY

MANIFEST.json at the archive root records every payload's SHA256. The core has
its own unchanged manifest. SOURCE_MAP.json records each newly included file's
campaign location. The ZIP payload was reopened and checked after writing.
Do not overwrite shared checkpoints, frozen data, or other collaborators' output.
Use separate run directories and one integrating owner for repository changes.

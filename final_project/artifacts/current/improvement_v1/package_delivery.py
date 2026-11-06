"""Build a checked collaborator ZIP from the portable core and explicit assets.

No chemistry calculations are run. The integrating owner supplies the final
receipt and a curated list of adaptation data, frozen models and fresh results.
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


README = """ECE 285 collaborator data and code — 6 November 2026

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
  python cvae/train.py --teacher uma --kind cvae --device cpu \\
    --epochs 2 --output-dir runs/new_smoke

This is an operability test, not a convergence experiment. Output directories
must be new. For the improved trainer, from the archive's top-level directory:

  pip install -r improvement_v1/requirements-learning.txt
  python improvement_v1/train.py --help
  python improvement_v1/train.py --data improvement_v1/training_original.npz \\
    --kind ar_cvae --condition-mode energy --orientation canonical \\
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
"""


TASKS = """Tasks for the ECE 285 midterm due 9 November

These are proposed owners and concrete outputs for the shared repository. Plan
roughly 6–8 focused hours per person before the midterm, then revisit the budget.
Presentation dates are 24 November and 1 December; the final report is due
12 December. The midterm is approximately five pages INCLUDING the revised
proposal and EXCLUDING references. It must cover the mathematical method/algorithm
and a discussion
of novelty, advantages and limitations. Do not assume the CVAE must outperform
the baselines for the project to be scientifically useful.

Ricardo — chemistry, data, symmetry and scoring

1. Verify the parent structure, occupancy convention, Li fraction, energy units
   and distinction between deletion, coordinate prediction, and relaxation.
   Check the supplied VESTA examples when included; record chemical concerns.
2. Audit grouped splits and canonical-mask novelty. Document exactly which
   symmetry operations define an orbit and which parent/cell the claim covers.
   Different model labels or trajectory frames are not independent configurations.
3. Write the short materials motivation and dataset/provenance paragraph. State
   explicitly which labels are UMA/MACE/other MLIPs and which archived labels are
   DFT. New large-cell MLIP labels must not be described as DFT truth.
4. Propose a small later DFT verification set spanning low energy, model
   disagreement and structural diversity. This task is selection/design only;
   submitting DFT is a separate decision. Sampler comparison remains the core.

Deliver: a checked data dictionary, a concise symmetry/provenance note, and the
materials motivation plus physical interpretation/limitations text for the report.

Beihao — generative model and controlled ablations

1. Run the short portable fit/reload check, then reproduce one included model's
   training settings in a NEW run directory. Preserve the shared frozen splits.
2. Explain encoder, latent variable, decoder, energy/composition conditions,
   exact Li-count handling and loss. Describe the actual implemented model,
   including whether it uses independent or autoregressive occupancy probabilities.
3. Compare latent KL, reconstruction and fresh prior-sample quality. Better
   reconstruction alone does not establish improved discovery. Distinguish three
   latent draws from three independently trained model seeds.
4. Review the completed controlled changes and choose at most one further
   ablation for the next development round. Freeze the hypothesis, controls,
   setting and evaluation budget before running it; use fresh final proposals
   after selection. Do not combine untracked architectural changes.

Deliver: one reproducible configuration/learning-curve figure, the mathematical
method paragraph, and an ablation table tied to measured oracle results.

Varad — baselines, validation, uncertainty and figures

1. Recompute the supplied low-tail yields from proposal IDs, canonical keys,
   frozen known history, MLIP scores and ORIGINAL training thresholds. Include
   every proposal attempt in the denominator; do not silently drop duplicates.
2. Compare random fixed-count sampling, the previous CVAE, new model variants,
   and low-energy training-mask mutation under equal attempt/oracle budgets.
   State seed count and report seed-specific results. Composition/seed groups
   and correlated optimizer frames are not interchangeable replications.
3. Plot energy quality against novelty/diversity and distance to training when
   those distances are available. Count all adaptive scoring calls in cost.
   Do not treat a novelty rate or different teacher threshold as DFT accuracy.
4. Check whether claimed improvements persist under the second MLIP teacher
   when available. Give failure cases and a clear continue/pivot criterion.

Deliver: a verified results CSV, one readable comparison figure with seed points,
and the error-analysis/validation and baseline sections of the midterm.

Shared integration before submission

Suggested sequence: by 7 November, each owner completes the input/provenance or
reproduction checks above and returns one draft paragraph plus concerns. On
8 November, integrate the measured figures, revised proposal and method draft;
resolve terminology and assign a single report editor. On 9 November, check the
five-page body, references, figure/source-table agreement and submission rules.

Meet once to agree on the final scope and scientific question. Use one person
to integrate changes, with separate branches/run directories for everyone else.
The report should distinguish observed results, interpretations, and proposed
next tests. Freeze the current figures and source tables before submission.
State that size/material transfer, DFT ground states, equilibrium thermodynamics
and a CVAE-specific geometry advantage have not been established unless new
direct evidence in the package actually demonstrates them.
"""


def build(output, stage, final_receipt, include):
    output, stage, final_receipt = map(Path, [output, stage, final_receipt])
    receipt = json.loads(final_receipt.read_text())
    assert receipt.get("status") in ["PASS", "COMPLETE"], "A completed integrating-owner receipt is required"
    assert not output.exists() and not stage.exists(), "Preserve prior deliveries; use new paths"
    assert not output.resolve().is_relative_to(stage.resolve()), "ZIP must be outside its staging tree"
    for entry in include:
        path = Path(entry)
        path = path if path.is_absolute() else ROOT / path
        assert not (path.is_dir() and stage.resolve().is_relative_to(path.resolve())), \
            "Do not include an ancestor of the staging directory"
    stage.mkdir(parents=True)
    source_map = {}
    core = ROOT / "collaborator_package"
    core_manifest = json.loads((core / "MANIFEST.json").read_text())["files"]
    for relative, checksum in core_manifest.items():
        assert sha(core / relative) == checksum, relative
        target = stage / "core" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(core / relative, target)
    shutil.copy2(core / "MANIFEST.json", stage / "core/MANIFEST.json")

    def copy(path, relative):
        path, relative = Path(path), Path(relative)
        assert not relative.is_absolute() and ".." not in relative.parts
        target = stage / relative
        if target.exists():
            assert sha(path) == sha(target), f"Conflicting destination {relative}"
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_map[str(relative)] = {"campaign_source": str(path.resolve()), "sha256": sha(path)}

    baseline = ["reports/preliminary_report.pdf", "reports/generation_thresholds.csv",
                "reports/generation_metrics.csv", "reports/generation_diversity.csv",
                "data/mask_manifest.csv", "data/parents/O3_N48_pilot.npz"]
    for relative in baseline:
        copy(ROOT / relative, relative)
    copy(final_receipt, "improvement_v1/final_receipt.json")
    for entry in include:
        path = Path(entry)
        path = path if path.is_absolute() else ROOT / path
        relative = path.resolve().relative_to(ROOT.resolve())
        assert path.exists(), path
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for f in files:
            if f.is_file() and "__pycache__" not in f.parts and f.suffix not in [".pyc", ".tmp"]:
                copy(f, f.resolve().relative_to(ROOT.resolve()))
    # Include the exact package-builder implementation for repeatability.
    copy(Path(__file__), "improvement_v1/package_delivery.py")
    (stage / "README.txt").write_text(README)
    (stage / "TASKS.txt").write_text(TASKS)
    (stage / "SOURCE_MAP.json").write_text(json.dumps(source_map, indent=2) + "\n")
    files = sorted(f for f in stage.rglob("*") if f.is_file())
    manifest = {str(f.relative_to(stage)): sha(f) for f in files}
    (stage / "MANIFEST.json").write_text(json.dumps({"files": manifest,
        "scope": "Portable core plus curated improvement data, frozen fits, fresh scores and report tasks.",
        "final_receipt_sha256": sha(final_receipt)}, indent=2) + "\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                z.write(f, Path(stage.name) / f.relative_to(stage))
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        prefix = stage.name + "/"
        expected = {prefix + f for f in manifest} | {prefix + "MANIFEST.json"}
        assert set(z.namelist()) == expected
        for relative, checksum in manifest.items():
            assert hashlib.sha256(z.read(prefix + relative)).hexdigest() == checksum, relative
    result = {"status": "PASS", "archive": str(output.resolve()), "bytes": output.stat().st_size,
        "sha256": sha(output), "payload_files": len(manifest), "manifest_files_checked": len(manifest),
        "checks": ["Original core payload hashes", "Reopened complete ZIP", "No extra or missing members",
                   "Every payload hash", "Completed final receipt included"],
        "scope": "Artifact integrity, not scientific superiority or independent physical validation."}
    Path(str(output) + ".sha256").write_text(f"{result['sha256']}  {output.name}\n")
    Path(str(output) + ".validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--stage-dir", type=Path, required=True)
    ap.add_argument("--final-receipt", type=Path, required=True)
    ap.add_argument("--include", nargs="+", required=True, help="Explicit campaign-relative files/directories")
    a = ap.parse_args()
    build(a.output, a.stage_dir, a.final_receipt, a.include)


if __name__ == "__main__":
    main()

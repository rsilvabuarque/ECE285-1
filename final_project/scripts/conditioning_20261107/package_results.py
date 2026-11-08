"""Curate the frozen required experiment packet; never run or relabel a model.

Run inside a compute allocation after final cached evaluation, the independent
required-cost audit, and the frozen report PDF are ready. The ZIP is a complete
cached replay payload. Large arrays/checkpoints stay out of Git via the packet's
own .gitignore. Foundation-model weights and private input documents are never
eligible sources. Raw E/F/stress shards remain at their audited HPC locations;
their hashes and stat bindings are preserved in the execution receipt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import zipfile


RELEASE_TAG = "ece285-conditioning-20261108"
LARGE_TABLES = {"attempt_outcomes.csv", "candidate_measurements.csv", "budget_curves.csv"}
FIT_FILES = (
    "best.pt", "best_validation_beta_1.pt", "best_validation_beta_0p1.pt",
    "best_validation_beta_1.pt.latent.json", "best_validation_beta_0p1.pt.latent.json",
    "configuration.json", "summary.json", "history.csv", "latent_history.jsonl", "condition_schema.json",
)
CORE_FITS = tuple(
    f"{prefix}_{seed}"
    for prefix in ("balance_0_d8", "balance_1_d8", "latent_d4", "latent_d16", "latent_d32",
                   "no_latent", "symmetry_augment", "graph_encoder", "graph_context")
    for seed in (20261106, 20261107, 20261108)
) + ("descriptor_20261106",)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def safe_relative(value):
    value = PurePosixPath(value)
    if value.is_absolute() or ".." in value.parts or any(":" in x for x in value.parts):
        raise ValueError(f"Unsafe payload path: {value}")
    return str(value)


class Curator:
    """An explicit source allowlist; no recursive copying of arbitrary run roots."""
    def __init__(self, root, stage):
        self.root, self.stage = Path(root).resolve(), Path(stage)
        self.sources = {}

    def add(self, source, relative):
        source = Path(source).resolve()
        relative = safe_relative(relative)
        if not source.is_file():
            raise FileNotFoundError(source)
        # Inputs are confined to this worktree. Foundation weights are external
        # to it, but the name/suffix checks also protect against future copying.
        if not source.is_relative_to(self.root):
            raise ValueError(f"Source outside worktree: {source}")
        parts = source.relative_to(self.root).parts
        if any(p in {"for_chat_gpt", ".git", ".venvs", ".venv", "models", ".models"} for p in parts):
            raise ValueError(f"Private/weight source excluded: {source}")
        if source.suffix == ".pt" and not source.is_relative_to(self.root / "runs/conditioning_20261107/fits"):
            raise ValueError("Only the required, fitted CVAE checkpoints may enter the packet")
        if source.suffix == ".pdf" and not (source.is_relative_to(self.root / "reports/conditioning_20261107")
                or source.is_relative_to(self.root / "runs/conditioning_20261107/evaluation")
                or source.is_relative_to(self.root / "artifacts/conditioning_20261107/required_packet")):
            raise ValueError(f"Only generated report/figure PDFs are eligible: {source}")
        if relative in self.sources and self.sources[relative] != source:
            raise ValueError(f"Conflicting payload source: {relative}")
        self.sources[relative] = source

    def tree(self, source, relative, suffixes=None, exclude_names=()):
        source = Path(source)
        if not source.is_dir():
            raise FileNotFoundError(source)
        for file in sorted(source.rglob("*")):
            if file.is_file() and file.name not in exclude_names and "__pycache__" not in file.parts:
                if suffixes is None or file.suffix in suffixes:
                    self.add(file, str(PurePosixPath(relative) / file.relative_to(source).as_posix()))

    def copy(self):
        for relative, source in sorted(self.sources.items()):
            target = self.stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def preserve_existing(stage, destination):
    """Preflight every conflict before making even the first destination change."""
    stage, destination = Path(stage), Path(destination)
    members = [p for p in sorted(stage.rglob("*")) if p.is_file()]
    for source in members:
        target = destination / source.relative_to(stage)
        if target.exists() and (not target.is_file() or digest(target) != digest(source)):
            raise FileExistsError(f"Preserving different existing packet file: {target}")
    for source in members:
        target = destination / source.relative_to(stage)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)


def portable_argv(argv, root):
    """Keep the measured optimizer settings, replacing only file/device paths."""
    mapping = {
        "--data": "data/training_round0.npz", "--parent": "data/parent.npz",
        "--condition-schema-in": "data/condition_schema_uma.json",
        "--descriptor-schema-in": "data/descriptor_schema.json",
        "--descriptor-values": "data/descriptor_values.npz",
    }
    result = list(argv)
    for i, value in enumerate(result[:-1]):
        if value in mapping:
            result[i + 1] = "{packet}/" + mapping[value]
        elif value == "--output-dir":
            result[i + 1] = "{output}/" + Path(result[i + 1]).name
        elif value == "--device":
            result[i + 1] = "{device}"
    if any(str(root) in token for token in result):
        raise ValueError("Training command still contains an unresolved worktree path")
    return result


def training_commands(root, run):
    commands = {}
    for path in sorted(run.glob("*_fit_tasks.json")):
        if any(x in path.name for x in ("smoke", "edit")):
            continue
        for task in read_json(path).get("tasks", []):
            argv = task["argv"]
            output = Path(argv[argv.index("--output-dir") + 1]).name
            if output not in CORE_FITS:
                continue
            portable = portable_argv(argv, root)
            if output in commands and commands[output]["argv"] != portable:
                raise ValueError(f"Distinct training definitions for required fit {output}")
            commands[output] = dict(module="cathode_cvae.learning.train_mask", argv=portable,
                original_task_manifests=sorted(set(commands.get(output, {}).get("original_task_manifests", [])
                    + [str(path.relative_to(root))])))
    if set(commands) != set(CORE_FITS):
        raise ValueError(f"Missing required portable fit commands: {set(CORE_FITS)-set(commands)}")
    return dict(scope="Reproduction settings, not a request to retrain or an extra fit authorization",
                placeholders=["{packet}", "{output}", "{device}"], fits=commands)


def check_ready(final_evaluation, matched_evaluation, cost_receipt, report_pdf):
    expected = [(final_evaluation, 34560, {"uma"}), (matched_evaluation, 20736, {"uma", "mace"})]
    for directory, attempts, teachers in expected:
        receipt = read_json(Path(directory) / "receipt.json")
        if receipt.get("status") != "PASS" or receipt.get("attempt_rows") != attempts or set(receipt.get("teachers", [])) != teachers:
            raise ValueError(f"Required final evaluation is not complete: {directory}")
        if not receipt.get("canonical_keys_recomputed"):
            raise ValueError("Final canonical keys must have been independently recomputed")
    cost = read_json(cost_receipt)
    if cost.get("findings") or not cost.get("evidence_stable_during_audit"):
        raise ValueError("Required execution audit is not a stable, finding-free snapshot")
    if cost.get("training", {}).get("eligible_matched_round0_4000_update_trajectories") != 28:
        raise ValueError("Required packet needs all 28 matched, 4000-update core fits")
    if not Path(report_pdf).is_file() or Path(report_pdf).read_bytes()[:4] != b"%PDF":
        raise ValueError("Frozen required report PDF is missing/invalid")
    return cost


def check_evaluation_bindings(directory, inputs):
    """Refuse a PASS receipt for stale or different assessment inputs."""
    bindings = read_json(Path(directory) / "receipt.json").get("source_hashes", {})
    for name, source in inputs.items():
        if bindings.get(name) != digest(source):
            raise ValueError(f"Final evaluation input changed or lacks its binding: {directory}/{name}")


def validate_fit(root, run, name):
    directory = run / "fits" / name
    summary = read_json(directory / "summary.json")
    if summary.get("status") != "FIT_COMPLETE" or summary.get("optimizer_updates") != 4000:
        raise ValueError(f"Required fit is not complete: {name}")
    for selection in ("1.0", "0.1"):
        item = summary["selection_checkpoints"][selection]
        if digest(directory / item["filename"]) != item["sha256"]:
            raise ValueError(f"Selected checkpoint changed: {name}/{item['filename']}")
    return directory


def raw_evidence_manifest(cost, root, run):
    """Reuse the independent audit's hashes, checking unchanged file stats now."""
    hashes = cost.get("evidence_file_hashes", {})
    stats = cost.get("evidence_file_stats", {})
    entries = {}
    for filename, checksum in hashes.items():
        path = Path(filename)
        if path.is_relative_to(run / "oracle"):
            if filename not in stats:
                raise ValueError("Required execution audit lacks raw-evidence stat bindings")
            actual, bound = path.stat(), stats[filename]
            if actual.st_size != bound["bytes"] or actual.st_mtime_ns != bound["mtime_ns"]:
                raise ValueError(f"Raw evidence changed after the required audit: {path}")
            entries[str(path.relative_to(root))] = dict(sha256=checksum, **bound)
    if not entries:
        raise ValueError("Required audit did not bind the raw oracle evidence")
    return dict(scope="HPC raw evidence retained; content hashes verified by the independent allocation audit; size/mtime rechecked during curation",
                runtime_dependency=False, files=entries)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--final-evaluation", type=Path)
    parser.add_argument("--matched-evaluation", type=Path)
    parser.add_argument("--uma-scores", type=Path)
    parser.add_argument("--mace-scores", type=Path)
    parser.add_argument("--cost-receipt", type=Path)
    parser.add_argument("--report-pdf", type=Path)
    parser.add_argument("--repository", default="rsilvabuarque/ECE285-1")
    args = parser.parse_args(argv)
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    root = args.root.resolve(); run = root / "runs/conditioning_20261107"
    output = root / "artifacts/conditioning_20261107"
    final_eval = args.final_evaluation or run / "evaluation/final"
    matched_eval = args.matched_evaluation or run / "evaluation/final_matched"
    uma = args.uma_scores or run / "collected/final/uma.npz"
    mace = args.mace_scores or run / "collected/final/mace.npz"
    cost_path = args.cost_receipt or root / "reports/conditioning_20261107/execution_required.json"
    pdf = args.report_pdf or output / "required_packet/main.pdf"
    archive = run / "ECE285_conditioning_20261108_required.zip"
    if archive.exists() or (output / "manifest.json").exists() or (output / "download.json").exists():
        raise FileExistsError("Required release is write-once; preserve the existing archive/manifest")
    cost = check_ready(final_eval, matched_eval, cost_path, pdf)
    common_bindings = {
        "parent": root / "artifacts/current/core/data/parents/O3_N48_pilot.npz",
        "training_keys": run / "data/training_keys.json", "history": run / "data/history_before_final.json",
        "historical_thresholds": root / "artifacts/current/reports/generation_thresholds.csv",
        "scores_uma": uma,
        "schema_uma": root / "artifacts/overnight_20261107/data/condition_schema_uma.json",
    }
    check_evaluation_bindings(final_eval, dict(common_bindings, attempts=run / "final/attempts.csv",
        candidates=run / "final/candidates.npz"))
    check_evaluation_bindings(matched_eval, dict(common_bindings, attempts=run / "final_matched_audit/attempts.csv",
        candidates=run / "final_matched_audit/candidates.npz", scores_mace=mace,
        schema_mace=root / "artifacts/overnight_20261107/data/condition_schema_mace.json"))
    with tempfile.TemporaryDirectory(prefix="required-payload-", dir=run) as tmp:
        stage = Path(tmp); curator = Curator(root, stage)
        # Preserve protocol/selection JSONs byte-for-byte, including the reviewed
        # core-versus-optional scope amendment. No execution-state diary enters it.
        for path in sorted(output.glob("*.json")):
            curator.add(path, path.name)
        inputs = {
            "data/training_round0.npz": root / "artifacts/current/improvement_v1/training_round0.npz",
            "data/parent.npz": root / "artifacts/current/core/data/parents/O3_N48_pilot.npz",
            "data/generation_thresholds.csv": root / "artifacts/current/reports/generation_thresholds.csv",
            "data/condition_schema_uma.json": root / "artifacts/overnight_20261107/data/condition_schema_uma.json",
            "data/condition_schema_mace.json": root / "artifacts/overnight_20261107/data/condition_schema_mace.json",
            "data/training_keys.json": run / "data/training_keys.json",
            "data/history_before_development.json": run / "data/history_before_development.json",
            "data/history_before_final.json": run / "data/history_before_final.json",
            "data/history_before_final_receipt.json": run / "data/history_before_final_receipt.json",
            "data/descriptor_schema.json": root / "reports/conditioning_20261107/parent_graph/descriptor_schema.json",
            "data/descriptor_values.npz": root / "reports/conditioning_20261107/parent_graph/descriptor_values.npz",
            "data/parent_spec.json": root / "reports/conditioning_20261107/parent_graph/parent_spec.json",
            "data/periodic_graph.json": root / "reports/conditioning_20261107/parent_graph/periodic_graph.json",
            "scores/uma.npz": uma, "scores/mace.npz": mace,
            "execution/required.json": cost_path,
            "required_packet/main.pdf": pdf,
        }
        for relative, source in inputs.items(): curator.add(source, relative)
        curator.tree(output / "required_packet", "required_packet",
                     suffixes={".pdf", ".tex", ".bib", ".json", ".md", ".sh"})
        for name in ("final_protocol.json", "final_generation_manifest.json", "cost_provenance_repair.json"):
            curator.add(run / name, "execution/" + name)
        for label in ("final", "final_matched_audit"):
            curator.tree(run / label, label, suffixes={".json", ".csv", ".npz"})
        for name in CORE_FITS:
            directory = validate_fit(root, run, name)
            for filename in FIT_FILES: curator.add(directory / filename, f"fits/{name}/{filename}")
        for path in sorted(run.glob("*_fit_tasks.json")):
            if not any(x in path.name for x in ("smoke", "edit")):
                curator.add(path, "execution/training_tasks/" + path.name)
        for name in ("final", "final_matched", "policy_cost_repair", "latent", "spatial_confirmed", "graph_context_confirmation", "descriptor"):
            directory = final_eval if name == "final" else matched_eval if name == "final_matched" else run / "evaluation" / name
            curator.tree(directory, "evaluation/" + name, suffixes={".csv", ".json", ".png", ".pdf"})
        for name in ("policy_diagnostics_v2", "latent_diagnostics", "latent_development", "parent_graph",
                     "spatial_confirmed_diagnostics", "final_structural_distances", "relaxed_target_audit",
                     "independent_final_metrics", "final_cache_history_check"):
            directory = root / "reports/conditioning_20261107" / name
            if directory.is_dir(): curator.tree(directory, "diagnostics/" + name, suffixes={".json", ".csv", ".png", ".pdf", ".npz"})
        report_root = root / "reports/conditioning_20261107"
        for path in sorted(report_root.glob("fresh_install_verification*")):
            if path.is_file() and path.suffix in {".json", ".log", ".txt", ".xml", ".py"}:
                curator.add(path, "verification/" + path.name)
        for filename in ("trained_causality_audit.json", "trained_causality_audit.py", "trained_causality_audit.command.txt",
                         "spatial_confirmed_diagnostics_manifest.json", "experiment_ledger.json", "SOURCE_REVIEW.md"):
            path = report_root / filename
            if path.is_file(): curator.add(path, "verification/" + filename)
        for label, sidecar in cost.get("sidecars", {}).items():
            if isinstance(sidecar, dict) and sidecar.get("path"):
                source = Path(sidecar["path"])
                if not source.is_absolute(): source = root / source
                curator.add(source, "execution/sidecars/" + source.name)
        # Raw arrays are not required to reproduce scalar tables. Keep the raw
        # registry/worker receipts and an audited hash inventory, not 1000s of
        # force arrays or foundation checkpoints.
        for filename in ("registry.json", "tasks.jsonl", "cache_uma.json", "cache_mace.json"):
            curator.add(run / "oracle" / filename, "raw_evidence/" + filename)
        curator.tree(run / "oracle/workers", "raw_evidence/workers", suffixes={".json"})
        score_receipts = sorted((run / "oracle/scores").glob("*/*.json"))
        if len(score_receipts) != cost["oracle"]["registered_tasks"]:
            raise ValueError("Required raw task receipts are incomplete")
        for path in score_receipts:
            curator.add(path, "raw_evidence/score_receipts/" + path.name)
        curator.tree(run / "executed_sources", "source/executed_sources", suffixes={".py", ".json"})
        for directory in ("src", "tests", "scripts/conditioning_20261107"):
            curator.tree(root / directory, "source/current/" + directory, suffixes={".py"})
        for filename in ("pyproject.toml", "README.md"):
            curator.add(root / filename, "source/current/" + filename)
        curator.copy()
        write_json(stage / "execution/portable_training_commands.json", training_commands(root, run))
        write_json(stage / "raw_evidence/manifest.json", raw_evidence_manifest(cost, root, run))
        jobs = {
            "final_uma": dict(attempts="final/attempts.csv", candidates="final/candidates.npz",
                teachers=["uma"], reference="evaluation/final"),
            "matched_uma_mace": dict(attempts="final_matched_audit/attempts.csv", candidates="final_matched_audit/candidates.npz",
                teachers=["uma", "mace"], reference="evaluation/final_matched"),
        }
        source_hashes = {str(p.relative_to(root)): digest(p) for p in sorted((root / "src").rglob("*.py"))}
        write_json(stage / "replay.json", dict(schema_version=1, jobs=jobs, parent="data/parent.npz",
            training_keys="data/training_keys.json", history="data/history_before_final.json",
            historical_thresholds="data/generation_thresholds.csv", source_hashes=source_hashes,
            scores={t: f"scores/{t}.npz" for t in ("uma", "mace")},
            schemas={t: f"data/condition_schema_{t}.json" for t in ("uma", "mace")},
            final_protocol="execution/final_protocol.json", cost_receipt="execution/required.json"))
        ignored = ["*.npz", "*.pt", "data/training_keys.json", "data/history*.json", "raw_evidence/**",
                   "final/attempts*.csv", "final_matched_audit/attempts*.csv", "evaluation/**/attempt_outcomes.csv",
                   "evaluation/**/candidate_measurements.csv", "evaluation/**/budget_curves.csv"]
        ignored += [str(p.relative_to(stage)) for p in sorted(stage.rglob("*")) if p.is_file() and p.stat().st_size > 2 * 1024 ** 2]
        (stage / ".gitignore").write_text("# Full payload is in the hash-verified release ZIP.\n" + "\n".join(dict.fromkeys(ignored)) + "\n")
        files = {str(p.relative_to(stage)): dict(bytes=p.stat().st_size, sha256=digest(p))
                 for p in sorted(stage.rglob("*")) if p.is_file()}
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        manifest = dict(schema_version=1, release_tag=RELEASE_TAG, created_UTC=datetime.now(timezone.utc).isoformat(),
            base_commit=commit, files=files, source_hashes=source_hashes, core_full_fit_count=28,
            scope="Required report snapshot; final UMA and prespecified matched MACE cached replay. No optional editor/NEB, new DFT, foundation weights, or private input documents.",
            raw_EFS_delivery="HPC retained, audited hash inventory in raw_evidence/manifest.json",
            cost_scope="Required-program snapshot. Active allocation cost is preliminary until all allocations are released; see execution/required.json.",
            source_provenance={rel: str(src.relative_to(root)) for rel, src in sorted(curator.sources.items())})
        write_json(stage / "manifest.json", manifest)
        # Preflight before creating an archive or adding anything to Git's tree.
        preserve_existing(stage, output)
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in sorted(stage.rglob("*")):
                if path.is_file(): bundle.write(path, path.relative_to(stage).as_posix())
        checksum = digest(archive)
        archive.with_suffix(".zip.sha256").write_text(checksum + "  " + archive.name + "\n")
        download = dict(url=f"https://github.com/{args.repository}/releases/download/{RELEASE_TAG}/{archive.name}",
            sha256=checksum, extract_to="artifacts/conditioning_20261107", bytes=archive.stat().st_size, files=len(files) + 1)
        with (output / "download.json").open("x") as f: f.write(json.dumps(download, indent=2) + "\n")
        print(json.dumps(dict(archive=str(archive), **download), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

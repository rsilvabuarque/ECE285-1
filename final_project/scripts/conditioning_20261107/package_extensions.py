"""Curate the optional results as a separate, core-dependent release asset.

This is an explicit file allowlist, not a recursive copy of the worktree. The
required ZIP/gate are inputs and are never rewritten. Original protocol bytes
retain their historical paths; replay supplies a separately recorded path map.
Global force/stress shards stay on HPC, with a hash/stat inventory in this ZIP.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

from replay_extensions import CORE_SHA256, RELEASE_TAG, digest, safe_relative, verify_archive

R = "runs/conditioning_20261107"
A = "artifacts/conditioning_20261107"
S = "scripts/conditioning_20261107"
REPLAY_TABLES = ("metrics.csv", "confusion.csv", "attempt_outcomes.csv", "candidate_measurements.csv",
    "budget_curves.csv", "summary.csv", "seed_metrics.csv", "summary_by_request.csv",
    "fixed_q20_low_request.csv", "allocation_realization.csv", "allocation_joint_realization.csv",
    "allocation_joint_summary.csv")
EDITOR_TABLES = ("attempt_outcomes.csv", "paired_energy_outcomes.csv", "metrics.csv", "seed_metrics.csv", "summary.csv")
TEMP_TABLES = ("attempt_outcomes.csv", "schedule_summary.csv", "requested_vs_measured.csv", "temperature_identity.csv")
LOW_TABLES = ("low_request_curves.csv", "low_request_curve_means.csv", "paired_request_energy.csv",
    "request_energy_summary.csv", "terminal_best_novel_low_children.csv", "terminal_minima_composition_means.csv", "query_accounting.csv")
POSTHOC_TABLES = ("requested_bin_energy_summary.csv", "requested_bin_energy_by_fit.csv",
    "requested_bin_energy_cells.csv", "LOW_request_paired_energy_cells.csv")
EDITOR_FIT_FILES = {"best.pt", "best_validation_beta_0p1.pt", "configuration.json", "summary.json",
    "history.csv", "latent_history.jsonl", "condition_schema.json"}
NEW_SOURCES = [f"src/cathode_cvae/learning/{x}.py" for x in ("edit_model", "train_edit", "generate_edit")] + [
    f"{S}/{x}.py" for x in ("prepare_edit_data", "run_edit_fits", "generate_edit_comparison",
    "evaluate_edit_comparison", "check_edit_results", "check_editor_checkpoint_aliases", "summarize_editor", "aggregate_prior_ablation", "check_aggregate_prior",
    "temperature_followup", "neb_common", "neb_prepare", "neb_run", "neb_mace_profile", "neb_summarize",
    "neb_allocation", "package_extensions", "replay_extensions")]


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def extract_destination(root, output):
    """Public extraction location follows the chosen in-project packet version."""
    return safe_relative(Path(output).resolve().relative_to(Path(root).resolve()).as_posix())


class ExtensionCurator:
    """Deduplicate immutable core members by hash, preserving source identities."""
    def __init__(self, root, core):
        self.root = Path(root).resolve(); self.core = core
        self.locator = {}; self.files = {}; self.hash_core = {}
        for member, binding in core["files"].items():
            self.hash_core.setdefault(binding["sha256"], member)
        for member, source in core.get("source_provenance", {}).items():
            self.locator[safe_relative(source)] = dict(packet="core", member=member, **core["files"][member])

    def add(self, source):
        source = Path(source).resolve()
        if not source.is_relative_to(self.root): raise ValueError(f"Source outside designated worktree: {source}")
        relative = source.relative_to(self.root).as_posix(); parts = Path(relative).parts
        if any(x in {"for_chat_gpt", ".git", ".venv", ".venvs", "models", ".models"} for x in parts):
            raise ValueError(f"Private or foundation-weight source excluded: {relative}")
        if source.name in {"instructions.txt", "AGENTS.md", "AGENTS.override.md"}: raise ValueError("Private instructions excluded")
        if not source.is_file(): raise FileNotFoundError(source)
        if source.suffix in {".model", ".pth", ".ckpt", ".bin", ".safetensors"}:raise ValueError("Foundation weights excluded")
        if source.suffix==".pdf" and not any(source.is_relative_to(self.root/p) for p in (
                "reports/conditioning_20261107", R+"/evaluation", A+"/required_packet")):
            raise ValueError("Only generated project report/figure PDFs are eligible")
        if source.suffix == ".pt":
            fits = {f"editor_{seed}" for seed in (20261106, 20261107, 20261108)}
            if source.parent.parent != self.root / R / "fits" or source.parent.name not in fits or source.name not in EDITOR_FIT_FILES:
                # Core checkpoints are allowed only as references to already
                # authenticated core members, never as new weight payloads.
                if relative not in self.locator or digest(source) != self.locator[relative]["sha256"]:
                    raise ValueError("Only the three new editor checkpoints are eligible")
        if source.suffix == ".npz" and "oracle/scores/" in relative: raise ValueError("Raw global E/F/stress shards stay on HPC")
        sha = digest(source); binding = dict(bytes=source.stat().st_size, sha256=sha)
        if relative in self.locator and self.locator[relative]["sha256"] == sha: return relative
        if relative in self.locator:
            old=self.locator[relative]
            # Preserve a prior authenticated version even when an explicitly
            # included mutable report/helper now occupies the same filename.
            # This virtual runtime alias is clearly distinct from an original
            # project path, and permits SHA-aware historical references.
            version=f".historical_versions/{old['sha256']}/{relative}"
            self.locator.setdefault(version,dict(**old,historical_original_relative=relative))
        if sha in self.hash_core:
            record = dict(packet="core", member=self.hash_core[sha], **binding)
        else:
            member = "project/" + safe_relative(relative)
            if member in self.files and self.files[member] != source: raise ValueError("Duplicate payload destination")
            self.files[member] = source; record = dict(packet="extensions", member=member, **binding)
        self.locator[relative] = record
        return relative

    def tree(self, relative, *, required=True, suffixes=None):
        path = self.root / relative
        if not path.is_dir():
            if required: raise FileNotFoundError(path)
            return
        for source in sorted(path.rglob("*")):
            if source.is_file() and "__pycache__" not in source.parts:
                if suffixes is None or source.suffix in suffixes or source.name in {"POSCAR", "CONTCAR"}: self.add(source)

    def protocol_sources(self, path):
        protocol = json.loads(Path(path).read_text())
        for source in protocol.get("sources", {}).values():
            if not isinstance(source, dict) or "path" not in source or "sha256" not in source: continue
            actual = Path(source["path"])
            if digest(actual) != source["sha256"]: raise ValueError(f"Frozen protocol source differs: {actual}")
            self.add(actual)


def pass_receipt(path, **expected):
    result = json.loads(Path(path).read_text())
    if result.get("status") != "PASS": raise ValueError(f"Incomplete science receipt: {path}")
    for key, value in expected.items():
        if result.get(key) != value: raise ValueError(f"Receipt {key} is not {value}: {path}")
    return result


def raw_inventory(curator, registry_path):
    """Bind retained task JSON to raw bytes without placing raw arrays in ZIP."""
    registry = json.loads(Path(registry_path).read_text()); rows = []
    for task in registry["tasks"]:
        path = curator.root / R / "oracle/scores" / task["model"] / (task["task_id"] + ".json")
        result = json.loads(path.read_text()); raw = path.with_suffix(".npz")
        if result.get("status") != "complete" or result["task"] != task: raise ValueError(f"Task not complete: {path}")
        actual = digest(raw)
        if actual != result["output_sha256"]: raise ValueError(f"Raw output changed: {raw}")
        curator.add(path); stat = raw.stat()
        rows.append(dict(task_id=task["task_id"], model=task["model"], path=str(raw), sha256=actual,
            bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, receipt=path.relative_to(curator.root).as_posix(),
            receipt_sha256=digest(path), logical_geometry_labels=len(task["expected_structure_ids"])))
    return dict(status="VERIFIED_AT_CURATION", raw_shards=rows,
        delivery="Raw global E/F/stress shards remain on HPC. Portable replay verifies archived inventory and task receipts, not absent raw array bytes.")


def collect(root, core, *, neb_run=None, neb_summary=None, neb_deferred=None):
    c = ExtensionCurator(root, core); root = c.root
    for relative, expected in core["source_hashes"].items():
        if digest(root / relative) != expected: raise ValueError(f"Required core source changed: {relative}")
    pass_receipt(root / R / "evaluation/editor/receipt.json", attempts=6912)
    pass_receipt(root / R / "evaluation/aggregate_prior/receipt.json", attempt_rows=6912)
    pass_receipt(root / R / "collected/optional_final/receipt.json")
    temp = "reports/conditioning_20261107/temperature_followup"
    pass_receipt(root / temp / "analysis/receipt.json", attempts=768, unique_scored_candidates=102)
    pass_receipt(root / "reports/conditioning_20261107/editor_low_request_v2/receipt.json", attempts=6912)
    gate = root / A / "required_gate.json"; gate_data = json.loads(gate.read_text())
    if gate_data.get("status") != "COMPLETE": raise ValueError("Required report gate is incomplete")
    c.add(gate); c.add(root / gate_data["report_pdf"]["path"])
    trees = [f"{R}/editor/data_v2", f"{R}/editor/assessment_v1", f"{R}/editor_fit_tasks.claims",
        f"{R}/executed_sources/editor_v1", f"{R}/aggregate_prior/preparation", f"{R}/aggregate_prior/assessment",
        f"{R}/evaluation/editor", f"{R}/evaluation/aggregate_prior", f"{R}/collected/optional_final", temp,
        "reports/conditioning_20261107/independent_editor_metrics", "reports/conditioning_20261107/independent_aggregate_prior", "reports/conditioning_20261107/editor_low_request_v2",
        "reports/conditioning_20261107/editor_checkpoint_alias_audit",
        "reports/conditioning_20261107/neb_pilot/prepared_v3"]
    for relative in trees: c.tree(relative, suffixes={".npz", ".json", ".jsonl", ".csv", ".pdf", ".png", ".py", ".md", ".log", ".txt", ".cif", ".vasp", ".POSCAR", ".CONTCAR"})
    # The report owner assembles this reviewed snapshot only after the optional
    # outcomes are ready. It contains the PDF and complete used build inputs.
    c.tree("reports/conditioning_20261107/final_report_snapshot")
    for relative in [f"{R}/editor_fit_tasks.json", f"{R}/editor/protocol.json", f"{R}/editor/registration_preflight.json", *NEW_SOURCES]: c.add(root / relative)
    for seed in (20261106, 20261107, 20261108):
        folder = root / R / "fits" / f"editor_{seed}"
        summary = json.loads((folder / "summary.json").read_text())
        cfg = json.loads((folder / "configuration.json").read_text())
        if cfg.get("experiment_phase") != "optional_editor" or cfg.get("maximum_updates") != 4000:
            raise ValueError("Editor fit does not match the bounded optional protocol")
        # Actual optimizer count, rather than the configured maximum, is bound
        # by both the fit summary and the preserved iteration history.
        if summary.get("optimizer_updates", summary.get("updates")) != 4000: raise ValueError("Editor fit did not complete 4000 updates")
        for name in EDITOR_FIT_FILES: c.add(folder / name)
    for source in sorted((root / "tests").glob("test_edit*.py")): c.add(source)
    for pattern in ("test_aggregate_prior*.py", "test_neb*.py", "test_extensions*.py", "test_checkpoint_alias*.py"):
        for source in sorted((root / "tests").glob(pattern)): c.add(source)
    c.add(root / "docs/AGGREGATE_PRIOR_ABLATION.md")
    for protocol in (f"{R}/editor/assessment_v1/protocol.json", f"{R}/aggregate_prior/preparation/protocol.json",
                     f"{R}/aggregate_prior/assessment/protocol.json", f"{temp}/protocol.json"):
        c.protocol_sources(root / protocol)
    # Hard-coded relative lookups used by the unchanged cached editor evaluator.
    for relative in (f"{R}/data/training_keys.json", "artifacts/current/reports/generation_thresholds.csv",
                     "reports/conditioning_20261107/parent_graph/periodic_graph.json",
                     f"{R}/oracle_snapshots/required/registry.json", f"{R}/oracle/cache_uma.npz", f"{R}/oracle/cache_uma.json"):
        c.add(root / relative)
    # Preserve original scalar-score receipts used in the saved editor/TEMP
    # analyses too. Replaying from the later union must not silently change an
    # original source binding; either scalar pool contains these same energies.
    c.tree(f"{R}/collected/editor", suffixes={".npz", ".json"})
    registry = root / R / "collected/optional_final/registry_snapshot.json"
    inventory = raw_inventory(c, registry)
    c.tree(f"{R}/oracle/workers", required=False, suffixes={".json"})
    for relative in ("reports/conditioning_20261107/aggregate_prior", "reports/conditioning_20261107/neb_pilot"):
        c.tree(relative, required=False, suffixes={".json", ".csv", ".pdf", ".png", ".log", ".md"})
    if bool(neb_run) == bool(neb_deferred): raise ValueError("Supply exactly one of --neb-run or --neb-deferred-receipt")
    neb = dict(status="DEFERRED")
    if neb_deferred:
        c.add(neb_deferred); neb["deferred_receipt"] = Path(neb_deferred).resolve().relative_to(root).as_posix()
    else:
        neb_run = Path(neb_run).resolve()
        if not neb_run.is_relative_to(root): raise ValueError("NEB run must be in the project")
        c.tree(neb_run.relative_to(root), suffixes={".npz", ".json", ".csv", ".pdf", ".png", ".log", ".txt", ".traj", ".cif", ".vasp", ".POSCAR", ".CONTCAR", ".xyz", ".extxyz"})
        receipts = list(neb_run.glob("*/receipt.json"))
        if not receipts: raise ValueError("No terminal NEB event/profile receipts; preserve failures and provide explicit defer receipt instead")
        if neb_summary is None: raise ValueError("--neb-summary is required for a completed/failed NEB run")
        c.tree(Path(neb_summary).resolve().relative_to(root), suffixes={".json", ".csv", ".pdf", ".png"})
        if not (Path(neb_summary) / "outcomes.csv").is_file(): raise ValueError("NEB saved-outcome summary missing")
        neb = dict(status="SAVED_OUTCOMES", run=neb_run.relative_to(root).as_posix(), reference=Path(neb_summary).resolve().relative_to(root).as_posix())
    jobs = dict(editor=dict(generated=f"{R}/editor/assessment_v1", reference=f"{R}/evaluation/editor", tables=list(EDITOR_TABLES)),
        aggregate=dict(generated=f"{R}/aggregate_prior/assessment", reference=f"{R}/evaluation/aggregate_prior", tables=list(REPLAY_TABLES)),
        temperature=dict(generated=temp, reference=f"{temp}/analysis", tables=list(TEMP_TABLES)),
        editor_low=dict(reference="reports/conditioning_20261107/editor_low_request_v2", tables=list(LOW_TABLES)),
        editor_posthoc=dict(reference="reports/conditioning_20261107/independent_editor_metrics", tables=list(POSTHOC_TABLES)), neb=neb)
    jobs["aggregate_independent"] = dict(reference="reports/conditioning_20261107/independent_aggregate_prior",
        tables=["metrics.csv", "confusion.csv", "seed_metrics.csv", "summary.csv", "original_q20_LOW_request.csv", "moment_membership_and_matrix_checks.csv"])
    for key, job in jobs.items():
        for name in job.get("tables", []): c.add(root / job["reference"] / name)
    return c, jobs, inventory


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--core-archive", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--neb-run", type=Path); p.add_argument("--neb-summary", type=Path); p.add_argument("--neb-deferred-receipt", type=Path)
    p.add_argument("--include-file", type=Path, action="append", default=[], help="Explicit additional generated report/receipt; same private/weight guards apply")
    p.add_argument("--attach-final-receipts", type=Path, nargs="+", help="After allocation release, attach small receipts outside the immutable ZIP")
    a = p.parse_args(argv)
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    extraction_path=extract_destination(a.root,a.output_dir)
    if a.attach_final_receipts:
        # This separate index binds closing accounting to an already prepared
        # ZIP. It never rewrites its scientific manifest or archive checksum.
        download = json.loads((a.output_dir/"download_ext.json").read_text())
        verify_archive(a.archive, download["sha256"], "manifest_ext.json")
        destination=a.output_dir/"closing_receipts"
        if destination.exists() or (a.output_dir/"final_receipts.json").exists(): raise FileExistsError("Preserve existing closing receipts")
        supplied=[]
        for source in a.attach_final_receipts:
            source=source.resolve()
            if not source.is_relative_to(a.root.resolve()) or source.suffix!=".json" or source.stat().st_size>8*1024**2:
                raise ValueError("Closing receipt must be a small project JSON")
            supplied.append(source)
        if len({p.name for p in supplied})!=len(supplied):raise ValueError("Closing receipt basenames must be distinct")
        destination.mkdir()
        for source in supplied:shutil.copyfile(source,destination/source.name)
        write_json(a.output_dir/"final_receipts.json",dict(status="ATTACHED_AFTER_SCIENTIFIC_ZIP",archive_sha256=download["sha256"],
            attached_UTC=datetime.now(timezone.utc).isoformat(),receipts={"closing_receipts/"+p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in supplied},
            archive_modified=False,scope="Closing accounting/completion evidence; original scientific archive and gate remain unchanged"))
        print(json.dumps(dict(status="CLOSING_RECEIPTS_ATTACHED",archive_modified=False,receipts=len(supplied))));return 0
    if a.output_dir.exists() or a.archive.exists(): raise FileExistsError("Preserve previous extension packet/archive")
    core = verify_archive(a.core_archive, CORE_SHA256, "manifest.json")
    if core.get("release_tag") != RELEASE_TAG or core.get("core_full_fit_count") != 28: raise ValueError("Unexpected required core release")
    c, jobs, inventory = collect(a.root, core, neb_run=a.neb_run, neb_summary=a.neb_summary, neb_deferred=a.neb_deferred_receipt)
    for source in a.include_file:c.add(source)
    a.output_dir.parent.mkdir(parents=True, exist_ok=True); a.archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="extensions-stage-", dir=a.output_dir.parent) as temp:
        stage = Path(temp) / "payload"; stage.mkdir()
        for member, source in sorted(c.files.items()):
            target = stage / member; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
        config = dict(schema_version=1, original_root=str(c.root), locator=c.locator, jobs=jobs,
            scores={x:f"{R}/collected/optional_final/{x}.npz" for x in ("uma", "mace")},
            training_tasks=f"{R}/editor_fit_tasks.json", raw_inventory="raw_shards_inventory.json",
            path_relocation="Only exact verified source paths may be redirected; original protocol/receipt bytes remain unchanged.")
        write_json(stage / "replay_ext.json", config); write_json(stage / "raw_shards_inventory.json", inventory)
        (stage / "README.txt").write_text("This is the optional companion to the immutable required release.\n"
            "Verify and replay with replay_extensions.py --core-archive CORE.zip --extensions-archive THIS.zip "
            "--download-ext download_ext.json --output-dir NEW_DIRECTORY.\n"
            "Replay never evaluates an MLIP or trains a network. Foundation weights and private PDFs are excluded.\n"
            "Original paths are provenance, not runtime requirements: verified sources are relocated into a new mirror.\n"
            "Raw global E/F/stress arrays remain on HPC; the inventory/receipts bind their curation-time hashes.\n"
            "Final allocation/cost receipts may be published separately after allocations close; they do not alter this scientific payload.\n")
        (stage / ".gitignore").write_text("*\n!.gitignore\n!manifest_ext.json\n!download_ext.json\n!README.txt\n")
        files = {path.relative_to(stage).as_posix():dict(bytes=path.stat().st_size, sha256=digest(path))
            for path in sorted(stage.rglob("*")) if path.is_file()}
        manifest = dict(schema_version=1, release_tag=RELEASE_TAG, created_UTC=datetime.now(timezone.utc).isoformat(),
            core_archive_sha256=CORE_SHA256, core_manifest_sha256=hashlib.sha256(zipfile.ZipFile(a.core_archive).read("manifest.json")).hexdigest(),
            core_full_fit_count=28, optional_editor_full_fit_count=3, aggregate_prior_new_updates=0,
            files=files, scientific_scope="Cached optional editor, aggregate-prior, temperature and bounded NEB results, including failures",
            final_accounting_scope="Separate closing receipts after allocation release; scientific ZIP is immutable",
            source_hashes={r:digest(c.root/r) for r in sorted(set(NEW_SOURCES)|{
                source.resolve().relative_to(c.root).as_posix() for source in a.include_file if source.suffix==".py"})})
        write_json(stage / "manifest_ext.json", manifest)
        with zipfile.ZipFile(a.archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path in sorted(stage.rglob("*")):
                if path.is_file(): z.write(path, path.relative_to(stage).as_posix())
        verify_archive(a.archive, digest(a.archive), "manifest_ext.json")
        shutil.copytree(stage, a.output_dir)
    download = dict(url=f"https://github.com/rsilvabuarque/ECE285-1/releases/download/{RELEASE_TAG}/{a.archive.name}",
        sha256=digest(a.archive), bytes=a.archive.stat().st_size, files=len(manifest["files"])+1,
        extract_to=extraction_path, core_archive_sha256=CORE_SHA256,
        manifest_sha256=digest(a.output_dir/"manifest_ext.json"), source_packet="Separate optional extension; required archive/gate unchanged")
    write_json(a.output_dir/"download_ext.json", download)
    print(json.dumps(dict(status="PREPARED_FOR_ROOT_REVIEW", **download), indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())

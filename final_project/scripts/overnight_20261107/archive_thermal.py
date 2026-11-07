"""Stream completed thermal-pilot evidence into one durable tar, without jobs.

Run only after MD and xPT writers stop. Model weights, environments, vendor
trees and symlinks are forbidden. Input files are hashed before copying and
again as they enter the tar; a final stat/path-set check detects active writers.
The internal content manifest omits itself; the external receipt hashes the
whole archive, avoiding a self-referential checksum.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import time


FORBIDDEN_PARTS = {"vendor", "env", "envs", ".venv", "venv", "site-packages", "checkpoints", ".git", "__pycache__", ".env", ".envrc"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".model", ".onnx", ".bin", ".pem", ".key"}
RUN_DIRECTORIES = {"tasks", "configs", "logs", "executed_sources", "selection", "environment_packages"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def stamp(path):
    info = path.stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def allowed_run_path(relative):
    if relative.parts[0] in RUN_DIRECTORIES: return True
    return len(relative.parts) == 1 and (relative.suffix == ".log" or
        (relative.suffix == ".json" and ("tasks" in relative.stem or "config" in relative.stem)))


def selected_files(root, *, run=False):
    """Walk explicit roots only; do not follow symlinks or discover other trees."""
    if root.is_symlink() or not root.is_dir(): raise ValueError(f"Expected a regular source directory: {root}")
    files = []
    for directory, subdirs, names in os.walk(root, followlinks=False):
        base = Path(directory)
        if run and base == root:
            # Explicitly unselected trees (notably vendor) are never opened.
            subdirs[:] = [name for name in subdirs if name in RUN_DIRECTORIES]
        for name in list(subdirs):
            p = base / name
            if p.is_symlink(): raise ValueError(f"Symlink source directory is forbidden: {p}")
            if name in FORBIDDEN_PARTS: raise ValueError(f"Excluded payload appeared inside selected source: {p}")
        for name in names:
            p = base / name; relative = p.relative_to(root)
            if run and not allowed_run_path(relative): continue
            if p.is_symlink(): raise ValueError(f"Symlink source file is forbidden: {p}")
            if not stat.S_ISREG(p.stat().st_mode): raise ValueError(f"Nonregular source file: {p}")
            if set(relative.parts) & FORBIDDEN_PARTS or p.suffix.lower() in FORBIDDEN_SUFFIXES:
                raise ValueError(f"Environment, secret-key or foundation-weight payload forbidden: {p}")
            files.append(p)
    return sorted(files)


def completion_checks(results, files, include_diagnostic_failures):
    statuses = []; noncomplete = []
    for path in files:
        if path.name != "receipt.json": continue
        receipt = json.loads(path.read_text())
        status = str(receipt.get("status", "MISSING_STATUS"))
        command = receipt.get("command")
        complete = status in {"COMPLETE", "CONVERGED", "PASS", "CALCULATED_DIAGNOSTIC", "PASS_COMPLETED_ARRAY_CONTRACT"}
        row = dict(path=str(path), command=command, status=status)
        statuses.append(row)
        if not complete:
            noncomplete.append(row)
            if not include_diagnostic_failures:
                raise RuntimeError(f"Unfinished/failed evidence {path}: {status}; wait for writers to stop or explicitly preserve terminated diagnostic failures")
    # A production/equilibration NPZ or live MD trace must have its producer
    # receipt. This prevents archiving a half-written run before its first NPZ.
    md_directories = {p.parent for p in files if p.name in {"production.npz", "equilibration.npz", "production_trace.csv", "equilibration_trace.csv"}}
    for directory in md_directories:
        path = directory / "receipt.json"
        if not path.is_file(): raise RuntimeError(f"MD evidence has no producer receipt: {directory}")
        receipt = json.loads(path.read_text())
        if receipt.get("command") != "md": raise RuntimeError(f"MD producer identity missing: {path}")
        if receipt.get("status") != "COMPLETE" and not include_diagnostic_failures:
            raise RuntimeError(f"MD is not COMPLETE: {path}")
    return statuses, noncomplete


class HashingReader:
    def __init__(self, stream): self.stream = stream; self.hash = hashlib.sha256(); self.bytes = 0
    def read(self, size=-1):
        value = self.stream.read(size); self.hash.update(value); self.bytes += len(value); return value


def exclusive_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def archive(args):
    if args.results.is_symlink() or args.run.is_symlink(): raise ValueError("Source roots cannot be symlinks")
    # Check the literal destinations before resolve(), including dangling links.
    for path in [args.archive, args.manifest, Path(str(args.archive) + ".sha256"), Path(str(args.archive) + ".partial")]:
        if path.exists() or path.is_symlink(): raise FileExistsError(f"Refusing existing destination: {path}")
    results = args.results.resolve(); run = args.run.resolve()
    target = args.archive.resolve(); manifest = args.manifest.resolve()
    checksum_path = Path(str(target) + ".sha256"); partial = Path(str(target) + ".partial")
    for path in [target, manifest, checksum_path, partial]:
        if path.exists() or path.is_symlink(): raise FileExistsError(f"Refusing existing destination: {path}")
    if target.suffix != ".tar": raise ValueError("Use an uncompressed .tar; native NPZ files are already compressed")
    if results == run or results in run.parents or run in results.parents: raise ValueError("Results and run roots must be separate")
    if results in target.parents or results in manifest.parents: raise ValueError("Outputs cannot be inside the selected results source")
    target.parent.mkdir(parents=True, exist_ok=True); manifest.parent.mkdir(parents=True, exist_ok=True)
    result_paths = selected_files(results); run_paths = selected_files(run, run=True)
    if not result_paths: raise ValueError("No native result files found")
    statuses, noncomplete = completion_checks(results, result_paths, args.include_diagnostic_failures)
    entries = {}; snapshots = {}
    for label, base, paths in [("results", results, result_paths), ("run", run, run_paths)]:
        for path in paths:
            key = str(Path(label) / path.relative_to(base)); before = stamp(path); checksum = sha(path)
            if stamp(path) != before: raise RuntimeError(f"Source changed during inventory: {path}")
            snapshots[key] = before
            entries[key] = dict(source_path=str(path), bytes=before[2], sha256=checksum, mtime_ns=before[3])
    content = dict(schema_version=1, created_unix=time.time(), scope="Native same-UMA thermal-pilot evidence; no environments, vendors or foundation weights",
                   source_roots=dict(results=str(results), run=str(run)), run_selection="Top-level *tasks*.json, *config*.json, *.log; tasks/configs/logs/executed_sources/selection/environment_packages directories only",
                   files=entries, file_count=len(entries), total_source_bytes=sum(x["bytes"] for x in entries.values()),
                   producer_receipts=statuses, included_diagnostic_failures=noncomplete,
                   writer_requirement="All source writers must have stopped; terminated incomplete evidence is explicitly diagnostic when opted in")
    content_bytes = (json.dumps(content, indent=2, allow_nan=False) + "\n").encode()
    source_path_sets = (set(map(str, result_paths)), set(map(str, run_paths)))
    fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    started = time.perf_counter()
    try:
        with os.fdopen(fd, "wb") as raw:
            with tarfile.open(fileobj=raw, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                info = tarfile.TarInfo("CONTENT_MANIFEST.json"); info.size = len(content_bytes); info.mode = 0o600
                tar.addfile(info, io.BytesIO(content_bytes))
                for key, entry in entries.items():
                    path = Path(entry["source_path"])
                    if path.is_symlink() or stamp(path) != snapshots[key]: raise RuntimeError(f"Source changed before tar copy: {path}")
                    info = tar.gettarinfo(str(path), arcname=key)
                    info.uid = 0; info.gid = 0; info.uname = ""; info.gname = ""
                    with path.open("rb") as source:
                        copied = HashingReader(source); tar.addfile(info, copied)
                    if copied.bytes != entry["bytes"] or copied.hash.hexdigest() != entry["sha256"] or stamp(path) != snapshots[key]:
                        raise RuntimeError(f"Source changed while entering tar: {path}")
            raw.flush(); os.fsync(raw.fileno())
        if source_path_sets != (set(map(str, selected_files(results))), set(map(str, selected_files(run, run=True)))):
            raise RuntimeError("Selected source path set changed during archiving")
        for key, entry in entries.items():
            if Path(entry["source_path"]).is_symlink() or stamp(Path(entry["source_path"])) != snapshots[key]:
                raise RuntimeError(f"Source changed after its tar entry: {entry['source_path']}")
        # A hard link publishes the complete archive atomically without ever
        # replacing an existing destination. Both paths are on the same CFS.
        os.link(partial, target); partial.unlink()
        archive_sha = sha(target)
        with checksum_path.open("x") as stream: stream.write(archive_sha + "  " + target.name + "\n")
        receipt = {**content, "status": "ARCHIVED_WITH_DIAGNOSTIC_FAILURES" if noncomplete else "ARCHIVED_COMPLETE_EVIDENCE",
            "archive": str(target), "archive_sha256": archive_sha, "archive_bytes": target.stat().st_size,
            "archive_format": "uncompressed PAX tar", "internal_manifest": "CONTENT_MANIFEST.json",
            "content_manifest_sha256": hashlib.sha256(content_bytes).hexdigest(),
            "internal_manifest_self_hash_omitted": True, "script_sha256": sha(__file__),
            "elapsed_seconds": time.perf_counter() - started,
            "verification": "Initial SHA256 plus SHA256 of exactly streamed tar bytes per file; matching pre/post/final inode,size,mtime,ctime; unchanged selected path sets; whole final archive SHA256"}
        exclusive_json(manifest, receipt)
        return receipt
    except BaseException as error:
        failure = Path(str(partial) + ".failure.json")
        if not failure.exists():
            exclusive_json(failure, dict(status="ARCHIVE_FAILED_PRESERVED_PARTIAL", error_type=type(error).__name__, error=str(error), partial=str(partial), completed_archive=str(target) if target.exists() else None))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True); p.add_argument("--run", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True); p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--include-diagnostic-failures", action="store_true", help="Preserve terminated failed/incomplete evidence explicitly; never use while writers are active")
    args = p.parse_args()
    if os.environ.get("NERSC_HOST") and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run durable evidence copying inside an authorized compute allocation")
    result = archive(args)
    print(json.dumps({k: result[k] for k in ["status", "archive", "archive_sha256", "archive_bytes", "file_count", "total_source_bytes"]}, indent=2))


if __name__ == "__main__": main()

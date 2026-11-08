"""Replay the required final tables from a verified, portable scalar-score packet.

This imports no MLIP and makes no oracle calls. Checkpoints are included so that
collaborators can inspect/retrain models separately; table replay needs only the
saved attempts, canonical candidates, scores, frozen schemas and novelty history.
On NERSC, run inside an allocation. Any other host follows its own site controls.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shlex
import sys


TABLES = (
    "metrics.csv", "confusion.csv", "attempt_outcomes.csv", "candidate_measurements.csv", "budget_curves.csv",
    "summary.csv", "seed_metrics.csv", "summary_by_request.csv", "fixed_q20_low_request.csv",
    "allocation_realization.csv", "allocation_joint_realization.csv", "allocation_joint_summary.csv",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def within(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts or any(":" in x for x in relative.parts):
        raise ValueError(f"Unsafe packet-relative path: {relative}")
    path = Path(root).joinpath(*relative.parts)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Packet symlink escapes the payload root")
    return path


def verify_packet(packet, source_root):
    """Validate every released member plus the runtime sources used for tables."""
    manifest_path = Path(packet) / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("release_tag") != "ece285-conditioning-20261108" or manifest.get("core_full_fit_count") != 28:
        raise ValueError("Not the required 28-fit conditioning packet")
    files = manifest.get("files", {})
    if not files or "replay.json" not in files:
        raise ValueError("Payload manifest lacks the portable replay definition")
    for name, binding in files.items():
        path = within(packet, name)
        if not path.is_file() or path.stat().st_size != binding["bytes"] or digest(path) != binding["sha256"]:
            raise ValueError(f"Payload binding failed: {name}")
    config = read_json(within(packet, "replay.json"))
    if config.get("source_hashes") != manifest.get("source_hashes"):
        raise ValueError("Replay source binding differs from the payload manifest")
    for relative, expected in config["source_hashes"].items():
        source = within(source_root, relative)
        if not source.is_file() or digest(source) != expected:
            raise ValueError(f"Runtime source differs from the released evaluator: {relative}")
    # Hashes are accidental-corruption/provenance checks. The outer ZIP checksum
    # in Git's download.json is the binding for the manifest itself.
    return manifest, config


def compare_csv(actual, expected, *, atol=1e-12, rtol=1e-10):
    """Compare identities exactly and numeric tables to a strict replay tolerance.

    Tolerances are in each column's own units and guard floating-point replay,
    not physical model accuracy. Integer and categorical identity/count fields
    are compared textually, so IDs, row order and attempt denominators cannot
    drift while passing a numerical comparison.
    """
    with Path(actual).open(newline="") as a, Path(expected).open(newline="") as b:
        left, right = csv.reader(a), csv.reader(b)
        from itertools import zip_longest
        count = 0
        for row_index, pair in enumerate(zip_longest(left, right)):
            ar, er = pair
            if ar is None or er is None or len(ar) != len(er):
                raise ValueError(f"Replay table shape changed: {Path(actual).name}, row {row_index}")
            for column, (x, y) in enumerate(zip(ar, er)):
                if x == y:
                    continue
                # An integer mismatch is always material. Floating notation
                # permits only roundoff-level differences, never missing values.
                if row_index == 0 or not (any(c in x for c in ".eE") and any(c in y for c in ".eE")):
                    raise ValueError(f"Replay table identity/count differs: {Path(actual).name}, row {row_index}, column {column}")
                try:
                    xx, yy = float(x), float(y)
                except ValueError as error:
                    raise ValueError(f"Replay table text differs: {Path(actual).name}, row {row_index}, column {column}") from error
                if not math.isfinite(xx) or not math.isfinite(yy) or not math.isclose(xx, yy, abs_tol=atol, rel_tol=rtol):
                    raise ValueError(f"Replay table numeric value differs: {Path(actual).name}, row {row_index}, column {column}")
            count += 1
    return dict(rows_including_header=count, byte_identical=digest(actual) == digest(expected),
                actual_sha256=digest(actual), reference_sha256=digest(expected))


def training_command(packet, name, output, device):
    config = read_json(within(packet, "execution/portable_training_commands.json"))
    if name not in config["fits"]:
        raise ValueError(f"Unknown fit {name}; available: {', '.join(sorted(config['fits']))}")
    fit = config["fits"][name]
    substitutions = {"{packet}": str(Path(packet).resolve()), "{output}": str(Path(output).resolve()), "{device}": device}
    args = []
    for token in fit["argv"]:
        for old, new in substitutions.items(): token = token.replace(old, new)
        args.append(token)
    return [sys.executable, "-m", fit["module"], *args]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--job", choices=("all", "final_uma", "matched_uma_mace"), default="all")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--plots", action="store_true", help="Render plots; correctness comparison uses the underlying tables")
    parser.add_argument("--print-training-command", metavar="FIT_NAME")
    parser.add_argument("--training-output", type=Path, default=Path("runs/reproduced_conditioning_fits"))
    parser.add_argument("--device", default="cuda", help="Used only when printing a training command; replay is CPU-only")
    args = parser.parse_args(argv)
    if args.print_training_command:
        # Printing is intentionally light and launches no training. Verify the
        # full packet in an allocation before executing this separate command.
        print(shlex.join(training_command(args.packet, args.print_training_command, args.training_output, args.device)))
        return 0
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    import cathode_cvae
    runtime = Path(cathode_cvae.__file__).resolve()
    if not runtime.is_relative_to((args.source_root / "src").resolve()):
        raise ValueError("Installed cathode_cvae does not resolve to --source-root; install this checkout with pip -e")
    manifest, config = verify_packet(args.packet, args.source_root)
    if args.verify_only:
        print(json.dumps(dict(status="VERIFIED", files=len(manifest["files"]),
            manifest_sha256=digest(args.packet / "manifest.json"), new_model_calls=0), indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required for cached replay")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from cathode_cvae.condition_evaluation import evaluate, plot_tables
    checked = {}
    for label, job in config["jobs"].items():
        if args.job != "all" and label != args.job:
            continue
        out = args.output_dir / label
        teachers = job["teachers"]
        receipt = evaluate(within(args.packet, job["attempts"]), within(args.packet, job["candidates"]),
            {t: within(args.packet, config["scores"][t]) for t in teachers},
            {t: within(args.packet, config["schemas"][t]) for t in teachers},
            within(args.packet, config["training_keys"]), within(args.packet, config["history"]),
            within(args.packet, config["historical_thresholds"]), out,
            within(args.packet, config["parent"]), within(args.packet, config["final_protocol"]),
            within(args.packet, config["cost_receipt"]))
        if receipt["status"] != "PASS":
            raise ValueError(f"Cached replay has incomplete energy coverage: {label}")
        comparisons = {}
        for name in TABLES:
            comparisons[name] = compare_csv(out / name, within(args.packet, job["reference"]) / name)
        if args.plots: plot_tables(out)
        checked[label] = dict(status="PASS", tables=comparisons, attempt_rows=receipt["attempt_rows"],
            teachers=receipt["teachers"], source_hashes=receipt["source_hashes"])
    result = dict(status="PASS", completed_UTC=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=digest(args.packet / "manifest.json"), scope="Cached table reproduction, not independent physical validation or generation resampling",
        jobs=checked, new_MLIP_calls=0, new_training_updates=0,
        numeric_replay_tolerance=dict(absolute=1e-12, relative=1e-10,
            units="Each floating column's units; exact categorical/integer identities and row order"))
    (args.output_dir / "replay_receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status="PASS", jobs=list(checked), new_MLIP_calls=0, new_training_updates=0), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

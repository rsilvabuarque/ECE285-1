"""Run the declared core pytest suite in an already installed fresh interpreter.

No installer, scientific training campaign, foundation-model inference, or
scheduler is invoked. The suite contains a synthetic two-update accounting unit
fixture, explicitly distinguished from a scientific fit. Outputs are write-once.
"""

import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources(root):
    files = {root / "pyproject.toml", root / "scripts/install.sh"}
    for folder in ("src", "tests", "scripts/conditioning_20261107", "reports/conditioning_20261107"):
        files.update((root / folder).rglob("*.py"))
    return {str(p.relative_to(root)): sha(p) for p in sorted(files) if p.is_file()}


def main():
    root = Path(__file__).resolve().parents[2]
    base = root / "reports/conditioning_20261107/fresh_install_verification"
    receipt_path = base.with_suffix(".json")
    if receipt_path.exists():
        raise FileExistsError("preserve existing verification receipts")
    start = time.perf_counter()
    before = sources(root)
    base.with_name(base.name + "_sources_before.json").write_text(json.dumps(before, indent=2) + "\n")
    packages = sorted((dict(name=d.metadata["Name"], version=d.version) for d in metadata.distributions()), key=lambda x: x["name"].lower())
    package_path = base.with_name(base.name + "_packages.json")
    package_path.write_text(json.dumps(packages, indent=2) + "\n")
    import cathode_cvae
    imported = Path(cathode_cvae.__file__).resolve()
    import_correct = imported.is_relative_to(root / "src")
    commands = []
    log_path = base.with_suffix(".log")
    xml_path = base.with_name(base.name + "_pytest.xml")
    with log_path.open("x", buffering=1) as log:
        log.write(json.dumps(dict(interpreter=sys.executable, prefix=sys.prefix, python=platform.python_version(),
            package_path=str(imported), import_correct=import_correct,
            source_files=len(before), CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"))) + "\n")
        for argv in ([sys.executable, "-m", "pip", "check"],
                     [sys.executable, "-m", "pytest", "-q", "tests", "--junitxml=" + str(xml_path)]):
            log.write("COMMAND " + json.dumps(argv) + "\n")
            then = time.perf_counter()
            completed = subprocess.run(argv, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
            commands.append(dict(argv=argv, returncode=completed.returncode, seconds=time.perf_counter()-then))
            log.write("RETURN " + json.dumps(commands[-1]) + "\n")
    after = sources(root)
    after_path = base.with_name(base.name + "_sources_after.json")
    after_path.write_text(json.dumps(after, indent=2) + "\n")
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    counts = {}
    if xml_path.exists():
        document = ET.parse(xml_path).getroot()
        suites = [document] if document.tag == "testsuite" else list(document.findall("testsuite"))
        counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    success = (import_correct and not changed and all(r["returncode"] == 0 for r in commands)
               and counts.get("tests", 0) > 0 and counts.get("failures") == counts.get("errors") == 0)
    result = dict(status="PASS" if success else "FAIL_OR_SOURCE_CHANGED", verified_UTC=datetime.now(timezone.utc).isoformat(),
        interpreter=sys.executable, python=platform.python_version(), prefix=sys.prefix, package_import=str(imported),
        import_from_requested_worktree=import_correct, commands=commands, junit_counts=counts,
        source_files=len(before), source_stable_during_checks=not changed, changed_source_files=changed,
        package_versions={k: metadata.version(k) for k in ("cathode-cvae", "numpy", "torch", "scipy", "ase", "spglib", "matplotlib", "pytest")},
        job_id=os.environ.get("SLURM_JOB_ID"), step_id=os.environ.get("SLURM_STEP_ID"), host=socket.gethostname(),
        environment={k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONPATH")},
        evidence=dict(log=dict(path=str(log_path), sha256=sha(log_path)),
            source_before=dict(path=str(base.with_name(base.name + "_sources_before.json")), sha256=sha(base.with_name(base.name + "_sources_before.json"))),
            source_after=dict(path=str(after_path), sha256=sha(after_path)),
            package_list=dict(path=str(package_path), sha256=sha(package_path)),
            junit=dict(path=str(xml_path), sha256=sha(xml_path)) if xml_path.exists() else None),
        runner_sha256=sha(__file__), elapsed_seconds=time.perf_counter()-start,
        scope="Fresh installed CORE package, full declared legacy+new unit suite and dependency check; no actual MLIP SDK/prediction parity is established by core mock/analytic tests",
        engineering_unit_scope="Includes one synthetic6-row/two-update trainer/reload/checkpoint fixture. It is not an LCO fit, new scientific trajectory, SCF or new dataset.")
    with receipt_path.open("x") as f:
        json.dump(result, f, indent=2); f.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "interpreter", "junit_counts", "changed_source_files", "package_versions", "elapsed_seconds")}, indent=2))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())

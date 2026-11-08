"""Independent static-oracle, fitting and allocation accounting audit.

Read-only with respect to experiment evidence. The output is a new receipt, not
an execution controller. Run substantial audits in a permitted allocation. A
changing registry or unfinished allocation is explicitly PRELIMINARY; passing
software/accounting checks does not establish physical accuracy or convergence.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time

UTC = timezone.utc
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(path):
    data = Path(path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", newline="") as f:
        if fields:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def parse_gpu_count(tres):
    """Count generic GPUs once; typed TRES are a breakdown, not extra devices."""
    parts = {}
    for entry in str(tres).split(","):
        if "=" in entry:
            key, value = entry.split("=", 1)
            if key.startswith("gres/gpu"):
                if not re.fullmatch(r"\d+", value):
                    raise ValueError("GPU AllocTRES counts must be integers")
                parts[key] = int(value)
    generic = parts.get("gres/gpu")
    typed = sum(v for k, v in parts.items() if k.startswith("gres/gpu:"))
    if generic is not None and typed and generic != typed:
        raise ValueError("generic and typed GPU TRES disagree; do not guess or double-count")
    return generic if generic is not None else typed


def parse_time(value):
    value = str(value).strip()
    if value in {"", "Unknown", "None", "N/A", "0"}:
        return None
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def allocation_audit(path, *, owned_jobs=None, expected_account="m5397_g", goal_start=None,
                     snapshot_utc=None):
    """Use top-level allocation rows only; exclude batch/extern/child steps.

    The caller must export Slurm times in UTC, e.g. ``env TZ=UTC sacct -X -P``.
    Running-row elapsed time is measured as of the export, not this audit's end.
    Existing allocations are attributed only from this goal's start onward.
    Both physical nodes and four-GPU equivalents are reported for shared jobs.
    """
    path = Path(path)
    text = path.read_text()
    first = text.splitlines()[0] if text.splitlines() else ""
    delimiter = "\t" if "\t" in first else "|" if "|" in first else ","
    raw = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    raw = [{str(k).strip(): str(v or "").strip() for k, v in row.items() if k is not None} for row in raw]
    requested = set(map(str, owned_jobs)) if owned_jobs else None
    snapshot = snapshot_utc or datetime.fromtimestamp(path.stat().st_mtime, UTC)
    rows, ignored, seen, findings = [], [], set(), []
    for r in raw:
        jid = r.get("JobIDRaw", r.get("JobID", ""))
        if not re.fullmatch(r"\d+", jid):
            ignored.append(jid)
            continue
        if requested is not None and jid not in requested:
            ignored.append(jid)
            continue
        if jid in seen:
            raise ValueError("duplicate top-level allocation row would double-count cost")
        seen.add(jid)
        account = r.get("Account", "")
        if account != expected_account:
            raise ValueError(f"allocation {jid} account {account!r} differs from declared account")
        gpus = parse_gpu_count(r.get("AllocTRES", ""))
        nodes = int(r.get("AllocNodes", r.get("NNodes", "0")) or 0)
        state = r.get("State", "").split()[0].rstrip("+")
        start, end = parse_time(r.get("Start")), parse_time(r.get("End"))
        elapsed = int(r.get("ElapsedRaw", "0") or 0)
        if elapsed < 0 or nodes < 0:
            raise ValueError("negative allocation duration/node count")
        terminal = state in TERMINAL
        if start is None:
            if elapsed:
                findings.append(dict(job_id=jid, finding="elapsed time but no start timestamp"))
            cost_seconds = 0.
            interval_end = None
        else:
            interval_end = end if end is not None else datetime.fromtimestamp(start.timestamp() + elapsed, UTC)
            if end is not None and abs((end - start).total_seconds() - elapsed) > 2:
                findings.append(dict(job_id=jid, finding="Start/End and ElapsedRaw disagree by >2s"))
            if not terminal and abs((snapshot - interval_end).total_seconds()) > 120:
                findings.append(dict(job_id=jid, finding="running Start+ElapsedRaw differs from export time by >120s; verify UTC export/freshness"))
            begin = max(start, goal_start) if goal_start else start
            cost_seconds = max(0., (interval_end - begin).total_seconds())
        rows.append(dict(job_id=jid, account=account, state=state, terminal=terminal,
            start_UTC=start.isoformat() if start else "", end_UTC=end.isoformat() if end else "",
            charged_through_UTC=interval_end.isoformat() if interval_end else "",
            full_allocation_elapsed_seconds=elapsed, charged_goal_seconds=cost_seconds,
            allocated_GPUs=gpus, physical_nodes=nodes,
            GPU_device_hours=gpus * cost_seconds / 3600,
            four_GPU_equivalent_node_hours=gpus * cost_seconds / 14400,
            physical_node_hours=nodes * cost_seconds / 3600,
            raw_AllocTRES=r.get("AllocTRES", "")))
    if requested is not None and requested != seen:
        findings.append(dict(finding="owned allocation IDs missing from accounting export", missing=sorted(requested - seen)))
    events = []
    for row in rows:
        s, e = parse_time(row["start_UTC"]), parse_time(row["charged_through_UTC"])
        if s is not None and e is not None and e > s:
            s = max(s, goal_start) if goal_start else s
            if e > s:
                events.extend([(s.timestamp(), 1, row["allocated_GPUs"]), (e.timestamp(), 0, -row["allocated_GPUs"])])
    current, peak = 0, 0
    for _, _, change in sorted(events):
        current += change; peak = max(peak, current)
    summary = dict(accounting_sha256=digest(path), timezone_assumption="UTC explicitly required for export",
        accounting_snapshot_UTC=snapshot.isoformat(), allocation_count=len(rows), ignored_child_or_out_of_scope_rows=ignored,
        listed_all_allocations_terminal=bool(rows) and all(r["terminal"] for r in rows),
        GPU_device_hours=sum(r["GPU_device_hours"] for r in rows),
        four_GPU_equivalent_node_hours=sum(r["four_GPU_equivalent_node_hours"] for r in rows),
        physical_node_hours=sum(r["physical_node_hours"] for r in rows),
        maximum_concurrently_allocated_GPUs=peak, findings=findings,
        timestamp_check="Bookkeeping tolerances:2s for integer-second Start/End versus ElapsedRaw;120s for a running row versus export snapshot. Findings require review, not a changed scientific tolerance.",
        scope="Allocated wall time includes startup, idle, failed work and CPU analysis; no child-step double counting; no monetary/credit estimate")
    return summary, rows


class Evidence:
    """Track bytes actually read and detect changed files at the end."""
    def __init__(self):
        self.files = {}
        self.stats = {}
        self.findings = []

    def bind(self, path, expected=None):
        path = Path(path)
        key = str(path.resolve())
        # Repeated fit configs bind the same large immutable parent/data. Their
        # bytes are checked once now and again in the closing mutation check.
        actual = self.files.get(key)
        if actual is None:
            actual = digest(path)
            self.files[key] = actual
            stat = path.stat()
            self.stats[key] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if expected is not None and actual != expected:
            self.findings.append(dict(path=str(path), finding="SHA256 differs from its binding receipt", expected=expected, actual=actual))
        return actual

    def read_json(self, path):
        data = Path(path).read_bytes()
        key = str(Path(path).resolve())
        self.files[key] = hashlib.sha256(data).hexdigest()
        stat = Path(path).stat()
        self.stats[key] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        return json.loads(data)

    def mutation_check(self):
        changed = []
        for p, sha in self.files.items():
            path = Path(p)
            if not path.is_file():
                changed.append(p); continue
            stat = path.stat()
            expected = self.stats[p]
            if stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"] or digest(path) != sha:
                changed.append(p)
        return changed


def expected_context(model, parent_sha, models):
    spec = models[model]
    return dict(parent_sha256=parent_sha, model=model, head=spec["head"], checkpoint_hashes=spec["files"],
        package_version=spec["version"], dtype="float32", tf32=False,
        geometry_protocol="pristine-derived deletion; unrelaxed; exact canonical parent coordinates", orientation="canonical")


def expected_physical_key(model, sid, parent_sha, models, n_sites):
    match = re.fullmatch(rf"mask{n_sites}_([0-9a-f]+)", sid)
    if match is None:
        raise ValueError("noncanonical physical structure ID")
    code = int(match.group(1), 16)
    if code >= 2 ** n_sites:
        raise ValueError("structure ID overflows parent site count")
    spec = models[model]
    body = dict(parent_sha256=parent_sha, mask=format(code, f"0{n_sites}b"),
                geometry_protocol="pristine-derived deletion; unrelaxed; exact canonical parent coordinates",
                model=model, head=spec["head"], checkpoint_hashes=spec["files"],
                package_version=spec["version"], dtype="float32", tf32=False)
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def audit_worker(value, model, models):
    spec = models[model]
    required = dict(model=model, head=spec["head"], package_version=spec["version"],
                    dtype="float32", tf32=False, nitro=False, compile=False, dispersion=False,
                    checkpoint_hashes={Path(k).name: v for k, v in spec["files"].items()})
    return {k: dict(expected=v, actual=value.get(k)) for k, v in required.items() if value.get(k) != v}


def oracle_audit(root, run, evidence, models):
    import numpy as np
    oracle = run / "oracle"
    registry_path = oracle / "registry.json"
    registry = evidence.read_json(registry_path)
    registry_sha = evidence.files[str(registry_path.resolve())]
    tasks = registry["tasks"]
    pairs = registry["physical_keys"]
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("oracle registry repeats task IDs")
    parent_paths = {t["parent_npz"] for t in tasks}
    if not parent_paths:
        parent_paths = {str(root / "artifacts/current/core/data/parents/O3_N48_pilot.npz")}
    if len(parent_paths) != 1:
        raise ValueError("single oracle queue contains multiple parent files")
    parent_path = Path(next(iter(parent_paths)))
    evidence.bind(parent_path, registry["parent_sha256"])
    with np.load(parent_path, allow_pickle=False) as z:
        mobile = z["target_indices"] if "target_indices" in z else z["li_indices"]
        full_atoms = len(z["numbers"]); permutations = z["permutations"]
    n_sites = len(mobile)
    if n_sites > 62:
        raise ValueError("this existing oracle registry uses integer IDs for at most62 sites")
    inverse = np.argsort(permutations, axis=1)
    powers = np.left_shift(np.uint64(1), np.arange(n_sites - 1, -1, -1, dtype=np.uint64))
    inputs = {}
    for t in tasks:
        path = Path(t["input_npz"])
        if str(path) in inputs:
            continue
        evidence.bind(path, t["input_sha256"])
        with np.load(path, allow_pickle=False) as z:
            ids = z["structure_ids"].astype(str)
            masks = z["masks"] if "masks" in z else 1-z["occupancy"]
            if "parent_sha256" in z and str(z["parent_sha256"].item()) != registry["parent_sha256"]:
                raise ValueError("input candidate parent identity differs from registry")
        if masks.shape != (len(ids), n_sites) or not np.isin(masks, [0, 1]).all() or len(set(ids)) != len(ids):
            raise ValueError("candidate input is not unique binary full-site masks")
        codes = (masks.astype(np.uint64) * powers).sum(axis=-1)
        for i, (sid, code) in enumerate(zip(ids, codes)):
            expected = f"mask{n_sites}_{int(code):0{(n_sites+3)//4}x}"
            if sid != expected:
                raise ValueError("candidate mask and physical scoring ID disagree")
        for start in range(0, len(masks), 64):
            block = masks[start:start+64].astype(np.uint64)
            canonical = (block[:, inverse] * powers).sum(axis=-1).min(axis=1)
            if not np.array_equal(canonical, codes[start:start+64]):
                raise ValueError("input geometry is not in the canonical physical frame")
        inputs[str(path)] = dict(ids=ids, codes=codes)
    worker_rows, workers = [], {}
    for path in sorted((oracle / "workers").glob("*.json")):
        w = evidence.read_json(path); model = w.get("model")
        if model not in models:
            evidence.findings.append(dict(path=str(path), finding="worker has unknown teacher")); continue
        mismatches = audit_worker(w, model, models)
        if mismatches:
            evidence.findings.append(dict(path=str(path), finding="worker inference context mismatch", mismatches=mismatches))
        identity = w.get("worker", path.stem)
        workers[identity] = w
        worker_rows.append(dict(worker=identity, model=model, package_version=w.get("package_version"),
            adapter_source_sha256=w.get("adapter_source_sha256"), worker_source_sha256=w.get("worker_source_sha256"),
            head=w.get("head"), dtype=w.get("dtype"), tf32=w.get("tf32"),
            job_id=w.get("job_id"), host=w.get("host"), context_matches=not mismatches,
            receipt_sha256=evidence.files[str(path.resolve())]))
    registered = Counter()
    fresh = set(); cached = set()
    for key, item in pairs.items():
        model, sid = item["model"], item["structure_id"]
        if expected_physical_key(model, sid, registry["parent_sha256"], models, n_sites) != key:
            raise ValueError("registered physical key does not bind the declared parent/teacher/mask/protocol")
        provenance = item["provenance"]
        if provenance not in {"new_scoring", "exact_geometry_cache"}:
            raise ValueError("unknown registry cost provenance")
        registered[model, provenance] += 1
        (fresh if provenance == "new_scoring" else cached).add((model, sid))
    cache_rows = []
    for model in sorted(m for m in models if (oracle / f"cache_{m}.npz").exists()):
        path, certificate = oracle / f"cache_{model}.npz", oracle / f"cache_{model}.json"
        c = evidence.read_json(certificate)
        if c.get("context") != expected_context(model, registry["parent_sha256"], models):
            raise ValueError("cache certificate physical context differs")
        evidence.bind(path, c["npz_sha256"])
        for source in c.get("sources", []):
            evidence.bind(source["path"], source["sha256"])
        for source in c.get("verified_worker_receipts", []):
            evidence.bind(source["path"], source["sha256"])
            ancestor = evidence.read_json(source["path"])
            if audit_worker(ancestor, model, models):
                raise ValueError("upstream cached teacher worker context differs")
        with np.load(path, allow_pickle=False) as z:
            ids, energies = z["structure_ids"].astype(str), z["energy_eV"]
            atoms, counts = z["n_atoms"], z["n_Li"]
        if len(set(ids)) != len(ids) or not np.isfinite(energies).all():
            raise ValueError("cache entries repeat IDs or contain unknown energies")
        available = {}
        for sid, atom_count, count in zip(ids, atoms, counts):
            code = int(sid.split("_")[-1], 16)
            expected_count = n_sites - code.bit_count()
            if int(count) != expected_count or int(atom_count) != full_atoms - n_sites + expected_count:
                raise ValueError("cached composition/count is inconsistent with its physical mask")
            available[sid] = True
        wanted = {sid for m, sid in cached if m == model}
        missing = wanted - available.keys()
        if missing:
            raise ValueError("registry claims cache entries absent from certified cache")
        cache_rows.append(dict(model=model, available_entries=len(ids), registered_reused_entries=len(wanted),
            cache_sha256=evidence.files[str(path.resolve())], certificate_sha256=evidence.files[str(certificate.resolve())]))
    task_rows, actual_calls, successes, failures, claimed_pairs = [], Counter(), Counter(), Counter(), set()
    task_pair_membership = set()
    for t in tasks:
        model, tid = t["model"], t["task_id"]
        d = inputs[t["input_npz"]]
        ids = d["ids"][t["indices"]].tolist()
        if ids != t["expected_structure_ids"] or len(set(t["indices"])) != len(ids):
            raise ValueError("task expected IDs/indices differ from its bound input")
        for sid in ids:
            if (model, sid) in task_pair_membership:
                raise ValueError("one fresh geometry-teacher pair appears in multiple oracle tasks")
            task_pair_membership.add((model, sid))
        data_path = oracle / "scores" / model / (tid + ".npz")
        json_path = data_path.with_suffix(".json")
        claim = oracle / "claims" / tid
        owner = evidence.read_json(claim / "owner.json") if (claim / "owner.json").exists() else {}
        status, success, failure, call_count, worker = "NOT_TERMINALLY_RECORDED", 0, 0, 0, owner.get("worker", "")
        if json_path.exists() and data_path.exists():
            saved = evidence.read_json(json_path)
            if saved["task"] != t:
                raise ValueError("saved raw oracle task differs from registry")
            evidence.bind(data_path, saved["output_sha256"])
            worker = saved["worker"]
            if worker not in workers or workers[worker].get("model") != model:
                raise ValueError("raw score lacks a matching teacher worker receipt")
            rr = saved["rows"]
            if [r["structure_id"] for r in rr] != ids:
                raise ValueError("raw per-geometry rows are not task-exact")
            with np.load(data_path, allow_pickle=False) as z:
                # Decode each compressed array once, not once per geometry.
                arrays = {k: z[k] for k in ("structure_ids", "status", "energy_eV", "n_Li", "n_atoms", "forces_parent_eV_A", "stress_eV_A3")}
                if arrays["structure_ids"].astype(str).tolist() != ids or arrays["status"].astype(str).tolist() != [r["status"] for r in rr]:
                    raise ValueError("raw score array coverage/status differs from its bound JSON")
                for i, row in enumerate(rr):
                    pair = (model, row["structure_id"])
                    if pair in claimed_pairs:
                        raise ValueError("fresh physical geometry-teacher evaluation repeated")
                    claimed_pairs.add(pair); call_count += 1; actual_calls[model] += 1
                    if row["status"] == "success":
                        success += 1; successes[model] += 1
                        expected_n = n_sites - int(row["structure_id"].split("_")[-1], 16).bit_count()
                        if int(arrays["n_Li"][i]) != expected_n or int(arrays["n_atoms"][i]) != full_atoms - n_sites + expected_n:
                            raise ValueError("successful raw score has incorrect physical atom/composition count")
                        if not np.isfinite(arrays["energy_eV"][i]) or not np.isfinite(arrays["forces_parent_eV_A"][i]).all() or not np.isfinite(arrays["stress_eV_A3"][i]).all():
                            raise ValueError("successful raw E/F/stress contains nonfinite values")
                        if float(arrays["energy_eV"][i]) != float(row["energy_eV"]):
                            raise ValueError("raw JSON and array energy disagree")
                    else:
                        failure += 1; failures[model] += 1
            status = "COMPLETE" if failure == 0 else "COMPLETE_WITH_FAILED_EVALUATIONS"
        elif (claim / "failure.json").exists():
            saved = evidence.read_json(claim / "failure.json")
            status = "TASK_FAILED_WITHOUT_COMPLETE_ROWS"
            evidence.findings.append(dict(task_id=tid, finding=status, error=saved.get("error")))
        elif owner:
            status = "CLAIMED_NOT_TERMINALLY_RECORDED"
        task_rows.append(dict(task_id=tid, model=model, status=status, registered_pairs=len(ids),
            recorded_logical_static_attempts=call_count, successful_EFS=success, failed_logical_attempts=failure,
            worker=worker, input_sha256=t["input_sha256"], output_sha256=evidence.files.get(str(data_path.resolve()), "")))
    if task_pair_membership != fresh:
        raise ValueError("fresh registry pairs and task memberships disagree")
    queue_path = oracle / "tasks.jsonl"
    if queue_path.exists():
        evidence.bind(queue_path)
        queued = [json.loads(line) for line in queue_path.read_text().splitlines() if line]
        if queued != tasks:
            evidence.findings.append(dict(path=str(queue_path), finding="published task queue and registry snapshot differ"))
    totals = {model: dict(registered_fresh_pairs=registered[model, "new_scoring"],
        registered_cache_reuses=registered[model, "exact_geometry_cache"],
        recorded_new_logical_static_attempts=actual_calls[model], successful_new_EFS=successes[model],
        failed_new_logical_attempts=failures[model],
        fresh_pairs_without_terminal_rows=registered[model, "new_scoring"] - actual_calls[model])
        for model in sorted({m for m, _ in registered})}
    summary = dict(registry_sha256_at_start=registry_sha, registered_tasks=len(tasks),
        complete_tasks=sum(r["status"] == "COMPLETE" for r in task_rows),
        noncomplete_tasks=sum(r["status"] != "COMPLETE" for r in task_rows),
        by_teacher=totals, cache_certificates=cache_rows,
        physical_context="Exact canonical parent geometry; pinned model/head/package/checkpoint, FP32 and TF32 off",
        counting_scope="Logical single-point attempts, each requesting E/F/stress; not instrumented neural-backbone forward counts; cache reuses are NOT fresh calls")
    return summary, task_rows, worker_rows


def flag_value(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def training_scope_contract(root, run, evidence, protocol):
    """Resolve the preserved local plan and its explicit CORE-cap amendment."""
    old = protocol.get("nonresource_science_bounds", {})
    inclusive = old.get("maximum_full_training_trajectories", 30)
    contract = dict(mode="original_inclusive_local_plan", maximum_core_full_trajectories=inclusive,
        maximum_optional_editor_full_trajectories=0, maximum_all_full_trajectories=inclusive,
        originally_planned_all_full_trajectories=old.get("planned_full_training_trajectories"),
        amendment=None)
    amendment_path = root / "artifacts/conditioning_20261107/protocol_amendment_graph_context_replication.json"
    if not amendment_path.exists():
        return contract
    amendment = evidence.read_json(amendment_path)
    original_path = root / "artifacts/conditioning_20261107/protocol.json"
    original_sha = evidence.bind(original_path)
    if amendment.get("original_protocol_sha256") != original_sha:
        raise ValueError("scope amendment does not bind the preserved original protocol")
    if amendment.get("core_full_fit_cap") != 30:
        raise ValueError("amendment may not change the user's literal CORE30 training cap")
    core, optional, total = (amendment.get(k) for k in ("planned_core_full_fits", "planned_optional_editor_full_fits", "planned_all_full_fits"))
    if core != 28 or optional != 3 or total != core + optional:
        raise ValueError("unrecognized phase plan: this amendment declares28core plus3optional-editor fits")
    if amendment.get("status") != "FROZEN_BEFORE_ADDITIONAL_GRAPH_CONTEXT_FITS" or parse_time(amendment.get("created_UTC", "")) is None:
        raise ValueError("scope amendment lacks its pre-experiment freeze/timestamp")
    instructions = run / "instructions.txt"
    expected = protocol.get("sources", {}).get("ECE285_next_goal_prompt_20261107.txt")
    if expected is not None:
        if evidence.bind(instructions, expected) != expected:
            raise ValueError("user instruction copy does not match the original protocol")
        text = instructions.read_text()
        if ("Cap the core experiment program at 30 new full training trajectories." not in text
                or "9. OPTIONAL EXTENSION PRIORITY 1: SEED-CONDITIONED EDIT CVAE" not in text):
            raise ValueError("instruction copy does not substantiate core/optional cap scope")
    return dict(mode="verified_core_scope_amendment", maximum_core_full_trajectories=30,
        maximum_optional_editor_full_trajectories=optional, maximum_all_full_trajectories=total,
        originally_planned_all_full_trajectories=old.get("planned_full_training_trajectories"),
        amended_planned_core_full_trajectories=core, amended_planned_all_full_trajectories=total,
        amendment=dict(path=str(amendment_path), sha256=evidence.files[str(amendment_path.resolve())],
                       created_UTC=amendment["created_UTC"], original_protocol_sha256=original_sha,
                       authority=amendment.get("authority")))


def verified_trainer_source(root, run, evidence, cfg, module_name):
    """Bind the declared trainer SHA to current or preserved executing source."""
    expected = cfg.get("trainer_sha256")
    if not expected:
        return None
    relative = Path(*module_name.split(".")).with_suffix(".py")
    candidates = [root / "src" / relative]
    snapshots = run / "executed_sources"
    if snapshots.exists():
        for directory in sorted(snapshots.iterdir()):
            if directory.is_dir():
                candidates.extend([directory / relative, directory / "src" / relative])
    for path in candidates:
        if path.is_file() and evidence.bind(path) == expected:
            return str(path.resolve())
    return None


def verified_editor_gate(root, evidence, cfg, owner):
    binding = cfg.get("required_report_gate")
    if not isinstance(binding, dict) or not binding.get("path") or not binding.get("sha256"):
        return None, "editor configuration lacks an explicit required-packet gate binding"
    path = Path(binding["path"])
    if not path.is_absolute():
        path = root / path
    if not path.is_file() or evidence.bind(path, binding["sha256"]) != binding["sha256"]:
        return None, "required-packet gate file is absent or its SHA differs"
    gate = evidence.read_json(path)
    if gate.get("status") not in {"PASS", "COMPLETE", "REQUIRED_CORE_COMPLETE", "READY_FOR_OPTIONAL_EXTENSIONS"}:
        return None, "required-packet gate is not complete"
    completed = parse_time(gate.get("completed_UTC", gate.get("created_UTC", "")))
    started = owner.get("started_unix", cfg.get("started_unix"))
    if started is None:
        value = cfg.get("training_started_UTC", cfg.get("started_UTC", ""))
        parsed = parse_time(value)
        started = parsed.timestamp() if parsed else None
    if completed is None or started is None or float(started) < completed.timestamp():
        return None, "editor training is not proven to start after the completed packet gate"
    report = gate.get("report_pdf")
    if isinstance(report, dict) and report.get("path") and report.get("sha256"):
        pdf = Path(report["path"])
        if not pdf.is_absolute():
            pdf = root / pdf
        if not pdf.is_file() or evidence.bind(pdf, report["sha256"]) != report["sha256"]:
            return None, "compiled gate-report snapshot is missing or changed"
    return dict(path=str(path), sha256=binding["sha256"], completed_UTC=completed.isoformat(),
                training_started_unix=float(started)), None


def classify_fit_phase(root, run, evidence, cfg, owner, task, stage, argv, summary, contract):
    """An editor directory name cannot exempt a mask fit from the CORE cap."""
    requested = cfg.get("maximum_updates", flag_value(argv, "--max-updates"))
    actual = summary.get("optimizer_updates")
    explicit_engineering = "ENGINEERING" in stage.upper() or task.get("experiment_phase") == "engineering"
    if explicit_engineering and requested is not None and int(requested) <= 8 and (actual is None or int(actual) <= 8):
        return dict(phase="engineering", verified=True, trainer_source=None, gate=None)
    if explicit_engineering and actual is not None and int(actual) > 8:
        evidence.findings.append(dict(finding="declared engineering task exceeds its bounded8-update scope; counted as core"))
    kind = cfg.get("kind")
    mask_kinds = {"dense", "dense_bce", "dense_count", "ar", "ar_cvae", "ar_baseline"}
    if kind in mask_kinds:
        if cfg.get("experiment_phase") == "optional_editor":
            evidence.findings.append(dict(finding="mask-training model incorrectly labelled optional editor; counted as core", kind=kind))
        source = verified_trainer_source(root, run, evidence, cfg, "cathode_cvae.learning.train_mask")
        return dict(phase="core", verified=source is not None, trainer_source=source, gate=None)
    optional_claim = cfg.get("experiment_phase") == "optional_editor"
    if optional_claim:
        valid_metadata = (cfg.get("trainer_module") == "cathode_cvae.learning.train_edit"
            and kind in {"seed_edit_cvae", "edit_cvae"})
        source = (verified_trainer_source(root, run, evidence, cfg, "cathode_cvae.learning.train_edit")
                  if valid_metadata else None)
        gate, problem = verified_editor_gate(root, evidence, cfg, owner) if valid_metadata else (None, "optional editor trainer/module/kind metadata are inconsistent")
        if contract["mode"] == "verified_core_scope_amendment" and source is not None and gate is not None:
            return dict(phase="optional_editor", verified=True, trainer_source=source, gate=gate)
        evidence.findings.append(dict(finding="optional editor phase not verified; conservatively counted as core",
            reason=problem or ("declared trainer source hash is unavailable" if source is None else "no verified scope amendment")))
    return dict(phase="core", verified=False, trainer_source=None, gate=None)


def check_phase_caps(rows, contract, evidence):
    scientific = [r for r in rows if r["category"] == "full_scientific_trajectory"]
    counts = {phase: sum(r["started"] and r["experiment_phase"] == phase for r in scientific)
              for phase in ("core", "optional_editor")}
    total = sum(r["started"] for r in scientific)
    for label, count, cap in (("core", counts["core"], contract["maximum_core_full_trajectories"]),
                             ("optional_editor", counts["optional_editor"], contract["maximum_optional_editor_full_trajectories"]),
                             ("all", total, contract["maximum_all_full_trajectories"])):
        if count > cap:
            evidence.findings.append(dict(finding=f"{label} full training trajectory bound exceeded", count=count, cap=cap))
    return dict(started_core_full_trajectories=counts["core"],
                started_optional_editor_full_trajectories=counts["optional_editor"], started_all_full_trajectories=total)


def bind_summary_checkpoint(root, directory, summary, phase, evidence):
    """Bind the trainer's selected file, preserving strict checks on its alias.

    The editor saved one checkpoint twice under different ZIP root names. Its
    summary hashes the named selection, whereas the historical mask trainer's
    summary hashes best.pt. No tensor tolerance or checkpoint rewrite is used.
    """
    expected = summary.get("checkpoint_sha256")
    if not expected:
        return
    if phase.get("phase") != "optional_editor" or not phase.get("verified"):
        evidence.bind(directory / "best.pt", expected)
        return
    selected = summary.get("selection_checkpoints", {}).get("0.1", {})
    if selected.get("filename") != "best_validation_beta_0p1.pt" or selected.get("sha256") != expected:
        evidence.findings.append(dict(path=str(directory), finding="editor summary does not bind its declared named selection"))
        return
    named = directory / selected["filename"]
    alias = directory / "best.pt"
    evidence.bind(named, expected)
    proof_path = root / "reports/conditioning_20261107/editor_checkpoint_alias_audit/receipt.json"
    if not proof_path.is_file():
        evidence.findings.append(dict(path=str(alias), finding="editor alias requires the independent exact payload audit"))
        return
    proof = evidence.read_json(proof_path)
    candidates = [r for r in proof.get("fits", []) if r.get("alias_checkpoint") == str(alias)]
    if proof.get("status") != "PASS_EXACT_EDITOR_CHECKPOINT_ALIAS_AUDIT" or len(candidates) != 1:
        evidence.findings.append(dict(path=str(alias), finding="editor alias proof is missing or ambiguous"))
        return
    row = candidates[0]
    valid = (row.get("named_checkpoint") == str(named) and row.get("named_sha256") == expected
        and all(row.get(k) is True for k in ("entire_deserialized_payload_bitwise_identical",
            "normalized_zip_member_payloads_identical", "summary_matches_named_selection",
            "generation_matches_named_selection", "protected_bytes_unchanged"))
        and row.get("normalized_member_inventory_named") == row.get("normalized_member_inventory_alias")
        and bool(row.get("normalized_member_inventory_named"))
        and row.get("unchanged_from_failed_preclosure", {}).get("actual") == row.get("alias_sha256")
        and row.get("unchanged_from_failed_preclosure", {}).get("expected") == expected)
    if not valid:
        evidence.findings.append(dict(path=str(alias), finding="editor alias exact payload proof is inconsistent"))
        return
    evidence.bind(alias, row["alias_sha256"])
    evidence.bind(root / "scripts/conditioning_20261107/check_editor_checkpoint_aliases.py", proof["source_sha256"])
    previous = proof["preserved_preclosure"]
    evidence.bind(previous["path"], previous["sha256"])


def fit_audit(root, run, evidence, protocol):
    contract = training_scope_contract(root, run, evidence, protocol)
    tasks = []
    by_output = {}
    definition_files = sorted(run.glob("*tasks.json"))
    for path in definition_files:
        value = evidence.read_json(path)
        for task in value.get("tasks", []):
            argv = task.get("argv")
            if not isinstance(argv, list) or "--output-dir" not in argv or "--max-updates" not in argv:
                continue
            directory = Path(flag_value(argv, "--output-dir"))
            if not directory.is_absolute():
                directory = root / directory
            claim = path.with_suffix(".claims") / task["id"]
            item = dict(path=path, task=task, argv=argv, directory=directory, claim=claim,
                        stage=str(value.get("stage", "")), definitions=[str(path)],
                        execution_claim_dirs=[str(claim)] if (claim / "owner.json").exists() or (claim / "receipt.json").exists() else [])
            key = str(directory.resolve())
            if key in by_output:
                prior = by_output[key]
                if prior["argv"] != argv:
                    raise ValueError("duplicate fit manifests specify DIFFERENT commands for one output; review actual execution ownership")
                definitions = prior["definitions"] + [str(path)]
                actual = prior["execution_claim_dirs"] + item["execution_claim_dirs"]
                # A combined worker list is often built from individual stage
                # manifests. Pick its real claim, not an unlaunched plan copy.
                if item["execution_claim_dirs"] and not prior["execution_claim_dirs"]:
                    by_output[key] = item
                by_output[key]["definitions"] = definitions
                by_output[key]["execution_claim_dirs"] = actual
            else:
                by_output[key] = item
    tasks.extend(by_output.values())
    # Include directly launched fits, without pretending they have worker receipts.
    for config in sorted(run.glob("fits/*/configuration.json")):
        if str(config.parent.resolve()) not in by_output:
            tasks.append(dict(path=None, task={}, argv=[], directory=config.parent, claim=None, stage="DIRECT_CONFIGURED_FIT",
                              definitions=[], execution_claim_dirs=[]))
    rows = []
    for t in tasks:
        directory, claim, argv = t["directory"], t["claim"], t["argv"]
        owner = evidence.read_json(claim / "owner.json") if claim and (claim / "owner.json").exists() else {}
        terminal = evidence.read_json(claim / "receipt.json") if claim and (claim / "receipt.json").exists() else {}
        if contract["amendment"] and owner.get("started_unix") is not None and any(
                "graph_context_replication_fit_tasks" in p for p in t["definitions"]):
            freeze_time = parse_time(contract["amendment"]["created_UTC"]).timestamp()
            if float(owner["started_unix"]) < freeze_time:
                evidence.findings.append(dict(path=str(directory), finding="additional graph-context replication started before its protocol amendment freeze"))
        def maybe_in_progress(path):
            if not path.exists():
                return {}
            try:
                return evidence.read_json(path)
            except json.JSONDecodeError:
                evidence.findings.append(dict(path=str(path), finding="JSON incomplete or invalid during snapshot; not accepted as a terminal fit"))
                return {}
        cfg = maybe_in_progress(directory / "configuration.json")
        summary = maybe_in_progress(directory / "summary.json")
        phase = classify_fit_phase(root, run, evidence, cfg, owner, t["task"], t["stage"], argv, summary, contract)
        engineering = phase["phase"] == "engineering"
        started = bool(owner or cfg or terminal or summary)
        status = "PLANNED"
        if started:
            status = "STARTED_NOT_TERMINALLY_RECORDED"
        if terminal.get("status") == "FAILED":
            status = "FAILED"
        elif summary.get("status") == "FIT_COMPLETE":
            status = "COMPLETE" if terminal.get("status") == "COMPLETE" else "COMPLETE_BY_SUMMARY_ONLY"
        if terminal.get("status") == "COMPLETE" and not summary:
            evidence.findings.append(dict(path=str(directory), finding="worker says complete but fit summary is missing"))
        if owner and t["path"] is not None and owner.get("tasks_sha256") != evidence.files[str(t["path"].resolve())]:
            evidence.findings.append(dict(path=str(directory), finding="fit task file changed after worker claimed it"))
        if len(t["execution_claim_dirs"]) > 1:
            evidence.findings.append(dict(path=str(directory), finding="multiple actual execution claims for the same output; inspect preserved launch failures",
                                          claim_directories=t["execution_claim_dirs"]))
        if cfg:
            for field, path_field in [("data_sha256", "data"), ("parent_sha256", "parent")]:
                if cfg.get(path_field) and cfg.get(field):
                    evidence.bind(cfg[path_field], cfg[field])
        if summary:
            for key, item in summary.get("selection_checkpoints", {}).items():
                evidence.bind(directory / item["filename"], item["sha256"])
            bind_summary_checkpoint(root, directory, summary, phase, evidence)
        mask_data = cfg.get("data_sha256") == protocol.get("data", {}).get("round0_sha256")
        eligible = (not engineering and mask_data and summary.get("optimizer_updates") == 4000
            and summary.get("stop_reason") == "optimizer_update_cap"
            and summary.get("train_rows") == 40157 and summary.get("val_rows") == 4967 and summary.get("test_rows") == 5123
            and summary.get("heldout_test_evaluated") is False and status in {"COMPLETE", "COMPLETE_BY_SUMMARY_ONLY"})
        rows.append(dict(fit=directory.name, stage=t["stage"], category="engineering" if engineering else "full_scientific_trajectory",
            experiment_phase=phase["phase"], phase_verified=phase["verified"], verified_trainer_source=phase["trainer_source"] or "",
            required_report_gate_json=json.dumps(phase["gate"]) if phase["gate"] else "",
            started=started, status=status, actual_optimizer_updates=summary.get("optimizer_updates", ""),
            requested_maximum_updates=cfg.get("maximum_updates", flag_value(argv, "--max-updates", "")),
            eligible_for_matched_round0_4000_update_comparison=eligible,
            train_rows=cfg.get("train_rows", ""), val_rows=cfg.get("val_rows", ""), test_rows=cfg.get("test_rows", ""),
            fit_seed=cfg.get("seed", ""), latent_dim=cfg.get("latent_dim", ""), kind=cfg.get("kind", ""),
            encoder_kind=cfg.get("encoder_kind", ""), graph_context=cfg.get("graph_context", ""),
            orientation=cfg.get("orientation", ""), class_balance=cfg.get("class_balance", ""),
            parameter_count=cfg.get("parameter_count", ""), train_seconds=summary.get("training_seconds", ""),
            configuration_sha256=evidence.files.get(str((directory / "configuration.json").resolve()), ""),
            trainer_source_sha256=cfg.get("trainer_sha256", ""), model_source_sha256=cfg.get("model_source_sha256", ""),
            task_file_sha256=evidence.files.get(str(t["path"].resolve()), "") if t["path"] else "",
            worker_job=owner.get("job", ""), command_json=json.dumps(argv),
            identical_planned_manifest_count=len(t["definitions"]),
            planned_definition_paths="|".join(t["definitions"]),
            actual_execution_claim_directories="|".join(t["execution_claim_dirs"]),
            direct_fit_has_no_task_receipt=bool(t["path"] is None)))
    full = [r for r in rows if r["category"] == "full_scientific_trajectory"]
    engineering = [r for r in rows if r["category"] == "engineering"]
    count = sum(r["started"] for r in full)
    phase_counts = check_phase_caps(rows, contract, evidence)
    return dict(planned_full_trajectories=len(full), started_full_trajectories=count,
                task_definition_files_at_start=[str(p.resolve()) for p in definition_files],
                protocol_planned_full_trajectories=protocol.get("nonresource_science_bounds", {}).get("planned_full_training_trajectories"),
                defined_full_trajectories_not_started=sum(not r["started"] for r in full),
                completed_full_trajectories=sum(r["status"].startswith("COMPLETE") for r in full),
                full_trajectories_without_terminal_summary=sum(r["started"] and not r["status"].startswith("COMPLETE") for r in full),
                engineering_trajectories_started=sum(r["started"] for r in engineering),
                engineering_trajectories_complete=sum(r["status"].startswith("COMPLETE") for r in engineering),
                eligible_matched_round0_4000_update_trajectories=sum(r["eligible_for_matched_round0_4000_update_comparison"] for r in rows),
                maximum_full_trajectories=contract["maximum_all_full_trajectories"],
                training_scope_contract=contract, **phase_counts,
                fitted_model_usefulness="Not inferred from successful training or this execution audit"), rows


def pilot_audit(run, evidence, models):
    rows = []
    for path in sorted((run / "pilot").glob("*/receipt.json")):
        value = evidence.read_json(path)
        worker = value["model_receipt"]; model = worker["model"]
        if audit_worker(worker, model, models):
            raise ValueError("engineering inference pilot context mismatch")
        rows.append(dict(model=model, receipt_sha256=evidence.files[str(path.resolve())],
            logical_static_attempts=len(value["rows"]), successful_EFS=sum(r.get("finite") is True for r in value["rows"]),
            wall_seconds=value.get("wall_seconds"), scope=value.get("scope")))
    return dict(rows=rows, logical_static_attempts=sum(r["logical_static_attempts"] for r in rows),
                scope="Separate recorded engineering EFS calls; not new benchmark candidates or neural-backbone forward counts")


def archive_auditor(output, source):
    path = Path(output).with_name(Path(output).stem + "_auditor.py")
    with path.open("xb") as f:
        f.write(source)
    return dict(path=str(path), sha256=hashlib.sha256(source).hexdigest())


def resource_findings(accounting, allocations, protocol):
    result = [dict(category="accounting", **item) for item in accounting["findings"]]
    limits = protocol["user_resource_override"]
    if accounting["maximum_concurrently_allocated_GPUs"] > limits["maximum_concurrent_GPUs"]:
        result.append(dict(finding="GPU concurrency authorization exceeded"))
    if max(accounting["physical_node_hours"], accounting["four_GPU_equivalent_node_hours"]) > limits["maximum_new_GPU_node_hours"]:
        result.append(dict(finding="new GPU-node-hour ceiling exceeded under at least one reported node convention"))
    hard_stop = parse_time(protocol.get("hard_stop_UTC", ""))
    if hard_stop:
        for row in allocations:
            end = parse_time(row["charged_through_UTC"])
            if end and end > hard_stop and row["charged_goal_seconds"] > 0:
                result.append(dict(job_id=row["job_id"], finding="GPU allocation extended beyond this goal's authorized elapsed window"))
    return result


def finalize_from(a, root, run, source):
    """Lightweight stdlib closure after the heavy audit and GPU release.

    Raw payload SHA256 checks belong to the preceding allocation audit. This
    closure hashes the heavy receipt and small control metadata, checks unchanged
    raw-file size/mtime, and computes final Slurm allocation accounting. It does
    not load arrays/models, evaluate an MLIP, or repeat the numerical analysis.
    """
    if a.accounting is None:
        raise ValueError("--finalize-from requires a final UTC accounting export")
    begin = time.perf_counter()
    prior, prior_sha = json_bytes(a.finalize_from)
    findings, changed, content_checked, stat_checked = [], [], [], []
    if not prior.get("evidence_stable_during_audit") or prior.get("findings"):
        findings.append(dict(finding="heavy audit was unstable or had unresolved findings"))
    expected_stats = prior.get("evidence_file_stats")
    if not isinstance(expected_stats, dict):
        raise ValueError("closing requires a new heavy audit containing payload stat signatures; the initial preliminary receipt remains historical")
    metadata_suffixes = {".json", ".jsonl", ".csv", ".tsv", ".py", ".txt", ".toml", ".yaml", ".yml"}
    for name, expected in prior["evidence_file_hashes"].items():
        path = Path(name)
        if not path.is_file():
            changed.append(name); continue
        if path.suffix.lower() in metadata_suffixes:
            if digest(path) != expected:
                changed.append(name)
            content_checked.append(name)
        else:
            stat = path.stat(); original = expected_stats.get(name)
            if original is None or stat.st_size != original["bytes"] or stat.st_mtime_ns != original["mtime_ns"]:
                changed.append(name)
            stat_checked.append(name)
    for label, row in prior.get("sidecars", {}).items():
        if not Path(row["path"]).is_file() or digest(row["path"]) != row["sha256"]:
            changed.append(row["path"])
    expected_tasks = prior["training"].get("task_definition_files_at_start", [])
    if [str(p.resolve()) for p in sorted(run.glob("*tasks.json"))] != expected_tasks:
        changed.append("FIT_TASK_DEFINITION_SET_CHANGED_AFTER_HEAVY_AUDIT")
    protocol_path = root / "artifacts/conditioning_20261107/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if prior["evidence_file_hashes"].get(str(protocol_path.resolve())) != digest(protocol_path):
        changed.append(str(protocol_path))
    accounting, rows = allocation_audit(a.accounting, owned_jobs=a.owned_jobs, expected_account=a.account,
        goal_start=parse_time(protocol["start_UTC"]),
        snapshot_utc=parse_time(a.accounting_snapshot_UTC) if a.accounting_snapshot_UTC else None)
    findings.extend(resource_findings(accounting, rows, protocol))
    previous_accounting = prior.get("allocations")
    if previous_accounting is None:
        findings.append(dict(finding="heavy audit lacks an independently checked UTC accounting snapshot"))
    else:
        allocation_sidecar = prior.get("sidecars", {}).get("allocations", {}).get("path")
        old = list(csv.DictReader(Path(allocation_sidecar).open())) if allocation_sidecar else []
        old_starts = {r["job_id"]: r["start_UTC"] for r in old}
        for row in rows:
            if row["job_id"] in old_starts and row["start_UTC"] != old_starts[row["job_id"]]:
                findings.append(dict(job_id=row["job_id"], finding="final allocation Start differs from verified heavy-audit UTC snapshot; check timezone/requeue history"))
        if set(old_starts) - {r["job_id"] for r in rows}:
            findings.append(dict(finding="some heavy-audit allocations are missing from final accounting"))
    science_terminal = (prior["oracle"]["noncomplete_tasks"] == 0
        and prior["training"]["full_trajectories_without_terminal_summary"] == 0
        and prior["training"]["defined_full_trajectories_not_started"] == 0)
    all_terminal = science_terminal and accounting["listed_all_allocations_terminal"]
    status = "PASS_EXECUTION_ACCOUNTING_FINAL" if all_terminal and not changed and not findings else "FAIL_FINAL_REQUIREMENTS"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    table_path = a.output.with_name(a.output.stem + "_allocations.csv")
    write_csv(table_path, rows)
    auditor = archive_auditor(a.output, source)
    result = dict(schema_version=1, status=status, finalized_UTC=datetime.now(UTC).isoformat(),
        requested_final=True, finalization_mode="stdlib accounting/control-metadata closure of allocated heavy audit",
        heavy_audit=dict(path=str(a.finalize_from), sha256=prior_sha, audited_UTC=prior.get("audited_UTC"),
                         auditor_sha256=prior.get("auditor_sha256")),
        oracle=prior["oracle"], training=prior["training"], engineering_inference=prior["engineering_inference"],
        allocations=accounting, all_listed_evidence_terminal=all_terminal,
        evidence_stable_after_heavy_audit=not changed, changed_files=changed, findings=findings,
        finalizer_checks=dict(control_metadata_content_hashes_checked=len(content_checked),
            raw_payload_stat_signatures_checked=len(stat_checked), raw_payload_content_rehashed=False,
            raw_payload_scope="Content hashes were verified inside the allocation. Closing checks unchanged size/mtime; it is not a second raw-content audit.",
            new_runtime_model_calls=0, arrays_or_models_loaded=False),
        sidecars=dict(allocations=dict(path=str(table_path), sha256=digest(table_path)), auditor_source=auditor),
        accounting_source=dict(path=str(a.accounting), sha256=digest(a.accounting)),
        finalizer_sha256=auditor["sha256"], elapsed_seconds=time.perf_counter() - begin,
        physical_scope=prior["physical_scope"], exclusions=prior["exclusions"])
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps(dict(status=status, allocations=accounting, findings=findings, changed_files=changed,
                         elapsed_seconds=result["elapsed_seconds"], output=str(a.output)), indent=2))
    return 0 if status == "PASS_EXECUTION_ACCOUNTING_FINAL" else 2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, default=Path("runs/conditioning_20261107"))
    p.add_argument("--accounting", type=Path)
    p.add_argument("--owned-jobs", nargs="+")
    p.add_argument("--account", default="m5397_g")
    p.add_argument("--accounting-snapshot-UTC")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--final", action="store_true", help="Require terminal oracle/fits/allocations and unchanged evidence")
    p.add_argument("--finalize-from", type=Path,
                   help="After GPU release, stdlib-only accounting closure using a stable allocated heavy audit")
    a = p.parse_args(argv)
    root = a.root.resolve(); run = a.run_dir if a.run_dir.is_absolute() else root / a.run_dir
    if a.output.exists():
        raise FileExistsError("preserve prior audit receipts; choose a new --output")
    source = Path(__file__).read_bytes()
    if a.finalize_from is not None:
        return finalize_from(a, root, run, source)
    sys.path.insert(0, str(root / "src"))
    from cathode_cvae.runtime import require_compute_allocation
    from cathode_cvae.mlip.registry import MODELS
    require_compute_allocation()
    begin = time.perf_counter(); evidence = Evidence()
    protocol = evidence.read_json(root / "artifacts/conditioning_20261107/protocol.json")
    oracle, oracle_rows, workers = oracle_audit(root, run, evidence, MODELS)
    fits, fit_rows = fit_audit(root, run, evidence, protocol)
    pilot = pilot_audit(run, evidence, MODELS)
    accounting, allocations = None, []
    if a.accounting is not None:
        evidence.bind(a.accounting)
        accounting, allocations = allocation_audit(a.accounting, owned_jobs=a.owned_jobs, expected_account=a.account,
            goal_start=parse_time(protocol["start_UTC"]),
            snapshot_utc=parse_time(a.accounting_snapshot_UTC) if a.accounting_snapshot_UTC else None)
        evidence.findings.extend(resource_findings(accounting, allocations, protocol))
    changed = evidence.mutation_check()
    task_definitions_at_end = [str(p.resolve()) for p in sorted(run.glob("*tasks.json"))]
    if task_definitions_at_end != fits["task_definition_files_at_start"]:
        changed.append("FIT_TASK_DEFINITION_SET_CHANGED_DURING_AUDIT")
    all_terminal = (oracle["noncomplete_tasks"] == 0 and fits["full_trajectories_without_terminal_summary"] == 0
        and fits["defined_full_trajectories_not_started"] == 0
        and accounting is not None and accounting["listed_all_allocations_terminal"])
    status = "PRELIMINARY"
    if changed:
        status = "PRELIMINARY_EVIDENCE_CHANGED_DURING_AUDIT"
    elif a.final:
        status = "PASS_EXECUTION_ACCOUNTING_FINAL" if all_terminal and not evidence.findings else "FAIL_FINAL_REQUIREMENTS"
    elif evidence.findings:
        status = "PRELIMINARY_WITH_FINDINGS"
    a.output.parent.mkdir(parents=True, exist_ok=True)
    sidecars = {}
    for label, rows in (("oracle_tasks", oracle_rows), ("fits", fit_rows), ("allocations", allocations), ("workers", workers)):
        path = a.output.with_name(a.output.stem + "_" + label + ".csv")
        write_csv(path, rows); sidecars[label] = dict(path=str(path), sha256=digest(path))
    sidecars["auditor_source"] = archive_auditor(a.output, source)
    result = dict(schema_version=1, status=status, audited_UTC=datetime.now(UTC).isoformat(),
        requested_final=a.final, evidence_stable_during_audit=not changed, changed_files=changed,
        findings=evidence.findings, oracle=oracle, training=fits, engineering_inference=pilot,
        allocations=accounting, all_listed_evidence_terminal=all_terminal,
        sidecars=sidecars, evidence_file_hashes=evidence.files, evidence_file_stats=evidence.stats,
        auditor_sha256=hashlib.sha256(source).hexdigest(),
        elapsed_seconds=time.perf_counter() - begin,
        physical_scope="Execution, exact identity/context, accounting and software receipts only; not DFT accuracy, statistical adequacy, global minima or transport validation",
        exclusions="Relaxation/MD/NEB force-call counters must be supplied by their own recorded pilots. This audit does not invent uninstrumented forward calls or count cache reuses as new EFS.")
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps(dict(status=status, oracle=oracle["by_teacher"], training=fits,
                         allocations=accounting, findings=len(evidence.findings), output=str(a.output)), indent=2))
    return 0 if not a.final or status == "PASS_EXECUTION_ACCOUNTING_FINAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())


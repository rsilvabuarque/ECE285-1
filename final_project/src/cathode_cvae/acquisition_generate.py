"""Execute static-energy acquisition schedules with a frozen multibin AR CVAE.

Physical temperature changes the allocation of requested energy bins only. It
never rescales latent noise or decoder probabilities. Every positive-count
request produces its declared number of attempt records, including failures;
zero-count requests remain in the allocation source and schedule receipt.
"""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .conditions import schema_identity, validate_schema
from .data import target_indices, validate_permutation_group
from .learning.generate_mask import attempt_records, load_model, resolve_request
from .learning.runtime import device_name, digest
from .learning.train_mask import canonicalize


def _identity(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _seed(base, run_id, request_index):
    value = _identity(dict(base_sampling_seed=int(base), allocation_run_id=run_id, request_index=request_index))
    return int(value[:16], 16) % (2**63 - 1)


def allocation_schedules(payload):
    """Validate declared integer budgets and retain pure/floored/mixture scopes."""
    if not isinstance(payload.get("plans"), list) or not payload["plans"]:
        raise ValueError("Allocation JSON must contain nonempty plans from the allocation command")
    seen = set()
    result = []
    for plan in payload["plans"]:
        plan_id = plan.get("allocation_id")
        schema_id = plan.get("schema_id")
        n = plan.get("n_target", plan.get("composition"))
        if not plan_id or not schema_id or n is None or isinstance(n, bool) or int(n) != n:
            raise ValueError("Each plan needs allocation_id, schema_id and integer n_target")
        for kind, groups in [("per_temperature", plan.get("per_temperature", [])),
                             ("temperature_mixture", plan.get("temperature_mixtures", []))]:
            for group in groups:
                policy = group["allocation_policy"]
                budget = group["budget"]
                if isinstance(budget, bool) or int(budget) != budget or budget < 0:
                    raise ValueError("Every schedule needs a nonnegative integer budget")
                requests = group.get("requests", [])
                if not requests:
                    raise ValueError("An allocation schedule has no bin requests")
                total = 0
                for r in requests:
                    count = r["attempts"]
                    if isinstance(count, bool) or int(count) != count or count < 0:
                        raise ValueError("Request attempt counts must be nonnegative integers")
                    total += int(count)
                    if r.get("schema_id") != schema_id or r.get("n_target") != n:
                        raise ValueError("Request and plan disagree on schema or composition")
                    if r.get("allocation_policy") != policy:
                        raise ValueError("Request and schedule policies differ")
                    temperature = float(r["temperature_K"])
                    probability = float(r["probability"])
                    if not np.isfinite(temperature) or temperature <= 0:
                        raise ValueError("Physical acquisition temperatures must be positive kelvin")
                    if not np.isfinite(probability) or not 0 <= probability <= 1:
                        raise ValueError("Allocation probabilities must lie in [0,1]")
                if total != budget or not np.isclose(sum(r["probability"] for r in requests), 1., atol=1e-12, rtol=1e-12):
                    raise ValueError("Schedule integer counts or probabilities do not match its declared budget")
                temperatures = sorted({float(r["temperature_K"]) for r in requests})
                if kind == "per_temperature":
                    if len(temperatures) != 1 or float(group["temperature_K"]) != temperatures[0]:
                        raise ValueError("A per-temperature schedule mixes physical temperatures")
                    temperature_tag = format(temperatures[0], ".12g")
                else:
                    temperature_tag = "mixture"
                schedule_key = (plan_id, policy, kind, temperature_tag)
                if schedule_key in seen:
                    raise ValueError("Repeated acquisition schedule identity")
                seen.add(schedule_key)
                result.append(dict(allocation_id=plan_id, schema_id=schema_id, n_target=int(n),
                    policy=policy, kind=kind, temperature_tag=temperature_tag,
                    temperatures_K=temperatures, budget=int(budget), requests=requests))
    if not result:
        raise ValueError("Allocation JSON contains no executable schedules")
    return result


def _source_hashes(allocation_path, parent_path):
    from .learning import generate_mask, models, train_mask
    return dict(allocation_sha256=digest(allocation_path), parent_sha256=digest(parent_path),
        generator_source_sha256=digest(__file__),
        canonicalizer_source_sha256=digest(train_mask.__file__),
        request_resolver_source_sha256=digest(generate_mask.__file__),
        model_source_sha256=digest(models.__file__))


def execute_allocation(allocation_path, checkpoint_paths, parent_path, output_dir,
                       device="auto", sampling_seed=20261107, threads=4):
    """Load each declared fit once and execute every prespecified schedule.

    `checkpoint_paths` maps the actual fitted seed to a trusted checkpoint. New
    geometry/protocol identities are canonical deletion masks under this exact
    parent. Models or schedules are never selected using resulting MLIP scores;
    this function performs no physical inference.
    """
    started = time.time()
    if int(threads) != threads or threads < 1 or isinstance(sampling_seed, bool) or int(sampling_seed) != sampling_seed or not 0 <= sampling_seed < 2**63:
        raise ValueError("Use positive threads and a nonnegative int64 base sampling seed")
    if not checkpoint_paths:
        raise ValueError("Provide at least one fitted checkpoint")
    hashes = _source_hashes(allocation_path, parent_path)
    payload = json.loads(Path(allocation_path).read_text())
    schedules = allocation_schedules(payload)
    declared_parent = payload.get("source", {}).get("parent_sha256")
    if declared_parent is not None and declared_parent != hashes["parent_sha256"]:
        raise ValueError("Allocation reference belongs to a different parent")
    with np.load(parent_path, allow_pickle=False) as archive:
        parent = {k: archive[k] for k in archive.files}
    n_sites = len(target_indices(parent))
    if not 1 <= n_sites <= 62:
        raise ValueError("Current canonicalization supports 1..62 removable sites")
    validate_permutation_group(parent["permutations"])
    if parent["permutations"].shape[1] != n_sites:
        raise ValueError("Parent site action and removable-site count differ")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=False)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ValueError("Deterministic CUDA sampling needs CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    device = device_name(device)
    torch.set_num_threads(int(threads))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    inverse = torch.as_tensor(np.argsort(parent["permutations"], axis=1), dtype=torch.long, device=device)
    records, candidate_map, schedule_receipts, model_receipts = [], {}, [], {}
    code_frequencies = Counter()
    protocol = None
    for fit_seed, checkpoint in checkpoint_paths.items():
        fit_seed = str(fit_seed)
        checkpoint = Path(checkpoint)
        checkpoint_sha = digest(checkpoint)
        model, cfg = load_model(checkpoint, device)
        model = model.to(device=device, dtype=torch.float32)
        actual_fit_seed = str(cfg.get("seed", cfg.get("fit_seed", "")))
        if actual_fit_seed != fit_seed:
            raise ValueError("Checkpoint fit seed differs from --checkpoints label; sampling seeds are not fitted models")
        if cfg.get("condition_mode") != "multibin" or model.condition_dim != 4 or not getattr(model, "has_latent", False):
            raise ValueError("Acquisition generation requires a fitted multibin autoregressive CVAE")
        if cfg.get("kind") not in {"ar", "ar_cvae"} or model.n_sites != n_sites:
            raise ValueError("Checkpoint architecture/site count differs from the required parent AR CVAE")
        if cfg.get("parent_sha256") != hashes["parent_sha256"]:
            raise ValueError("Checkpoint does not certify this exact parent hash")
        schema = validate_schema(cfg["condition_schema"])
        schema_id = schema_identity(schema)
        if any(schedule["schema_id"] != schema_id for schedule in schedules):
            raise ValueError("Allocation schema differs from a checkpoint's frozen condition schema")
        geometry_protocol = cfg.get("geometry_protocol", "pristine-derived deletion; unrelaxed")
        if protocol is not None and geometry_protocol != protocol:
            raise ValueError("Checkpoints mix incompatible geometry protocols")
        protocol = geometry_protocol
        model_receipts[fit_seed] = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
            fit_seed=fit_seed, condition_schema_id=schema_id, condition_schema=schema,
            parent_sha256=cfg["parent_sha256"], geometry_protocol=geometry_protocol,
            condition_mode=cfg["condition_mode"], dtype="float32", tf32=False)
        for schedule in schedules:
            run_body = dict(allocation_id=schedule["allocation_id"], fit_seed=fit_seed,
                allocation_policy=schedule["policy"], schedule_kind=schedule["kind"],
                temperature_tag=schedule["temperature_tag"])
            run_id = "allocation_" + _identity(run_body)
            schedule_generated = schedule_failed = 0
            request_receipts = []
            for index, request in enumerate(schedule["requests"]):
                count = int(request["attempts"])
                seed = _seed(sampling_seed, run_id, index)
                n = schedule["n_target"]
                requested = request["requested_bin"]
                request_receipt = dict(requested_bin=requested, temperature_K=float(request["temperature_K"]),
                    attempts=count, probability=float(request["probability"]), sampling_seed=seed)
                if count == 0:
                    request_receipt["status"] = "ZERO_ALLOCATION"
                    request_receipts.append(request_receipt)
                    continue
                error, effective, ids, keys = None, "", [], []
                try:
                    vector, requested, effective, resolved_id, _ = resolve_request(cfg, n, requested_bin=requested)
                    if resolved_id != schema_id:
                        raise ValueError("Resolved request changed the frozen schema identity")
                    c = torch.as_tensor(vector, dtype=torch.float32, device=device).repeat(count, 1)
                    generator = torch.Generator(device=device).manual_seed(seed)
                    with torch.no_grad():
                        raw = model.sample(c, n, generator=generator)
                        if raw.shape != (count, n_sites) or not bool(((raw == 0) | (raw == 1)).all()) or not bool((raw.sum(1) == n).all()):
                            raise RuntimeError("Constrained sampler returned an invalid count or occupancy")
                        canonical, codes = canonicalize(raw, inverse)
                    occupancy = canonical.cpu().numpy().astype(np.uint8)
                    width = (n_sites + 3) // 4
                    keys = [format(int(code), f"0{n_sites}b") for code in codes]
                    ids = [f"mask{n_sites}_{int(code):0{width}x}" for code in codes]
                    for code, row in zip(codes, occupancy):
                        code = int(code)
                        if code in candidate_map and not np.array_equal(candidate_map[code], row):
                            raise RuntimeError("Canonical code collides across different actual occupancies")
                        candidate_map[code] = row
                        code_frequencies[code] += 1
                except (ValueError, KeyError, RuntimeError, FloatingPointError) as failure:
                    error = f"{type(failure).__name__}: {failure}"
                batch = attempt_records(count, ids, keys, method="multibin", fit_seed=actual_fit_seed,
                    sampling_seed=seed, n_target=n, requested_bin=requested, effective_model_class=effective,
                    schema_id=schema_id, checkpoint_sha256=checkpoint_sha, parent_sha256=hashes["parent_sha256"],
                    geometry_protocol=geometry_protocol, temperature_K=float(request["temperature_K"]),
                    allocation_policy=schedule["policy"], allocation_weight=float(request["probability"]), error=error)
                for row in batch:
                    row.update(attempt_id=f"{run_id}_request{index}_attempt{row['attempt']}",
                        replicate_id=fit_seed, allocation_run_id=run_id, allocation_id=schedule["allocation_id"],
                        schedule_kind=schedule["kind"], base_sampling_seed=int(sampling_seed))
                records.extend(batch)
                schedule_generated += count if error is None else 0
                schedule_failed += count if error is not None else 0
                request_receipt.update(status="GENERATED" if error is None else "FAILED_REQUEST", error=error)
                request_receipts.append(request_receipt)
            if schedule_generated + schedule_failed != schedule["budget"]:
                raise RuntimeError("Executed schedule did not retain its entire declared attempt budget")
            schedule_receipts.append(dict(run_body, allocation_run_id=run_id, n_target=schedule["n_target"],
                temperatures_K=schedule["temperatures_K"], declared_budget=schedule["budget"],
                generated_attempts=schedule_generated, failed_requests=schedule_failed,
                requests=request_receipts))
        if digest(checkpoint) != checkpoint_sha:
            raise RuntimeError("A model checkpoint changed during generation")
    if not records:
        raise ValueError("All schedules have zero budget; no generation attempts were requested")
    if _source_hashes(allocation_path, parent_path) != hashes:
        raise RuntimeError("Allocation, parent or generation source changed during execution")
    codes = sorted(candidate_map)
    occupancy = (np.stack([candidate_map[c] for c in codes]) if codes else np.empty((0, n_sites), np.uint8))
    keys = np.asarray([format(c, f"0{n_sites}b") for c in codes], dtype=f"<U{n_sites}")
    width = (n_sites + 3) // 4
    ids = np.asarray([f"mask{n_sites}_{c:0{width}x}" for c in codes], dtype=f"<U{len(str(n_sites))+5+width}")
    n = occupancy.sum(1).astype(np.int64)
    np.savez_compressed(output/"candidates.npz", structure_ids=ids, masks=1-occupancy,
        occupancy=occupancy, canonical_masks=keys, n_Li=n, n_target=n,
        n_atoms=len(parent["numbers"])-n_sites+n, parent_sha256=np.asarray(hashes["parent_sha256"]),
        geometry_protocol=np.asarray(protocol), orientation=np.asarray("canonical"),
        attempt_frequency=np.asarray([code_frequencies[c] for c in codes], dtype=np.int64),
        sampling_measure=np.asarray("CVAE proposal-biased acquisition requests; not a uniform reference pool"))
    with (output/"attempts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    finished = time.time()
    generated = sum(r["status"] == "generated" for r in records)
    receipt = dict(status="GENERATED" if generated == len(records) else "GENERATED_WITH_FAILED_REQUESTS" if generated else "FAILED_REQUESTS",
        started_unix=started, generation_completed_unix=finished, elapsed_seconds=finished-started,
        attempts=len(records), generated_attempts=generated, failed_requests=len(records)-generated,
        unique_scoring_geometries=len(codes), schedules=len(schedule_receipts),
        zero_count_bin_requests=sum(r["attempts"]==0 for s in schedule_receipts for r in s["requests"]),
        base_sampling_seed=int(sampling_seed), device=device, torch=torch.__version__,
        dtype="float32", tf32=False, deterministic_algorithms=True,
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        source_hashes=hashes, models=model_receipts, schedule_receipts=schedule_receipts,
        output_sha256=dict(attempts= digest(output/"attempts.csv"), candidates=digest(output/"candidates.npz")),
        no_physical_labels_generated=True, model_and_plan_hashes_stable_through_generation=True,
        physical_temperature_enters_decoder=False,
        interpretation="Prespecified bin-acquisition attempts only; measured energy ranges require subsequent exact-geometry teacher evaluations")
    (output/"generation_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:receipt[k] for k in ["status","attempts","generated_attempts","failed_requests","unique_scoring_geometries","schedules"]},indent=2))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Actual fitted seed=trusted-checkpoint.pt")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    parser.add_argument("--sampling-seed", type=int, default=20261107)
    parser.add_argument("--threads", type=int, default=4)
    a=parser.parse_args(argv)
    from .runtime import require_compute_allocation
    require_compute_allocation()
    if any("=" not in v for v in a.checkpoints):parser.error("Use fitted-seed=checkpoint-path")
    checkpoints=dict(v.split("=",1) for v in a.checkpoints)
    if len(checkpoints)!=len(a.checkpoints):parser.error("Each fitted seed must appear once")
    return execute_allocation(a.allocation,checkpoints,a.parent,a.output_dir,a.device,a.sampling_seed,a.threads)


if __name__=="__main__":main()

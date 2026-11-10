r"""Stage and merge a protected, fixed-schema feedback acquisition round.

The input generator must use ``scripts/overnight_20261108/generate_controls.py``
and preserve its frozen ``attempts.csv``, ``primary_candidates.npz`` and
``receipt.json``. Stage one method/replicate before *any* scoring, assigning a
deterministic TRAIN/VAL role to each canonical vacancy orbit. Then register the
staged ``candidates.npz`` with ``cathode_cvae.pool_scoring``. Merge reads the
resulting task-level score and worker receipts, never a bare energy array.

Example, from the project root inside an appropriate compute allocation::

    python -m cathode_cvae.iterative_acquisition stage \
      --parent PARENT.npz --base-data TRAINING.npz --schema SCHEMA.json \
      --generation-dir GENERATED --known-history HISTORY.json \
      --protected-history PROTECTED.json --method shell_cvae --replicate 1 \
      --round 1 --role-seed 4 --counts 12 24 36 --bins low middle high \
      --attempts-per-cell 128 --queries-per-cell 64 --output-dir STAGED
    python -m cathode_cvae.pool_scoring register --root ORACLE \
      --input STAGED/candidates.npz --parent PARENT.npz --models uma
    # Start workers, then collect into COLLECTED; retain task and worker receipts.
    python -m cathode_cvae.iterative_acquisition merge --stage-dir STAGED \
      --parent PARENT.npz --base-data TRAINING.npz --schema SCHEMA.json \
      --oracle-root ORACLE --collection-receipt COLLECTED/receipt.json \
      --output-dir MERGED --new-weight 1

For a subsequent round, pass MERGED/training.npz as --base-data and its
merge_receipt.json as --previous-merge-receipt. Build the next history from the
complete prior history and every acquisition identity, including failures and
other policy arms. Role hashing is parent/seed/orbit based and independent of
the acquired energy, method and round. Stage all matched arms before scoring so
cross-arm identical configurations can safely share an actual oracle query.

Only the fixed parent/geometry and pinned UMA training teacher are supported.
MACE can be scored separately as a held-out second-teacher assessment.
Original H0 TRAIN/VAL/TEST rows remain byte-value identical, and acquired rows
can only be TRAIN or VAL; no existing role or saved input is modified.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .conditions import classify, load_schema
from .data import sha256, target_indices, validate_permutation_group
from .mlip.registry import MODELS
from .pool_scoring import PROTOCOL, physical_key
from .runtime import require_compute_allocation


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _binding(path):
    path = Path(path).absolute()
    return {"path": str(path), "sha256": sha256(path)}


def _assert_binding(binding, path, label):
    if binding["sha256"] != sha256(path):
        raise ValueError(f"{label} bytes changed after the frozen stage")


def _keys(path, sites):
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict):
        fields = ("keys", "canonical_masks", "known_keys", "canonical_keys")
        names = [key for key in fields if key in value]
        if len(names) != 1:
            raise ValueError("History dictionary must name exactly one canonical-key array")
        value = value[names[0]]
    if not isinstance(value, list):
        raise ValueError("History must contain an array of vacancy strings")
    keys = set(value)
    if any(not isinstance(key, str) or len(key) != sites or set(key) - {"0", "1"} for key in keys):
        raise ValueError("History contains a nonbinary key or a different parent size")
    return keys


def _parent(path):
    with np.load(path, allow_pickle=False) as archive:
        permutations = np.asarray(archive["permutations"], dtype=np.int64)
        numbers = np.asarray(archive["numbers"])
        mobile = np.asarray(target_indices(archive), dtype=np.int64)
    sites = permutations.shape[1]
    if sites != 48 or mobile.shape != (sites,) or len(set(mobile.tolist())) != sites:
        raise ValueError("Current scoring queue requires a 48-site parent and explicit mobile indices")
    if np.any(mobile < 0) or np.any(mobile >= len(numbers)):
        raise ValueError("Mobile-site indices are outside the parent")
    if permutations.ndim != 2 or not np.array_equal(
        np.sort(permutations, axis=1), np.tile(np.arange(sites), (len(permutations), 1))
    ):
        raise ValueError("Parent permutations are invalid")
    validate_permutation_group(permutations)
    return permutations, len(numbers) - sites


def _canonical_vacancy(occupancy, permutations, block=128):
    """Exact minimum vacancy bit strings under the saved site permutations."""
    x = np.asarray(occupancy)
    sites = permutations.shape[1]
    if x.ndim != 2 or x.shape[1] != sites or not np.isin(x, [0, 1]).all():
        raise ValueError("Occupancy must be binary with one column per parent site")
    inv = np.argsort(permutations, axis=1)
    powers = (1 << np.arange(sites - 1, -1, -1, dtype=np.int64))
    result = []
    for start in range(0, len(x), block):
        transformed = (1 - x[start:start + block].astype(np.uint8))[:, inv]
        codes = np.einsum("bpn,n->bp", transformed, powers, optimize=True).min(axis=1)
        result.extend(format(int(code), f"0{sites}b") for code in codes)
    return np.asarray(result, dtype=f"<U{sites}")


def _data(path, permutations):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {"occupancy", "energy_eV_atom", "split", "weight", "structure_ids"}
    if not required <= arrays.keys() or not ({"n_Li", "n_target"} & arrays.keys()):
        raise ValueError("Training NPZ lacks required occupancy, count, energy, role or weight arrays")
    count_key = "n_target" if "n_target" in arrays else "n_Li"
    n = np.asarray(arrays[count_key])
    x = np.asarray(arrays["occupancy"])
    rows = len(n)
    for name in required | {count_key}:
        if len(arrays[name]) != rows:
            raise ValueError(f"Training field {name} has the wrong number of rows")
    if x.shape != (rows, permutations.shape[1]) or not np.array_equal(x.sum(axis=1), n):
        raise ValueError("Training occupancy and requested counts disagree")
    if not np.isfinite(arrays["energy_eV_atom"]).all() or not np.isfinite(arrays["weight"]).all() or np.any(arrays["weight"] <= 0):
        raise ValueError("Training energy/weight values must be finite and weights positive")
    roles = arrays["split"].astype(str)
    if set(roles) - {"train", "val", "test"} or "train" not in roles or "val" not in roles:
        raise ValueError("Dataset roles must be train/val/test with nonempty train and val")
    if len(set(arrays["structure_ids"].astype(str))) != rows:
        raise ValueError("Dataset structure IDs are not unique")
    keys = _canonical_vacancy(x, permutations)
    if len(set(keys.tolist())) != rows:
        raise ValueError("Dataset repeats a canonical orbit, potentially crossing protected roles")
    if "canonical_masks" in arrays and not np.array_equal(keys, arrays["canonical_masks"].astype(str)):
        raise ValueError("Saved canonical orbit identities do not match the actual occupancies")
    if "n_Li" in arrays and "n_target" in arrays and not np.array_equal(arrays["n_Li"], arrays["n_target"]):
        raise ValueError("n_Li and n_target disagree")
    return arrays, keys, roles, n


def _schema(schema_path, parent_path, origin_sha):
    schema = load_schema(schema_path)
    provenance = schema.get("provenance", {})
    model = MODELS["uma"]
    expected = {
        "parent_sha256": sha256(parent_path),
        "teacher": "uma",
        "teacher_head": model["head"],
        "geometry_protocol": PROTOCOL,
        "data_sha256": origin_sha,
    }
    for name, value in expected.items():
        if provenance.get(name) != value:
            raise ValueError(f"Frozen schema has incompatible {name}")
    recorded = provenance.get("teacher_checkpoint_hashes")
    if recorded != model["files"]:
        raise ValueError("Frozen schema uses a different UMA checkpoint")
    if schema["n_sites"] != 48:
        raise ValueError("Frozen condition schema uses another parent size")
    return schema


def _base_lineage(base_path, schema_path, parent_path, round_index, previous_receipt):
    base_sha = sha256(base_path)
    if round_index == 1:
        if previous_receipt is not None:
            raise ValueError("Round one must not claim a previous acquisition")
        origin_sha = base_sha
    else:
        if previous_receipt is None:
            raise ValueError("A later round requires the previous merge receipt")
        previous = json.loads(Path(previous_receipt).read_text())
        if previous.get("status") != "MERGED_ITERATIVE_ROUND" or previous.get("round") != round_index - 1:
            raise ValueError("Previous merge receipt has the wrong status or round")
        if previous["outputs"]["training.npz"] != base_sha:
            raise ValueError("Previous merged data bytes do not match the base data")
        _assert_binding(previous["inputs"]["parent"], parent_path, "Parent")
        _assert_binding(previous["inputs"]["schema"], schema_path, "Schema")
        origin_sha = previous["origin_data_sha256"]
    schema = _schema(schema_path, parent_path, origin_sha)
    return origin_sha, schema


def _role(parent_sha, role_seed, key, val_fraction):
    digest = hashlib.sha256(f"iterative-role-v1|{parent_sha}|{role_seed}|{key}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "val" if value < val_fraction else "train"


def _generation_inputs(generation_dir):
    generation_dir = Path(generation_dir)
    receipt_path = generation_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "FROZEN_BEFORE_SCORING" or not receipt.get("no_energy_filtering"):
        raise ValueError("Generation receipt must freeze unfiltered attempts before scoring")
    paths = {name: generation_dir / name for name in
             ("attempts.csv", "primary_candidates.npz", "raw_attempts.npz", "protocol.json")}
    for name, path in paths.items():
        expected = (receipt.get("protocol_sha256") if name == "protocol.json"
                    else receipt.get("outputs", {}).get(name))
        if expected != sha256(path):
            raise ValueError(f"Generation receipt does not bind {name}")
    return receipt_path, paths


def stage(*, parent_path, base_data_path, schema_path, generation_dir, known_histories,
          protected_histories, method, replicate, round_index, role_seed, counts, bins,
          attempts_per_cell, queries_per_cell, val_fraction, output_dir,
          previous_merge_receipt=None, allowed_swaps=(1, 2, 3)):
    """Freeze raw attempt selection and role assignment before oracle scoring."""
    if round_index < 1 or attempts_per_cell < 1 or not 0 <= queries_per_cell <= attempts_per_cell:
        raise ValueError("Invalid round, attempt budget or query budget")
    if not math.isfinite(val_fraction) or not 0 <= val_fraction < 1:
        raise ValueError("New VAL fraction must lie in [0,1)")
    if not known_histories or not protected_histories:
        raise ValueError("Explicit known and protected history sources are both required")
    if len(set(counts)) != len(counts) or len(set(bins)) != len(bins) or not counts or not bins:
        raise ValueError("Counts and bins must be nonempty and unique")
    if any(n < 0 or n > 48 for n in counts) or set(bins) - {"low", "middle", "high"}:
        raise ValueError("Invalid composition or requested energy class")
    if not allowed_swaps or any(not isinstance(k, int) or k < 0 for k in allowed_swaps):
        raise ValueError("Allowed mutation swap budgets must be nonnegative integers")
    permutations, host_atoms = _parent(parent_path)
    parent_sha = sha256(parent_path)
    origin_sha, schema = _base_lineage(base_data_path, schema_path, parent_path, round_index, previous_merge_receipt)
    base, base_keys, base_roles, _ = _data(base_data_path, permutations)
    known = set(base_keys.tolist())
    protected = set(base_keys[np.isin(base_roles, ["val", "test"])].tolist())
    declared_external_protected = set()
    for path in known_histories:
        known.update(_keys(path, 48))
    for path in protected_histories:
        declared_external_protected.update(_keys(path, 48))
    protected.update(declared_external_protected)
    known.update(protected)
    # Histories produced by the project's registry are canonical vacancy keys.
    # For an explicitly supplied history that also contains a noncanonical
    # representative, candidate-level orbit checks still prevent leakage.
    known_codes = {int(key, 2) for key in known}
    protected_codes = {int(key, 2) for key in protected}
    external_codes = {int(key, 2) for key in declared_external_protected}
    inverse_permutations = np.argsort(permutations, axis=1)
    powers = 1 << np.arange(47, -1, -1, dtype=np.int64)
    orbit_status = {}

    def variants(key):
        bits = np.asarray(list(key), dtype=np.uint8)
        return np.einsum("pn,n->p", bits[inverse_permutations], powers, optimize=True)

    def membership(key):
        if key in protected:
            return "protected"
        if key not in orbit_status:
            transformed = variants(key)
            if any(int(code) in protected_codes for code in transformed):
                orbit_status[key] = "protected"
            elif key in known or any(int(code) in known_codes for code in transformed):
                orbit_status[key] = "known"
            else:
                orbit_status[key] = "fresh"
        return orbit_status[key]
    if previous_merge_receipt is not None:
        previous = json.loads(Path(previous_merge_receipt).read_text())
        if previous["role_seed"] != role_seed or previous["new_validation_fraction"] != val_fraction:
            raise ValueError("Role assignment policy changed between acquisition rounds")
        old_stage = Path(previous["inputs"]["stage_receipt"]["path"])
        _assert_binding(previous["inputs"]["stage_receipt"], old_stage, "Previous stage receipt")
        old_receipt = json.loads(old_stage.read_text())
        old_roles_path = old_stage.parent / "assigned_roles.json"
        if sha256(old_roles_path) != old_receipt["outputs"]["assigned_roles.json"]:
            raise ValueError("Previous acquisition identities have changed")
        if any(membership(key) == "fresh" for key in json.loads(old_roles_path.read_text())):
            raise ValueError("Known history omits prior acquired or failed candidate orbits")
    receipt_path, generated = _generation_inputs(generation_dir)
    generation_receipt = json.loads(receipt_path.read_text())
    protocol = json.loads(generated["protocol.json"].read_text())
    if protocol.get("status") != "FROZEN_BEFORE_GENERATION_AND_SCORING" or not protocol.get("no_energy_filtering"):
        raise ValueError("Generation protocol was not frozen before unfiltered proposals")
    for name, actual in (("parent", parent_sha), ("data", sha256(base_data_path)), ("schema", sha256(schema_path))):
        expected = protocol.get("inputs", {}).get(name, {}).get("sha256")
        if expected != actual:
            raise ValueError(f"Generated attempts do not bind the declared {name}")
    if protocol.get("condition_schema_id") != schema["identity"]:
        raise ValueError("Generated attempts have a different condition schema")
    generation_history = protocol.get("inputs", {}).get("history", {})
    generation_history_path = Path(generation_history.get("path", ""))
    if (not generation_history_path.is_file()
            or generation_history.get("sha256") != sha256(generation_history_path)
            or not _keys(generation_history_path, 48) <= known):
        raise ValueError("Known history must include the frozen generator history")
    manifest_binding = protocol.get("manifest", {})
    manifest_path = Path(manifest_binding.get("path", ""))
    if not manifest_path.is_file() or manifest_binding.get("sha256") != sha256(manifest_path):
        raise ValueError("Generated protocol does not bind its frozen model manifest")
    manifest = json.loads(manifest_path.read_text())
    if not str(manifest.get("status", "")).startswith("FROZEN"):
        raise ValueError("Generation manifest was not frozen before proposals")
    entries = [item for item in manifest.get("models", []) if item.get("method") == str(method)
               and str(item.get("replicate_id", item.get("fit_seed", ""))) == str(replicate)]
    if len(entries) != 1:
        raise ValueError("Frozen manifest does not uniquely identify the requested method/replicate")
    is_mutation = entries[0].get("sampler") == "mutation"
    if not entries[0].get("sampler"):
        expected_checkpoint = entries[0].get("expected_checkpoint_sha256", entries[0].get("checkpoint_sha256"))
        checkpoints = [item for item in generation_receipt.get("checkpoints", [])
                       if item.get("method") == str(method)
                       and str(item.get("replicate_id", item.get("fit_seed", ""))) == str(replicate)]
        if (not entries[0].get("checkpoint") or not expected_checkpoint or len(checkpoints) != 1
                or checkpoints[0].get("actual_checkpoint_sha256") != expected_checkpoint):
            raise ValueError("Generated model attempts lack a matching frozen checkpoint hash")
    elif entries[0]["sampler"] not in {"mutation", "uniform"}:
        raise ValueError("Unknown generation sampler in frozen manifest")
    with generated["attempts.csv"].open(newline="") as stream:
        selected_rows = [row for row in csv.DictReader(stream)
                         if row.get("method") == str(method) and row.get("replicate_id") == str(replicate)]
    expected_cells = {(int(n), str(b)) for n in counts for b in bins}
    cells = Counter((int(row["n_target"]), row["requested_bin"]) for row in selected_rows)
    if set(cells) != expected_cells or any(cells[cell] != attempts_per_cell for cell in expected_cells):
        raise ValueError("Attempt counts do not match the predeclared count/bin cells")
    if len({row["attempt_id"] for row in selected_rows}) != len(selected_rows):
        raise ValueError("Attempt identities are duplicated")
    try:
        raw_rows = np.asarray([int(row["raw_row"]) for row in selected_rows], dtype=np.int64)
    except (KeyError, ValueError):
        raise ValueError("Every attempt must identify its raw generated row") from None
    if len(set(raw_rows.tolist())) != len(raw_rows):
        raise ValueError("Multiple attempts point to one raw generated row")
    with np.load(generated["raw_attempts.npz"], allow_pickle=False) as archive:
        raw_x = np.asarray(archive["occupancy"])
        raw_seed = np.asarray(archive["seed_occupancy"])
    if (raw_x.ndim != 2 or raw_x.shape[1] != 48 or raw_seed.shape != raw_x.shape
            or np.any(raw_rows < 0) or np.any(raw_rows >= len(raw_x))):
        raise ValueError("Raw attempt or mutation-seed array has an invalid shape/index")
    valid_indices = np.asarray([i for i, row in enumerate(selected_rows) if row.get("status") == "generated"], dtype=int)
    if len(valid_indices):
        actual_x = raw_x[raw_rows[valid_indices]]
        actual_keys = _canonical_vacancy(actual_x, permutations)
        if any(actual_keys[k] != selected_rows[i].get("canonical_mask", "")
               or int(actual_x[k].sum()) != int(selected_rows[i]["n_target"])
               for k, i in enumerate(valid_indices)):
            raise ValueError("Raw generated occupancies disagree with attempt masks/counts")
    if is_mutation:
        for i in valid_indices:
            row = selected_rows[i]
            try:
                seed_index = int(row["seed_source_row"])
                swaps = int(row["swap_count"])
            except (KeyError, ValueError):
                raise ValueError("Mutation attempts must identify the TRAIN seed and swap budget") from None
            if (not 0 <= seed_index < len(base_roles) or base_roles[seed_index] != "train"
                    or row.get("seed_source_sha256") != sha256(base_data_path)
                    or row.get("seed_structure_id") != str(base["structure_ids"][seed_index])):
                raise ValueError("Mutation seed does not belong to the bound TRAIN dataset")
            seed = np.asarray(base["occupancy"][seed_index])
            child = raw_x[raw_rows[i]]
            if not np.array_equal(raw_seed[raw_rows[i]], seed):
                raise ValueError("Raw mutation seed differs from its declared TRAIN row")
            removed = np.flatnonzero((seed == 1) & (child == 0)).tolist()
            added = np.flatnonzero((seed == 0) & (child == 1)).tolist()
            if (swaps not in allowed_swaps or len(removed) != swaps or len(added) != swaps
                    or row.get("removed_sites", "") != "|".join(map(str, removed))
                    or row.get("added_sites", "") != "|".join(map(str, added))):
                raise ValueError("Mutation child does not match its declared edit budget/sites")
    with np.load(generated["primary_candidates.npz"], allow_pickle=False) as archive:
        pool_masks = np.asarray(archive["masks"])
        pool_ids = archive["structure_ids"].astype(str)
        if str(archive["parent_sha256"].item()) != parent_sha:
            raise ValueError("Generated candidate parent binding differs")
    if pool_masks.ndim != 2 or pool_masks.shape[1] != 48 or len(pool_ids) != len(pool_masks):
        raise ValueError("Generated candidate pool shape is invalid")
    if not np.isin(pool_masks, [0, 1]).all() or len(set(pool_ids)) != len(pool_ids):
        raise ValueError("Candidate pool contains invalid masks or duplicate IDs")
    pool = {sid: mask for sid, mask in zip(pool_ids, pool_masks)}
    chosen = {}
    query_by_cell = Counter()
    statuses = Counter()
    annotated = []
    for source in selected_rows:
        row = dict(source)
        n = int(row["n_target"])
        cell = (n, row["requested_bin"])
        key, sid = row.get("canonical_mask", ""), row.get("structure_id", "")
        role = ""
        if row.get("status") != "generated":
            status = "generation_failed_or_invalid"
        else:
            if len(key) != 48 or set(key) - {"0", "1"} or key.count("0") != n:
                raise ValueError("Generated canonical vacancy string violates the requested count")
            if sid != f"mask48_{int(key, 2):012x}" or sid not in pool or not np.array_equal(pool[sid], np.asarray(list(key), dtype=np.uint8)):
                raise ValueError("Attempt ID, candidate mask and canonical string disagree")
            member = membership(key)
            if member != "fresh":
                status = member
            elif key in chosen:
                status = "duplicate_in_round"
            elif query_by_cell[cell] >= queries_per_cell:
                status = "cell_query_budget_exhausted"
            else:
                check = _canonical_vacancy((1 - pool[sid])[None], permutations)[0]
                if check != key:
                    raise ValueError("Selected candidate is not in canonical parent orientation")
                role = _role(parent_sha, role_seed, key, val_fraction)
                chosen[key] = {"structure_id": sid, "n_target": n, "role": role,
                               "requested_bin": row["requested_bin"], "first_attempt_id": row["attempt_id"]}
                query_by_cell[cell] += 1
                status = "selected_for_query"
        statuses[status] += 1
        row.update(stage_status=status, assigned_role=role)
        annotated.append(row)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    selected = list(chosen.items())
    masks = np.asarray([list(key) for key, _ in selected], dtype=np.uint8).reshape(-1, 48)
    counts_out = 48 - masks.sum(axis=1, dtype=np.int64)
    np.savez_compressed(output / "candidates.npz", masks=masks, occupancy=1-masks,
        structure_ids=np.asarray([record["structure_id"] for _, record in selected], dtype="<U19"),
        n_target=counts_out, n_Li=counts_out, n_atoms=host_atoms+counts_out,
        parent_sha256=np.asarray(parent_sha))
    with (output / "attempts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(annotated[0]))
        writer.writeheader()
        writer.writerows(annotated)
    _write_json(output / "assigned_roles.json", {key: record for key, record in selected})
    receipt = {
        "status": "STAGED_BEFORE_SCORING", "round": int(round_index), "method": str(method),
        "replicate": str(replicate), "created_unix": time.time(),
        "source_sha256": sha256(__file__),
        "origin_data_sha256": origin_sha, "condition_schema_id": schema["identity"],
        "counts": list(counts), "bins": list(bins), "attempts_per_cell": attempts_per_cell,
        "queries_per_cell": queries_per_cell, "role_seed": int(role_seed),
        "allowed_mutation_swap_budgets": list(allowed_swaps),
        "mutation_swap_counts": dict(Counter(int(selected_rows[i]["swap_count"]) for i in valid_indices)) if is_mutation else {},
        "new_validation_fraction": val_fraction, "parent_sha256": parent_sha,
        "attempts": len(annotated), "selected_unique_orbits": len(selected),
        "base_role_external_protected_overlap_counts": {
            role: sum(key in declared_external_protected
                      or any(int(code) in external_codes for code in variants(key))
                      for key in base_keys[base_roles == role])
            for role in ("train", "val", "test")},
        "stage_status_counts": dict(statuses),
        "queries_by_cell": {f"{n}:{b}": query_by_cell[(n,b)] for n,b in sorted(expected_cells)},
        "inputs": {"parent": _binding(parent_path), "base_data": _binding(base_data_path),
                   "schema": _binding(schema_path), "generation_receipt": _binding(receipt_path),
                   "generation_manifest": _binding(manifest_path),
                   "generation_history": _binding(generation_history_path),
                   "generation_files": {name: _binding(path) for name, path in generated.items()},
                   "known_histories": [_binding(path) for path in known_histories],
                   "protected_histories": [_binding(path) for path in protected_histories],
                   "previous_merge_receipt": None if previous_merge_receipt is None else _binding(previous_merge_receipt)},
        "outputs": {name: sha256(output / name) for name in ("candidates.npz", "attempts.csv", "assigned_roles.json")},
        "policy": "Roles are decided before score access; all attempts, underfilled cells and query-cap exclusions remain recorded.",
    }
    _write_json(output / "stage_receipt.json", receipt)
    return receipt


def _check_worker(path, teacher):
    worker = json.loads(Path(path).read_text())
    spec = MODELS[teacher]
    checks = {"model": teacher, "head": spec["head"], "package_version": spec["version"],
              "dtype": "float32", "tf32": False}
    for name, expected in checks.items():
        if worker.get(name) != expected:
            raise ValueError(f"Scoring worker has incompatible {name}")
    actual_hashes = {Path(name).name: value for name, value in worker.get("checkpoint_hashes", {}).items()}
    wanted_hashes = {Path(name).name: value for name, value in spec["files"].items()}
    if actual_hashes != wanted_hashes:
        raise ValueError("Scoring worker has different checkpoint bytes")
    return worker


def _scores(stage_dir, stage_receipt, oracle_root, collection_receipt, teacher):
    """Verify selected geometries against raw, pinned task-level score evidence."""
    if teacher != "uma":
        raise ValueError("Round training uses the frozen UMA condition schema")
    root = Path(oracle_root)
    registry_path = root / "registry.json"
    registry = json.loads(registry_path.read_text())
    collection = json.loads(Path(collection_receipt).read_text())
    if collection.get("status") not in {"PASS", "INCOMPLETE"} or collection.get("registry_sha256") != sha256(registry_path):
        raise ValueError("Collection receipt does not bind the current oracle registry")
    if registry.get("parent_sha256") != stage_receipt["parent_sha256"]:
        raise ValueError("Oracle registry uses a different parent")
    roles = json.loads((Path(stage_dir) / "assigned_roles.json").read_text())
    expected = {record["structure_id"]: (key, record) for key, record in roles.items()}
    for sid, (key, _) in expected.items():
        physical = physical_key(stage_receipt["parent_sha256"], key, teacher)
        record = registry.get("physical_keys", {}).get(physical)
        if not record or record.get("model") != teacher or record.get("structure_id") != sid:
            raise ValueError("A staged candidate is missing its pinned physical oracle key")
        # New training roles must be frozen before the relevant raw score was
        # acquired. The archived cache certificate does not establish this time.
        if record.get("provenance") != "new_scoring":
            raise ValueError("A pre-existing cache score cannot establish pre-scoring role assignment")
    selected = {}
    task_count = 0
    worker_bindings = {}
    task_bindings = {}
    for task in registry.get("tasks", []):
        if task.get("model") != teacher:
            continue
        wanted = set(task.get("expected_structure_ids", [])) & set(expected)
        if not wanted:
            continue
        task_count += 1
        tid = task["task_id"]
        score_path = root / "scores" / teacher / f"{tid}.npz"
        metadata_path = score_path.with_suffix(".json")
        owner_path = root / "claims" / tid / "owner.json"
        if not score_path.is_file() or not metadata_path.is_file() or not owner_path.is_file():
            raise ValueError(f"Acquisition task {tid} has not produced terminal score evidence")
        metadata = json.loads(metadata_path.read_text())
        owner = json.loads(owner_path.read_text())
        if metadata.get("status") not in {"complete", "complete_with_failures"} or metadata.get("task") != task:
            raise ValueError("Oracle task receipt differs from its registered task")
        if sha256(score_path) != metadata.get("output_sha256"):
            raise ValueError("Task score bytes differ from their receipt")
        if task.get("geometry_protocol") != PROTOCOL or task.get("parent_sha256") != stage_receipt["parent_sha256"]:
            raise ValueError("Task geometry or parent differs from the staged experiment")
        if sha256(task["input_npz"]) != task.get("input_sha256") or sha256(task["parent_npz"]) != task.get("parent_sha256"):
            raise ValueError("Oracle task input bytes changed")
        if owner.get("worker") != metadata.get("worker") or float(owner.get("started_unix", 0)) < stage_receipt["created_unix"]:
            raise ValueError("Oracle task began before role assignment or worker identity differs")
        worker_path = root / "workers" / f'{metadata["worker"]}.json'
        _check_worker(worker_path, teacher)
        worker_bindings[str(worker_path.absolute())] = sha256(worker_path)
        with np.load(task["input_npz"], allow_pickle=False) as archive:
            original_ids = archive["structure_ids"].astype(str)[task["indices"]]
            original_masks = np.asarray(archive["masks"])[task["indices"]]
        with np.load(score_path, allow_pickle=False) as archive:
            score_ids = archive["structure_ids"].astype(str)
            status = archive["status"].astype(str)
            energy = archive["energy_eV"]
            n_atoms = archive["n_atoms"]
            n_li = archive["n_Li"]
        if (not np.array_equal(score_ids, original_ids) or list(score_ids) != task["expected_structure_ids"]
                or not (len(score_ids) == len(status) == len(energy) == len(n_atoms) == len(n_li))):
            raise ValueError("Oracle task IDs or score array lengths differ")
        task_rows = metadata.get("rows")
        if not isinstance(task_rows, list) or len(task_rows) != len(score_ids):
            raise ValueError("Oracle task receipt is missing per-structure outcomes")
        if any(row.get("structure_id") != sid or row.get("status") != state
               for row, sid, state in zip(task_rows, score_ids, status)):
            raise ValueError("Oracle receipt outcomes disagree with the score arrays")
        if (metadata["status"] == "complete") != bool(np.all(status == "success")):
            raise ValueError("Oracle task completion status disagrees with row outcomes")
        task_bindings[tid] = {"score": _binding(score_path), "score_receipt": _binding(metadata_path),
                              "claim_owner": _binding(owner_path), "task_input": _binding(task["input_npz"])}
        for idx, sid in enumerate(score_ids):
            if sid not in wanted:
                continue
            if sid in selected:
                raise ValueError("A physical acquisition has multiple raw score tasks")
            key, record = expected[sid]
            if not np.array_equal(original_masks[idx], np.asarray(list(key), dtype=np.uint8)):
                raise ValueError("Scored task geometry disagrees with the staged vacancy mask")
            state = status[idx]
            if state == "success":
                if (not np.isfinite(energy[idx]) or int(n_li[idx]) != record["n_target"]
                        or float(task_rows[idx].get("energy_eV", np.nan)) != float(energy[idx])
                        or int(task_rows[idx].get("n_atoms", -1)) != int(n_atoms[idx])
                        or int(task_rows[idx].get("n_Li", -1)) != int(n_li[idx])):
                    raise ValueError("Score energy or lithium count is invalid")
                selected[sid] = {"status": "success", "energy_eV": float(energy[idx]),
                                 "n_atoms": int(n_atoms[idx]), "n_target": int(n_li[idx]),
                                 "physical_key": physical_key(stage_receipt["parent_sha256"], key, teacher),
                                 "task_id": tid, "worker": metadata["worker"]}
            elif state == "failed":
                selected[sid] = {"status": "failed", "physical_key": physical_key(stage_receipt["parent_sha256"], key, teacher),
                                 "task_id": tid, "worker": metadata["worker"],
                                 "error_type": task_rows[idx].get("error_type", "unknown"),
                                 "error": task_rows[idx].get("error", "")}
            else:
                raise ValueError("Unknown or nonterminal oracle score status")
    if set(selected) != set(expected):
        raise ValueError("Not every staged geometry has a terminal, certified score")
    return selected, task_count, worker_bindings, task_bindings


def merge(*, stage_dir, parent_path, base_data_path, schema_path, oracle_root,
          collection_receipt, output_dir, teacher="uma", new_weight=1.0,
          new_target_fraction=None):
    """Append successful, preassigned development labels to a new training NPZ.

    ``energy_eV`` is the scalar total energy from the verified UMA task; the
    saved training quantity divides it by the verified number of present atoms.
    If specified, a target fraction reweights new TRAIN rows separately within
    each (composition, frozen energy-bin) cell. Original rows are never changed.
    """
    if not math.isfinite(new_weight) or new_weight <= 0:
        raise ValueError("New-row weight must be finite and positive")
    if new_target_fraction is not None and (not math.isfinite(new_target_fraction) or not 0 < new_target_fraction < 1):
        raise ValueError("New target weight fraction must lie strictly between zero and one")
    if new_target_fraction is not None and new_weight != 1.0:
        raise ValueError("Specify a fixed new weight OR a target fraction, not both")
    stage_dir = Path(stage_dir)
    stage_path = stage_dir / "stage_receipt.json"
    record = json.loads(stage_path.read_text())
    if record.get("status") != "STAGED_BEFORE_SCORING":
        raise ValueError("Acquisition stage is not frozen before scoring")
    for name in ("candidates.npz", "attempts.csv", "assigned_roles.json"):
        if sha256(stage_dir / name) != record["outputs"][name]:
            raise ValueError(f"Staged {name} changed before merge")
    for name, path in (("parent", parent_path), ("base_data", base_data_path), ("schema", schema_path)):
        _assert_binding(record["inputs"][name], path, name)
    for kind in ("known_histories", "protected_histories"):
        for binding in record["inputs"][kind]:
            _assert_binding(binding, binding["path"], kind)
    _assert_binding(record["inputs"]["generation_manifest"],
                    record["inputs"]["generation_manifest"]["path"], "Generation manifest")
    _assert_binding(record["inputs"]["generation_history"],
                    record["inputs"]["generation_history"]["path"], "Generation history")
    for name, binding in record["inputs"]["generation_files"].items():
        _assert_binding(binding, binding["path"], f"Generation {name}")
    if record["inputs"]["previous_merge_receipt"] is not None:
        binding = record["inputs"]["previous_merge_receipt"]
        _assert_binding(binding, binding["path"], "Previous merge receipt")
    _assert_binding(record["inputs"]["generation_receipt"], record["inputs"]["generation_receipt"]["path"], "Generation receipt")
    permutations, host_atoms = _parent(parent_path)
    schema = _schema(schema_path, parent_path, record["origin_data_sha256"])
    if schema["identity"] != record["condition_schema_id"]:
        raise ValueError("Frozen condition schema identity changed")
    base, base_keys, base_roles, base_n = _data(base_data_path, permutations)
    role_data = json.loads((stage_dir / "assigned_roles.json").read_text())
    if len(role_data) != record["selected_unique_orbits"] or set(role_data) & set(base_keys.tolist()):
        raise ValueError("Assigned acquisition orbits are duplicated or already in the base dataset")
    for key, item in role_data.items():
        expected_role = _role(record["parent_sha256"], record["role_seed"], key,
                              record["new_validation_fraction"])
        if item["role"] != expected_role or item["role"] not in {"train", "val"}:
            raise ValueError("Assigned role does not match the pre-score deterministic policy")
    scored, task_count, workers, task_bindings = _scores(stage_dir, record, oracle_root, collection_receipt, teacher)
    success = [(key, item, scored[item["structure_id"]]) for key, item in role_data.items()
               if scored[item["structure_id"]]["status"] == "success"]
    failed = [(key, item, scored[item["structure_id"]]) for key, item in role_data.items()
              if scored[item["structure_id"]]["status"] == "failed"]
    for _, item, score in success:
        if score["n_atoms"] != host_atoms + item["n_target"]:
            raise ValueError("Total-energy denominator disagrees with the parent and occupancy")
    new_n = np.asarray([item["n_target"] for _, item, _ in success], dtype=np.int64)
    new_e = np.asarray([score["energy_eV"] / score["n_atoms"] for _, _, score in success], dtype=np.float64)
    new_roles = np.asarray([item["role"] for _, item, _ in success], dtype="<U5")
    new_bins = classify(schema, new_n, new_e) if len(success) else np.asarray([], dtype="<U6")
    new_weights = np.full(len(success), new_weight, dtype=np.float64)
    weight_cells = {}
    if new_target_fraction is not None:
        old_train = np.flatnonzero(base_roles == "train")
        old_bins = classify(schema, base_n[old_train], base["energy_eV_atom"][old_train])
        for n in record["counts"]:
            for b in ("low", "middle", "high"):
                new_sel = (new_n == n) & (new_bins == b) & (new_roles == "train")
                old_sel = (base_n[old_train] == n) & (old_bins == b)
                n_new = int(new_sel.sum())
                if not n_new:
                    weight_cells[f"{n}:{b}"] = {"new_train": 0, "target_reached": False}
                    continue
                old_total = float(np.asarray(base["weight"])[old_train[old_sel]].sum())
                if not math.isfinite(old_total) or old_total <= 0:
                    raise ValueError("Targeted new weight fraction needs existing TRAIN support in every added cell")
                assigned = (new_target_fraction / (1-new_target_fraction)) * old_total / n_new
                new_weights[new_sel] = assigned
                weight_cells[f"{n}:{b}"] = {"new_train": n_new, "old_weight": old_total,
                    "new_weight_each": assigned, "target_fraction": new_target_fraction,
                    "target_reached": True}
        # New validation rows retain a base weight of one and never influence
        # training reweighting or class-threshold fitting.
        new_weights[new_roles == "val"] = 1.0
    new_x = (1 - np.asarray([list(key) for key, _, _ in success], dtype=np.uint8).reshape(-1, 48))
    additions = {
        "occupancy": new_x, "n_Li": new_n, "n_target": new_n,
        "energy_eV_atom": new_e, "split": new_roles, "weight": new_weights,
        "structure_ids": np.asarray([item["structure_id"] for _, item, _ in success], dtype="<U19"),
        "canonical_masks": np.asarray([key for key, _, _ in success], dtype="<U48"),
        "source": np.asarray([f'iterative_r{record["round"]}_{record["method"]}' for _ in success]),
    }
    merged = {}
    rows = len(base_n)
    for name, values in base.items():
        if values.ndim == 0:
            merged[name] = values
        elif len(values) == rows and name in additions:
            merged[name] = np.concatenate((values, additions[name]), axis=0)
        else:
            raise ValueError(f"Cannot safely append unrecognized dataset field {name}")
    if "canonical_masks" not in merged:
        merged["canonical_masks"] = np.concatenate((base_keys, additions["canonical_masks"]))
    # Verify every original field is unchanged, including all protected VAL/TEST
    # values and row order. Newly appended rows retain their preassigned roles.
    for name, original in base.items():
        actual = merged[name]
        if original.ndim == 0:
            if not np.array_equal(original, actual):
                raise AssertionError(f"Original scalar field {name} changed")
        elif not np.array_equal(original, actual[:rows]):
            raise AssertionError(f"Original rows of {name} changed")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "training.npz", **merged)
    labels = []
    for index, (key, item, score) in enumerate(success):
        labels.append({"canonical_mask": key, "structure_id": item["structure_id"],
            "role": item["role"], "n_target": item["n_target"], "requested_bin": item["requested_bin"],
            "measured_bin": str(new_bins[index]), "status": "success", "energy_eV": score["energy_eV"],
            "n_atoms": score["n_atoms"], "energy_eV_atom": float(new_e[index]),
            "weight": float(new_weights[index]), "physical_key": score["physical_key"],
            "task_id": score["task_id"], "error_type": "", "error": ""})
    for key, item, score in failed:
        labels.append({"canonical_mask": key, "structure_id": item["structure_id"],
            "role": item["role"], "n_target": item["n_target"], "requested_bin": item["requested_bin"],
            "measured_bin": "", "status": "failed", "energy_eV": "", "n_atoms": "",
            "energy_eV_atom": "", "weight": "", "physical_key": score["physical_key"],
            "task_id": score["task_id"], "error_type": score["error_type"], "error": score["error"]})
    fields = ["canonical_mask", "structure_id", "role", "n_target", "requested_bin", "measured_bin",
              "status", "energy_eV", "n_atoms", "energy_eV_atom", "weight", "physical_key", "task_id",
              "error_type", "error"]
    with (output / "labels.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(labels)
    # Broadcast a certified physical score to every matching raw attempt while
    # preserving the original row count and stage status; no duplicate attempt
    # is promoted to a second unique oracle call.
    label_lookup = {row["structure_id"]: row for row in labels}
    with (stage_dir / "attempts.csv").open(newline="") as stream:
        attempts = list(csv.DictReader(stream))
    joined = []
    for attempt in attempts:
        row = dict(attempt)
        label = label_lookup.get(attempt.get("structure_id"), {})
        row.update(oracle_status=label.get("status", "not_queried"),
                   energy_eV_atom=label.get("energy_eV_atom", ""),
                   measured_bin=label.get("measured_bin", ""),
                   oracle_physical_key=label.get("physical_key", ""))
        joined.append(row)
    with (output / "attempts_scored.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)
    receipt = {
        "status": "MERGED_ITERATIVE_ROUND", "round": record["round"], "method": record["method"],
        "replicate": record["replicate"], "role_seed": record["role_seed"],
        "source_sha256": sha256(__file__),
        "new_validation_fraction": record["new_validation_fraction"],
        "origin_data_sha256": record["origin_data_sha256"], "condition_schema_id": schema["identity"],
        "teacher": teacher, "teacher_head": MODELS[teacher]["head"],
        "teacher_checkpoint_hashes": MODELS[teacher]["files"], "geometry_protocol": PROTOCOL,
        "score_total_energy_units": "eV", "training_energy_units": "eV per present atom",
        "attempts": len(joined), "staged_unique_oracle_keys": len(role_data),
        "terminal_successful_orbits": len(success), "terminal_failed_orbits": len(failed),
        "score_tasks_used": task_count, "new_train_rows": int(np.sum(new_roles == "train")),
        "new_val_rows": int(np.sum(new_roles == "val")), "new_test_rows": 0,
        "base_split_counts": dict(Counter(base_roles.tolist())),
        "base_role_external_protected_overlap_counts": record["base_role_external_protected_overlap_counts"],
        "output_split_counts": dict(Counter(merged["split"].astype(str).tolist())),
        "new_weight": new_weight, "new_target_fraction": new_target_fraction,
        "weight_by_training_cell": weight_cells,
        "inputs": {"stage_receipt": _binding(stage_path), "parent": _binding(parent_path),
                   "base_data": _binding(base_data_path), "schema": _binding(schema_path),
                   "oracle_registry": _binding(Path(oracle_root) / "registry.json"),
                   "collection_receipt": _binding(collection_receipt)},
        "worker_receipt_hashes": workers,
        "task_evidence": task_bindings,
        "outputs": {name: sha256(output / name) for name in ("training.npz", "labels.csv", "attempts_scored.csv")},
        "accounting": "Attempt rows retained; physical oracle keys can be deduplicated across arms and rounds. Terminal failures stay visible and are not fitted.",
        "weight_scope": "Target fractions, when requested, are within each occupied count/frozen-bin TRAIN cell before class balancing; validation rows retain weight one.",
    }
    _write_json(output / "merge_receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("stage", help="Freeze attempt selection and new roles before scoring")
    for name in ("parent", "base-data", "schema", "generation-dir", "output-dir"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--known-history", type=Path, action="append", required=True)
    prepare.add_argument("--protected-history", type=Path, action="append", required=True)
    prepare.add_argument("--previous-merge-receipt", type=Path)
    prepare.add_argument("--method", required=True)
    prepare.add_argument("--replicate", required=True)
    prepare.add_argument("--round", dest="round_index", type=int, required=True)
    prepare.add_argument("--role-seed", type=int, required=True)
    prepare.add_argument("--counts", type=int, nargs="+", required=True)
    prepare.add_argument("--bins", nargs="+", choices=["low", "middle", "high"], required=True)
    prepare.add_argument("--attempts-per-cell", type=int, required=True)
    prepare.add_argument("--queries-per-cell", type=int, required=True)
    prepare.add_argument("--allowed-swaps", type=int, nargs="+", default=[1, 2, 3],
        help="Predeclared mutation edit budgets; record and verify every seed/child pair")
    prepare.add_argument("--val-fraction", type=float, default=0.0,
        help="New-orbit VAL fraction; default 0 keeps the original validation set unchanged")
    finish = sub.add_parser("merge", help="Verify oracle provenance and append terminal labels")
    for name in ("stage-dir", "parent", "base-data", "schema", "oracle-root", "collection-receipt", "output-dir"):
        finish.add_argument("--" + name, type=Path, required=True)
    finish.add_argument("--teacher", choices=["uma"], default="uma")
    weights = finish.add_mutually_exclusive_group()
    weights.add_argument("--new-weight", type=float, default=1.0)
    weights.add_argument("--new-target-fraction", type=float,
        help="Target new-data TRAIN weight fraction within each composition/energy-bin cell")
    args = parser.parse_args(argv)
    require_compute_allocation()
    if args.command == "stage":
        receipt = stage(parent_path=args.parent, base_data_path=args.base_data,
            schema_path=args.schema, generation_dir=args.generation_dir,
            known_histories=args.known_history, protected_histories=args.protected_history,
            method=args.method, replicate=args.replicate, round_index=args.round_index,
            role_seed=args.role_seed, counts=args.counts, bins=args.bins,
            attempts_per_cell=args.attempts_per_cell, queries_per_cell=args.queries_per_cell,
            val_fraction=args.val_fraction, output_dir=args.output_dir,
            previous_merge_receipt=args.previous_merge_receipt, allowed_swaps=args.allowed_swaps)
        summary = {key: receipt[key] for key in ("status", "round", "method", "replicate",
                   "attempts", "selected_unique_orbits", "stage_status_counts", "queries_by_cell")}
    else:
        receipt = merge(stage_dir=args.stage_dir, parent_path=args.parent,
            base_data_path=args.base_data, schema_path=args.schema, oracle_root=args.oracle_root,
            collection_receipt=args.collection_receipt, output_dir=args.output_dir,
            teacher=args.teacher, new_weight=args.new_weight,
            new_target_fraction=args.new_target_fraction)
        summary = {key: receipt[key] for key in ("status", "round", "method", "replicate",
                   "attempts", "staged_unique_oracle_keys", "terminal_successful_orbits",
                   "terminal_failed_orbits", "new_train_rows", "new_val_rows")}
    print(json.dumps(summary, indent=2))
    return receipt


if __name__ == "__main__":
    main()

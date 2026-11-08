"""Freeze the complete primary assessment before generating or scoring it.

This command reads only protected inputs, completed development evidence and
checkpoints. It never reads a final pool, performs inference, or submits a job.
History records known parent-mask orbits, not a cache of relaxed energies.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import uuid

import numpy as np

from cathode_cvae.acquisition import canonical_keys
from cathode_cvae.conditions import load_schema
from cathode_cvae.data import target_indices, validate_permutation_group
from cathode_cvae.runtime import require_compute_allocation


PROJECT = Path(__file__).resolve().parents[2]
NAMESPACE = "conditioning-final-20261107-v1"
SEEDS = (20261106, 20261107, 20261108)
COUNTS = (12, 24, 36)
ATTEMPTS_PER_CELL = 128
PROTOCOL = "pristine-derived deletion; unrelaxed; exact canonical parent coordinates"
MACE_FAMILIES = ("cvae_d4", "symmetry", "graph_encoder", "graph_context", "uniform", "mutation")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):h.update(block)
    return h.hexdigest()


def named_path(value):
    if "=" not in value:raise argparse.ArgumentTypeError("Use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:raise argparse.ArgumentTypeError("NAME and PATH must be nonempty")
    return name, Path(path)


def family_specs():
    """All comparisons stay frozen regardless of the development recommendation."""
    rows = []
    variants = [(f"cvae_d{d}", "balance_1_d8" if d == 8 else f"latent_d{d}", d, "dense", False, "canonical") for d in [4,8,16,32]]
    variants += [("ar_baseline", "no_latent", 0, "dense", False, "canonical"),
                 ("symmetry", "symmetry_augment", 4, "dense", False, "augment"),
                 ("graph_encoder", "graph_encoder", 4, "graph", False, "canonical"),
                 ("graph_context", "graph_context", 4, "graph", True, "canonical")]
    for method, folder, dimension, encoder, context, orientation in variants:
        for seed in SEEDS:
            rows.append(dict(method=method, fit_seed=seed, replicate_id=seed,
                fit_folder=f"{folder}_{seed}", latent_dim=dimension,
                encoder_kind=encoder, graph_context=context, orientation=orientation))
    for sampler in ["uniform", "mutation"]:
        rows.extend(dict(method=sampler, replicate_id=seed, sampler=sampler) for seed in SEEDS)
    return rows


def validate_keys(keys, n_sites, permutations, *, context):
    """Exact orbit identity, with no integer overflow or zero-padding adaptation."""
    keys = list(map(str, keys))
    if any(len(k) != n_sites or set(k)-{"0", "1"} for k in keys):
        raise ValueError(context+": keys must be exact-length vacancy-one strings; no cell-size padding")
    if not keys:return set()
    vacancy = np.frombuffer("".join(keys).encode("ascii"), dtype=np.uint8).reshape(len(keys), n_sites)-ord("0")
    actual = canonical_keys(1-vacancy, permutations)
    if not np.array_equal(actual, np.asarray(keys)):
        raise ValueError(context+": a mask is not canonical under the supplied actual parent symmetry")
    return set(keys)


def load_pool(path, parent_sha, n_sites, permutations, host_atoms):
    before = digest(path)
    with np.load(path, allow_pickle=False) as z:
        if "parent_sha256" not in z or str(z["parent_sha256"].item()) != parent_sha:
            raise ValueError("Candidate pool belongs to another or unidentified parent: "+str(path))
        if "masks" in z:
            masks = z["masks"]
            if "occupancy" in z and not np.array_equal(masks, 1-z["occupancy"]):
                raise ValueError("Candidate vacancy/present conventions disagree")
        elif "occupancy" in z:masks = 1-z["occupancy"]
        else:raise ValueError("Candidate pool has neither masks nor occupancy")
        if masks.ndim != 2 or masks.shape[1] != n_sites or not len(masks) or not np.isin(masks, [0, 1]).all():
            raise ValueError("Candidate pool must contain nonempty binary masks on this parent, without padding")
        ids = z["structure_ids"].astype(str)
        if ids.shape != (len(masks),) or len(set(ids)) != len(ids):raise ValueError("Candidate IDs must be unique per physical pool")
        counts = n_sites-masks.sum(1)
        for field in ["n_Li", "n_target"]:
            if field in z and not np.array_equal(z[field], counts):raise ValueError("Candidate target counts disagree with masks")
        if "n_atoms" in z and not np.array_equal(z["n_atoms"], host_atoms+counts):
            raise ValueError("Candidate atom counts disagree with fixed host chemistry")
    keys = ["".join(map(str, r.astype(int))) for r in masks]
    if len(set(keys)) != len(keys):raise ValueError("Candidate pools must have unique canonical orbits")
    known = validate_keys(keys, n_sites, permutations, context=str(path))
    if digest(path) != before:raise ValueError("Candidate pool changed during history freezing")
    return known, dict(path=str(path.resolve()), sha256=before, rows=len(keys), parent_sha256=parent_sha,
        role="All declared candidates, including diagnostics and mutation seed orbits; mask knowledge only")


def decode_audit_orbit(value, n_sites):
    """The paired-audit column stores hexadecimal *uint8 vacancy bytes*."""
    try:raw = bytes.fromhex(value)
    except ValueError as error:raise ValueError("Paired-audit orbit is not hexadecimal uint8 bytes") from error
    if len(raw) != n_sites or any(b not in {0, 1} for b in raw):
        raise ValueError("Paired-audit orbit must decode to exactly the parent site count; never pad another cell")
    return "".join(map(str, raw))


def geometry_keys(csv_path, receipt_path, data_path, parent_sha, n_sites, permutations):
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "PASS":raise ValueError("Paired geometry audit is incomplete")
    bindings = receipt.get("input_hashes", {})
    parent_hashes = [v for k, v in bindings.items() if k.endswith("parents/O3_N48_pilot.npz")]
    data_hashes = [v for k, v in bindings.items() if k.endswith("geometry_uma_fixed.npz")]
    if parent_hashes != [parent_sha] or data_hashes != [digest(data_path)]:
        raise ValueError("Geometry audit parent/source hashes differ from the actual source")
    with csv_path.open() as stream:rows = list(csv.DictReader(stream))
    with np.load(data_path, allow_pickle=False) as z:
        occupancy, sid, cohort, n = z["occupancy"], z["structure_ids"].astype(str), z["cohort"].astype(str), z["n_Li"]
    if occupancy.ndim != 2 or occupancy.shape[1] != n_sites:
        raise ValueError("Geometry archive has another mobile sublattice; no padding is allowed")
    keys, frames = [], []
    for row in rows:
        index = int(row["deletion_frame"])
        if not 0 <= index < len(occupancy) or cohort[index] != "deletion" or sid[index] != row["structure_id"]:
            raise ValueError("Geometry audit does not identify the exact deletion source frame")
        if int(n[index]) != int(row["n_Li"]):raise ValueError("Geometry deletion count differs from audit")
        keys.append(decode_audit_orbit(row["initial_occupation_orbit"], n_sites));frames.append(index)
    if len(rows) != receipt["lineages"] or len(set(frames)) != len(frames):
        raise ValueError("Geometry lineage coverage differs from the completed audit")
    actual = canonical_keys(occupancy[frames], permutations)
    if not np.array_equal(actual, keys):raise ValueError("Geometry audit keys differ from actual deletion masks")
    return set(keys), dict(path=str(csv_path.resolve()), sha256=digest(csv_path), rows=len(rows),
        audit_receipt_sha256=digest(receipt_path), geometry_archive_sha256=digest(data_path),
        encoding="Hexadecimal uint8 vacancy bytes decoded to exactly the parent site count",
        role="Pristine deletion-mask knowledge only; no relaxed coordinates, energies, forces or stresses enter a static cache")


def source_manifests(run):
    records = []
    for path in sorted((run/"executed_sources").glob("*/manifest.json")):
        files = json.loads(path.read_text())
        for relative, expected in files.items():
            p = PurePosixPath(relative)
            if p.is_absolute() or ".." in p.parts or p.parts[0] != "cathode_cvae":
                raise ValueError("Unsafe executed source-manifest path")
            if digest(path.parent/relative) != expected:raise ValueError("Executed source snapshot changed")
        records.append(dict(path=str(path.resolve()), sha256=digest(path), files=files))
    if not records:raise ValueError("No immutable executed-source manifests found")
    return records


def checkpoint_entry(spec, run, parent_sha, data_sha, split_sha, schema_id, snapshots):
    folder = run/"fits"/spec["fit_folder"]
    config_path, summary_path = folder/"configuration.json", folder/"summary.json"
    cfg, summary = json.loads(config_path.read_text()), json.loads(summary_path.read_text())
    expected = dict(model_format_version=2, kind="ar_baseline" if spec["latent_dim"] == 0 else "ar_cvae",
        n_sites=48, condition_mode="multibin", condition_dim=4, latent_dim=spec["latent_dim"],
        hidden_dim=64, encoder_kind=spec["encoder_kind"], graph_context=spec["graph_context"],
        orientation=spec["orientation"], seed=spec["fit_seed"], parent_sha256=parent_sha,
        data_sha256=data_sha, split_sha256=split_sha, condition_schema_id=schema_id,
        train_rows=40157, val_rows=4967, test_rows=5123, counts=list(COUNTS),
        teacher="uma", teacher_head="omat", dtype="float32", tf32=False,
        geometry_protocol=PROTOCOL, class_balance=True, beta_final=.1,
        maximum_updates=4000, count_likelihood=True)
    for key, value in expected.items():
        if cfg.get(key) != value:raise ValueError(f"{spec['fit_folder']}: incompatible {key}: {cfg.get(key)!r} != {value!r}")
    if cfg.get("descriptor_schema") is not None:raise ValueError("Descriptor pilot is not part of this primary final manifest")
    if summary.get("status") != "FIT_COMPLETE" or summary.get("optimizer_updates") != 4000 or summary.get("heldout_test_evaluated") is not False:
        raise ValueError("Final comparisons need completed matched training and protected test preservation")
    choice = summary["selection_checkpoints"]["0.1"]
    if choice["filename"] != "best_validation_beta_0p1.pt":raise ValueError("Unexpected selected checkpoint rule")
    checkpoint = folder/choice["filename"]
    if digest(checkpoint) != choice["sha256"]:raise ValueError("Checkpoint bytes differ from completed fit receipt")
    from cathode_cvae.learning.runtime import trusted_checkpoint
    stored = trusted_checkpoint(checkpoint,"cpu")
    embedded = stored.get("configuration",{})
    for key,value in expected.items():
        if embedded.get(key) != value:raise ValueError("Checkpoint-embedded configuration disagrees with frozen "+key)
    for key in ["trainer_sha256","model_source_sha256"]:
        if embedded.get(key) != cfg.get(key):raise ValueError("Embedded and sidecar source hashes disagree")
    if stored.get("selection_beta") != .1 or stored.get("epoch") != choice["epoch"]:
        raise ValueError("Checkpoint state does not match its declared validation selection")
    matches = [r for r in snapshots if r["files"].get("cathode_cvae/learning/train_mask.py") == cfg.get("trainer_sha256")
               and r["files"].get("cathode_cvae/learning/models.py") == cfg.get("model_source_sha256")]
    if not matches:raise ValueError("Checkpoint trainer/model source hashes have no preserved executed-source match")
    task_receipts = list(run.glob(f"*.claims/{spec['fit_folder']}/receipt.json"))
    if len(task_receipts) != 1:raise ValueError("Fit must have one authoritative completed task receipt")
    task = json.loads(task_receipts[0].read_text())
    task_list = task_receipts[0].parent.parent.with_suffix(".json")
    if task.get("status") != "COMPLETE" or task.get("tasks_sha256") != digest(task_list):
        raise ValueError("Fit execution receipt does not match its immutable task list")
    result = dict(method=spec["method"], fit_seed=spec["fit_seed"], replicate_id=spec["replicate_id"],
        checkpoint=str(checkpoint.resolve()), expected_checkpoint_sha256=choice["sha256"],
        frozen_configuration_sha256=digest(config_path), frozen_summary_sha256=digest(summary_path),
        trainer_sha256=cfg["trainer_sha256"], model_source_sha256=cfg["model_source_sha256"],
        matching_executed_source_manifest_hashes=[r["sha256"] for r in matches],
        fit_task_receipt=str(task_receipts[0].resolve()), fit_task_receipt_sha256=digest(task_receipts[0]),
        latent_dim=spec["latent_dim"], model_role="AR baseline, no latent" if spec["latent_dim"] == 0 else "CVAE",
        selected_checkpoint_epoch=stored["epoch"],selected_checkpoint_optimizer_updates=stored.get("optimizer_updates"),
        orientation=spec["orientation"], encoder_kind=spec["encoder_kind"], graph_context=spec["graph_context"],
        family_fit_replicates=3)
    return result


def publish_once(run, outputs, protocol_path):
    """Publish staged files without replacing any existing evidence; protocol last."""
    paths = list(outputs)
    if protocol_path not in outputs:raise ValueError("A protocol commit marker is required")
    if any(p.exists() or p.is_symlink() for p in paths):
        raise FileExistsError("Final freeze outputs already exist; preserve the original assessment")
    stage = run/(".freeze_final_"+uuid.uuid4().hex)
    stage.mkdir(parents=True, exist_ok=False)
    staged = {}
    for i, path in enumerate(paths):
        temporary = stage/str(i)
        with temporary.open("xb") as stream:
            stream.write(outputs[path]);stream.flush();os.fsync(stream.fileno())
        staged[path] = temporary
    for path in [p for p in paths if p != protocol_path]+[protocol_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        os.link(staged[path], path)
    shutil.rmtree(stage)


def freeze(args):
    root = args.root.resolve(); run = root/"runs/conditioning_20261107"
    protocol_path, manifest_path = run/"final_protocol.json", run/"final_generation_manifest.json"
    history_path, history_receipt_path = run/"data/history_before_final.json", run/"data/history_before_final_receipt.json"
    if any(p.exists() or p.is_symlink() for p in [protocol_path,manifest_path,history_path,history_receipt_path]):
        raise FileExistsError("A final freeze already exists; it cannot be overwritten")
    final_output = args.final_output_dir or run/"final"
    if any(p.exists() for p in [final_output,run/"final",run/"development/final",run/"collected/final",run/"evaluation/final"]):
        raise ValueError("A final-output/score directory already exists; do not freeze after generation")
    initial = json.loads((root/"artifacts/conditioning_20261107/protocol.json").read_text())
    parent_path = root/"artifacts/current/core/data/parents/O3_N48_pilot.npz"
    data_path = root/"artifacts/current/improvement_v1/training_round0.npz"
    parent_sha, data_sha = digest(parent_path), digest(data_path)
    if parent_sha != initial["data"]["parent_sha256"] or data_sha != initial["data"]["round0_sha256"]:
        raise ValueError("Protected parent or round-zero bytes changed")
    with np.load(parent_path,allow_pickle=False) as z:parent = {k:z[k] for k in z.files}
    n_sites = len(target_indices(parent)); permutations = parent["permutations"]
    if n_sites != 48:raise ValueError("This prespecified assessment is the 48-site parent; never pad another cell")
    validate_permutation_group(permutations)
    with np.load(data_path,allow_pickle=False) as z:
        ids, roles, protected_keys = z["structure_ids"].astype(str), z["split"].astype(str), z["canonical_masks"].astype(str)
    if Counter(roles) != Counter(train=40157,val=4967,test=5123) or len(set(protected_keys)) != len(protected_keys):
        raise ValueError("Protected orbit roles/counts changed")
    split_sha = hashlib.sha256("\n".join(f"{s}\t{r}" for s,r in zip(ids,roles)).encode()).hexdigest()
    schema_path = root/"artifacts/overnight_20261107/data/condition_schema_uma.json"
    schema = load_schema(schema_path)
    snapshots = source_manifests(run)
    specs = family_specs()
    models = [checkpoint_entry(s, run, parent_sha, data_sha, split_sha, schema["identity"], snapshots)
              if "checkpoint" not in s and "fit_folder" in s else s for s in specs]
    if len(models) != 30 or len(models)*len(COUNTS)*3*ATTEMPTS_PER_CELL != 34560:
        raise AssertionError("Final family/replicate budget changed")
    recommendation_family = {"graph_context_pilot":"graph_context"}.get(args.recommendation_family,args.recommendation_family)
    if recommendation_family not in {r["method"] for r in models}:
        raise ValueError("Development recommendation must name a frozen comparison family")
    recommendations = args.recommendation_source or [p for p in [root/"artifacts/conditioning_20261107/selected_policy.json",
        root/"artifacts/conditioning_20261107/selected_models.json"] if p.exists()]
    recommendation_sources = [dict(path=str(p.resolve()),sha256=digest(p)) for p in recommendations]
    base_path = run/"data/history_before_development.json"
    base = json.loads(base_path.read_text())
    if not isinstance(base,list) or len(base) != 239867:raise ValueError("Starting history differs from its declared frozen scope")
    history = validate_keys(base,n_sites,permutations,context="Original frozen development history")
    if not set(protected_keys) <= history:raise ValueError("Every protected data role must already be known history")
    sources = [dict(name="history_before_development",path=str(base_path.resolve()),sha256=digest(base_path),rows=len(base),new_orbits=len(history))]
    completion_items = args.completion_receipt or []
    if len({k for k,_ in completion_items}) != len(completion_items):raise ValueError("Completion receipt names must be unique")
    completion = dict(completion_items)
    pools = {name:run/f"development/{name}/candidates.npz" for name in ["policy","latent","spatial","descriptor"]}
    for name in pools:
        receipt_path = completion.get(name,run/f"evaluation/{name}/receipt.json")
        r = json.loads(receipt_path.read_text())
        if r.get("status") != "PASS":raise ValueError("Required development evidence is incomplete: "+name)
        pool, record = load_pool(pools[name],parent_sha,n_sites,permutations,len(parent["numbers"])-n_sites)
        record.update(name=name,completion_receipt=str(receipt_path.resolve()),completion_receipt_sha256=digest(receipt_path),new_orbits=len(pool-history))
        sources.append(record);history.update(pool)
    additional = args.additional_candidates or []
    if len({k for k,_ in additional}) != len(additional) or set(k for k,_ in additional)&set(pools):
        raise ValueError("Additional candidate pool names must be unique and distinct from required stages")
    for name, path in additional:
        pool, record = load_pool(path,parent_sha,n_sites,permutations,len(parent["numbers"])-n_sites)
        if name in completion:
            receipt_path = completion[name]
            if json.loads(receipt_path.read_text()).get("status") != "PASS":
                raise ValueError("Explicit additional development evidence is incomplete: "+name)
            record.update(completion_receipt=str(receipt_path.resolve()),completion_receipt_sha256=digest(receipt_path))
        record.update(name="explicit_additional_"+name,new_orbits=len(pool-history));sources.append(record);history.update(pool)
    supplied_pool_names = set(pools)|{name for name,_ in args.additional_candidates or []}
    if set(completion)-supplied_pool_names:raise ValueError("A completion receipt has no corresponding declared development pool")
    audit_dir = root/"reports/conditioning_20261107/relaxed_target_audit"
    geom, record = geometry_keys(audit_dir/"paired_lineages.csv",audit_dir/"receipt.json",
        root/"artifacts/current/core/data/geometry_uma_fixed.npz",parent_sha,n_sites,permutations)
    record.update(name="paired_geometry_deletion_masks_only",new_orbits=len(geom-history));sources.append(record);history.update(geom)
    history_bytes = (json.dumps(sorted(history))+"\n").encode()
    history_sha = hashlib.sha256(history_bytes).hexdigest()
    manifest = dict(status="FROZEN_BEFORE_FINAL_GENERATION",namespace=NAMESPACE,
        count_per_composition_bin_replicate=ATTEMPTS_PER_CELL,counts=list(COUNTS),requested_bins=["low","middle","high"],
        expected_primary_attempts=34560,diagnostics=False,models=models,
        expected_parent_sha256=parent_sha,expected_training_data_sha256=data_sha,
        expected_condition_schema_id=schema["identity"],history_path=str(history_path.resolve()),
        history_sha256=history_sha)
    manifest_bytes = (json.dumps(manifest,indent=2)+"\n").encode()
    audit = dict(status="FROZEN_BEFORE_FINAL_GENERATION",mask_convention="one=vacancy; exactly48mobile-site bits",
        parent_sha256=parent_sha,protected_data_sha256=data_sha,protected_split_sha256=split_sha,
        all_protected_roles_already_in_starting_history=True,base_history_orbits=len(base),final_history_orbits=len(history),
        sources=sources,history_sha256=history_sha,
        energy_cache_entries_imported=0,geometry_scope="Only parent mask identities; relaxed properties never relabeled as static deletions",
        source_sha256=digest(__file__))
    final = dict(status="FROZEN_BEFORE_FINAL_GENERATION",frozen_UTC=datetime.now(timezone.utc).isoformat(),
        namespace=NAMESPACE,source_sha256=digest(__file__),counts=list(COUNTS),requested_bins=["low","middle","high"],
        attempts_per_cell=ATTEMPTS_PER_CELL,model_or_sampler_replicate_entries=30,expected_primary_attempts=34560,
        diagnostic_attempts=0,recommendation_family_declared_on_development=recommendation_family,
        supplied_development_family_name=args.recommendation_family,
        development_family_aliases={"graph_context_pilot":"graph_context"},
        recommendation_scope="Declared before final generation; all comparisons remain regardless of recommendation; final labels cannot select a new winner",
        recommendation_sources=recommendation_sources,
        final_output_directory=str(final_output.resolve()),parent_sha256=parent_sha,data_sha256=data_sha,split_sha256=split_sha,
        condition_schema_sha256=digest(schema_path),condition_schema_id=schema["identity"],
        history_path=str(history_path.resolve()),history_sha256=history_sha,history_orbits=len(history),
        generation_manifest_path=str(manifest_path.resolve()),generation_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        family_replicates=dict(Counter(r["method"] for r in models)),
        UMA=dict(scope="All frozen primary attempts",expected_attempts=34560),
        MACE=dict(scope="Select by method membership only, never by energy",families=list(MACE_FAMILIES),
                  expected_attempts=20736,exactly_matched_UMA_structures=True),
        source_snapshots=snapshots,
        provenance_scope="Trainer/model hashes checked against preserved snapshots; complete task receipts retained. Transitive execution provenance is not guessed from a matching top-level hash.",
        protected_test_labels_used_for_selection=False,new_DFT=False,new_MLIP_finetuning=False,
        input_protocol_sha256=digest(root/"artifacts/conditioning_20261107/protocol.json"))
    audit_bytes = (json.dumps(audit,indent=2)+"\n").encode()
    final["history_receipt_sha256"] = hashlib.sha256(audit_bytes).hexdigest()
    outputs = {history_path:history_bytes,history_receipt_path:audit_bytes,
               manifest_path:manifest_bytes,protocol_path:(json.dumps(final,indent=2)+"\n").encode()}
    publish_once(run,outputs,protocol_path)
    return dict(status=final["status"],protocol=str(protocol_path),protocol_sha256=digest(protocol_path),
                manifest=str(manifest_path),manifest_sha256=digest(manifest_path),history=str(history_path),
                history_sha256=history_sha,history_orbits=len(history),primary_attempts=34560,MACE_attempts=20736)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=PROJECT)
    p.add_argument("--recommendation-family",required=True)
    p.add_argument("--recommendation-source",type=Path,action="append")
    p.add_argument("--completion-receipt",type=named_path,action="append")
    p.add_argument("--additional-candidates",type=named_path,action="append")
    p.add_argument("--final-output-dir",type=Path)
    args=p.parse_args(argv);require_compute_allocation()
    print(json.dumps(freeze(args),indent=2))


if __name__ == "__main__":main()

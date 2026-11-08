"""Unseal the original paired cohort TEST labels only after the final gate.

Classification uses the two already-frozen TRAIN schemas. Failed relaxations
remain in the 400-per-composition denominators, with unknown endpoint classes.
This script produces assessment-only CSVs; it never updates learning inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from paired_common import atomic_json, check_prepared, geometry_hash, identity, load_npz, sha256, verify_teacher_receipt


def check_gate(gate_path, joint, raw, static, paired):
    """Authenticate final approval before reading any TEST energy value."""
    gate = json.loads(Path(gate_path).read_text())
    if gate["status"] != "APPROVED_FINAL_SCORING_AFTER_ALL_PROPOSALS_AND_INDEPENDENT_RAW_AUDIT":
        raise ValueError("Final-only unsealing has not been approved")
    sources = dict(joint_protocol=joint, independent_raw_audit=raw, static_receipt=static, paired_receipt=paired)
    loaded = {}
    for name, path in sources.items():
        if sha256(path) != gate[name+"_sha256"]:
            raise ValueError("Final gate input changed: "+name)
        loaded[name] = json.loads(Path(path).read_text())
    if loaded["independent_raw_audit"]["status"] != "PASS":
        raise ValueError("Raw proposal audit has not passed")
    if loaded["static_receipt"]["status"] != "FROZEN_BEFORE_SCORING":
        raise ValueError("Static proposal cohort is not frozen")
    if loaded["paired_receipt"]["status"] != "FROZEN_PROPOSALS_AND_ENDPOINT_TASKS_NO_SCFS":
        raise ValueError("Endpoint proposal cohort is not frozen")
    if loaded["joint_protocol"].get("no_more_model_selection") is not True:
        raise ValueError("Model selection boundary is not closed")
    return dict(path=str(Path(gate_path).absolute()), sha256=sha256(gate_path),
                approved_UTC=gate["created_UTC"], sources={name:dict(path=str(Path(path).absolute()),sha256=sha256(path)) for name,path in sources.items()})


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def initial_energy_or_unknown(arrays, record):
    """An interrupted zero-call task has no energy array or scalar label."""
    if "initial_energy_eV" not in arrays:
        if record.get("initial_energy_eV") is not None:
            raise ValueError("Initial scalar energy has no matching raw array")
        return None
    value=float(arrays["initial_energy_eV"])
    if not np.isfinite(value):
        if record.get("initial_energy_eV") is not None:
            raise ValueError("Nonfinite initial array disagrees with scalar label")
        return None
    if value!=record.get("initial_energy_eV"):
        raise ValueError("Initial energy scalar mismatch")
    return value


def main(argv=None):
    from cathode_cvae.runtime import require_compute_allocation
    from cathode_cvae.conditions import classify, load_schema
    from scipy.stats import spearmanr, kendalltau
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["prepared", "queue", "collection", "scoring-gate", "joint-protocol", "raw-audit", "static-receipt", "paired-receipt", "output-dir"]:
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args(argv); require_compute_allocation()
    gate = check_gate(a.scoring_gate, a.joint_protocol, a.raw_audit, a.static_receipt, a.paired_receipt)
    _, context = check_prepared(a.prepared)
    cohort = load_npz(a.prepared/"cohort.npz"); parent = load_npz(a.prepared/"parent.npz")
    tasks = json.loads((a.prepared/"tasks_all.json").read_text())["tasks"]
    sealed_path = a.collection/"sealed_test_manifest.json"
    sealed = json.loads(sealed_path.read_text())
    cohort_sha = sha256(a.prepared/"cohort.npz")
    if sealed["status"] != "TEST_ENERGIES_SEALED" or sealed["energy_values_read"] is not False or sealed["original_cohort_roles_sha256"] != cohort_sha:
        raise ValueError("Original TEST sealing provenance changed")
    seals = {r["task_id"]:r for r in sealed["records"]}
    schemas = {}; bindings = {}
    for target in ["deletion", "endpoint"]:
        path = a.collection/f"condition_schema_{target}.json"; schema = load_schema(path)
        training = a.collection/f"learning_{target}.npz"
        if schema["provenance"]["data_sha256"] != sha256(training) or schema["provenance"]["endpoint_context_hash"] != context["context_hash"]:
            raise ValueError("Frozen TRAIN schema binding changed")
        schemas[target] = schema
        bindings[target] = dict(path=str(path.absolute()), sha256=sha256(path), identity=schema["identity"], training_data_sha256=sha256(training))
    test_indices = np.flatnonzero(cohort["split"].astype(str) == "test")
    if len(test_indices) != 1200 or set(seals) != {tasks[i]["task_id"] for i in test_indices}:
        raise ValueError("Original 1200 TEST identities changed")
    rows, sources, workers = [], [], {}
    for i in test_indices:
        task = tasks[int(i)]; seal = seals[task["task_id"]]
        directory = a.queue/"results"/task["task_id"]
        result_path, arrays_path = directory/"result.json", directory/"arrays.npz"
        claim_path = a.queue/"claims"/(task["task_id"]+".json")
        for path, expected in [(result_path,seal["result_sha256"]),(arrays_path,seal["arrays_sha256"]),(claim_path,seal["claim_sha256"])]:
            if sha256(path) != expected: raise ValueError("Sealed raw evidence changed: "+str(path))
        result, claim = json.loads(result_path.read_text()), json.loads(claim_path.read_text())
        if result["context_hash"] != context["context_hash"] or result["physical_key"] != task["physical_key"] or claim["task"]["physical_key"] != task["physical_key"]:
            raise ValueError("TEST task physical/context binding differs")
        worker_id = claim["worker"]
        if worker_id != seal["worker"]: raise ValueError("TEST claim worker changed")
        if worker_id not in workers:
            path = a.queue/"workers"/(worker_id+".json"); worker=json.loads(path.read_text())
            verify_teacher_receipt(worker["model_receipt"], context)
            workers[worker_id] = dict(path=str(path.absolute()),sha256=sha256(path),model=worker["model_receipt"])
        if result.get("model_receipt_sha256", identity(workers[worker_id]["model"])) != identity(workers[worker_id]["model"]):
            raise ValueError("Result inference binding differs from worker")
        n = int(cohort["n_target"][i]); natoms = int(result["n_atoms"])
        row = dict(cohort_row=int(i),task_id=task["task_id"],structure_id=str(cohort["structure_ids"][i]),
                   n_target=n,split="test",status=result["status"],endpoint_usable=bool(result["endpoint_labels_usable"]),
                   deletion_energy_eV=None,endpoint_energy_eV=None,deletion_energy_eV_atom=None,endpoint_energy_eV_atom=None,
                   deletion_bin="unknown",endpoint_bin="unknown",endpoint_fmax_eV_A=result["endpoint_force_max_eV_A"])
        with np.load(arrays_path, allow_pickle=False) as z:
            if not np.array_equal(z["occupancy"], cohort["occupancy"][i]) or not np.array_equal(z["cell_A"],parent["cell_A"]):
                raise ValueError("Original TEST mask/cell changed")
            slots = z["parent_indices"].astype(int)
            if not np.array_equal(z["numbers"],parent["numbers"][slots]) or not np.array_equal(z["initial_positions_A"],parent["positions_A"][slots]):
                raise ValueError("TEST initial geometry differs from pristine deletion")
            if geometry_hash(z["numbers"],z["initial_positions_A"],z["cell_A"]) != result["initial_geometry_hash"]:
                raise ValueError("Initial geometry hash mismatch")
            e0 = initial_energy_or_unknown(z,result)
            if e0 is not None:
                row.update(deletion_energy_eV=e0,deletion_energy_eV_atom=e0/natoms,
                           deletion_bin=str(classify(schemas["deletion"],[n],[e0/natoms])[0]))
            if result["endpoint_labels_usable"]:
                e1 = float(z["endpoint_energy_eV"])
                fmax=float(np.linalg.norm(z["endpoint_forces_eV_A"],axis=1).max())
                if not np.isfinite(e1) or e1 != result["endpoint_energy_eV"] or fmax != result["endpoint_force_max_eV_A"] or not fmax < context["optimizer"]["fmax_eV_A"]:
                    raise ValueError("Usable TEST endpoint energy/physical force mismatch")
                if geometry_hash(z["numbers"],z["endpoint_positions_A"],z["cell_A"]) != result["endpoint_literal_geometry_hash"]:
                    raise ValueError("TEST endpoint geometry mismatch")
                row.update(endpoint_energy_eV=e1,endpoint_energy_eV_atom=e1/natoms,
                           endpoint_bin=str(classify(schemas["endpoint"],[n],[e1/natoms])[0]))
        rows.append(row)
        sources.append(dict(task_id=task["task_id"],result_sha256=seal["result_sha256"],arrays_sha256=seal["arrays_sha256"],claim_sha256=seal["claim_sha256"]))
    summaries, transitions = [], []
    for n in sorted({r["n_target"] for r in rows}):
        rr=[r for r in rows if r["n_target"]==n]
        if len(rr)!=400:raise ValueError("400-per-count denominator changed")
        known=[r for r in rr if r["deletion_energy_eV_atom"] is not None and r["endpoint_energy_eV_atom"] is not None]
        x=np.array([r["deletion_energy_eV_atom"] for r in known]);y=np.array([r["endpoint_energy_eV_atom"] for r in known])
        lo0=sum(r["deletion_bin"]=="low" for r in rr);lo1=sum(r["endpoint_bin"]=="low" for r in rr)
        inter=sum(r["deletion_bin"]==r["endpoint_bin"]=="low" for r in rr)
        summaries.append(dict(n_target=n,original_TEST_attempts=400,known_pairs=len(known),
            statuses=json.dumps(dict(Counter(r["status"] for r in rr))),
            deletion_known=sum(r["deletion_bin"]!="unknown" for r in rr),endpoint_known=sum(r["endpoint_bin"]!="unknown" for r in rr),
            deletion_class_counts=json.dumps(dict(Counter(r["deletion_bin"] for r in rr))),
            endpoint_class_counts=json.dumps(dict(Counter(r["endpoint_bin"] for r in rr))),
            spearman=float(spearmanr(x,y).statistic),kendall=float(kendalltau(x,y).statistic),
            deletion_low=lo0,endpoint_low=lo1,low_intersection=inter,
            endpoint_LOW_yield_given_deletion_LOW_all_attempts=inter/lo0 if lo0 else None,
            endpoint_LOW_recall_by_deletion_LOW=inter/lo1 if lo1 else None,
            mean_relaxation_delta_eV_cell=float(np.mean([r["endpoint_energy_eV"]-r["deletion_energy_eV"] for r in known])),
            mean_relaxation_delta_meV_atom=float(np.mean(1000*(y-x)))))
        for start in ["low","middle","high","unknown"]:
            for end in ["low","middle","high","unknown"]:
                denom=sum(r["deletion_bin"]==start for r in rr)
                count=sum(r["deletion_bin"]==start and r["endpoint_bin"]==end for r in rr)
                transitions.append(dict(n_target=n,deletion_bin=start,endpoint_bin=end,attempts=count,deletion_bin_attempts=denom,row_fraction=count/denom if denom else None))
    # Recheck the assessment boundary and TRAIN schemas after all reads.
    if sha256(a.scoring_gate)!=gate["sha256"]:raise ValueError("Final approval changed during audit")
    for target,binding in bindings.items():
        if sha256(a.collection/f"condition_schema_{target}.json")!=binding["sha256"] or sha256(a.collection/f"learning_{target}.npz")!=binding["training_data_sha256"]:
            raise ValueError("Frozen training inputs changed during audit")
    a.output_dir.mkdir(parents=True,exist_ok=False)
    write_csv(a.output_dir/"assessment_only_TEST_labels.csv",rows)
    write_csv(a.output_dir/"composition_summary.csv",summaries)
    write_csv(a.output_dir/"bin_transitions.csv",transitions)
    receipt=dict(status="PASS_FINAL_ONLY_TEST_SUPPORT_AUDIT",completed_UTC=datetime.now(timezone.utc).isoformat(),
        gate=gate,source_sha256=sha256(__file__),sealed_manifest_sha256=sha256(sealed_path),original_cohort_sha256=cohort_sha,
        original_TEST_rows=1200,statuses=dict(Counter(r["status"] for r in rows)),known_pairs=sum(r["endpoint_usable"] for r in rows),
        schemas=bindings,raw_sources=sources,worker_sources=workers,TEST_energy_unsealed_for_final_assessment_only=True,
        training_data_or_schemas_modified=False,new_threshold_fits=0,new_training_updates=0,new_model_calls=0,
        scope="No resplit, top-up, rerun or model selection. Rank correlations use known pairs; support, class transitions and LOW yields retain all400 original attempts/count, including interrupted unknown endpoints.")
    atomic_json(a.output_dir/"receipt.json",receipt)
    print(json.dumps(dict(status=receipt["status"],statuses=receipt["statuses"],summaries=summaries),indent=2))


if __name__=="__main__":
    main()

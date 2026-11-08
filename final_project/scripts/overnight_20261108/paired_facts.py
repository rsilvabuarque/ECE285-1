"""Freeze compact scalar endpoint facts for CPU-only offsite metric replay.

Raw E/F/stress/trajectory arrays stay separately archived. This producer checks
their hashes and scalar/force consistency once on HPC; replay authenticates this
compact immutable snapshot and recomputes classes, novelty and all denominators.
"""
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from paired_common import atomic_json,check_prepared,identity,sha256


def main(argv=None):
    from cathode_cvae.runtime import require_compute_allocation
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["prepared","tasks","queue","assessment-preflight","output-dir"]:p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args(argv);require_compute_allocation()
    _,context=check_prepared(args.prepared)
    preflight=json.loads(args.assessment_preflight.read_text())
    if preflight.get("status")!="ALL_PROPOSALS_FROZEN_BEFORE_NEW_ENDPOINT_SCORING":
        raise ValueError("Compact FINAL facts require the prescoring proposal freeze")
    tasks=json.loads(args.tasks.read_text())
    if tasks["context_hash"]!=context["context_hash"]:raise ValueError("Compact facts have a different numerical endpoint context")
    records={};missing=[]
    keep=["physical_key","context_hash","status","converged","endpoint_labels_usable","initial_geometry_hash",
          "endpoint_literal_geometry_hash","n_atoms","n_target","initial_energy_eV","endpoint_energy_eV",
          "endpoint_force_max_eV_A","n_steps","calculator_EFS_attempts","calculator_EFS_successes",
          "calculator_EFS_failures","arrays_sha256","model_receipt_sha256"]
    for task in tasks["tasks"]:
        directory=args.queue/"results"/task["task_id"];path=directory/"result.json"
        if not path.exists():missing.append(task["task_id"]);continue
        raw=json.loads(path.read_text());arrays=directory/"arrays.npz"
        if raw["physical_key"]!=task["physical_key"] or raw["context_hash"]!=context["context_hash"]:
            raise ValueError("Source endpoint belongs to a different physical query")
        if sha256(arrays)!=raw["arrays_sha256"]:raise ValueError("Raw endpoint arrays changed")
        with np.load(arrays,allow_pickle=False) as z:
            if raw["initial_energy_eV"] is not None and float(z["initial_energy_eV"])!=raw["initial_energy_eV"]:
                raise ValueError("Initial scalar differs from its raw data")
            if raw["endpoint_energy_eV"] is not None:
                if float(z["endpoint_energy_eV"])!=raw["endpoint_energy_eV"]:raise ValueError("Endpoint scalar differs from raw data")
                if float(np.linalg.norm(z["endpoint_forces_eV_A"],axis=1).max())!=raw["endpoint_force_max_eV_A"]:
                    raise ValueError("Endpoint physical force differs from its raw data")
        claim_path=args.queue/"claims"/(task["task_id"]+".json");claim=json.loads(claim_path.read_text())
        record={key:raw.get(key) for key in keep}
        mapping=raw.get("mapping")
        record["mapping"]={key:mapping.get(key) for key in ["endpoint_occupation_orbit","mobile_assignment_changes",
            "host_assignment_changes","mobile_nearest_collisions","host_nearest_collisions"]} if mapping else None
        records[task["physical_key"]]=dict(record=record,canonical_vacancy_key=task["canonical_vacancy_key"],
            claimed_unix=claim["claimed_unix"],source=dict(task_id=task["task_id"],result_sha256=sha256(path),
                arrays_sha256=raw["arrays_sha256"],claim_sha256=sha256(claim_path)))
    if missing:raise RuntimeError("No biased partial facts snapshot: terminal results missing for "+str(len(missing))+" tasks")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    facts=dict(status="FROZEN_COMPLETE_ENDPOINT_FACTS",schema="paired-endpoint-facts-v1",frozen_unix=time.time(),
        frozen_UTC=datetime.now(timezone.utc).isoformat(),context=context,context_hash=context["context_hash"],records=records,
        source_tasks_sha256=sha256(args.tasks),assessment_preflight_sha256=sha256(args.assessment_preflight),
        source_sha256=sha256(__file__),raw_arrays_rehashed_by_producer=True,
        replay_scope="Scalar facts/identity/denominator replay. Full raw coordinate/force arrays remain separately archived; offsite replay does not pretend to rehash absent raw arrays.")
    facts["identity"]=identity(facts);atomic_json(args.output_dir/"facts.json",facts)
    atomic_json(args.output_dir/"receipt.json",dict(status=facts["status"],facts_sha256=sha256(args.output_dir/"facts.json"),
        physical_queries=len(records),new_MLIP_calls=0,new_training_updates=0,raw_arrays_in_core_payload=False))
    print(json.dumps(dict(status=facts["status"],physical_queries=len(records),facts_sha256=sha256(args.output_dir/"facts.json")),indent=2))
    return facts


if __name__=="__main__":main()

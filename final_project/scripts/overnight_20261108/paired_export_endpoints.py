"""Export a predeclared TRAIN endpoint subset for ordinary portable MLIP score.

This reads coordinates/identities only. MACE single points at these UMA
endpoints are distinct from independently optimized MACE structures.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from paired_common import atomic_json, atomic_npz, check_prepared, geometry_hash, identity, load_npz, sha256


def export(args):
    from cathode_cvae.runtime import require_compute_allocation
    from ase import Atoms
    from ase.io import write
    require_compute_allocation()
    prep, context=check_prepared(args.prepared)
    cohort=load_npz(args.prepared/"cohort.npz")
    tasks=json.loads((args.prepared/"tasks_all.json").read_text())["tasks"]
    indices=[]
    selection_path=getattr(args,"selection_manifest",None)
    if selection_path is not None:
        selection=json.loads(selection_path.read_text())
        if selection["status"]!="FROZEN_BEFORE_SELECTED_ENDPOINT_LABELS_INSPECTED" or args.per_count!=selection["per_count"]:
            raise ValueError("MACE export must retain the frozen selection status/count")
        if identity({k:v for k,v in selection.items() if k!="identity"})!=selection["identity"]:
            raise ValueError("Hash-selected MACE manifest changed")
        if (selection["primary_context_hash"]!=context["context_hash"] or
                selection["source_cohort"]["sha256"]!=sha256(args.prepared/"cohort.npz")):
            raise ValueError("MACE selection belongs to a different frozen cohort")
        for item in selection["selected"]:
            row=item["cohort_row"]
            if str(cohort["canonical_masks"][row])!=item["canonical_mask"] or cohort["split"][row]!="train":
                raise ValueError("Hash-selected MACE row/role differs from authoritative cohort")
            indices.append(row)
    else:
        for n in prep["counts"]:
            rows=[int(i) for i in cohort["pilot_indices"] if cohort["n_target"][i]==n]
            if args.per_count>len(rows):raise ValueError("Export exceeds prespecified TRAIN pilot")
            indices.extend(rows[:args.per_count])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    structures=[];records=[]
    for row in indices:
        if cohort["split"][row]!="train":raise ValueError("Secondary-teacher subset must retain TRAIN role")
        task=tasks[row];directory=args.queue/"results"/task["task_id"]
        header=directory/"status.json"
        record=json.loads(header.read_text())
        if record["context_hash"]!=context["context_hash"]:raise ValueError("Primary endpoint context differs")
        path=directory/"arrays.npz"
        if sha256(path)!=record["arrays_sha256"]:raise ValueError("Endpoint source bytes changed")
        binding=dict(cohort_row=row,structure_id=str(cohort["structure_ids"][row]),
            primary_task_id=task["task_id"],split="train",n_target=int(cohort["n_target"][row]),
            source_arrays_path=str(path.absolute()),source_arrays_sha256=record["arrays_sha256"],
            status=record["status"],exported=bool(record["endpoint_labels_usable"]))
        if record["endpoint_labels_usable"]:
            with np.load(path,allow_pickle=False) as z:
                xyz=z["endpoint_positions_A"];cell=z["cell_A"];numbers=z["numbers"];slots=z["parent_indices"]
            atoms=Atoms(numbers=numbers,positions=xyz,cell=cell,pbc=True)
            atoms.info.update(structure_id=binding["structure_id"],cohort_row=row,geometry_source="UMA fixed-cell endpoint; MACE single point only")
            atoms.new_array("parent_indices",np.asarray(slots,int))
            structures.append(atoms)
            binding["literal_geometry_hash"]=geometry_hash(numbers,xyz,cell)
        records.append(binding)
    if structures:
        write(str(args.output_dir/"endpoints.extxyz"),structures,format="extxyz")
        atomic_npz(args.output_dir/"endpoints.npz", structure_ids=np.array([s.info["structure_id"] for s in structures]),
            offsets=np.r_[0,np.cumsum([len(s) for s in structures])],
            numbers=np.concatenate([s.numbers for s in structures]),
            positions=np.concatenate([s.positions for s in structures]),
            cell=np.array([s.cell.array for s in structures]),
            parent_indices=np.concatenate([s.arrays["parent_indices"] for s in structures]),
            parent_sha256=np.array(context["parent_sha256"]))
    result=dict(status="COMPLETE_EXPORT" if all(r["exported"] for r in records) else "RETAINED_ENDPOINT_FAILURES",
        exported_UTC=datetime.now(timezone.utc).isoformat(),prespecified=len(indices),exported=len(structures),
        energy_labels_read=False,per_count=args.per_count,
        selection="Frozen full-TRAIN hash selection" if selection_path is not None else "First frozen TRAIN pilot rows/count; no energy-based selection",
        selection_manifest=dict(path=str(selection_path.absolute()),sha256=sha256(selection_path)) if selection_path is not None else None,
        primary_context_hash=context["context_hash"],parent_sha256=context["parent_sha256"],
        extxyz_sha256=sha256(args.output_dir/"endpoints.extxyz") if structures else None,
        inference_npz_sha256=sha256(args.output_dir/"endpoints.npz") if structures else None,
        geometry_precision="Use endpoints.npz for inference: preserves every stored coordinate bit. extxyz is a rounded visualization export only.",
        records=records,source_sha256=sha256(__file__),
        next_command="python -m cathode_cvae.mlip.cli score --model mace --input ENDPOINTS.npz --parent PREPARED/parent.npz --output NEW_DIRECTORY --checkpoint MACE_CHECKPOINT --device cuda",
        interpretation="MACE E/F/stress on exact UMA endpoints, not independently minimized MACE labels or DFT validation")
    atomic_json(args.output_dir/"manifest.json",result)
    print(json.dumps({k:result[k] for k in ["status","prespecified","exported","energy_labels_read"]},indent=2))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared",type=Path,required=True)
    p.add_argument("--queue",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--per-count",type=int,default=32)
    p.add_argument("--selection-manifest",type=Path,help="Use immutable full-TRAIN hash selection instead of the original pilot subset")
    return export(p.parse_args(argv))


if __name__=="__main__":main()

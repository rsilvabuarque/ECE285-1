"""Freeze identity-only FINAL history from explicit exposed pools/geometries.

The exposure manifest lists pool_files, coordinate_files and status_roots. Paths
may be strings or dictionaries with path/sha256; coordinate dictionaries may
also name positions_key and indices_path. No energy array is accessed, including
in TEST lineages. A complete primary accepted-frame mapping audit is required.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from paired_common import atomic_json,canonical,check_prepared,load_npz,mobile_indices,sha256
from paired_mapping_audit import audit_frames,nearest_parent_separations


def read_keys(path):
    value=json.loads(Path(path).read_text())
    return set(value if isinstance(value,list) else next(value[k] for k in
        ["keys","canonical_masks","known_keys","canonical_keys"] if k in value))


def checked_path(spec,base):
    path=Path(spec["path"] if isinstance(spec,dict) else spec)
    if not path.is_absolute():path=base/path
    digest=sha256(path)
    if isinstance(spec,dict) and spec.get("sha256",digest)!=digest:
        raise ValueError("Declared exposure source changed: "+str(path))
    return path,digest


def pool_keys(path,parent,expected_parent_sha256=None):
    """Read only binary identity arrays; malformed draws remain non-configurations."""
    if Path(path).suffix==".json":return read_keys(path),dict(kind="identity_json",invalid_rows=0)
    result=set();invalid=0;sites=len(mobile_indices(parent))
    with np.load(path,allow_pickle=False) as z:
        if expected_parent_sha256 is not None and "parent_sha256" in z and str(z["parent_sha256"].item())!=expected_parent_sha256:
            raise ValueError("Exposure pool is bound to a different physical parent")
        if "canonical_masks" in z:
            keys=z["canonical_masks"].astype(str)
            for key in keys:
                if len(key)!=sites or set(key)-{"0","1"}:invalid+=1
                else:result.add(key)
        else:
            fields=[k for k in ["occupancy","seed_occupancy"] if k in z]
            if not fields and "masks" in z:fields=["masks"]
            if not fields:raise ValueError("Exposure pool has no explicit parent mask identity array")
            for field in fields:
                values=z[field]
                if values.ndim!=2 or values.shape[1]!=sites:
                    raise ValueError("Exposure pool belongs to a different site domain")
                for value in values:
                    if not np.isin(value,[0,1]).all():invalid+=1;continue
                    occupancy=1-value if field=="masks" else value
                    key,_=canonical(occupancy,parent["permutations"]);result.add(key)
    return result,dict(kind="mask_npz",invalid_rows=invalid)


def coordinate_keys(path,parent,minimum,*,positions_key=None,indices_path=None):
    """Only coordinate/species/cell/original-index keys are read from the NPZ."""
    with np.load(path,allow_pickle=False) as z:
        name=positions_key or next((k for k in ["trajectory_positions_A","positions_A","endpoint_positions_A","initial_positions_A"] if k in z),None)
        if name is None:raise ValueError("No explicit coordinate array in exposed geometry")
        xyz=z[name]
        numbers=z["numbers"]
        cell=z["cell_A"] if "cell_A" in z else z["cell"]
        slots=z["parent_indices"] if "parent_indices" in z else z["parent_slot"] if "parent_slot" in z else None
    if slots is None and indices_path is not None:
        with np.load(indices_path,allow_pickle=False) as z:slots=z["parent_indices"] if "parent_indices" in z else z["parent_slot"]
    if slots is None:raise ValueError("Exposed geometry lacks continuous original atom indices; do not rematch atoms independently")
    slots=np.asarray(slots,int)
    if len(set(slots.tolist()))!=len(slots) or not np.array_equal(numbers,parent["numbers"][slots]):
        raise ValueError("Exposed geometry species/order do not match original parent indices")
    xyz=np.asarray(xyz,float)
    if xyz.ndim==2:xyz=xyz[None]
    if xyz.shape[1:]!=(len(slots),3) or not np.isfinite(xyz).all():raise ValueError("Invalid exposed coordinate array")
    if not np.array_equal(cell,parent["cell_A"]):
        raise ValueError("This FINAL history requires the declared fixed parent cell")
    occupancy=np.isin(mobile_indices(parent),slots).astype(np.uint8)
    initial,_=canonical(occupancy,parent["permutations"])
    audit=audit_frames(parent,slots,xyz,minimum)
    keys={initial}|{row["endpoint_occupation_orbit"] for row in audit["full_mapping_events"]}
    return keys,{k:v for k,v in audit.items() if k not in ["full_mapping_events","certificate"]}


def freeze(args):
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    prep,context=check_prepared(args.prepared)
    parent=load_npz(args.prepared/"parent.npz");cohort=load_npz(args.prepared/"cohort.npz")
    sites=len(mobile_indices(parent));history=read_keys(args.intake_history)
    history.update(cohort["canonical_masks"].astype(str))
    minimum=nearest_parent_separations(parent)
    manifest=json.loads(args.exposure_manifest.read_text());base=args.exposure_manifest.parent
    args.output_dir.mkdir(parents=True,exist_ok=False)
    audit_receipt=json.loads((args.mapping_audit/"receipt.json").read_text())
    if (audit_receipt["status"]!="PASS_COMPLETE" or audit_receipt["context_hash"]!=context["context_hash"] or
            audit_receipt["lineages"]!=len(cohort["occupancy"])):
        raise ValueError("Complete primary accepted-frame audit is required before FINAL history")
    primary=[]
    tasks=json.loads((args.prepared/"tasks_all.json").read_text())["tasks"]
    for task in tasks:
        header_path=args.paired_queue/"results"/task["task_id"]/"status.json"
        header=json.loads(header_path.read_text())
        mapping_path=args.mapping_audit/(task["task_id"]+".json")
        mapped=json.loads(mapping_path.read_text())
        if header["arrays_sha256"]!=mapped["arrays_sha256"] or header["context_hash"]!=context["context_hash"]:
            raise ValueError("Primary frame audit and terminal header disagree")
        history.update(row["endpoint_occupation_orbit"] for row in mapped["full_mapping_events"])
        if header.get("mapping"):history.add(header["mapping"]["endpoint_occupation_orbit"])
        primary.append(dict(task_id=task["task_id"],status=header["status"],arrays_sha256=header["arrays_sha256"],
            header_sha256=sha256(header_path),mapping_sha256=sha256(mapping_path)))
    pools=[];coordinates=[]
    for spec in manifest.get("pool_files",[]):
        path,digest=checked_path(spec,base);keys,info=pool_keys(path,parent,context["parent_sha256"])
        history.update(keys)
        if sha256(path)!=digest:raise ValueError("Exposure mask file changed while read")
        pools.append(dict(path=str(path.absolute()),sha256=digest,unique_keys=len(keys),**info))
    coordinate_specs=list(manifest.get("coordinate_files",[]))
    for raw_root in manifest.get("status_roots",[]):
        root=Path(raw_root);root=root if root.is_absolute() else base/root
        for header_path in sorted((root/"results").glob("*/status.json")):
            header=json.loads(header_path.read_text());path=header_path.parent/"arrays.npz"
            coordinate_specs.append(dict(path=str(path),sha256=header["arrays_sha256"],positions_key="trajectory_positions_A"))
            if header.get("mapping"):history.add(header["mapping"]["endpoint_occupation_orbit"])
    for spec in coordinate_specs:
        path,digest=checked_path(spec,base);options=spec if isinstance(spec,dict) else {}
        index_path=None
        if options.get("indices_path"):
            index_path=Path(options["indices_path"]);index_path=index_path if index_path.is_absolute() else base/index_path
        index_digest=sha256(index_path) if index_path is not None else None
        keys,info=coordinate_keys(path,parent,minimum,positions_key=options.get("positions_key"),indices_path=index_path)
        history.update(keys)
        if sha256(path)!=digest:raise ValueError("Exposed coordinate file changed while read; freeze a stable saved version")
        if index_path is not None and sha256(index_path)!=index_digest:raise ValueError("Original atom-index companion changed while read")
        coordinates.append(dict(path=str(path.absolute()),sha256=digest,unique_keys=len(keys),
            index_companion=dict(path=str(index_path.absolute()),sha256=index_digest) if index_path is not None else None,**info))
    if any(len(k)!=sites or set(k)-{"0","1"} for k in history):raise ValueError("Final history mixes parent domains")
    atomic_json(args.output_dir/"history.json",sorted(history))
    record=dict(status="FROZEN_IDENTITIES_ONLY",frozen_UTC=datetime.now(timezone.utc).isoformat(),
        history_sha256=sha256(args.output_dir/"history.json"),intake_history_sha256=sha256(args.intake_history),
        cohort_sha256=sha256(args.prepared/"cohort.npz"),parent_sha256=context["parent_sha256"],
        primary_context_hash=context["context_hash"],known_orbits=len(history),primary_lineages=len(primary),
        primary_statuses=dict(Counter(r["status"] for r in primary)),primary_mapping_sources=primary,
        mapping_audit_receipt=dict(path=str((args.mapping_audit/"receipt.json").absolute()),sha256=sha256(args.mapping_audit/"receipt.json")),
        exposure_manifest=dict(path=str(args.exposure_manifest.absolute()),sha256=sha256(args.exposure_manifest)),
        declared_exposure_manifest_verified=True,pool_sources=pools,coordinate_sources=coordinates,
        energy_fields_read=0,new_MLIP_calls=0,new_training_updates=0,source_sha256=sha256(__file__),
        dependencies={name:sha256(Path(__file__).with_name(name)) for name in ["paired_common.py","paired_mapping_audit.py"]},
        primary_raw_arrays_rehashed_here=False,
        scope="Original masks plus all declared DEV and available saved coordinate mappings. Complete primary frame audit binds raw arrays; TEST energies never accessed. Occupation exposure does not create new independent training rows or certify PES basins.")
    atomic_json(args.output_dir/"receipt.json",record)
    print(json.dumps({k:record[k] for k in ["status","known_orbits","primary_lineages","primary_statuses","energy_fields_read"]},indent=2))
    return record


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["prepared","intake-history","exposure-manifest","paired-queue","mapping-audit","output-dir"]:
        p.add_argument("--"+name,type=Path,required=True)
    return freeze(p.parse_args(argv))


if __name__=="__main__":main()

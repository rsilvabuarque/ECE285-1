"""Deduplicate exact exported UMA coordinates while retaining both SP scopes.

Original selection/export manifests are never edited. Derived scope manifests
retain every prespecified row, including nonusable endpoints, and bind each
usable row to the one physical scoring identity in the full-precision union.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
from paired_common import atomic_json,atomic_npz,geometry_hash,sha256


def main(argv=None):
    from cathode_cvae.runtime import require_compute_allocation
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exports",nargs="+",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args(argv);require_compute_allocation()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    unique={};id_hash={};scopes={};parent_sha=None;context=None
    for directory in args.exports:
        scope=directory.name
        if scope in scopes:raise ValueError("Distinct export scopes need distinct directory names")
        manifest_path=directory/"manifest.json";manifest=json.loads(manifest_path.read_text())
        if parent_sha is None:parent_sha=manifest["parent_sha256"];context=manifest["primary_context_hash"]
        if manifest["parent_sha256"]!=parent_sha or manifest["primary_context_hash"]!=context:
            raise ValueError("MACE-SP scopes use different source geometry protocols")
        payload={}
        if manifest["exported"]:
            path=directory/"endpoints.npz"
            if sha256(path)!=manifest["inference_npz_sha256"]:raise ValueError("Full-precision export changed")
            with np.load(path,allow_pickle=False) as z:
                for i,sid in enumerate(z["structure_ids"].astype(str)):
                    lo,hi=z["offsets"][i:i+2]
                    payload[sid]=dict(numbers=z["numbers"][lo:hi],positions=z["positions"][lo:hi],
                        cell=z["cell"][i],parent_indices=z["parent_indices"][lo:hi])
        records=[]
        for raw in manifest["records"]:
            row=dict(raw);sid=row["structure_id"];row["scoring_structure_id"]=None
            if row["exported"]:
                d=payload[sid];key=geometry_hash(d["numbers"],d["positions"],d["cell"])
                if key!=row["literal_geometry_hash"]:raise ValueError("Export coordinate identity differs from selected source")
                if sid in id_hash and id_hash[sid]!=key:raise ValueError("One initial structure ID maps to inconsistent endpoint coordinates")
                id_hash[sid]=key
                if key not in unique:unique[key]=dict(d,structure_id=sid)
                row["scoring_structure_id"]=unique[key]["structure_id"]
            records.append(row)
        derived=dict(manifest,records=records,
            derived_from_original_manifest=dict(path=str(manifest_path.absolute()),sha256=sha256(manifest_path)),
            scoring_scope="Exact-coordinate union; original selection denominator retained")
        atomic_json(args.output_dir/("scope_"+scope+".json"),derived)
        scopes[scope]=dict(prespecified=manifest["prespecified"],usable=manifest["exported"],
            failed_or_nonusable=manifest["prespecified"]-manifest["exported"],
            derived_manifest_sha256=sha256(args.output_dir/("scope_"+scope+".json")))
    values=list(unique.values())
    if values:
        atomic_npz(args.output_dir/"endpoints.npz",structure_ids=np.array([x["structure_id"] for x in values]),
            offsets=np.r_[0,np.cumsum([len(x["numbers"]) for x in values])],
            numbers=np.concatenate([x["numbers"] for x in values]),positions=np.concatenate([x["positions"] for x in values]),
            cell=np.array([x["cell"] for x in values]),parent_indices=np.concatenate([x["parent_indices"] for x in values]),
            parent_sha256=np.array(parent_sha))
    record=dict(status="FROZEN_EXACT_COORDINATE_SP_UNION",completed_UTC=datetime.now(timezone.utc).isoformat(),
        scopes=scopes,unique_physical_coordinates=len(values),parent_sha256=parent_sha,primary_context_hash=context,
        inference_npz_sha256=sha256(args.output_dir/"endpoints.npz") if values else None,
        original_manifests_unchanged=True,energies_read=False,new_MLIP_calls=0,source_sha256=sha256(__file__))
    atomic_json(args.output_dir/"receipt.json",record);print(json.dumps(record,indent=2))
    return record


if __name__=="__main__":main()

"""Label-free symmetry-aware distances for the frozen FINAL proposal pool.

The nearest distance to a fixed TRAIN subset is an upper bound on the nearest
distance to all TRAIN orbits. All proposals remain rows, including repeated
orbits. Mutation's raw parent-child alignment is preserved separately from its
minimum distance over the actual fixed-parent symmetry group.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from cathode_cvae.data import validate_permutation_group
from cathode_cvae.runtime import require_compute_allocation


PROJECT=Path(__file__).resolve().parents[2]
REFERENCE_NAMESPACE="conditioning-final-distance-reference-20261107-v1"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):h.update(block)
    return h.hexdigest()


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("x",newline="") as f:
        if fields:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def pack(x):
    """One native uint64 per binary row; unused low bits remain exactly zero."""
    x=np.asarray(x,np.uint8)
    if x.ndim!=2 or not 1<=x.shape[1]<=62 or not np.isin(x,[0,1]).all():
        raise ValueError("This benchmark audit requires 1..62 binary parent sites")
    packed=np.zeros((len(x),8),np.uint8)
    bits=np.packbits(x,axis=1,bitorder="big")
    packed[:,:bits.shape[1]]=bits
    return packed.view(">u8").ravel().astype(np.uint64)


def orbit_codes(x,inverse):
    return np.stack([pack(x[:,p]) for p in inverse],axis=1)


def canonical_codes(x,inverse):
    # Vacancy-one lexicographic minimum agrees with the generator's keys.
    best=np.full(len(x),np.iinfo(np.uint64).max(),np.uint64)
    for p in inverse:np.minimum(best,pack(1-x[:,p]),out=best)
    return best


def bounded_nearest(query,reference,inverse,*,deadline,batch=64):
    images=np.unique(orbit_codes(reference,inverse))
    codes=pack(query);distance=np.empty(len(query),np.uint8)
    for first in range(0,len(query),batch):
        if time.monotonic()>deadline:raise TimeoutError("Declared CPU wall-time bound reached; preserve partial evidence")
        xor=np.bitwise_xor(codes[first:first+batch,None],images[None])
        distance[first:first+batch]=np.bitwise_count(xor).min(axis=1)
    return distance,len(images)


def independent_small_check():
    """Compare packed arithmetic with direct bit differences on all 4-bit rows."""
    x=np.asarray([[int(c) for c in format(i,"04b")] for i in range(16)],np.uint8)
    inverse=np.asarray([np.roll(np.arange(4),j) for j in range(4)])
    images=orbit_codes(x,inverse)
    fast=np.bitwise_count(pack(x)[:,None,None]^images[None]).min(axis=2)
    direct=np.asarray([[min(np.count_nonzero(a!=b[p]) for p in inverse) for b in x] for a in x])
    if not np.array_equal(fast,direct):raise AssertionError("Packed distances fail independent direct-bit comparison")
    if pack(np.asarray([[1]*48],np.uint8))[0].item().bit_count()!=48:
        raise AssertionError("48-site packed representation lost bits")
    return dict(status="PASS",direct_pair_checks=256,synthetic_sites=4,
        operations=4,packing_48_site_bitcount=48)


def summaries(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r["method"],r["replicate_id"],r["n_Li"],r["requested_bin"])].append(r)
    cells=[]
    for key,rr in sorted(groups.items()):
        out=dict(zip(("method","replicate_id","n_Li","requested_bin"),key))
        d=[r["nearest_TRAIN_reference_orbit_Hamming"] for r in rr if r["nearest_TRAIN_reference_orbit_Hamming"]!=""]
        out.update(attempts=len(rr),distance_known=len(d),exact_valid_attempts=sum(r["exact_valid"] for r in rr),
            mean_nearest_TRAIN_reference_orbit_Hamming=float(np.mean(d)) if d else "",
            median_nearest_TRAIN_reference_orbit_Hamming=float(np.median(d)) if d else "",
            minimum_nearest_TRAIN_reference_orbit_Hamming=min(d) if d else "",
            maximum_nearest_TRAIN_reference_orbit_Hamming=max(d) if d else "",
            zero_reference_distance_attempts=sum(q==0 for q in d))
        for field in ["raw_seed_child_Hamming","symmetry_minimum_seed_child_Hamming"]:
            values=[r[field] for r in rr if r[field]!=""]
            if values:out["mean_"+field]=float(np.mean(values))
        cells.append(out)
    byseed=defaultdict(list)
    for r in cells:byseed[(r["method"],r["replicate_id"])].append(r)
    seeds=[]
    for (method,seed),rr in sorted(byseed.items()):
        out=dict(method=method,replicate_id=seed,condition_cells=len(rr),attempts=sum(r["attempts"] for r in rr))
        for field in ["mean_nearest_TRAIN_reference_orbit_Hamming","mean_raw_seed_child_Hamming",
                      "mean_symmetry_minimum_seed_child_Hamming"]:
            values=[r[field] for r in rr if r.get(field,"")!=""]
            if values:out[field]=float(np.mean(values))
        seeds.append(out)
    return cells,seeds


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=PROJECT)
    p.add_argument("--output-dir",type=Path)
    p.add_argument("--reference-size",type=int,default=1024)
    p.add_argument("--max-seconds",type=float,default=300)
    a=p.parse_args(argv);require_compute_allocation()
    start=time.monotonic();deadline=start+a.max_seconds-5
    root=a.root.resolve();run=root/"runs/conditioning_20261107"
    output=a.output_dir or root/"reports/conditioning_20261107/final_structural_distances"
    output.mkdir(parents=True,exist_ok=False)
    rows=[];before={};source_paths=[]
    try:
        selfcheck=independent_small_check()
        protocol_path=run/"final_protocol.json";generation=run/"final/receipt.json"
        attempts=run/"final/attempts.csv";raw_path=run/"final/raw_attempts.npz"
        data_path=root/"artifacts/current/improvement_v1/training_round0.npz"
        parent_path=root/"artifacts/current/core/data/parents/O3_N48_pilot.npz"
        source_paths=[protocol_path,generation,attempts,raw_path,data_path,parent_path]
        before={str(p.relative_to(root)):sha(p) for p in source_paths}
        protocol=json.loads(protocol_path.read_text());generated=json.loads(generation.read_text())
        if generated["manifest_sha256"]!=protocol["generation_manifest_sha256"]:
            raise ValueError("Final proposals do not match the pre-generation protocol")
        for path in [attempts,raw_path]:
            if sha(path)!=generated["outputs"][path.name]:raise ValueError("Final proposal bytes changed")
        if sha(data_path)!=protocol["data_sha256"] or sha(parent_path)!=protocol["parent_sha256"]:
            raise ValueError("Protected training or parent bytes changed")
        with np.load(parent_path,allow_pickle=False) as z:perms=z["permutations"]
        validate_permutation_group(perms);inverse=np.argsort(perms,axis=1)
        if perms.shape!=(192,48):raise ValueError("Expected the actual192-operation48-site parent")
        with np.load(data_path,allow_pickle=False) as z:
            split=z["split"].astype(str);counts=z["n_Li"];ids=z["structure_ids"].astype(str)
            keys=z["canonical_masks"].astype(str);training_occupancy=z["occupancy"]
        with np.load(raw_path,allow_pickle=False) as z:
            raw=z["occupancy"];raw_seeds=z["seed_occupancy"]
        rows=read(attempts)
        if len(rows)!=34560 or raw.shape!=(34560,48) or raw_seeds.shape!=raw.shape:
            raise ValueError("Unexpected final attempt budget or raw alignment")
        budget=Counter((r["method"],r["replicate_id"],int(r["n_Li"]),r["requested_bin"]) for r in rows)
        if len(budget)!=270 or set(budget.values())!={128}:
            raise ValueError("Need ten families, three replicates, three counts and three bins")
        if sorted(int(r["raw_row"]) for r in rows)!=list(range(len(raw))):
            raise ValueError("Final raw row identities are not a bijection")
        references={};reference_rows=[];train_rows={}
        for n in [12,24,36]:
            eligible=np.flatnonzero((split=="train")&(counts==n));train_rows[n]=len(eligible)
            ranked=sorted(eligible,key=lambda i:hashlib.sha256(
                f"{REFERENCE_NAMESPACE}|{protocol['parent_sha256']}|{n}|{keys[i]}".encode()).digest())
            selected=ranked[:min(a.reference_size,len(ranked))]
            if len(selected)<min(1024,len(eligible)):raise ValueError("Declared audit uses1024referenceorbits/count")
            references[n]=1-np.asarray([[int(c) for c in keys[i]] for i in selected],np.uint8)
            reference_rows.extend(dict(n_Li=n,structure_id=str(ids[i]),canonical_vacancy_key=str(keys[i]),
                source_row=int(i),source_role="train",rank=j) for j,i in enumerate(selected))
        # The exact reference is written before distances are computed. It uses
        # hashes and TRAIN identities only, without final geometry or energy ranks.
        write(output/"TRAIN_reference.csv",reference_rows)
        (output/"reference_protocol.json").write_text(json.dumps(dict(
            status="FROZEN_BEFORE_DISTANCE_COMPUTATION",namespace=REFERENCE_NAMESPACE,
            selection="Lowest SHA256(namespace|parent_sha256|count|canonicalkey), TRAINonly",
            reference_rows=len(reference_rows),reference_size_per_count={n:len(x) for n,x in references.items()},
            training_rows_per_count=train_rows,reference_sha256=sha(output/"TRAIN_reference.csv"),
            protected_training_data_sha256=sha(data_path),energy_fields_read=0),indent=2)+"\n")
        observed_canonical=canonical_codes(raw,inverse)
        supplied_codes=np.asarray([int(r["canonical_mask"],2)<<16 for r in rows],np.uint64)
        indices=np.asarray([int(r["raw_row"]) for r in rows])
        if not np.array_equal(supplied_codes,observed_canonical[indices]):
            raise ValueError("Attempt canonical keys disagree with the actual raw masks and parent operations")
        unique_keys={n:sorted({r["canonical_mask"] for r in rows if int(r["n_Li"])==n}) for n in [12,24,36]}
        queries={n:1-np.asarray([[int(c) for c in key] for key in unique_keys[n]],np.uint8) for n in unique_keys}
        distance_maps={};image_counts={}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures={n:executor.submit(bounded_nearest,queries[n],references[n],inverse,deadline=deadline) for n in queries}
            for n,future in futures.items():
                distances,images=future.result();distance_maps[n]=dict(zip(unique_keys[n],map(int,distances)));image_counts[n]=images
                print(f"DISTANCES_COMPLETE count={n} uniqueorbits={len(distances)} referenceimages={images}",flush=True)
        mutation_rows=[r for r in rows if r["method"]=="mutation"]
        mutation_indices=np.asarray([int(r["raw_row"]) for r in mutation_rows])
        child=raw[mutation_indices];seeds=raw_seeds[mutation_indices]
        raw_distance=np.count_nonzero(child!=seeds,axis=1)
        seed_distance=np.bitwise_count(pack(child)[:,None]^orbit_codes(seeds,inverse)).min(axis=1)
        seed_stats={}
        for j,r in enumerate(mutation_rows):
            source_row=int(r["seed_source_row"]);k=int(r["swap_count"])
            if split[source_row]!="train" or str(ids[source_row])!=r["seed_structure_id"]:
                raise ValueError("Mutation seed is not its recorded protected TRAIN identity")
            if r["seed_source_sha256"]!=protocol["data_sha256"] or not np.array_equal(seeds[j],training_occupancy[source_row]):
                raise ValueError("Mutation seed lost its actual source orientation")
            removed=np.asarray([int(s) for s in r["removed_sites"].split("|")])
            added=np.asarray([int(s) for s in r["added_sites"].split("|")])
            actual_removed=np.flatnonzero((seeds[j]==1)&(child[j]==0))
            actual_added=np.flatnonzero((seeds[j]==0)&(child[j]==1))
            if raw_distance[j]!=2*k or not np.array_equal(np.sort(removed),actual_removed) or not np.array_equal(np.sort(added),actual_added):
                raise ValueError("Mutation raw child does not implement the recorded exact k swaps")
            if seed_distance[j]>raw_distance[j]:raise AssertionError("Orbitminimum exceeded the identity-operation distance")
            seed_stats[int(r["raw_row"])]=(int(raw_distance[j]),int(seed_distance[j]))
        attempt_output=[]
        for r in rows:
            index=int(r["raw_row"]);n=int(r["n_Li"])
            valid=bool(np.isin(raw[index],[0,1]).all() and raw[index].sum()==n and r["status"]=="generated")
            seed_raw,seed_orbit=seed_stats.get(index,("",""))
            attempt_output.append(dict(attempt_id=r["attempt_id"],method=r["method"],replicate_id=r["replicate_id"],
                fit_seed=r["fit_seed"],n_Li=n,requested_bin=r["requested_bin"],raw_row=index,
                canonical_mask=r["canonical_mask"],exact_valid=valid,
                nearest_TRAIN_reference_orbit_Hamming=distance_maps[n][r["canonical_mask"]] if valid else "",
                TRAIN_reference_orbits=len(references[n]),
                seed_structure_id=r.get("seed_structure_id",""),swap_count=r.get("swap_count",""),
                raw_seed_child_Hamming=seed_raw,symmetry_minimum_seed_child_Hamming=seed_orbit))
        write(output/"attempt_distances.csv",attempt_output)
        write(output/"unique_orbit_distances.csv",[dict(n_Li=n,canonical_mask=key,
            nearest_TRAIN_reference_orbit_Hamming=value) for n,lookup in distance_maps.items() for key,value in lookup.items()])
        cells,seed_summary=summaries(attempt_output)
        write(output/"metrics_by_condition_seed.csv",cells);write(output/"metrics_by_seed.csv",seed_summary)
        (output/"METHOD.md").write_text(
            "For occupancy vectors a,b, d_G(a,b)=min_g sum_j |a_j-b_{g^-1(j)}|, over all192 actual retained parent operations. "
            "The comparison uses all periodic parent translations/point symmetries; it is not Hamming distance between canonical representatives. "
            "For fixed TRAIN reference R, d_R(a)=min_{b in R}d_G(a,b). Since R is a subset of TRAIN, d_R(a)>=d_TRAIN(a). "
            "Thus this is an upper bound, not a certified distance from the full training set. No positive lower bound on full-TRAIN distance follows. "
            "References are1024 deterministic SHA-ranked canonical TRAIN orbits/count. Units are differing mobile-site occupations, not angstrom. "
            "For equal counts, Hamming/2 is the minimum number of simultaneous removal/insertion swaps under that alignment. "
            "Mutation raw seeds/children retain their original joint orientation, and their raw distance must be exactly2k; its orbit minimum may be smaller, including zero. "
            "All attempts, including repeated orbits, remain in tables and averages. Within each fit seed, summaries weight each composition/bin cell equally. "
            "No energy labels, MLIP calls, new structures, optimization, novelty reclassification, or model selection were used. "
            "The reference identities were frozen before distance computation; the already fixed final proposal pool and assessment decision remain unchanged.\n")
        after={str(q.relative_to(root)):sha(q) for q in source_paths}
        if before!=after:raise ValueError("Protected input changed during the label-free audit")
        result=dict(status="PASS",attempts=len(rows),condition_seed_groups=len(cells),method_seed_groups=len(seed_summary),
            families=sorted({r["method"] for r in rows}),family_replicates=dict(Counter(r["method"] for r in seed_summary)),
            valid_attempts=sum(r["exact_valid"] for r in attempt_output),unique_generated_orbits=sum(len(x) for x in unique_keys.values()),
            mutation_seed_checks=len(mutation_rows),mutation_raw_distances_exact_2k=True,
            actual_parent_operations=len(perms),reference_orbits_per_count={n:len(x) for n,x in references.items()},
            reference_images_per_count=image_counts,reference_namespace=REFERENCE_NAMESPACE,
            metric="Exact symmetry-aware minimum site Hamming distance to the fixed TRAIN subset; upper bound on full-TRAIN nearest distance",
            all_attempts_preserved=True,novelty_scope="Training/history novelty remains the separate main assessment metric",
            elapsed_seconds=time.monotonic()-start,maximum_seconds=a.max_seconds,device="cpu",cpu_workers=3,
            numpy=np.__version__,job_id=os.environ.get("SLURM_JOB_ID"),step_id=os.environ.get("SLURM_STEP_ID"),
            independent_selfcheck=selfcheck,input_hashes=before,inputs_unchanged=True,
            energy_fields_read=0,new_MLIP_calls=0,new_training_trajectories=0,selection_changed=False,
            source_sha256=sha(__file__),output_hashes={q.name:sha(q) for q in output.iterdir() if q.is_file()})
        with (output/"receipt.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
        print(json.dumps(result,indent=2),flush=True)
    except Exception as error:
        with (output/"failure_receipt.json").open("x") as f:json.dump(dict(status="FAILED_PRESERVED",error=str(error),
            error_type=type(error).__name__,traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start,
            source_sha256=sha(__file__),input_hashes=before,new_MLIP_calls=0,new_training_trajectories=0),f,indent=2)
        raise


if __name__=="__main__":main()

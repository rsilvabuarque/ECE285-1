"""Freeze matched learned-edit and uniform-mutation proposals before scoring.

Every paired trial shares its raw heldout seed, exact one/two/three swap count,
and removal/addition uniform streams. Canonicalization is for physical scoring
and novelty only; seed and child retain one common orientation for editing.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np

from cathode_cvae.acquisition import canonical_keys
from cathode_cvae.conditions import load_schema
from cathode_cvae.data import sha256
from cathode_cvae.learning.edit_model import encode_condition
from cathode_cvae.runtime import require_compute_allocation

COUNTS = (12,24,36)
BINS = ("low","middle","high")
FIT_SEEDS = (20261106,20261107,20261108)
NAMESPACE = "conditioning-heldout-seed-edit-20261107-v1"


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def seed_for(*values):
    return int.from_bytes(hashlib.sha256("|".join(map(str,values)).encode()).digest()[:4],"little")


def paired_trials(bank, fit_seed, n, requested, count, namespace=NAMESPACE):
    """Uniform eligible seed and k draws, without looking at target outcomes."""
    rows=np.flatnonzero((bank["n_Li"]==n)&(bank["seed_bin"].astype(str)==requested))
    if not len(rows):raise ValueError("Unsupported heldout seed cell")
    rng=np.random.default_rng(seed_for(namespace,fit_seed,n,requested,"seed-and-edit-budget"))
    chosen=rng.choice(rows,count,replace=True)
    k=rng.integers(1,4,size=count)
    if np.any(np.minimum(n,bank["occupancy"].shape[1]-n)<k):raise ValueError("Unavailable exact edit budget")
    return chosen,k


def verify_edit(seed, child, removed, added, k):
    """Check exact raw edits before any symmetry transformation."""
    seed=np.asarray(seed);child=np.asarray(child);removed=np.asarray(removed);added=np.asarray(added);k=np.asarray(k)
    if seed.ndim!=2 or child.shape!=seed.shape or removed.shape!=seed.shape or added.shape!=seed.shape:
        raise ValueError("Raw seed/child/edit shapes differ")
    if k.shape!=(len(seed),) or not np.isin(k,[1,2,3]).all():raise ValueError("One exact swap budget is required per seed")
    if any(not np.isin(a,[0,1]).all() for a in [seed,child,removed,added]):raise ValueError("Editing arrays must be binary")
    if not np.array_equal(removed,((seed==1)&(child==0)).astype(np.uint8)):
        raise ValueError("Removed sites differ from actual raw seed-to-child changes")
    if not np.array_equal(added,((seed==0)&(child==1)).astype(np.uint8)):
        raise ValueError("Added sites differ from actual raw seed-to-child changes")
    if not np.array_equal(removed.sum(1),k) or not np.array_equal(added.sum(1),k):raise ValueError("Wrong distinct removal/addition count")
    if not np.array_equal(seed.sum(1),child.sum(1)) or not np.array_equal((seed!=child).sum(1),2*k):
        raise ValueError("Editing lost composition or the raw2k constraint")


def supported_trials(cfg,n,requested,k):
    """A mathematical sampler does not establish training support for a condition."""
    if "supported_requests" not in cfg:raise ValueError("Editor checkpoint lacks explicit TRAIN condition support")
    available={(int(r["n_Li"]),int(r["swap_count"]),str(r["child_bin"]))
               for r in cfg["supported_requests"] if int(r["train_rows"])>0}
    return np.asarray([(int(n),int(budget),str(requested)) in available for budget in k])


def load_bank(path,parent_sha,permutations):
    with np.load(path,allow_pickle=False) as z:bank={k:z[k] for k in z.files}
    required={"occupancy","seed_ids","seed_bin","n_Li","canonical_seed_keys","component_ids","H0_source_rows","split","parent_sha256","condition_schema_id"}
    if not required<=set(bank):raise ValueError("Assessment seed bank lacks explicit heldout lineage provenance")
    x=bank["occupancy"]
    if x.ndim!=2 or x.shape[1]!=48 or not np.isin(x,[0,1]).all() or not np.array_equal(x.sum(1),bank["n_Li"]):
        raise ValueError("Seed bank must contain valid48site raw occupations")
    if str(bank["parent_sha256"].item())!=parent_sha or not np.all(bank["split"].astype(str)=="heldout"):
        raise ValueError("All assessment seeds must be heldout editor lineages on this parent")
    keys=canonical_keys(x,permutations)
    if not np.array_equal(keys,bank["canonical_seed_keys"].astype(str)) or len(set(keys))!=len(keys):
        raise ValueError("Heldout seed identities are not unique actual parent orbits")
    return bank


def freeze_history(root,output,parent_sha):
    fp=root/"runs/conditioning_20261107/final_protocol.json"
    final=json.loads(fp.read_text());source=Path(final["history_path"])
    if sha256(source)!=final["history_sha256"]:raise ValueError("Core frozen history changed")
    history=set(json.loads(source.read_text()))
    pool=root/"runs/conditioning_20261107/final/candidates.npz"
    generation=json.loads((pool.parent/"receipt.json").read_text())
    if generation["outputs"]["candidates.npz"]!=sha256(pool):raise ValueError("Core final proposal pool changed")
    with np.load(pool,allow_pickle=False) as z:
        if str(z["parent_sha256"].item())!=parent_sha:raise ValueError("Core pool is another parent")
        masks=z["masks"]
    history.update("".join(map(str,r)) for r in masks)
    path=output/"history.json";write_json(path,sorted(history))
    return path,len(history),dict(core_final_protocol=dict(path=str(fp),sha256=sha256(fp)),
        core_history=dict(path=str(source),sha256=sha256(source)),core_final_pool=dict(path=str(pool),sha256=sha256(pool)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument("--checkpoint",action="append",required=True,help="Actual fitseed=checkpoint; repeat for all three fits")
    p.add_argument("--seedbank",type=Path);p.add_argument("--pair-data",type=Path)
    p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--count",type=int,default=128)
    p.add_argument("--namespace",default=NAMESPACE);p.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    a=p.parse_args();require_compute_allocation()
    import torch
    from cathode_cvae.learning.edit_model import uniform_edits
    from cathode_cvae.learning.generate_edit import load_edit_model
    root=a.root.resolve();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    if not 1<=a.count<=128:raise ValueError("The bounded comparison supports at most128attempts/cell")
    checkpoints={int(s.split("=",1)[0]):Path(s.split("=",1)[1]).resolve() for s in a.checkpoint}
    if set(checkpoints)!=set(FIT_SEEDS) or len(a.checkpoint)!=3:raise ValueError("Supply exactly the three prespecified actual fit seeds")
    seedbank=a.seedbank or root/"runs/conditioning_20261107/editor/data_v2/assessment_seeds.npz"
    pairs=a.pair_data or seedbank.parent/"edit_pairs.npz"
    preparation=seedbank.parent/"preparation_receipt.json"
    prep=json.loads(preparation.read_text())
    if prep["status"]!="READY_FOR_BOUNDED_EDITOR":raise ValueError("Editor data/support audit has not passed")
    if sha256(seedbank)!=prep["outputs"]["assessment_seeds.npz"] or sha256(pairs)!=prep["outputs"]["edit_pairs.npz"]:
        raise ValueError("Prepared seed/pair identities changed")
    gate=root/"artifacts/conditioning_20261107/required_gate.json"
    if json.loads(gate.read_text())["status"]!="COMPLETE":raise ValueError("Required report gate has not completed")
    parent=root/"artifacts/current/core/data/parents/O3_N48_pilot.npz";parent_sha=sha256(parent)
    with np.load(parent,allow_pickle=False) as z:permutations=z["permutations"]
    bank=load_bank(seedbank,parent_sha,permutations)
    schema_path=root/"artifacts/overnight_20261107/data/condition_schema_uma.json";schema=load_schema(schema_path)
    if str(bank["condition_schema_id"].item())!=schema["identity"]:raise ValueError("Seed bank condition schema differs")
    with np.load(pairs,allow_pickle=False) as z:
        split=z["split"].astype(str);components=z["component_ids"].astype(str)
        train_identities=set(z["canonical_seed_keys"][split=="train"].astype(str))|set(z["canonical_child_keys"][split=="train"].astype(str))
        val_identities=set(z["canonical_seed_keys"][split=="val"].astype(str))|set(z["canonical_child_keys"][split=="val"].astype(str))
        if set(bank["canonical_seed_keys"].astype(str))&(train_identities|val_identities):raise ValueError("Heldout seed identity crosses editor training/validation roles")
        if set(bank["component_ids"].astype(str))&set(components[np.isin(split,["train","val"])]):raise ValueError("Heldout seed component crosses roles")
    history,history_count,prior_sources=freeze_history(root,out,parent_sha)
    paths=dict(parent=parent,seedbank=seedbank,pair_data=pairs,preparation=preparation,schema=schema_path,required_gate=gate,history=history)
    sources={name:dict(path=str(path.resolve()),sha256=sha256(path)) for name,path in paths.items()}
    sources.update(prior_sources)
    checkpoint_records={str(seed):dict(path=str(path),sha256=sha256(path)) for seed,path in checkpoints.items()}
    protocol=dict(status="FROZEN_BEFORE_GENERATION_AND_SCORING",created_UTC=datetime.now(timezone.utc).isoformat(),
        namespace=a.namespace,experiment_phase="optional_editor",fit_seeds=list(FIT_SEEDS),counts=list(COUNTS),requested_bins=list(BINS),
        count_per_composition_bin_fit_method=a.count,methods=["learned_edit","mutation"],
        expected_attempts=2*3*3*3*a.count,expected_paired_trials=3*3*3*a.count,sources=sources,checkpoints=checkpoint_records,
        known_history_orbits=history_count,history_scope="Core pre-final history plus all core final proposal identities; no core final energies used for selection",
        paired_seed_policy="Uniform with replacement among nested-heldout seed families with matching composition and H0seedbin=requested child bin",
        paired_edit_policy="Same actual seed and independently uniform k in1/2/3 per paired trial; same removal/addition uniforms; latent only for learned editor",
        geometry_protocol="pristine-derived deletion; unrelaxed; exact canonical parent coordinates",
        representation="Inference/learning preserve raw seed-child alignment; independent canonicalization used only for physical scoring and novelty",
        requested_condition="[nLi/48,onehot(child energy bin),k/3]", source_sha256=sha256(__file__),
        all_attempts_retained=True,includes_seed_SCFS=True,no_MACE_required=True,no_final_energy_feedback=True,
        limitations="Nested-heldout editor seed families are known H0TRAIN structures, not globally novel seeds; one paired random stream per independent editor fit")
    write_json(out/"protocol.json",protocol)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    records=[];raw_seed=[];raw_child=[];raw_remove=[];raw_add=[];raw_ur=[];raw_ua=[];raw_z=[]
    candidates={};candidate_roles={};failures=[];started=time.time()
    for fit_seed,checkpoint in checkpoints.items():
        model,cfg=load_edit_model(checkpoint,a.device)
        if cfg["parent_sha256"]!=parent_sha or cfg["data_sha256"]!=sha256(pairs) or cfg["condition_schema_id"]!=schema["identity"]:
            raise ValueError("Editor checkpoint parent/data/schema differs from frozen comparison")
        actual_seed=cfg.get("seed",cfg.get("fit_seed"))
        if actual_seed!=fit_seed:raise ValueError("Checkpoint actual fit seed differs from its requested label")
        for n in COUNTS:
            for requested in BINS:
                chosen,k=paired_trials(bank,fit_seed,n,requested,a.count,a.namespace)
                seed=bank["occupancy"][chosen].astype(np.uint8)
                c=encode_condition(np.full(a.count,n),requested,k,48)
                rng_seed=seed_for(a.namespace,fit_seed,n,requested,"shared-subset-uniforms")
                g=torch.Generator(device=a.device).manual_seed(rng_seed)
                ur=torch.rand((a.count,48),device=a.device,generator=g)
                ua=torch.rand((a.count,48),device=a.device,generator=g)
                z=torch.randn((a.count,model.latent_dim),device=a.device,generator=g)
                st=torch.tensor(seed,dtype=torch.float32,device=a.device);ct=torch.tensor(c,device=a.device);kt=torch.tensor(k,device=a.device)
                fingerprints=dict(uniforms_remove_sha256=hashlib.sha256(ur.cpu().numpy().tobytes()).hexdigest(),
                    uniforms_add_sha256=hashlib.sha256(ua.cpu().numpy().tobytes()).hexdigest())
                ur_cpu=ur.cpu().numpy();ua_cpu=ua.cpu().numpy();z_cpu=z.cpu().numpy()
                seed_keys=bank["canonical_seed_keys"][chosen].astype(str)
                for key in seed_keys:
                    candidates[key]=f"mask48_{int(key,2):012x}";candidate_roles.setdefault(key,set()).add("seed")
                for method in ["learned_edit","mutation"]:
                    supported=supported_trials(cfg,n,requested,k) if method=="learned_edit" else np.ones(a.count,dtype=bool)
                    valid=np.flatnonzero(supported)
                    errors=np.where(supported,"","Unsupported TRAIN condition (composition, child bin, swap budget)").astype(object)
                    result={name:np.full((a.count,48),-1,np.int8) for name in ["child","removed","added"]}
                    child_keys=np.full(a.count,"",dtype="<U48")
                    try:
                        if len(valid):
                            vi=torch.as_tensor(valid,device=a.device)
                            with torch.no_grad():
                                sampled=(model.sample(st[vi],ct[vi],kt[vi],z=z[vi],uniforms_remove=ur[vi],uniforms_add=ua[vi]) if method=="learned_edit" else
                                         uniform_edits(st[vi],kt[vi],uniforms_remove=ur[vi],uniforms_add=ua[vi]))
                            sampled={name:value.cpu().numpy() for name,value in sampled.items()}
                            verify_edit(seed[valid],sampled["child"],sampled["removed"],sampled["added"],k[valid])
                            for name,value in sampled.items():result[name][valid]=value.astype(np.uint8)
                            child_keys[valid]=canonical_keys(sampled["child"],permutations)
                    except (ValueError,RuntimeError,FloatingPointError,AssertionError) as failure:
                        errors[valid]=f"{type(failure).__name__}: {failure}";child_keys[valid]=""
                    if np.any(errors!=""):
                        failures.append(dict(fit_seed=fit_seed,n_Li=n,requested_bin=requested,method=method,
                            failed_attempts=int(np.count_nonzero(errors!="")),errors=sorted(set(errors[errors!=""].tolist()))))
                    for i,source_row in enumerate(chosen):
                        error=str(errors[i])
                        key=str(child_keys[i]);seed_key=str(seed_keys[i]);pair=f"{a.namespace}-{fit_seed}-{n}-{requested}-{i}"
                        if not error:
                            candidates[key]=f"mask48_{int(key,2):012x}";candidate_roles.setdefault(key,set()).add("child")
                            child=result["child"][i];removed=result["removed"][i];added=result["added"][i]
                        else:child=removed=added=np.full(48,-1,np.int8)
                        row=dict(attempt_id=f"{pair}-{method}",paired_trial_id=pair,method=method,fit_seed=fit_seed,replicate_id=fit_seed,
                            n_target=n,n_Li=n,requested_bin=requested,effective_model_class=requested,status="generated" if not error else "failed_request",
                            error=error,structure_id=candidates[key] if key else "",canonical_mask=key,
                            seed_structure_id=str(bank["seed_ids"][source_row]),seed_scoring_id=candidates[seed_key],canonical_seed_mask=seed_key,
                            heldout_seed_bank_row=int(source_row),H0_source_row=int(bank["H0_source_rows"][source_row]),
                            seed_component_id=str(bank["component_ids"][source_row]),seed_bin=str(bank["seed_bin"][source_row]),swap_count=int(k[i]),
                            removed_sites="|".join(map(str,np.flatnonzero(removed==1))),added_sites="|".join(map(str,np.flatnonzero(added==1))),
                            raw_row=len(raw_seed),sampling_seed=rng_seed,parent_sha256=parent_sha,condition_schema_id=schema["identity"],
                            checkpoint_sha256=checkpoint_records[str(fit_seed)]["sha256"] if method=="learned_edit" else "",
                            seedbank_sha256=sources["seedbank"]["sha256"],geometry_protocol=protocol["geometry_protocol"],**fingerprints)
                        records.append(row);raw_seed.append(seed[i]);raw_child.append(child);raw_remove.append(removed);raw_add.append(added)
                        raw_ur.append(ur_cpu[i]);raw_ua.append(ua_cpu[i]);raw_z.append(z_cpu[i] if method=="learned_edit" else np.full(model.latent_dim,np.nan))
        if sha256(checkpoint)!=checkpoint_records[str(fit_seed)]["sha256"]:raise ValueError("Checkpoint changed during generation")
    assert len(records)==protocol["expected_attempts"]
    keys=sorted(candidates);masks=np.array([[int(v) for v in key] for key in keys],np.uint8)
    np.savez_compressed(out/"candidates.npz",masks=masks,occupancy=1-masks,structure_ids=np.array([candidates[key] for key in keys]),
        n_Li=48-masks.sum(1),n_atoms=192-masks.sum(1),parent_sha256=np.asarray(parent_sha),
        scoring_role=np.array(["|".join(sorted(candidate_roles[key])) for key in keys]))
    np.savez_compressed(out/"raw_attempts.npz",seed_occupancy=np.array(raw_seed,np.int8),occupancy=np.array(raw_child,np.int8),
        removed=np.array(raw_remove,np.int8),added=np.array(raw_add,np.int8),
        uniforms_remove=np.array(raw_ur,np.float32),uniforms_add=np.array(raw_ua,np.float32),latent=np.array(raw_z,np.float32))
    write_csv(out/"attempts.csv",records)
    for source in sources.values():
        if sha256(source["path"])!=source["sha256"]:raise ValueError("Frozen comparison source changed")
    receipt=dict(status="FROZEN_BEFORE_SCORING",attempts=len(records),paired_trials=protocol["expected_paired_trials"],
        unique_query_geometries=len(keys),unique_seed_geometries=sum("seed" in r for r in candidate_roles.values()),
        unique_child_geometries=sum("child" in r for r in candidate_roles.values()),
        failed_attempts=sum(r["status"]!="generated" for r in records),failures=failures,
        method_attempts=dict(Counter(r["method"] for r in records)),raw_orientation_preserved=True,
        baseline_and_model_share_actual_seed_budget_and_uniforms=True,no_postgeneration_energy_filter=True,
        fit_replicates=3,source_sha256=sha256(__file__),protocol_sha256=sha256(out/"protocol.json"),
        outputs={name:sha256(out/name) for name in ["attempts.csv","raw_attempts.npz","candidates.npz"]},
        elapsed_seconds=time.time()-started,device=a.device,torch=torch.__version__,dtype="float32",tf32=False)
    write_json(out/"receipt.json",receipt);print(json.dumps(receipt,indent=2))


if __name__=="__main__":main()

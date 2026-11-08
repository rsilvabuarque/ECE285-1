"""Frozen, paired occupancy proposals with explicit inputs and no energy filtering.

All primary attempts, including repeats and unsupported requests, are retained.
Canonicalization merges only physical queries; it never reduces denominators.
The optional alternate-latent probes are recorded separately from primary rows.
"""
import argparse,csv,hashlib,inspect,json,os,time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
from cathode_cvae.conditions import classify,load_schema
from cathode_cvae.data import sha256,target_indices
from cathode_cvae.learning.generate_mask import load_model,resolve_request,canonical_masks,attempt_records
from cathode_cvae.learning.models import sample_exact_count
from cathode_cvae.runtime import require_compute_allocation


def seed_for(*parts):return int.from_bytes(hashlib.sha256("|".join(map(str,parts)).encode()).digest()[:4],"little")
def fingerprint(array):return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
def write_csv(path,rows):
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
def read_keys(path):
    value=json.loads(Path(path).read_text())
    if isinstance(value,dict):value=next(value[k] for k in ("keys","canonical_masks","known_keys","canonical_keys") if k in value)
    return set(map(str,value))


def validate_manifest(manifest,bindings,namespace,count,diagnostics):
    if not str(manifest.get("status","")).startswith("FROZEN"):raise ValueError("Manifest must be frozen before proposals")
    aliases={"parent":"expected_parent_sha256","data":"expected_training_data_sha256","schema":"expected_condition_schema_sha256","history":"history_sha256"}
    for name,digest in bindings.items():
        expected=manifest.get("inputs",{}).get(name,{}).get("sha256",manifest.get(aliases[name],manifest.get(name+"_sha256")))
        if expected is not None and expected!=digest:raise ValueError("Frozen "+name+" binding differs")
    if manifest.get("namespace",namespace)!=namespace:raise ValueError("Frozen namespace differs")
    if manifest.get("count_per_composition_bin_replicate",manifest.get("count_per_group",count))!=count:raise ValueError("Frozen draw budget differs")
    if manifest.get("diagnostics",diagnostics)!=diagnostics:raise ValueError("Frozen diagnostic scope differs")
    labels=[(r["method"],r.get("replicate_id",r.get("fit_seed"))) for r in manifest["models"]]
    if len(labels)!=len(set(labels)):raise ValueError("Duplicate model/method replicate in frozen manifest")
    expected=len(manifest["models"])*len(manifest["counts"])*len(manifest.get("requested_bins",["low","middle","high"]))*count
    if manifest.get("expected_primary_attempts",expected)!=expected:raise ValueError("Manifest attempt budget inconsistent")
    return expected


def fixed_count_uniform(uniforms,n):
    return sample_exact_count(torch.zeros_like(uniforms),n,uniforms=uniforms)


def mutation_children(seeds,swaps,remove_uniforms,add_uniforms):
    """Exact-k removals/additions selected from the ORIGINAL aligned seed."""
    x=np.asarray(seeds).copy();removed=np.zeros_like(x);added=np.zeros_like(x)
    for i,(seed,k) in enumerate(zip(seeds,swaps)):
        occupied=np.flatnonzero(seed);vacant=np.flatnonzero(1-seed)
        if k>min(len(occupied),len(vacant)):raise ValueError("Requested edit budget exceeds seed support")
        take=occupied[np.argsort(remove_uniforms[i,occupied],kind="stable")[:k]]
        give=vacant[np.argsort(add_uniforms[i,vacant],kind="stable")[:k]]
        removed[i,take]=1;added[i,give]=1;x[i,take]=0;x[i,give]=1
    return x,removed,added


def model_draw(model,cfg,n,requested,z,uniforms,schema,descriptor_range=None):
    try:
        vector,*_=resolve_request(cfg,n,requested,None,schema,descriptor_range)
        condition=torch.tensor(vector,device=z.device).repeat(len(z),1)
        with torch.no_grad():result=model.sample(condition,n,z=z[:,:model.latent_dim],uniforms=uniforms).cpu().numpy()
        return result,None
    except (ValueError,RuntimeError,FloatingPointError) as failure:
        return None,f"{type(failure).__name__}: {failure}"


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","parent","data","condition-schema","history","output-dir"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--source-root",type=Path);p.add_argument("--namespace");p.add_argument("--count",type=int,default=64)
    p.add_argument("--diagnostics",action="store_true");p.add_argument("--device",default="cpu",choices=["cpu","cuda"]);p.add_argument("--threads",type=int,default=4)
    a=p.parse_args(argv);require_compute_allocation()
    if a.count<1:raise ValueError("Positive attempt budget required")
    torch.set_num_threads(a.threads);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source_root=(a.source_root or a.manifest.parent).resolve()
    def resolve(path):
        path=Path(path);return path if path.is_absolute() else source_root/path
    manifest=json.loads(a.manifest.read_text());namespace=a.namespace or manifest["namespace"]
    sources=dict(parent=a.parent,data=a.data,schema=a.condition_schema,history=a.history)
    bindings={k:sha256(v) for k,v in sources.items()};expected=validate_manifest(manifest,bindings,namespace,a.count,a.diagnostics)
    schema=load_schema(a.condition_schema)
    if manifest.get("expected_condition_schema_id",schema["identity"])!=schema["identity"]:raise ValueError("Condition schema identity differs")
    with np.load(a.parent,allow_pickle=False) as z:parent={k:z[k] for k in z.files}
    permutations=parent["permutations"];sites=permutations.shape[1];host_atoms=len(parent["numbers"])-sites
    if len(target_indices(parent))!=sites or schema["n_sites"]!=sites:raise ValueError("Parent/schema target dimensions differ")
    if any(int(n)!=n or not 0<=n<=sites for n in manifest["counts"]):raise ValueError("Frozen counts must be integers within parent size")
    history=read_keys(a.history)
    if any(len(k)!=sites or set(k)-{"0","1"} for k in history):raise ValueError("History belongs to another parent dimension")
    with np.load(a.data,allow_pickle=False) as z:
        data={k:z[k] for k in ("occupancy","split","structure_ids","energy_eV_atom")}
        data["n_target"]=z["n_target"] if "n_target" in z else z["n_Li"]
    train=np.flatnonzero(data["split"].astype(str)=="train");train_labels=classify(schema,data["n_target"][train],data["energy_eV_atom"][train])
    train_keys,_=canonical_masks(data["occupancy"][train],permutations)
    if not set(train_keys)<=history:raise ValueError("Frozen novelty history omits known TRAIN masks")
    a.output_dir.mkdir(parents=True,exist_ok=False)
    protocol=dict(status="FROZEN_BEFORE_GENERATION_AND_SCORING",namespace=namespace,source_sha256=sha256(__file__),
        manifest=dict(path=str(a.manifest.resolve()),sha256=sha256(a.manifest)),
        inputs={k:dict(path=str(v.resolve()),sha256=bindings[k]) for k,v in sources.items()},
        condition_schema_id=schema["identity"],expected_primary_attempts=expected,diagnostics=a.diagnostics,
        counts=manifest["counts"],requested_bins=manifest.get("requested_bins",["low","middle","high"]),count_per_group=a.count,
        latent_stream_width=int(manifest.get("latent_stream_width",32)),
        random_scope="Common actual standard-normal latent and AR uniforms within replicate/count across model arms and requests; mutation has shared seed-choice/k/edit streams across requests",
        geometry_protocol="pristine-derived deletion; unrelaxed; exact canonical parent coordinates",no_energy_filtering=True)
    write_json(a.output_dir/"protocol.json",protocol)
    rows=[];raw=[];seed_raw=[];latent_raw=[];uniform_raw=[];remove_raw=[];add_raw=[];models=[];started=time.perf_counter()
    bins=manifest.get("requested_bins",["low","middle","high"]);counts=manifest["counts"]
    width=int(manifest.get("latent_stream_width",32))

    def append(x,entry,n,requested,diagnostic,common_seed,cfg,cksha,z,u,seeds=None,extra=None,error=None,remove_u=None,add_u=None):
        batch=a.count;values=np.full((batch,sites),np.nan,np.float32) if x is None else np.asarray(x)
        if values.shape!=(batch,sites):raise ValueError("Sampler returned wrong shape")
        valid=np.isin(values,[0,1]).all(1)&(values.sum(1)==n)
        keys=np.full(batch,"",dtype=f"<U{sites}");ids=np.full(batch,"",dtype=f"<U{12+(sites+3)//4}")
        if valid.any():
            kk,_=canonical_masks(values[valid],permutations);keys[valid]=kk
            ids[valid]=[f"mask{sites}_{int(k,2):0{(sites+3)//4}x}" for k in kk]
        mode=None if cfg is None else cfg.get("condition_mode","energy")
        effective="frozen-sampler" if cfg is None else requested if mode=="multibin" else "unconditioned" if mode=="composition" else "low" if requested=="low" else "rest"
        rr=attempt_records(batch,ids,keys,method=entry["method"],fit_seed=entry.get("fit_seed",""),sampling_seed=common_seed,
            n_target=n,requested_bin=requested,effective_model_class=effective,schema_id=schema["identity"],
            checkpoint_sha256=cksha,parent_sha256=bindings["parent"],geometry_protocol=protocol["geometry_protocol"])
        rep=entry.get("replicate_id",entry.get("fit_seed"));actual_dim=0 if cfg is None else int(cfg["latent_dim"])
        actual_z=np.zeros((batch,width),np.float32)
        if actual_dim:actual_z[:,:actual_dim]=z[:,:actual_dim]
        for i,row in enumerate(rr):
            j=len(raw);raw.append(values[i].astype(np.float32));seed_raw.append(np.full(sites,np.nan,np.float32) if seeds is None else seeds[i]);latent_raw.append(actual_z[i]);uniform_raw.append(u[i])
            remove_raw.append(np.full(sites,np.nan,np.float32) if remove_u is None else remove_u[i]);add_raw.append(np.full(sites,np.nan,np.float32) if add_u is None else add_u[i])
            row.update(attempt_id=f"{namespace}-{entry['method']}-{rep}-{n}-{requested}-{diagnostic}-{i}",replicate_id=rep,
                diagnostic=diagnostic,raw_row=j,paired_id=f"{namespace}-{rep}-{n}-{i}",n_target=n,
                status=("generated" if valid[i] and error is None else "failed_request" if error else "invalid_mask"),error=error or "",
                allocation_policy=namespace,allocation_run_id=f"{entry['method']}-{n}",allocation_weight=1/len(bins),
                latent_dim=actual_dim,latents_sha256=fingerprint(actual_z[:,:actual_dim]),uniforms_sha256=fingerprint(u),
                energy_target=entry.get("energy_target","deletion" if cfg is None else cfg.get("energy_target","deletion")))
            if extra:row.update(extra[i])
            rows.append(row)

    for entry in manifest["models"]:
        ck=resolve(entry["checkpoint"]) if entry.get("checkpoint") else None
        if ck:
            cksha=sha256(ck);wanted=entry.get("expected_checkpoint_sha256",entry.get("checkpoint_sha256"))
            if wanted is None or wanted!=cksha:raise ValueError("Every checkpoint needs a matching frozen SHA")
            model,cfg=load_model(ck,a.device)
            if cfg.get("parent_sha256") not in {None,bindings["parent"]}:raise ValueError("Checkpoint parent differs")
            if cfg.get("data_sha256") not in {None,bindings["data"]}:raise ValueError("Checkpoint fitting data differs from supplied data")
            if cfg.get("seed")!=entry.get("fit_seed"):raise ValueError("Manifest fit seed differs from checkpoint")
            if model.latent_dim>width:raise ValueError("Frozen common latent stream is too narrow")
            models.append(dict(**entry,actual_checkpoint_sha256=cksha,
                fit_trainer_sha256=cfg.get("trainer_sha256"),fit_model_source_sha256=cfg.get("model_source_sha256"),
                runtime_model_source_sha256=sha256(inspect.getsourcefile(type(model))),
                checkpoint_format_version=cfg.get("model_format_version"),
                compatibility_scope="Strict versioned architecture/state reload; runtime generation source is separately bound, never inferred by padding weights"))
        else:model=None;cfg=None;cksha="not-applicable"
        for n in counts:
            rep=entry.get("replicate_id",entry.get("fit_seed"));common_seed=seed_for(namespace,rep,n,"paired-model-stream")
            generator=torch.Generator(device=a.device).manual_seed(common_seed)
            zz=torch.randn((a.count,width),generator=generator,device=a.device);uu=torch.rand((a.count,sites),generator=generator,device=a.device)
            z=zz.cpu().numpy();u=uu.cpu().numpy()
            rng=np.random.default_rng(seed_for(namespace,rep,n,"mutation-stream"))
            choices=rng.random(a.count);swaps=rng.integers(1,min(3,n,sites-n)+1,size=a.count) if 0<n<sites else np.zeros(a.count,int)
            remove_u=rng.random((a.count,sites)).astype(np.float32);add_u=rng.random((a.count,sites)).astype(np.float32)
            for requested in bins:
                if model is not None:
                    x,error=model_draw(model,cfg,n,requested,zz,uu,schema,entry.get("descriptor_range"))
                    append(x,entry,n,requested,"primary",common_seed,cfg,cksha,z,u,error=error)
                    if a.diagnostics and requested=="low" and model.latent_dim:
                        g=torch.Generator(device=a.device).manual_seed(seed_for(common_seed,"latent-B"));zb=torch.randn((a.count,width),generator=g,device=a.device)
                        xb,error_b=model_draw(model,cfg,n,requested,zb,uu,schema,entry.get("descriptor_range"))
                        append(xb,entry,n,requested,"latent_B",common_seed,cfg,cksha,zb.cpu().numpy(),u,error=error_b)
                elif entry.get("sampler")=="uniform":
                    append(fixed_count_uniform(uu,n).cpu().numpy(),entry,n,requested,"primary",common_seed,None,cksha,z,u)
                elif entry.get("sampler")=="mutation":
                    pool=train[(data["n_target"][train]==n)&(train_labels==requested)]
                    if not len(pool):append(None,entry,n,requested,"primary",common_seed,None,cksha,z,u,error="No TRAIN seed supports this requested bin");continue
                    chosen=pool[np.minimum((choices*len(pool)).astype(int),len(pool)-1)]
                    seeds=data["occupancy"][chosen].astype(np.uint8)
                    x,removed,added=mutation_children(seeds,swaps,remove_u,add_u)
                    extra=[dict(seed_structure_id=str(data["structure_ids"][i]),seed_source_row=int(i),seed_source_sha256=bindings["data"],swap_count=int(k),
                        removed_sites="|".join(map(str,np.flatnonzero(r))),added_sites="|".join(map(str,np.flatnonzero(g)))) for i,k,r,g in zip(chosen,swaps,removed,added)]
                    append(x,entry,n,requested,"primary",common_seed,None,cksha,z,u,seeds,extra,remove_u=remove_u,add_u=add_u)
                else:raise ValueError("Unknown frozen sampler")
        if ck and sha256(ck)!=cksha:raise ValueError("Checkpoint changed during generation")
    primary=[r for r in rows if r["diagnostic"]=="primary"]
    if len(primary)!=expected:raise ValueError("Actual primary denominator differs from frozen budget")
    def save_pool(name,selected):
        codes=sorted({int(r["canonical_mask"],2) for r in selected if r["status"]=="generated"})
        masks=np.asarray([[(code>>i)&1 for i in range(sites-1,-1,-1)] for code in codes],np.uint8).reshape(-1,sites)
        np.savez_compressed(a.output_dir/name,masks=masks,occupancy=1-masks,
            structure_ids=np.asarray([f"mask{sites}_{code:0{(sites+3)//4}x}" for code in codes]),
            n_target=sites-masks.sum(1),n_Li=sites-masks.sum(1),n_atoms=host_atoms+sites-masks.sum(1),parent_sha256=np.asarray(bindings["parent"]))
        return len(codes)
    unique=save_pool("candidates.npz",rows)
    save_pool("primary_candidates.npz",primary)
    save_pool("diagnostic_candidates.npz",[r for r in rows if r["diagnostic"]!="primary"])
    raw_arrays=dict(occupancy=np.asarray(raw),seed_occupancy=np.asarray(seed_raw),latents=np.asarray(latent_raw),
        uniforms=np.asarray(uniform_raw),remove_uniforms=np.asarray(remove_raw),add_uniforms=np.asarray(add_raw))
    np.savez_compressed(a.output_dir/"raw_attempts.npz",**raw_arrays)
    for name,mask in [("raw_primary.npz",np.asarray([r["diagnostic"]=="primary" for r in rows])),
                      ("raw_diagnostics.npz",np.asarray([r["diagnostic"]!="primary" for r in rows]))]:
        np.savez_compressed(a.output_dir/name,global_raw_row=np.flatnonzero(mask),**{k:v[mask] for k,v in raw_arrays.items()})
    write_csv(a.output_dir/"attempts.csv",primary);write_csv(a.output_dir/"attempts_all.csv",rows)
    if any(sha256(sources[k])!=digest for k,digest in bindings.items()):raise ValueError("Frozen input changed during generation")
    receipt=dict(status="FROZEN_BEFORE_SCORING",primary_attempts=len(primary),total_attempts=len(rows),unique_geometries=unique,
        failed_attempts=sum(r["status"]!="generated" for r in rows),elapsed_seconds=time.perf_counter()-started,
        source_sha256=sha256(__file__),protocol_sha256=sha256(a.output_dir/"protocol.json"),checkpoints=models,
        runtime_loader_source=dict(path=inspect.getsourcefile(load_model),sha256=sha256(inspect.getsourcefile(load_model))),
        outputs={name:sha256(a.output_dir/name) for name in ("attempts.csv","attempts_all.csv","raw_attempts.npz","raw_primary.npz","raw_diagnostics.npz","candidates.npz","primary_candidates.npz","diagnostic_candidates.npz")},
        no_energy_filtering=True,new_MLIP_calls=0,new_training_updates=0)
    write_json(a.output_dir/"receipt.json",receipt);print(json.dumps(receipt,indent=2));return receipt


if __name__=="__main__":main()

"""Bounded posthoc Gaussian proposals from TRAIN aggregate-posterior moments.

This changes latent sampling for frozen dense CVAEs, not their trained prior or
weights. The class-conditional Gaussian matches two analytic moments of a
mixture; it is not the mixture itself. Generation uses common base Gaussian
noise and AR uniforms, but condition-specific transforms change actual z.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time

IMPORT_STARTED=time.perf_counter()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch

from cathode_cvae.acquisition import canonical_keys
from cathode_cvae.conditions import classify,encode_requests,load_schema
from cathode_cvae.learning.generate_mask import load_model,resolve_request,attempt_records
from cathode_cvae.runtime import require_compute_allocation


ROOT=Path(__file__).resolve().parents[2]
NAMESPACE="conditioning-aggregate-prior-20261107-v1"
SEEDS=(20261106,20261107,20261108)
COUNTS=(12,24,36)
BINS=("low","middle","high")
GATE_SHA="c6a7105dd9578c5a1caac1c8bc6acfb2bc134e8fd1ce04e2a0c8def232f91bf3"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""):h.update(b)
    return h.hexdigest()


def dump(path,value):
    with Path(path).open("x") as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n")


def write_csv(path,rows):
    import csv
    with Path(path).open("x",newline="") as f:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def seed_for(*values):return int.from_bytes(hashlib.sha256("|".join(map(str,values)).encode()).digest()[:4],"little")


def gaussian_moments(mu,logvar):
    """Population total covariance: E[diag(sigma²)] + Cov(mu), denominator N."""
    mu=np.asarray(mu,np.float64);logvar=np.asarray(logvar,np.float64)
    if mu.ndim!=2 or mu.shape!=logvar.shape or not len(mu) or not np.isfinite(mu).all() or not np.isfinite(logvar).all():
        raise ValueError("Finite, nonempty matching posterior parameter arrays required")
    with np.errstate(over="ignore",under="ignore"):
        posterior_variance=np.exp(logvar)
    if not np.isfinite(posterior_variance).all() or not (posterior_variance>0).all():
        raise ValueError("Posterior variances must remain finite and strictly positive; no jitter repair")
    center=mu.mean(0);residual=mu-center
    covariance=np.einsum("ni,nj->ij",residual,residual)/len(mu)+np.diag(posterior_variance.mean(0))
    if not np.isfinite(covariance).all() or not np.array_equal(covariance,covariance.T):
        raise ValueError("Analytic covariance is nonfinite or asymmetric")
    cholesky=np.linalg.cholesky(covariance)
    eigenvalues=np.linalg.eigvalsh(covariance)
    if not np.all(eigenvalues>0):raise ValueError("Analytic covariance is not strictly positive definite")
    return center,covariance,cholesky,eigenvalues


def training_examples(data):
    """Select TRAIN before consuming energies or encoding any configurations."""
    selected=np.flatnonzero(np.asarray(data["split"]).astype(str)=="train")
    if not len(selected):raise ValueError("No protected TRAIN examples")
    keys=np.asarray(data["canonical_masks"])[selected].astype(str)
    x=1-np.asarray([[int(b) for b in key] for key in keys],np.uint8)
    n=np.asarray(data["n_Li"])[selected];energy=np.asarray(data["energy_eV_atom"])[selected]
    ids=np.asarray(data["structure_ids"])[selected].astype(str)
    if x.shape[1]!=48 or not np.array_equal(x.sum(1),n) or len(set(keys))!=len(keys):
        raise ValueError("TRAIN canonical occupancy/count identities differ")
    return dict(occupancy=x,n_Li=n,energy_eV_atom=energy,structure_ids=ids,canonical_masks=keys)


def inputs(root):
    run=root/"runs/conditioning_20261107"
    return dict(parent=root/"artifacts/current/core/data/parents/O3_N48_pilot.npz",
        data=root/"artifacts/current/improvement_v1/training_round0.npz",
        schema=root/"artifacts/overnight_20261107/data/condition_schema_uma.json",
        frozen_models=run/"final_generation_manifest.json",gate=root/"artifacts/conditioning_20261107/required_gate.json")


def prepare(root,out,batch_size=512):
    start=time.perf_counter();paths=inputs(root);before={name:sha(p) for name,p in paths.items()}
    if before["gate"]!=GATE_SHA or json.loads(paths["gate"].read_text())["status"]!="COMPLETE":
        raise ValueError("Required packet must remain complete and unchanged")
    manifest=json.loads(paths["frozen_models"].read_text())
    models=[r for r in manifest["models"] if r["method"]=="cvae_d4"]
    if len(models)!=3 or {r["fit_seed"] for r in models}!=set(SEEDS):raise ValueError("Use the exact three frozen dense D4 fits")
    schema=load_schema(paths["schema"])
    with np.load(paths["data"],allow_pickle=False) as z:
        data=training_examples({k:z[k] for k in ["split","canonical_masks","n_Li","energy_eV_atom","structure_ids"]})
    labels=classify(schema,data["n_Li"],data["energy_eV_atom"])
    c=encode_requests(schema,data["n_Li"],labels,mode="multibin")
    if len(data["occupancy"])!=40157:raise ValueError("Protected TRAIN row count differs")
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    all_mu=[];all_cov=[];all_chol=[];fit_ids=[];counts=[];bins=[];sample_sizes=[];records=[];checkpoint_records=[]
    calls=0;encoded=0;encoder_seconds=0.;load_seconds=0.;moments_seconds=0.
    for entry in sorted(models,key=lambda r:r["fit_seed"]):
        checkpoint=Path(entry["checkpoint"]);expected=entry["expected_checkpoint_sha256"]
        if sha(checkpoint)!=expected:raise ValueError("Frozen dense checkpoint changed")
        t=time.perf_counter();model,cfg=load_model(checkpoint,"cpu");load_seconds+=time.perf_counter()-t
        if (cfg["latent_dim"]!=4 or cfg["n_sites"]!=48 or cfg["condition_dim"]!=4 or cfg.get("encoder_kind")!="dense"
                or cfg.get("graph_context") or cfg["parent_sha256"]!=before["parent"] or cfg["data_sha256"]!=before["data"]
                or cfg["condition_schema_id"]!=schema["identity"]):
            raise ValueError("This is not the fixed dense D4/H0/schema comparison")
        mus=[];lvs=[]
        with torch.inference_mode():
            for first in range(0,len(c),batch_size):
                x=torch.tensor(data["occupancy"][first:first+batch_size],dtype=torch.float32)
                condition=torch.tensor(c[first:first+batch_size]);t=time.perf_counter()
                mu,lv=model.encode(x,condition);encoder_seconds+=time.perf_counter()-t
                calls+=1;encoded+=len(x);mus.append(mu.numpy().copy());lvs.append(lv.numpy().copy())
        mu=np.concatenate(mus);lv=np.concatenate(lvs)
        for n in COUNTS:
            for label in BINS:
                selected=(data["n_Li"]==n)&(labels==label);t=time.perf_counter()
                m,S,L,eigen=gaussian_moments(mu[selected],lv[selected]);moments_seconds+=time.perf_counter()-t
                all_mu.append(m);all_cov.append(S);all_chol.append(L);fit_ids.append(entry["fit_seed"])
                counts.append(n);bins.append(label);sample_sizes.append(int(selected.sum()))
                records.append(dict(fit_seed=entry["fit_seed"],n_Li=n,energy_bin=label,TRAIN_rows=int(selected.sum()),
                    mean=m.tolist(),covariance=S.tolist(),cholesky=L.tolist(),eigenvalues=eigen.tolist(),
                    marginal_posterior_variance_mean=np.exp(lv[selected].astype(np.float64)).mean(0).tolist()))
        if sha(checkpoint)!=expected:raise ValueError("Frozen dense weights changed during moment extraction")
        checkpoint_records.append(dict(fit_seed=entry["fit_seed"],path=str(checkpoint),sha256=expected,
            model_source_sha256=cfg["model_source_sha256"],trainer_sha256=cfg["trainer_sha256"]))
    np.savez_compressed(out/"moments.npz",means=np.asarray(all_mu),covariances=np.asarray(all_cov),
        cholesky=np.asarray(all_chol),fit_seeds=np.asarray(fit_ids),n_Li=np.asarray(counts),energy_bins=np.asarray(bins),
        TRAIN_rows=np.asarray(sample_sizes),parent_sha256=np.asarray(before["parent"]),data_sha256=np.asarray(before["data"]),
        condition_schema_id=np.asarray(schema["identity"]))
    dump(out/"moment_groups.json",records)
    after={name:sha(p) for name,p in paths.items()}
    if before!=after:raise ValueError("Protected moment-preparation inputs changed")
    protocol=dict(status="FROZEN_BEFORE_NEW_PRIOR_PROPOSALS",created_UTC=datetime.now(timezone.utc).isoformat(),
        namespace=NAMESPACE,hypothesis="TRAIN class-conditional Gaussian aggregate-posterior moments improve prior generation for active dense latents whose standard-prior condition response is weak",
        changed_factor="Posthoc Gaussian latent proposal versus standardN(0,I); identical fixed checkpoint, request, baseepsilon and ARuniforms",
        approximation="Analytic first two moments of q_aggregate=sum_i q(z|TRAINmask_i,c)/N; replace mixture with one full4D Gaussian percomposition/trueTRAINbin",
        not_a_trained_conditional_prior=True,not_exact_aggregate_mixture=True,NN_optimizer_updates=0,
        training_scope="Protected H0TRAIN only; no validation/test/final labels or encoded masks in moment estimates",
        methods=["standard_prior","aggregate_prior"],counts=list(COUNTS),requested_bins=list(BINS),fit_seeds=list(SEEDS),
        attempts_per_count_bin_fit_method=128,total_attempts=6912,teachers=["uma","mace"],maximum_logical_static_queries=13824,
        coupling="Same baseepsilon and ARuniforms within eachfit/count acrossrequests/methods; actualaggregatez differs acrossconditions, so this is NOT a same-z condition-response test",
        moment_estimator="Population denominator N, fullcovariance E[diag(exp(logvar))]+E[(mu-m)(mu-m)^T]; strictfinite/positivevariance/PD checks; no shrinkage/jitter/tuning",
        moment_dtype="float64 derived from FP32 CPU encoder outputs",proposal_transform_dtype="float32 for fixed FP32 decoder",
        geometry_protocol="pristine-derived deletion; unrelaxed; exact canonical parent coordinates",
        model_recommendation_unchanged=True,checkpoints=checkpoint_records,
        inputs={name:dict(path=str(p),sha256=before[name]) for name,p in paths.items()},
        moments_sha256=sha(out/"moments.npz"),source_sha256=sha(__file__))
    dump(out/"protocol.json",protocol)
    result=dict(status="PASS_TRAIN_ONLY_MOMENTS",groups=len(records),encoder_calls=calls,encoded_TRAIN_rows=encoded,
        unique_TRAIN_rows_per_fit=len(c),NN_optimizer_updates=0,weights_written=False,weights_unchanged=True,
        checkpoint_load_seconds=load_seconds,encoder_seconds=encoder_seconds,moment_calculation_seconds=moments_seconds,
        elapsed_seconds=time.perf_counter()-start,import_seconds=start-IMPORT_STARTED,device="cpu",threads=4,
        source_sha256=sha(__file__),input_hashes=before,inputs_unchanged=True,
        moments_sha256=sha(out/"moments.npz"),protocol_sha256=sha(out/"protocol.json"),new_MLIP_calls=0,
        job_id=os.environ.get("SLURM_JOB_ID"),step_id=os.environ.get("SLURM_STEP_ID"))
    dump(out/"receipt.json",result);print(json.dumps(result,indent=2));return result


def known_history(root,parent_sha):
    run=root/"runs/conditioning_20261107";core=json.loads((run/"final_protocol.json").read_text())
    base=Path(core["history_path"])
    if sha(base)!=core["history_sha256"]:raise ValueError("Pre-final history changed")
    history=set(json.loads(base.read_text()));registry_path=run/"oracle/registry.json"
    registry=json.loads(registry_path.read_text());registry_sha=sha(registry_path)
    if registry["parent_sha256"]!=parent_sha:raise ValueError("Registered prior masks belong to another parent")
    for value in registry["physical_keys"].values():
        match=re.fullmatch(r"mask48_([0-9a-fA-F]{12})",value["structure_id"])
        if not match:raise ValueError("Registered geometry is not the exact48site parent namespace; never pad another cell")
        history.add(format(int(match.group(1),16),"048b"))
    if sha(registry_path)!=registry_sha:raise ValueError("Registry changed during history freezing")
    return sorted(history),dict(base_history=dict(path=str(base),sha256=sha(base)),
        registered_prior_geometries=dict(path=str(registry_path),sha256=registry_sha,geometry_model_pairs=len(registry["physical_keys"])),
        scope="All registered prior core/editor/temperature mask identities plus frozen base history, irrespective of teacher/cache status; no energy values used")


def transformed_latents(epsilon,center,cholesky):
    """Shared standard epsilon, with actual class-specific transformed z."""
    return epsilon@cholesky.T+center


def generate(root,prepared,out,device="cuda",count=128):
    start=time.perf_counter()
    if count!=128:raise ValueError("The frozen ablation uses exactly128attempts percell/method")
    prior_protocol=json.loads((prepared/"protocol.json").read_text());prior_receipt=json.loads((prepared/"receipt.json").read_text())
    if prior_receipt["status"]!="PASS_TRAIN_ONLY_MOMENTS" or prior_protocol["source_sha256"]!=sha(__file__):
        raise ValueError("Use the unchanged prepared ablation implementation")
    if sha(prepared/"moments.npz")!=prior_protocol["moments_sha256"]:raise ValueError("Moment bytes changed")
    paths=inputs(root);parent_sha=sha(paths["parent"]);schema=load_schema(paths["schema"])
    with np.load(paths["parent"],allow_pickle=False) as z:perms=z["permutations"]
    with np.load(prepared/"moments.npz",allow_pickle=False) as z:moments={k:z[k] for k in z.files}
    lookup={(int(s),int(n),str(b)):(m,L) for s,n,b,m,L in zip(moments["fit_seeds"],moments["n_Li"],moments["energy_bins"],moments["means"],moments["cholesky"])}
    out.mkdir(parents=True,exist_ok=False);history,sources=known_history(root,parent_sha)
    dump(out/"history.json",history)
    protocol=dict(prior_protocol,status="FROZEN_BEFORE_GENERATION_AND_SCORING",generation_frozen_UTC=datetime.now(timezone.utc).isoformat(),
        moment_preparation_protocol_sha256=sha(prepared/"protocol.json"),history_sha256=sha(out/"history.json"),
        history_orbits=len(history),history_sources=sources,all_prespecified_attempts_scored_with_both_teachers=True)
    dump(out/"protocol.json",protocol)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    rows=[];raw=[];eps_rows=[];z_rows=[];u_rows=[];weights_before={}
    for entry in prior_protocol["checkpoints"]:
        checkpoint=Path(entry["path"]);weights_before[str(checkpoint)]=sha(checkpoint)
        if weights_before[str(checkpoint)]!=entry["sha256"]:raise ValueError("Frozen dense weights changed")
        model,cfg=load_model(checkpoint,device);fit=entry["fit_seed"]
        for n in COUNTS:
            random_seed=seed_for(NAMESPACE,fit,n,"base-noise-and-AR")
            rng=torch.Generator(device=device).manual_seed(random_seed)
            epsilon=torch.randn((count,4),generator=rng,device=device);uniforms=torch.rand((count,48),generator=rng,device=device)
            for requested in BINS:
                vector,*_=resolve_request(cfg,n,requested,None,schema);condition=torch.tensor(vector,device=device).repeat(count,1)
                m,L=lookup[fit,n,requested];center=torch.tensor(m,dtype=torch.float32,device=device);factor=torch.tensor(L,dtype=torch.float32,device=device)
                alternatives={"standard_prior":epsilon,"aggregate_prior":transformed_latents(epsilon,center,factor)}
                for method,z in alternatives.items():
                    with torch.inference_mode():x=model.sample(condition,n,z=z,uniforms=uniforms).cpu().numpy()
                    if not np.isin(x,[0,1]).all() or not np.all(x.sum(1)==n):raise ValueError("Exact-count proposal validity lost")
                    keys=canonical_keys(x,perms);ids=[f"mask48_{int(k,2):012x}" for k in keys]
                    attempt=attempt_records(count,ids,keys,method=method,fit_seed=fit,sampling_seed=random_seed,n_target=n,
                        requested_bin=requested,effective_model_class=requested,schema_id=schema["identity"],
                        checkpoint_sha256=entry["sha256"],parent_sha256=parent_sha,geometry_protocol=protocol["geometry_protocol"])
                    eps=epsilon.cpu().numpy();actual=z.cpu().numpy();u=uniforms.cpu().numpy()
                    for i,r in enumerate(attempt):
                        index=len(raw);raw.append(x[i]);eps_rows.append(eps[i]);z_rows.append(actual[i]);u_rows.append(u[i])
                        r.update(attempt_id=f"{NAMESPACE}-{method}-{fit}-{n}-{requested}-{i}",replicate_id=fit,diagnostic="primary",raw_row=index,
                            paired_id=f"{NAMESPACE}-{fit}-{n}-{requested}-{i}",allocation_policy=NAMESPACE,allocation_run_id=f"{method}-{n}",allocation_weight=1/3,
                            base_epsilon_sha256=hashlib.sha256(eps.tobytes()).hexdigest(),latents_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                            uniforms_sha256=hashlib.sha256(u.tobytes()).hexdigest(),latent_proposal=method,
                            coupling_scope="Same baseepsilon/ARuniforms; class-specific actualz changes, not a same-z condition test")
                        rows.append(r)
        if sha(checkpoint)!=entry["sha256"]:raise ValueError("Frozen dense weights changed during sampling")
    if len(rows)!=6912:raise AssertionError("The prespecified two-method proposal budget changed")
    keys=sorted({r["canonical_mask"] for r in rows});masks=np.asarray([[int(b) for b in key] for key in keys],np.uint8)
    np.savez_compressed(out/"candidates.npz",masks=masks,occupancy=1-masks,structure_ids=np.asarray([f"mask48_{int(k,2):012x}" for k in keys]),
        n_Li=48-masks.sum(1),n_atoms=192-masks.sum(1),parent_sha256=np.asarray(parent_sha))
    np.savez_compressed(out/"raw_attempts.npz",occupancy=np.asarray(raw,np.uint8),base_epsilon=np.asarray(eps_rows),latents=np.asarray(z_rows),uniforms=np.asarray(u_rows))
    write_csv(out/"attempts.csv",rows)
    receipt=dict(status="FROZEN_BEFORE_SCORING",attempts=len(rows),unique_geometries=len(keys),teachers=["uma","mace"],
        NN_optimizer_updates=0,weights_unchanged=True,encoder_calls_during_generation=0,history_orbits=len(history),
        all_attempts_retained=True,source_sha256=sha(__file__),protocol_sha256=sha(out/"protocol.json"),
        outputs={name:sha(out/name) for name in ["attempts.csv","raw_attempts.npz","candidates.npz","history.json"]},
        elapsed_seconds=time.perf_counter()-start,new_MLIP_calls=0)
    dump(out/"receipt.json",receipt);print(json.dumps(receipt,indent=2));return receipt


def evaluate(root,generated,out,uma,mace):
    from cathode_cvae.condition_evaluation import evaluate as core_evaluate,plot_tables
    receipt=json.loads((generated/"receipt.json").read_text())
    for name,expected in receipt["outputs"].items():
        if sha(generated/name)!=expected:raise ValueError("Frozen prior-ablation proposals/history changed")
    result=core_evaluate(generated/"attempts.csv",generated/"candidates.npz",{"uma":uma,"mace":mace},
        {teacher:root/f"artifacts/overnight_20261107/data/condition_schema_{teacher}.json" for teacher in ["uma","mace"]},
        root/"runs/conditioning_20261107/data/training_keys.json",generated/"history.json",
        root/"artifacts/current/reports/generation_thresholds.csv",out,inputs(root)["parent"])
    if result["status"]!="PASS":raise ValueError("Preserve unknown energies; do not report a complete ablation from cache hits")
    plot_tables(out);print(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    for name in ["prepare","generate","evaluate"]:
        s=sub.add_parser(name);s.add_argument("--root",type=Path,default=ROOT);s.add_argument("--output-dir",type=Path,required=True)
        if name=="prepare":s.add_argument("--batch-size",type=int,default=512)
        elif name=="generate":
            s.add_argument("--prepared",type=Path,required=True);s.add_argument("--device",default="cuda");s.add_argument("--count",type=int,default=128)
        else:
            s.add_argument("--generated",type=Path,required=True);s.add_argument("--scores-uma",type=Path,required=True);s.add_argument("--scores-mace",type=Path,required=True)
    a=p.parse_args();require_compute_allocation();root=a.root.resolve()
    try:
        if a.command=="prepare":prepare(root,a.output_dir,a.batch_size)
        elif a.command=="generate":generate(root,a.prepared,a.output_dir,a.device,a.count)
        else:evaluate(root,a.generated,a.output_dir,a.scores_uma,a.scores_mace)
    except Exception as error:
        if a.output_dir.exists() and not (a.output_dir/"failure_receipt.json").exists():
            import traceback
            dump(a.output_dir/"failure_receipt.json",dict(status="FAILED_PRESERVED",error=str(error),traceback=traceback.format_exc(),
                source_sha256=sha(__file__),NN_optimizer_updates=0))
        raise


if __name__=="__main__":main()

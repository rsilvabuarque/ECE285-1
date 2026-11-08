"""Actual-checkpoint count and causal-decoder probes at fixed supplied z,c.

No full-mask encoder may be called. Source hashes, selected checkpoint bytes and
the completed trajectory count are bound separately. No energy labels are read.
"""
import argparse,inspect,json,traceback
from pathlib import Path
import numpy as np
import torch
from cathode_cvae.data import sha256
from cathode_cvae.learning.generate_mask import load_model,resolve_request,canonical_masks
from cathode_cvae.learning.runtime import trusted_checkpoint
from cathode_cvae.runtime import require_compute_allocation


def audit(model,x,c,z,uniforms,steps):
    calls=[0]
    def forbidden(*args,**kwargs):
        calls[0]+=1;raise AssertionError("Encoder called inside a fixed-latent decoder audit")
    model.encode=forbidden
    if hasattr(model,"encoder"):model.encoder.forward=forbidden
    context=getattr(model,"graph_context",None);n=model.n_sites
    atol=float(2*n*np.finfo(np.float32).eps/(1-n*np.finfo(np.float32).eps))
    with torch.no_grad():
        teacher=context.teacher_features(x) if context is not None else x.new_zeros((len(x),n,0))
        step=torch.stack([context.step_features(x,i) for i in range(n)],1) if context is not None else teacher
    records=[];gradients=[]
    for i in steps:
        variable=x.detach().clone().requires_grad_(True);logits=model.decode_logits(z,c,variable)
        gradient,=torch.autograd.grad(logits[:,i].sum(),variable)
        altered=x.clone();altered[:,i:]=altered[:,i:].flip(1)
        with torch.no_grad():changed=model.decode_logits(z,c,altered)[:,i]
        record=dict(site=i,future_gradient_exact_zero=bool(torch.count_nonzero(gradient[:,i:])==0),
            future_gradient_max_abs=float(gradient[:,i:].abs().max()),past_gradient_max_abs=float(gradient[:,:i].abs().max()) if i else 0.,
            future_permutation_preserves_count=bool(torch.equal(altered.sum(1),x.sum(1))),
            future_permutation_logits_exactly_unchanged=bool(torch.equal(logits[:,i].detach(),changed)))
        if context is not None:
            with torch.no_grad():
                record["future_permutation_teacher_features_exactly_unchanged"]=bool(torch.equal(teacher[:,i],context.teacher_features(altered)[:,i]))
                record["future_permutation_step_features_exactly_unchanged"]=bool(torch.equal(step[:,i],context.step_features(altered,i)))
        records.append(record);gradients.append(gradient.detach().numpy())
    with torch.no_grad():
        n_target=torch.round(c[:,0]*n).long()
        a=model.sample(c,n_target,z=z,uniforms=uniforms);b=model.sample(c,n_target,z=z,uniforms=uniforms)
    positive=any(r["past_gradient_max_abs"]>0 for r in records if r["site"])
    exact=bool(torch.equal(a.sum(1),n_target.to(a.dtype))) and bool(torch.isin(a,a.new_tensor([0,1])).all())
    feature_match=bool(torch.allclose(teacher,step,rtol=0,atol=atol))
    passed=(calls[0]==0 and positive and exact and torch.equal(a,b) and feature_match and not model.gru.bidirectional
        and all(r["future_gradient_exact_zero"] and r["future_permutation_logits_exactly_unchanged"]
            and r.get("future_permutation_teacher_features_exactly_unchanged",True)
            and r.get("future_permutation_step_features_exactly_unchanged",True) for r in records))
    result=dict(status="PASS" if passed else "FAILED",encoder_calls=calls[0],exact_count=exact,reproducible=bool(torch.equal(a,b)),
        positive_past_gradient_control=positive,teacher_step_features_agree=feature_match,
        feature_max_abs_error=float((teacher-step).abs().max()) if teacher.numel() else 0.,feature_atol=atol,
        feature_tolerance_scope="Dimensionless FP32 reduction allowance2*gamma_N; future gradients/invariance still exact zero/equality",
        gradient_sites=records,decoder_source=dict(path=inspect.getsourcefile(type(model)),sha256=sha256(inspect.getsourcefile(type(model)))))
    return result,dict(gradients=np.stack(gradients),z=z.numpy(),conditions=c.numpy(),sample=a.numpy())


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest",type=Path);g.add_argument("--checkpoint",type=Path,action="append")
    p.add_argument("--data",type=Path,required=True);p.add_argument("--parent",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--role",choices=["train","val"],default="val");p.add_argument("--masks-per-count",type=int,default=3)
    p.add_argument("--expected-updates",type=int,default=4000);p.add_argument("--seed",type=int,default=20261108);p.add_argument("--threads",type=int,default=4)
    a=p.parse_args(argv);require_compute_allocation();torch.set_num_threads(a.threads);torch.use_deterministic_algorithms(True)
    if a.output.exists() or a.output.with_suffix(".npz").exists():raise FileExistsError("Preserve the previous trained audit")
    with np.load(a.parent,allow_pickle=False) as z:permutations=z["permutations"]
    with np.load(a.data,allow_pickle=False) as z:
        role=z["split"].astype(str);counts=z["n_target"] if "n_target" in z else z["n_Li"]
        ids=z["structure_ids"].astype(str);occupancy=z["occupancy"]
    chosen=np.concatenate([np.flatnonzero((role==a.role)&(counts==n))[:a.masks_per_count] for n in sorted(set(counts.tolist()))])
    if not len(chosen):raise ValueError("No non-test masks available for trained probes")
    keys,_=canonical_masks(occupancy[chosen],permutations)
    x=torch.tensor(1-np.asarray([[int(v) for v in k] for k in keys],np.uint8),dtype=torch.float32)
    entries=json.loads(a.manifest.read_text())["models"] if a.manifest else [dict(checkpoint=str(p)) for p in a.checkpoint]
    results=[];arrays=dict(occupancy=x.numpy(),structure_ids=ids[chosen],role=role[chosen]);data_sha=sha256(a.data);parent_sha=sha256(a.parent)
    for index,entry in enumerate(entries):
        if not entry.get("checkpoint"):continue
        path=Path(entry["checkpoint"]);digest=sha256(path)
        try:
            expected=entry.get("expected_checkpoint_sha256",entry.get("checkpoint_sha256",digest))
            if expected!=digest:raise ValueError("Checkpoint differs from frozen manifest")
            summary=json.loads(path.with_name("summary.json").read_text())
            if summary["optimizer_updates"]!=a.expected_updates:raise ValueError("Trajectory is not terminal at the required update count")
            named=[r for r in summary.get("selection_checkpoints",{}).values() if r.get("filename")==path.name]
            if named and named[0]["sha256"]!=digest:raise ValueError("Named selected checkpoint differs from its terminal summary binding")
            if path.name=="best.pt" and summary.get("checkpoint_sha256",digest)!=digest:raise ValueError("Primary selected checkpoint differs from summary")
            model,cfg=load_model(path,"cpu")
            selected=trusted_checkpoint(path,"cpu")
            if cfg.get("parent_sha256")!=parent_sha or cfg.get("data_sha256")!=data_sha:raise ValueError("Actual probe parent/data differs from fit")
            c=torch.tensor(np.asarray([resolve_request(cfg,int(n),"low")[0] for n in counts[chosen]]))
            rng=torch.Generator(device="cpu").manual_seed(a.seed);z=torch.randn((len(x),model.latent_dim),generator=rng);u=torch.rand(x.shape,generator=rng)
            steps=sorted(set([0,model.n_sites//4,model.n_sites//2,3*model.n_sites//4,model.n_sites-1]))
            record,raw=audit(model,x,c,z,u,steps)
            record.update(checkpoint=str(path.resolve()),checkpoint_sha256=digest,completed_updates=summary["optimizer_updates"],
                selected_checkpoint_updates=selected.get("optimizer_updates"),selected_epoch=selected.get("epoch"),
                selected_beta=selected.get("selection_beta"),fit_seed=cfg["seed"],kind=cfg["kind"],
                latent_dim=model.latent_dim,encoder_kind=cfg["encoder_kind"],context_spec=cfg.get("decoder_context_spec"),
                fit_model_source_sha256=cfg.get("model_source_sha256"),checkpoint_unchanged=sha256(path)==digest)
            if not record["checkpoint_unchanged"]:record["status"]="FAILED"
            for name,value in raw.items():arrays[f"model{index}_{name}"]=value
            results.append(record)
        except Exception as error:results.append(dict(status="FAILED",checkpoint=str(path),checkpoint_sha256=digest,error=repr(error),traceback=traceback.format_exc()))
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output.with_suffix(".npz"),**arrays)
    receipt=dict(status="PASS" if results and all(r["status"]=="PASS" for r in results) else "FAILED",models=results,
        input_role=a.role,input_masks=len(x),energy_labels_loaded=False,data_sha256=data_sha,parent_sha256=parent_sha,
        source_sha256=sha256(__file__),raw_sha256=sha256(a.output.with_suffix(".npz")),new_MLIP_calls=0,new_optimizer_updates=0,
        scope="Fixed supplied latent and condition; full-mask encoder forbidden; numerical/software causality, not physical validation")
    a.output.write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps(dict(status=receipt["status"],models=len(results),input_masks=len(x))))
    if receipt["status"]!="PASS":raise SystemExit(1)
    return receipt


if __name__=="__main__":main()

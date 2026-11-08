"""Independent causal-decoder checks on all three trained graph-context models.

Uses actual protected TRAIN masks and fixed supplied latent/condition vectors.
The full-mask encoder is forbidden during these decoder tests. There are no
optimizer steps, checkpoint writes, physical labels, or MLIP evaluations.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from cathode_cvae.learning.generate_mask import load_model, resolve_request
from cathode_cvae.runtime import require_compute_allocation


PROJECT = Path(__file__).resolve().parents[2]
STEPS = (0,12,24,36,47)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):h.update(block)
    return h.hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=PROJECT)
    p.add_argument("--output",type=Path)
    p.add_argument("--seed",type=int,default=20261107)
    a=p.parse_args(argv);require_compute_allocation()
    root=a.root.resolve()
    output=a.output or root/"reports/conditioning_20261107/trained_causality_audit.json"
    gradient_output=output.with_name(output.stem+".gradients.npz")
    if output.exists() or gradient_output.exists():raise FileExistsError("Preserve the existing trained audit")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    data=root/"artifacts/current/improvement_v1/training_round0.npz"
    parent=root/"artifacts/current/core/data/parents/O3_N48_pilot.npz"
    data_sha,parent_sha=digest(data),digest(parent)
    with np.load(data,allow_pickle=False) as z:
        role=z["split"].astype(str);counts=z["n_Li"]
        ids=z["structure_ids"].astype(str);keys=z["canonical_masks"].astype(str)
    chosen=np.concatenate([np.flatnonzero((role=="train")&(counts==n))[:3] for n in [12,24,36]])
    if len(chosen)!=9:raise ValueError("Actual training cohort does not support all declared count probes")
    occupancy=1-np.asarray([[int(v) for v in key] for key in keys[chosen]],np.uint8)
    if not np.array_equal(occupancy.sum(1),counts[chosen]):raise ValueError("Invalid canonical mask convention")
    x=torch.tensor(occupancy,dtype=torch.float32)
    results=[];arrays={"input_occupancy":occupancy,"input_counts":counts[chosen],"input_structure_ids":ids[chosen]}
    for fit_seed in [20261106,20261107,20261108]:
        checkpoint=root/f"runs/conditioning_20261107/fits/graph_context_{fit_seed}/best_validation_beta_0p1.pt"
        checkpoint_sha=digest(checkpoint)
        try:
            model,cfg=load_model(checkpoint,"cpu")
            if cfg["parent_sha256"]!=parent_sha or cfg["data_sha256"]!=data_sha:
                raise ValueError("Trained checkpoint parent/data differs from the actual probe source")
            if cfg.get("encoder_kind")!="graph" or not cfg.get("graph_context"):
                raise ValueError("This is not the declared graph-encoder plus causal-context model")
            summary=json.loads(checkpoint.with_name("summary.json").read_text())
            if summary["optimizer_updates"]!=4000:raise ValueError("Audit requires the completed 4000-update trajectory")
            c=torch.tensor(np.asarray([resolve_request(cfg,int(n),"low")[0] for n in counts[chosen]]))
            rng=torch.Generator(device="cpu").manual_seed(a.seed)
            z=torch.randn((len(x),model.latent_dim),generator=rng)
            uniforms=torch.rand(x.shape,generator=rng)
            calls=[0]
            def forbidden_encoder(*args,**kwargs):
                calls[0]+=1
                raise AssertionError("Full-mask encoder entered a fixed-latent decoder audit")
            model.encode=forbidden_encoder
            model.encoder.forward=forbidden_encoder
            model.eval()
            context=model.graph_context
            # Two normalized positive sums can follow different reduction paths.
            # 2*gamma_n is a conservative dimensionless FP32 roundoff allowance;
            # future-gradient and finite-future invariance below still require
            # exact zero/equality, independently of this feature comparison.
            eps=np.finfo(np.float32).eps;n=model.n_sites
            feature_atol=float(2*n*eps/(1-n*eps))
            with torch.no_grad():
                teacher=context.teacher_features(x)
                step=torch.stack([context.step_features(x,i) for i in range(n)],dim=1)
            feature_error=float((teacher-step).abs().max())
            feature_agrees=bool(torch.allclose(teacher,step,rtol=0,atol=feature_atol))
            gradients=[];step_records=[]
            for i in STEPS:
                variable=x.detach().clone().requires_grad_(True)
                logits=model.decode_logits(z,c,variable)
                gradient,=torch.autograd.grad(logits[:,i].sum(),variable)
                future=gradient[:,i:]
                future_exact_zero=bool(torch.count_nonzero(future)==0)
                past_max=float(gradient[:,:i].abs().max()) if i else 0.
                altered=x.clone();altered[:,i:]=altered[:,i:].flip(1)
                with torch.no_grad():
                    future_features=context.teacher_features(altered)[:,i]
                    step_future_features=context.step_features(altered,i)
                    changed_logits=model.decode_logits(z,c,altered)[:,i]
                unchanged_teacher=bool(torch.equal(teacher[:,i],future_features))
                unchanged_step=bool(torch.equal(step[:,i],step_future_features))
                unchanged_logits=bool(torch.equal(logits[:,i].detach(),changed_logits))
                step_records.append(dict(site=i,future_gradient_exact_zero=future_exact_zero,
                    future_gradient_max_abs=float(future.abs().max()),past_gradient_max_abs=past_max,
                    future_permutation_rows_actually_changed=int(torch.any(altered!=x,dim=1).sum()),
                    future_permutation_preserves_count=bool(torch.equal(altered.sum(1),x.sum(1))),
                    future_permutation_teacher_features_exactly_unchanged=unchanged_teacher,
                    future_permutation_step_features_exactly_unchanged=unchanged_step,
                    future_permutation_logits_exactly_unchanged=unchanged_logits))
                gradients.append(gradient.detach().numpy())
            positive=next(r for r in step_records if r["site"]==24)["past_gradient_max_abs"]>0
            with torch.no_grad():
                sample1=model.sample(c,torch.tensor(counts[chosen]),z=z,uniforms=uniforms)
                sample2=model.sample(c,torch.tensor(counts[chosen]),z=z,uniforms=uniforms)
            reproducible=bool(torch.equal(sample1,sample2))
            exact_count=bool(torch.equal(sample1.sum(1),torch.tensor(counts[chosen],dtype=sample1.dtype)))
            unidirectional=not model.gru.bidirectional
            passed=(feature_agrees and positive and reproducible and exact_count and unidirectional and calls[0]==0
                and all(r["future_gradient_exact_zero"] and r["future_permutation_teacher_features_exactly_unchanged"]
                        and r["future_permutation_step_features_exactly_unchanged"]
                        and r["future_permutation_logits_exactly_unchanged"] for r in step_records))
            record=dict(fit_seed=fit_seed,status="PASS" if passed else "FAILED",checkpoint=str(checkpoint.resolve()),
                checkpoint_sha256=checkpoint_sha,checkpoint_unchanged=digest(checkpoint)==checkpoint_sha,
                trajectory_updates=summary["optimizer_updates"],encoder_calls=calls[0],
                teacher_step_max_absolute_feature_difference=feature_error,teacher_step_feature_atol=feature_atol,
                teacher_step_features_agree=feature_agrees,gradient_steps=step_records,
                positive_past_gradient_control_at_site24=positive,sample_reproducible=reproducible,
                sample_exact_count=exact_count,GRU_unidirectional=unidirectional,
                decoder_source_sha256=digest(inspect.getsourcefile(type(model))),
                graph_context_source_sha256=digest(inspect.getsourcefile(type(context))))
            if not record["checkpoint_unchanged"]:record["status"]="FAILED"
            results.append(record)
            prefix=f"seed{fit_seed}_"
            arrays[prefix+"gradients_by_tested_site"]=np.stack(gradients)
            arrays[prefix+"latent"]=z.numpy();arrays[prefix+"conditions"]=c.numpy()
            arrays[prefix+"sample"]=sample1.numpy().astype(np.uint8)
            arrays[prefix+"teacher_features"]=teacher.numpy();arrays[prefix+"step_features"]=step.numpy()
        except Exception as error:
            results.append(dict(fit_seed=fit_seed,status="FAILED",error_type=type(error).__name__,error=str(error),
                traceback=traceback.format_exc(),checkpoint_sha256=checkpoint_sha,
                checkpoint_unchanged=digest(checkpoint)==checkpoint_sha))
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(gradient_output,**arrays)
    receipt=dict(status="PASS" if all(r["status"]=="PASS" for r in results) else "FAILED",
        checkpoints=len(results),input_masks=len(x),input_roles="TRAIN only; no test or validation labels",
        counts=[12,24,36],masks_per_count=3,fixed_latent_seed=a.seed,fixed_request="low",
        gradient_sites=list(STEPS),gradient_scope="d logit_i / d x_j at fixed supplied z,c; every j>=i must be exactly zero",
        feature_tolerance_provenance="Dimensionless neighbor fractions; 2*gamma_n with FP32 epsilon and n=48 bounds differing positive reduction paths; causality gradients/future perturbations require exact zero/equality",
        source_review="Decoder uses explicit z,c, previous occupancy, site index, and strict-past graph features. The GRU is unidirectional; reviewed decoder/context methods store no occupation-dependent hidden cache. Both full-mask encoder entry points are forbidden at runtime.",
        limitations="Finite trained-model probes plus source inspection, not a theorem of physical accuracy or equivariance; no optimizer or MLIP calls",
        optimizer_steps=0,new_training_trajectories=0,new_MLIP_calls=0,weights_written=False,
        dtype="float32",device="cpu",torch=torch.__version__,source_sha256=digest(__file__),
        parent_sha256=parent_sha,data_sha256=data_sha,raw_gradients_sha256=digest(gradient_output),models=results)
    with output.open("x") as stream:stream.write(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2))
    if receipt["status"]!="PASS":raise SystemExit(1)


if __name__=="__main__":main()

"""Bounded optional seed-edit CVAE training after the required-report gate.

Conditions describe the actual observed CHILD energy bin, never the source
sampler's request. Editor TRAIN/VAL/heldout roles are connected seed/child
identity families nested inside the preserved project data boundaries.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from . import edit_model
from .edit_model import SeedEditCVAE,encode_condition,edit_targets,EDIT_FORMAT_VERSION
from .models import gaussian_kl
from .runtime import device_name
from ..conditions import load_schema,classify
from ..runtime import require_compute_allocation


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def validate_pairs(data):
    """Enforce count/edit/role alignment without independently rotating pairs."""
    seed,child=data["seeds"],data["children"];n=data["n_Li"];k=data["swap_count"]
    roles=data["split"].astype(str);components=data["component_ids"].astype(str)
    if seed.ndim!=2 or child.shape!=seed.shape or not np.isin(seed,[0,1]).all() or not np.isin(child,[0,1]).all():
        raise ValueError("Learning needs same-parent binary raw seed/child pairs")
    if not np.array_equal(seed.sum(1),n) or not np.array_equal(child.sum(1),n) or not np.isin(k,[1,2,3]).all():
        raise ValueError("A pair lost composition or its supported swap count")
    if not np.array_equal(np.count_nonzero(seed!=child,axis=1),2*k):
        raise ValueError("A copied seed or nonlocal recorded budget is not an exact edit target")
    if not set(roles)<={"train","val","heldout"} or not all(np.any(roles==r) for r in ["train","val","heldout"]):
        raise ValueError("Need disjoint nonempty TRAIN, VAL and heldout seed-family roles")
    identities={};component_roles={};identity_components={}
    for a,b,component,role in zip(data["canonical_seed_keys"].astype(str),data["canonical_child_keys"].astype(str),components,roles):
        if component in component_roles and component_roles[component]!=role:raise ValueError("A connected component crosses editor roles")
        component_roles[component]=role
        for identity in [a,b]:
            if identity in identities and identities[identity]!=role:raise ValueError("A shared seed/child identity crosses editor roles")
            if identity in identity_components and identity_components[identity]!=component:raise ValueError("One identity appears in multiple components")
            identities[identity]=role;identity_components[identity]=component
    if len(set(data["pair_ids"].astype(str)))!=len(seed):raise ValueError("Pair identities repeat")
    return dict(rows=len(seed),n_sites=seed.shape[1],split_rows=dict(Counter(roles)),
        component_count=len(component_roles),identity_role_bridges=0)


def train_weights(n,k,bins,roles):
    """Inverse observed class frequency within each(n,k), TRAIN only."""
    weight=np.ones(len(n),np.float64);support=[]
    for count,budget in sorted(set(zip(n[roles=="train"].tolist(),k[roles=="train"].tolist()))):
        selected=np.flatnonzero((roles=="train")&(n==count)&(k==budget))
        classes=Counter(bins[selected].tolist())
        for label,number in sorted(classes.items()):
            group=selected[bins[selected]==label]
            weight[group]/=len(classes)*(number/len(selected))
            support.append(dict(n_Li=int(count),swap_count=int(budget),child_bin=label,train_rows=number))
    weight/=weight[roles=="train"].mean()
    return weight,support


def latent_statistics(mu,lv):
    values=.5*(mu.square()+lv.exp()-lv-1)
    kl=values.mean(0);variance=mu.var(0,unbiased=False)
    return dict(total_KL_nats=float(kl.sum()),mean_KL_nats_per_dimension=float(kl.mean()),
        KL_nats_by_dimension=kl.tolist(),posterior_mean_by_dimension=mu.mean(0).tolist(),
        posterior_mean_variance_by_dimension=variance.tolist(),posterior_variance_mean_by_dimension=lv.exp().mean(0).tolist(),
        active_components={str(c):int((variance>c).sum()) for c in [.001,.01,.1]},
        active_definition="Unweighted VAL posterior-mean variance > dimensionless descriptive cutoff; no physical tolerance")


def arguments(argv):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True);p.add_argument("--data",type=Path,required=True)
    p.add_argument("--preparation-receipt",type=Path,required=True);p.add_argument("--parent",type=Path,required=True)
    p.add_argument("--condition-schema-in",type=Path,required=True)
    p.add_argument("--required-gate",type=Path,required=True);p.add_argument("--required-gate-sha256",required=True)
    p.add_argument("--seed",type=int,required=True);p.add_argument("--counts",type=int,nargs="+",default=[12,24,36])
    p.add_argument("--latent-dim",type=int,default=4);p.add_argument("--hidden-dim",type=int,default=128)
    p.add_argument("--orientation",choices=["original","joint-augment"],default="joint-augment")
    p.add_argument("--max-updates",type=int,default=4000);p.add_argument("--epochs",type=int,default=1000)
    p.add_argument("--batch-size",type=int,default=512);p.add_argument("--learning-rate",type=float,default=.001)
    p.add_argument("--beta-final",type=float,default=.1);p.add_argument("--beta-warmup",type=int,default=20)
    p.add_argument("--validation-beta",type=float,default=.1);p.add_argument("--validation-samples",type=int,default=2)
    p.add_argument("--class-balance",action="store_true");p.add_argument("--max-seconds",type=float,default=1200)
    p.add_argument("--device",default="auto");p.add_argument("--threads",type=int,default=4)
    p.add_argument("--output-dir",type=Path,required=True)
    return p.parse_args(argv)


def main(argv=None):
    a=arguments(argv);require_compute_allocation();a.device=device_name(a.device)
    if min(a.max_updates,a.epochs,a.batch_size,a.beta_warmup,a.validation_samples,a.latent_dim,a.hidden_dim)<1:
        raise ValueError("Positive bounded training settings required")
    if a.validation_beta!=.1:raise ValueError("This optional pilot has one predeclared selection beta0.1")
    root=a.root.resolve();gate_path=a.required_gate.resolve()
    if digest(gate_path)!=a.required_gate_sha256:raise ValueError("Required-report gate SHA differs")
    gate=json.loads(gate_path.read_text());started_utc=datetime.now(timezone.utc)
    completed=datetime.fromisoformat(gate["completed_UTC"])
    if gate["status"]!="COMPLETE" or started_utc<=completed:raise ValueError("Editor fit cannot precede the completed required-report gate")
    report=root/gate["report_pdf"]["path"]
    if digest(report)!=gate["report_pdf"]["sha256"]:raise ValueError("Compiled required-report snapshot differs")
    prepared=json.loads(a.preparation_receipt.read_text());data_sha=digest(a.data);parent_sha=digest(a.parent)
    if prepared["status"]!="READY_FOR_BOUNDED_EDITOR" or prepared["identity_role_bridges"]!=0:
        raise ValueError("Pair provenance/support audit is not ready")
    if prepared["outputs"][a.data.name]!=data_sha:raise ValueError("Prepared pair bytes differ from the audited data")
    if prepared["required_report_gate"]["sha256"]!=a.required_gate_sha256:raise ValueError("Pair preparation used another gate")
    with np.load(a.data,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    data_check=validate_pairs(data)
    if str(data["parent_sha256"].item())!=parent_sha:raise ValueError("Pair and supplied parent differ")
    schema=load_schema(a.condition_schema_in)
    if schema["identity"]!=str(data["condition_schema_id"].item()) or schema["provenance"]["parent_sha256"]!=parent_sha:
        raise ValueError("Pair conditions use a different frozen parent/teacher schema")
    if schema["provenance"]["data_sha256"]!=str(data["H0_sha256"].item()):raise ValueError("Condition reference differs from protected H0")
    if not np.array_equal(classify(schema,data["n_Li"],data["child_energy_eV_atom"]),data["child_bin"].astype(str)):
        raise ValueError("Editor target label is not its actual measured CHILD energy bin")
    if set(data["n_Li"].tolist())!=set(a.counts):raise ValueError("Declared compositions differ from the prepared pairs")
    roles=data["split"].astype(str);n=data["n_Li"];k=data["swap_count"];bins=data["child_bin"].astype(str)
    weights,support=train_weights(n,k,bins,roles)
    if not a.class_balance:weights[:]=1.
    train=np.flatnonzero(roles=="train");val=np.flatnonzero(roles=="val");heldout=np.flatnonzero(roles=="heldout")
    split_sha=hashlib.sha256("\n".join(f"{i}\t{r}" for i,r in zip(data["pair_ids"].astype(str),roles)).encode()).hexdigest()
    with np.load(a.parent,allow_pickle=False) as z:perms=z["permutations"]
    if perms.shape[1]!=data_check["n_sites"]:raise ValueError("Parent and pair site counts disagree")
    a.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(a.threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    seed=torch.tensor(data["seeds"],dtype=torch.float32,device=a.device)
    child=torch.tensor(data["children"],dtype=torch.float32,device=a.device)
    c=torch.tensor(encode_condition(n,bins,k,data_check["n_sites"]),device=a.device)
    swaps=torch.tensor(k,dtype=torch.long,device=a.device);w=torch.tensor(weights,dtype=torch.float32,device=a.device)
    train_t=torch.tensor(train,device=a.device);val_t=torch.tensor(val,device=a.device)
    inverse=torch.tensor(np.argsort(perms,axis=1),device=a.device)
    model=SeedEditCVAE(data_check["n_sites"],5,a.latent_dim,a.hidden_dim).to(a.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=a.learning_rate)
    noise_generator=torch.Generator(device=a.device).manual_seed(a.seed+8911)
    noise=torch.randn((a.validation_samples,len(val),a.latent_dim),device=a.device,generator=noise_generator)
    config=dict(kind="seed_edit_cvae",edit_format_version=EDIT_FORMAT_VERSION,variational_model=True,
        experiment_phase="optional_editor",trainer_module="cathode_cvae.learning.train_edit",
        trainer_sha256=digest(__file__),model_source_sha256=digest(edit_model.__file__),
        required_report_gate=dict(path=str(gate_path.relative_to(root)),sha256=a.required_gate_sha256),
        started_UTC=started_utc.isoformat(),training_started_UTC=started_utc.isoformat(),started_unix=started_utc.timestamp(),
        seed=a.seed,n_sites=data_check["n_sites"],condition_dim=5,latent_dim=a.latent_dim,hidden_dim=a.hidden_dim,
        parameter_count=sum(p.numel() for p in model.parameters()),counts=list(a.counts),swap_counts=[1,2,3],
        data=str(a.data.resolve()),data_sha256=data_sha,preparation_receipt_sha256=digest(a.preparation_receipt),
        parent=str(a.parent.resolve()),parent_sha256=parent_sha,H0_sha256=str(data["H0_sha256"].item()),
        split_sha256=split_sha,train_rows=len(train),val_rows=len(val),heldout_rows=len(heldout),
        condition_schema=schema,condition_schema_id=schema["identity"],supported_requests=support,
        condition_encoding="[present_count/n_sites,onehot(measured CHILD energy bin low/middle/high),swap_count/3]",
        teacher="uma",teacher_head="omat",dtype="float32",precision="float32",tf32=False,
        geometry_protocol="pristine-derived deletion; unrelaxed; exact canonical parent coordinates",
        orientation=a.orientation,augmentation="Same retained parent operation applied jointly to seed and child; conditions invariant",
        class_balance=a.class_balance,class_balance_scope="TRAIN only within each(count,swapcount) over observed actual CHILD bins; total TRAIN weight preserved",
        loss="Exact-k removal NLL + exact-k addition NLL conditioned on removals + beta*Gaussian KL",
        KL_convention="Per-row sum across latent components then row-weighted mean; never divided by latent dimension",
        validation_weighting="Unweighted VAL rows, fixed seeded Monte-Carlo posterior noise; heldout excluded",
        optimizer="Fresh Adam, no warmstart or optimizer-state reuse",learning_rate=a.learning_rate,batch_size=a.batch_size,
        beta_final=a.beta_final,beta_warmup_epochs=a.beta_warmup,validation_beta=a.validation_beta,
        validation_samples=a.validation_samples,validation_noise_seed=a.seed+8911,
        maximum_updates=a.max_updates,maximum_epochs=a.epochs,maximum_seconds=a.max_seconds,
        protected_test_evaluated=False,heldout_seed_families_evaluated=False,data_check=data_check)
    dump(a.output_dir/"configuration.json",config)
    (a.output_dir/"condition_schema.json").write_text(json.dumps(schema,indent=2)+"\n")
    started=time.perf_counter();updates=0;best=math.inf;selection=None;stop="maximum_epochs"
    import csv
    with (a.output_dir/"history.csv").open("x",newline="") as history,(a.output_dir/"latent_history.jsonl").open("x") as latent_stream:
        writer=None
        for epoch in range(1,a.epochs+1):
            model.train();order=train_t[torch.randperm(len(train_t),device=a.device)];totals=np.zeros(4)
            beta=a.beta_final*min(1,epoch/a.beta_warmup)
            for ids in order.split(a.batch_size):
                ss,cc=seed[ids],child[ids]
                if a.orientation=="joint-augment":
                    operation=inverse[torch.randint(len(inverse),(len(ids),),device=a.device)]
                    ss=ss.gather(1,operation);cc=cc.gather(1,operation)
                mu,lv=model.encode(ss,cc,c[ids]);z=mu+torch.exp(.5*lv)*torch.randn_like(mu)
                rec=model.reconstruction(ss,cc,c[ids],z,swaps[ids]);kl=gaussian_kl(mu,lv)
                objective=rec+beta*kl;loss=(objective*w[ids]).mean()
                if not torch.isfinite(loss):raise FloatingPointError("Nonfinite editor loss; preserve this fit")
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step();updates+=1
                totals+=np.asarray([float((rec*w[ids]).sum().detach()),float((kl*w[ids]).sum().detach()),
                    float((objective*w[ids]).sum().detach()),float(w[ids].sum())])
                if updates>=a.max_updates:break
            model.eval();val_rec=0.;val_kl=0.;mus=[];lvs=[]
            with torch.no_grad():
                for offset in range(0,len(val_t),a.batch_size):
                    ids=val_t[offset:offset+a.batch_size];mu,lv=model.encode(seed[ids],child[ids],c[ids]);rec=mu.new_zeros(len(ids))
                    for eps in noise[:,offset:offset+len(ids)]:
                        rec+=model.reconstruction(seed[ids],child[ids],c[ids],mu+torch.exp(.5*lv)*eps,swaps[ids])/a.validation_samples
                    val_rec+=float(rec.sum());val_kl+=float(gaussian_kl(mu,lv).sum());mus.append(mu.cpu());lvs.append(lv.cpu())
            val_rec/=len(val);val_kl/=len(val);score=val_rec+a.validation_beta*val_kl
            if not math.isfinite(score):raise FloatingPointError("Nonfinite editor VAL objective")
            stats=latent_statistics(torch.cat(mus),torch.cat(lvs))
            record=dict(epoch=epoch,optimizer_updates=updates,beta=beta,train_reconstruction=totals[0]/totals[3],
                train_KL=totals[1]/totals[3],train_objective=totals[2]/totals[3],val_reconstruction=val_rec,val_KL=val_kl,
                val_objective=score,elapsed_seconds=time.perf_counter()-started)
            if writer is None:writer=csv.DictWriter(history,fieldnames=list(record));writer.writeheader()
            writer.writerow(record);history.flush();latent_stream.write(json.dumps(dict(epoch=epoch,optimizer_updates=updates,pooled=stats))+"\n");latent_stream.flush()
            if score<best:
                best=score;selected=dict(epoch=epoch,optimizer_updates=updates,selection_beta=.1,
                    val_objective=score,filename="best_validation_beta_0p1.pt")
                checkpoint=dict(state_dict=model.state_dict(),configuration=config,epoch=epoch,optimizer_updates=updates,
                    selection_beta=.1,latent_statistics=stats)
                torch.save(checkpoint,a.output_dir/selected["filename"]);torch.save(checkpoint,a.output_dir/"best.pt")
                selected["sha256"]=digest(a.output_dir/selected["filename"]);selection=selected
            print(json.dumps(record),flush=True)
            if updates>=a.max_updates:stop="maximum_updates";break
            if time.perf_counter()-started>=a.max_seconds:stop="time_cap";break
    summary=dict(status="FIT_COMPLETE" if updates==a.max_updates else "FIT_PARTIAL",optimizer_updates=updates,
        stop_reason=stop,training_seconds=time.perf_counter()-started,epochs_completed=epoch,
        checkpoint_sha256=selection["sha256"],selection_checkpoints={"0.1":selection},
        parameter_count=config["parameter_count"],heldout_test_evaluated=False,heldout_seed_families_evaluated=False,
        source_inputs_unchanged=digest(a.data)==data_sha and digest(a.parent)==parent_sha,
        experiment_phase="optional_editor",kind="seed_edit_cvae",new_MLIP_calls=0)
    dump(a.output_dir/"summary.json",summary)
    if not summary["source_inputs_unchanged"]:raise ValueError("Editor inputs changed during fitting")
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=="__main__":main()

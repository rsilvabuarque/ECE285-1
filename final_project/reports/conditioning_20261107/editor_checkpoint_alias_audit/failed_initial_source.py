"""Read-only exact audit of editor named-selection and best.pt archive aliases.

This never constructs a network or optimizer. Both archives are loaded with
weights_only=True onto CPU; every nested field and every tensor byte is checked.
Different outer ZIP root names are investigated separately from semantic data.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,math,os,shlex,struct,sys,time,zipfile
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(2**20),b""):h.update(block)
    return h.hexdigest()


def exact_equal(left,right,torch,path="checkpoint",counts=None):
    """No floating tolerance: signed zeros and tensor storage bytes matter."""
    if counts is None:counts=dict(tensors=0,tensor_elements=0,scalar_fields=0,containers=0)
    if type(left) is not type(right):raise ValueError(f"Type mismatch:{path}")
    if isinstance(left,torch.Tensor):
        if left.layout!=torch.strided or right.layout!=torch.strided:raise ValueError("Unexpected non-dense checkpoint tensor")
        if left.dtype!=right.dtype or left.shape!=right.shape or left.stride()!=right.stride():raise ValueError(f"Tensor metadata mismatch:{path}")
        a=left.detach().cpu().contiguous();b=right.detach().cpu().contiguous()
        if a.reshape(-1).view(torch.uint8).numpy().tobytes()!=b.reshape(-1).view(torch.uint8).numpy().tobytes():raise ValueError(f"Tensor byte mismatch:{path}")
        if (a.is_floating_point() or a.is_complex()) and not bool(torch.isfinite(a).all()):raise ValueError(f"Nonfinite saved tensor:{path}")
        counts["tensors"]+=1;counts["tensor_elements"]+=a.numel();return counts
    if isinstance(left,dict):
        if list(left)!=list(right):raise ValueError(f"Dictionary keys/order mismatch:{path}")
        counts["containers"]+=1
        for key in left:exact_equal(left[key],right[key],torch,path+"."+str(key),counts)
    elif isinstance(left,(list,tuple)):
        if len(left)!=len(right):raise ValueError(f"Sequence length mismatch:{path}")
        counts["containers"]+=1
        for i,(a,b) in enumerate(zip(left,right)):exact_equal(a,b,torch,path+f"[{i}]",counts)
    elif isinstance(left,float):
        if not math.isfinite(left) or struct.pack("!d",left)!=struct.pack("!d",right):raise ValueError(f"Float byte mismatch:{path}")
        counts["scalar_fields"]+=1
    elif left is None or isinstance(left,(str,int,bool,bytes)):
        if left!=right:raise ValueError(f"Scalar mismatch:{path}")
        counts["scalar_fields"]+=1
    else:raise ValueError(f"Unexpected serialized value type at {path}:{type(left).__name__}")
    return counts


def normalized_members(path):
    with zipfile.ZipFile(path) as z:
        result={};prefixes=set()
        for member in z.namelist():
            parts=member.split("/",1)
            if len(parts)!=2 or parts[1] in result:raise ValueError("Unrecognized checkpoint ZIP hierarchy")
            prefixes.add(parts[0]);data=z.read(member)
            result[parts[1]]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        if len(prefixes)!=1:raise ValueError("Checkpoint has multiple archive roots")
        return sorted(prefixes),result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--preclosure",type=Path,required=True,help="Preserved failed auditor evidence, never rewritten")
    a=p.parse_args(argv)
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    import torch
    torch.set_num_threads(2);started=time.perf_counter();root=a.root.resolve();run=root/"runs/conditioning_20261107"
    if a.output.exists():raise FileExistsError("Preserve previous alias audit")
    generation=json.loads((run/"editor/assessment_v1/protocol.json").read_text());results=[]
    preclosure=json.loads(a.preclosure.read_text());preclosure_sha=sha(a.preclosure)
    findings={r["path"]:r for r in preclosure["findings"]}
    for seed in (20261106,20261107,20261108):
        folder=run/"fits"/f"editor_{seed}";summary=json.loads((folder/"summary.json").read_text());selection=summary["selection_checkpoints"]["0.1"]
        filename=selection["filename"]
        if filename!="best_validation_beta_0p1.pt":raise ValueError("Unexpected named editor selection file")
        named=folder/filename;alias=folder/"best.pt";before={str(p):sha(p) for p in (named,alias,folder/"summary.json",folder/"configuration.json")}
        finding=findings[str(alias)]
        if finding["actual"]!=before[str(alias)] or finding["expected"]!=before[str(named)]:
            raise ValueError("Current bytes differ from the preserved failed audit; investigate before any repair")
        left=torch.load(named,map_location="cpu",weights_only=True);right=torch.load(alias,map_location="cpu",weights_only=True)
        compared=exact_equal(left,right,torch)
        expected=json.loads((folder/"configuration.json").read_text())
        # Serialized configuration and adjacent JSON have the same primitive
        # values; JSON can change insertion order, so compare canonical text.
        if json.dumps(left["configuration"],sort_keys=True)!=json.dumps(expected,sort_keys=True):raise ValueError("Adjacent fit configuration differs from checkpoint")
        if set(left)!={"state_dict","configuration","epoch","optimizer_updates","selection_beta","latent_statistics"}:
            raise ValueError("Unexpected checkpoint field set; audit all new fields before proceeding")
        for key in ("epoch","optimizer_updates","selection_beta"):
            if left[key]!=selection[key]:raise ValueError(f"Selected {key} differs from named checkpoint")
        cfg=left["configuration"]
        if cfg["kind"]!="seed_edit_cvae" or cfg["experiment_phase"]!="optional_editor":raise ValueError("Alias exception must remain editor-specific")
        if before[str(named)]!=summary["checkpoint_sha256"] or before[str(named)]!=selection["sha256"]:raise ValueError("Summary does not bind its named selection bytes")
        training_source=root/"src/cathode_cvae/learning/train_edit.py"
        if sha(training_source)!=cfg["trainer_sha256"]:raise ValueError("Training source differs from saved configuration")
        declared=generation["checkpoints"][str(seed)]
        if Path(declared["path"])!=named or declared["sha256"]!=before[str(named)]:raise ValueError("Generation was not bound to the audited named selection")
        prefix_a,members_a=normalized_members(named);prefix_b,members_b=normalized_members(alias)
        if before!={str(p):sha(p) for p in (named,alias,folder/"summary.json",folder/"configuration.json")}:
            raise ValueError("Protected checkpoint or metadata changed during read-only audit")
        results.append(dict(fit_seed=seed,named_checkpoint=str(named),named_sha256=before[str(named)],
            alias_checkpoint=str(alias),alias_sha256=before[str(alias)],outer_bytes_identical=before[str(named)]==before[str(alias)],
            entire_deserialized_payload_bitwise_identical=True,compared=compared,checkpoint_keys=list(left),
            selected_epoch=left["epoch"],selected_optimizer_updates=left["optimizer_updates"],completed_optimizer_updates=summary["optimizer_updates"],
            optimizer_state_saved=False,optimizer_scope="Optimizer UPDATE COUNT is stored; neither alias contains Adam state tensors",
            selection_beta=left["selection_beta"],latent_statistics=left["latent_statistics"],
            named_zip_prefix=prefix_a,alias_zip_prefix=prefix_b,normalized_zip_member_payloads_identical=members_a==members_b,
            normalized_member_inventory_named=members_a,normalized_member_inventory_alias=members_b,
            source_sha256=cfg["trainer_sha256"],summary_matches_named_selection=True,generation_matches_named_selection=True,
            protected_bytes_unchanged=True,unchanged_from_failed_preclosure=dict(path=finding["path"],expected=finding["expected"],actual=finding["actual"])))
    result=dict(status="PASS_EXACT_EDITOR_CHECKPOINT_ALIAS_AUDIT",audited_UTC=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-started,torch=str(torch.__version__),load_policy="torch.load(map_location='cpu',weights_only=True)",
        scientific_scope="Archive alias integrity only; no inference, optimizer update, altered source, or new physical validation",
        fits=results,new_MLIP_calls=0,new_encoder_calls=0,new_training_updates=0,source_sha256=sha(__file__),
        preserved_preclosure=dict(path=str(a.preclosure.resolve()),sha256=preclosure_sha,status=preclosure["status"]),
        execution=dict(python=sys.executable,argv=sys.argv,command=shlex.join([sys.executable,*sys.argv]),allocation=os.environ.get("SLURM_JOB_ID"),step=os.environ.get("SLURM_STEP_ID")),
        proposed_auditor_repair="For verified optional_editor/seed_edit_cvae only, bind summary.checkpoint_sha256 to selection_checkpoints['0.1'].filename; independently hash best.pt and require complete deserialized identity. Preserve the previous failed receipt/source.")
    if sha(a.preclosure)!=preclosure_sha:raise ValueError("Preserved failed audit changed")
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    (a.output.parent/"audit_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(status=result["status"],fits=len(results),normalized_zip_payloads_identical=[r["normalized_zip_member_payloads_identical"] for r in results],elapsed_seconds=result["elapsed_seconds"]),indent=2));return 0


if __name__=="__main__":raise SystemExit(main())

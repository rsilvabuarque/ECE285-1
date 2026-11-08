"""Load and sample the separately versioned seed-edit CVAE without an MLIP."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import edit_model
from .edit_model import EDIT_FORMAT_VERSION, SeedEditCVAE, encode_condition
from .runtime import device_name, trusted_checkpoint
from ..runtime import require_compute_allocation


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_edit_model(checkpoint,device="cpu"):
    value=trusted_checkpoint(checkpoint,device);cfg=value.get("configuration")
    required={"kind","edit_format_version","n_sites","condition_dim","latent_dim","hidden_dim","model_source_sha256"}
    if not isinstance(cfg,dict) or not required<=set(cfg):raise ValueError("Editor checkpoint lacks explicit dimensions/version")
    if cfg["kind"]!="seed_edit_cvae" or cfg["edit_format_version"]!=EDIT_FORMAT_VERSION or cfg["condition_dim"]!=5:
        raise ValueError("Unsupported editor checkpoint kind/version/condition encoding")
    if cfg["model_source_sha256"]!=digest(edit_model.__file__):
        raise ValueError("Use the preserved matching editor model source for this checkpoint")
    model=SeedEditCVAE(n_sites=cfg["n_sites"],condition_dim=cfg["condition_dim"],
        latent_dim=cfg["latent_dim"],hidden_dim=cfg["hidden_dim"]).to(device)
    model.load_state_dict(value["state_dict"],strict=True);model.eval()
    return model,cfg


def require_supported_requests(cfg,n_target,energy_bin,swap_count):
    """Reject condition triples without an observed TRAIN-pair example."""
    n,b,k=np.broadcast_arrays(np.asarray(n_target),np.asarray(energy_bin).astype(str),np.asarray(swap_count))
    supported={(int(r["n_Li"]),int(r["swap_count"]),str(r["child_bin"])) for r in cfg.get("supported_requests",[]) if r["train_rows"]>0}
    absent={(int(nn),int(kk),str(bb)) for nn,bb,kk in zip(n.flat,b.flat,k.flat) if (int(nn),int(kk),str(bb)) not in supported}
    if absent:raise ValueError(f"Editor request has no protected TRAIN-pair support: {sorted(absent)}")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--parent",type=Path,required=True)
    p.add_argument("--seed-file",type=Path,required=True);p.add_argument("--requested-bin",choices=["low","middle","high"],required=True)
    p.add_argument("--swaps",type=int,choices=[1,2,3],required=True);p.add_argument("--draws-per-seed",type=int,default=1)
    p.add_argument("--random-seed",type=int,default=20261108);p.add_argument("--device",default="auto")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv);require_compute_allocation()
    if a.output.exists():raise FileExistsError("Preserve existing edit attempts")
    if a.draws_per_seed<1:raise ValueError("Use at least one draw per supplied seed")
    device=device_name(a.device);torch.set_num_threads(4)
    model,cfg=load_edit_model(a.checkpoint,device)
    if digest(a.parent)!=cfg["parent_sha256"]:raise ValueError("Supplied parent differs from this fixed-parent editor")
    with np.load(a.seed_file,allow_pickle=False) as z:
        seed=z["occupancy"] if "occupancy" in z else z["seeds"]
        ids=z["seed_ids"].astype(str) if "seed_ids" in z else z["structure_ids"].astype(str)
    if seed.shape!=(len(ids),model.n_sites) or not np.isin(seed,[0,1]).all():raise ValueError("Invalid supplied seed identities/masks")
    seed=np.repeat(seed,a.draws_per_seed,axis=0);ids=np.repeat(ids,a.draws_per_seed)
    n=seed.sum(1);k=np.full(len(seed),a.swaps)
    require_supported_requests(cfg,n,a.requested_bin,k)
    c=encode_condition(n,a.requested_bin,k,model.n_sites)
    rng=torch.Generator(device=device).manual_seed(a.random_seed)
    with torch.no_grad():
        result=model.sample(torch.tensor(seed,dtype=torch.float32,device=device),
            torch.tensor(c,device=device),torch.tensor(k,device=device),generator=rng)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,seeds=seed.astype(np.uint8),children=result["child"].cpu().numpy().astype(np.uint8),
        removed=result["removed"].cpu().numpy().astype(np.uint8),added=result["added"].cpu().numpy().astype(np.uint8),
        seed_ids=ids,n_Li=n,swap_count=k,requested_bin=np.full(len(seed),a.requested_bin),
        checkpoint_sha256=np.asarray(digest(a.checkpoint)),parent_sha256=np.asarray(cfg["parent_sha256"]),
        random_seed=np.asarray(a.random_seed),seed_source_sha256=np.asarray(digest(a.seed_file)))
    print(json.dumps(dict(status="GENERATED_UNSCORED",attempts=len(seed),output=str(a.output),
        raw_distance_exact_2k=True,energy_known=False,new_MLIP_calls=0),indent=2))


if __name__=="__main__":main()

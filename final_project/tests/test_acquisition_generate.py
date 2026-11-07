"""Small untrained network fixtures test schedule execution and provenance only."""

import csv
import json

import numpy as np
import pytest
import torch

from cathode_cvae.acquisition import make_allocation_plan
from cathode_cvae.acquisition_generate import allocation_schedules, execute_allocation, main
from cathode_cvae.conditions import fit_schema, schema_identity
from cathode_cvae.data import sha256
from cathode_cvae.learning.models import make_model


def fixture(tmp_path, *, sparse=False, requested_support=None):
    parent=tmp_path/"parent.npz"
    permutations=np.stack([np.roll(np.arange(4),i) for i in range(4)])
    np.savez(parent,numbers=[27,3,3,3,3],positions_A=np.arange(15).reshape(5,3)/10,
             cell_A=np.eye(3)*4,target_indices=[1,2,3,4],permutations=permutations)
    schema=fit_schema(np.full(10,2),np.zeros(10) if sparse else np.linspace(-2,2,10),
                      ["train"]*10,n_sites=4,provenance=dict(parent_sha256=sha256(parent)))
    cfg=dict(kind="ar_cvae",n_sites=4,condition_mode="multibin",condition_dim=4,
             latent_dim=2,hidden_dim=8,seed=7,parent_sha256=sha256(parent),condition_schema=schema,
             geometry_protocol="synthetic fixed-parent deletion; unrelaxed")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(15)
        model=make_model("ar_cvae",n_sites=4,condition_dim=4,latent_dim=2,hidden_dim=8)
    checkpoint=tmp_path/"best.pt";torch.save(dict(state_dict=model.state_dict(),configuration=cfg),checkpoint)
    support=["low","middle","high"] if requested_support is None else requested_support
    energies=np.linspace(0,.2,len(support))
    plan=make_allocation_plan(energies,support,np.ones(len(support)),[400.,500.],support,5,
        measure_kind="uniform_raw_sample",composition=2,schema_id=schema_identity(schema))
    payload=dict(schema_version="static-acquisition-run-v1",source=dict(parent_sha256=sha256(parent)),plans=[plan])
    allocation=tmp_path/"allocation.json";allocation.write_text(json.dumps(payload))
    return allocation,checkpoint,parent,payload


def test_all_schedules_exact_counts_zero_bins_and_joint_mixture_identity(tmp_path,monkeypatch):
    import cathode_cvae.acquisition_generate as module
    allocation,checkpoint,parent,payload=fixture(tmp_path)
    original=module.load_model;calls=[];contexts=[]
    def counted(path,device):
        calls.append(str(path));model,cfg=original(path,device);sample=model.sample
        def observe(c,n,**kwargs):
            contexts.append(c[0].cpu().numpy().copy())
            return sample(c,n,**kwargs)
        model.sample=observe
        return model,cfg
    monkeypatch.setattr(module,"load_model",counted)
    output=tmp_path/"output"
    result=execute_allocation(allocation,{"7":checkpoint},parent,output,device="cpu",sampling_seed=9,threads=1)
    assert calls==[str(checkpoint)]
    assert result["attempts"]==30 and result["schedules"]==6
    assert result["generated_attempts"]==30 and result["failed_requests"]==0
    assert result["zero_count_bin_requests"]>0
    assert result["tf32"] is False and result["physical_temperature_enters_decoder"] is False
    assert set(tuple(c) for c in contexts)<={(.5,1.,0.,0.),(.5,0.,1.,0.),(.5,0.,0.,1.)}
    with (output/"attempts.csv").open() as f:rows=list(csv.DictReader(f))
    assert len({r["attempt_id"] for r in rows})==30
    assert {r["fit_seed"] for r in rows}=={"7"}
    assert all(r["allocation_run_id"] and r["condition_schema_id"] for r in rows)
    mixture=[s for s in result["schedule_receipts"] if s["schedule_kind"]=="temperature_mixture"]
    assert len(mixture)==2
    for schedule in mixture:
        rr=[r for r in rows if r["allocation_run_id"]==schedule["allocation_run_id"]]
        assert len(rr)==5
        assert set(map(float,[r["temperature_K"] for r in rr]))=={400.,500.}
    with np.load(output/"candidates.npz",allow_pickle=False) as z:
        assert (z["occupancy"].sum(1)==2).all()
        assert (z["n_atoms"]==3).all()
        assert np.array_equal(z["masks"],1-z["occupancy"])
        assert len(set(z["structure_ids"]))==len(z["structure_ids"])<30
        assert z["attempt_frequency"].sum()==30
        for sid,key in zip(z["structure_ids"],z["canonical_masks"]):assert sid==f"mask4_{int(key,2):01x}"
    again=tmp_path/"again"
    execute_allocation(allocation,{"7":checkpoint},parent,again,device="cpu",sampling_seed=9,threads=1)
    assert (output/"attempts.csv").read_bytes()==(again/"attempts.csv").read_bytes()
    with np.load(output/"candidates.npz") as x,np.load(again/"candidates.npz") as y:
        for k in x.files:np.testing.assert_array_equal(x[k],y[k])


def test_unsupported_positive_requests_remain_in_attempt_denominator(tmp_path):
    allocation,checkpoint,parent,payload=fixture(tmp_path,sparse=True,requested_support=["high"])
    out=tmp_path/"out"
    receipt=execute_allocation(allocation,{"7":checkpoint},parent,out,device="cpu",threads=1)
    assert receipt["status"]=="FAILED_REQUESTS"
    assert receipt["attempts"]==receipt["failed_requests"]==30
    with (out/"attempts.csv").open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==30 and {r["status"] for r in rows}=={"failed_request"}
    assert all(r["error"] and not r["scoring_structure_id"] for r in rows)
    with np.load(out/"candidates.npz") as z:assert z["occupancy"].shape==(0,4)


def test_strict_checkpoint_seed_schema_and_parent_binding(tmp_path):
    allocation,checkpoint,parent,payload=fixture(tmp_path)
    with pytest.raises(ValueError,match="fit seed"):
        execute_allocation(allocation,{"8":checkpoint},parent,tmp_path/"bad_seed",device="cpu",threads=1)
    payload["plans"][0]["schema_id"]="different"
    for group in [*payload["plans"][0]["per_temperature"],*payload["plans"][0]["temperature_mixtures"]]:
        for request in group["requests"]:request["schema_id"]="different"
    allocation.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match="schema differs"):
        execute_allocation(allocation,{"7":checkpoint},parent,tmp_path/"bad_schema",device="cpu",threads=1)
    other=tmp_path/"other.npz";np.savez(other,numbers=[28,3,3,3,3])
    with pytest.raises(ValueError,match="different parent"):
        execute_allocation(allocation,{"7":checkpoint},other,tmp_path/"bad_parent",device="cpu",threads=1)


def test_full_cli_uses_seed_checkpoint_mapping_and_writes_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr("cathode_cvae.runtime.require_compute_allocation",lambda:None)
    allocation,checkpoint,parent,payload=fixture(tmp_path)
    out=tmp_path/"cli"
    result=main(["--allocation",str(allocation),"--checkpoints",f"7={checkpoint}","--parent",str(parent),
                 "--output-dir",str(out),"--device","cpu","--sampling-seed","19","--threads","1"])
    assert result["attempts"]==30
    assert json.loads((out/"generation_receipt.json").read_text())["no_physical_labels_generated"] is True
    bad=json.loads(allocation.read_text());bad["plans"][0]["per_temperature"][0]["budget"]+=1
    with pytest.raises(ValueError,match="declared budget"):allocation_schedules(bad)

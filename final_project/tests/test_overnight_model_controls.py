"""Causal, serialization and update-clock controls; fixtures are synthetic."""
import json

import numpy as np
import pytest
import torch

from cathode_cvae.graph import CausalGraphContext, build_periodic_graph, fixed_nonphysical_context_spec
from cathode_cvae.parent_spec import ParentSpec
from cathode_cvae.learning.models import make_model
from cathode_cvae.learning.generate_mask import load_model, main as generate
from cathode_cvae.learning.train_mask import main as train, sha256
from cathode_cvae.learning.schedules import beta_at_update, schedule_metadata


def graph_fixture():
    # Na sites exercise the parent-driven interface without hard-coded Li/Co.
    parent=ParentSpec(numbers=[11]*6+[8],positions_A=[[0,0,0],[1,0,0],[0,1.2,0],
        [1.3,1.1,0],[.3,.4,1.7],[1.5,.7,1.5],[.5,.5,.8]],cell_A=np.eye(3)*4,
        mobile_indices=np.arange(6),permutations=np.arange(6)[None],pbc=[True]*3)
    return build_periodic_graph(parent,cutoff_A=2.3)


def test_reference_update_clock_exactly_reproduces_old_h0_epoch_exposure():
    for u in range(4000):
        old=beta_at_update(mode="legacy_epoch",final=.1,update=u,epoch=u//79+1,warmup_epochs=20,reference_updates_per_epoch=79)
        new=beta_at_update(mode="reference_updates",final=.1,update=u,epoch=u//8+1,warmup_epochs=20,reference_updates_per_epoch=79)
        assert old==new
    assert beta_at_update(mode="reference_updates",final=.1,update=0,epoch=1,warmup_epochs=20,reference_updates_per_epoch=79)==.1*(1/20)
    assert beta_at_update(mode="reference_updates",final=.1,update=1501,epoch=1,warmup_epochs=20,reference_updates_per_epoch=79)==.1
    spec=schedule_metadata("reference_updates",.1,20,79)
    assert spec["nominal_warmup_updates"]==1580 and spec["full_weight_start_update_one_based"]==1502


def test_fixed_nonphysical_control_preserves_capacity_weights_and_causality():
    torch.set_num_threads(1);graph=graph_fixture();spec=fixed_nonphysical_context_spec(graph,20261108)
    assert spec==fixed_nonphysical_context_spec(graph,20261108)
    physical=CausalGraphContext(graph);scrambled=CausalGraphContext(graph,spec["association"],spec["site_permutation"])
    assert physical.to_spec()["kernels_sha256"]!=spec["kernels_sha256"]
    for a,b in zip(physical.kernels,scrambled.kernels):
        assert torch.allclose(a.flatten().sort().values,b.flatten().sort().values,atol=8*torch.finfo(torch.float32).eps,rtol=0)
    torch.manual_seed(44);a=make_model("ar_baseline",n_sites=6,condition_dim=4,graph_context=True,graph_data=graph.to_dict())
    torch.manual_seed(44);b=make_model("ar_baseline",n_sites=6,condition_dim=4,graph_context=True,graph_data=graph.to_dict(),context_spec=spec)
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    assert all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters()))
    x=torch.tensor([[1.,0.,1.,0.,0.,1.]],requires_grad=True);c=torch.tensor([[.5,1.,0.,0.]])
    for i in (0,2,5):
        assert torch.allclose(scrambled.teacher_features(x)[:,i],scrambled.step_features(x,i))
        gradient=torch.autograd.grad(b.decode_logits(None,c,x)[0,i],x,retain_graph=True)[0]
        assert torch.count_nonzero(gradient[:,i:])==0
    assert torch.count_nonzero(torch.autograd.grad(b.decode_logits(None,c,x)[0,3],x)[0][:,:3])>0
    with pytest.raises(ValueError,match="outside the retained"):
        CausalGraphContext(graph,"fixed_nonphysical",np.arange(6))


@pytest.mark.parametrize("kind,encoder",[("ar_cvae","graph"),("ar_cvae","dense"),("ar_baseline","dense")])
def test_new_context_width_reload_exact_counts_and_buffer_binding(tmp_path,kind,encoder):
    graph=graph_fixture();spec=fixed_nonphysical_context_spec(graph,20261108)
    latent=0 if kind=="ar_baseline" else 4
    model=make_model(kind,n_sites=6,condition_dim=4,latent_dim=latent,hidden_dim=128,
        encoder_kind=encoder,graph_context=True,graph_data=graph.to_dict(),context_spec=spec)
    cfg=dict(model_format_version=3,kind=kind,n_sites=6,condition_dim=4,latent_dim=latent,hidden_dim=128,
        encoder_kind=encoder,graph_context=True,graph=graph.to_dict(),decoder_context_spec=spec,
        energy_target="endpoint",label_energy_protocol="synthetic endpoint",proposal_geometry_protocol="pristine-derived deletion",
        beta_schedule=schedule_metadata("reference_updates",.1,20,79))
    path=tmp_path/"checkpoint.pt";torch.save(dict(state_dict=model.state_dict(),configuration=cfg),path)
    loaded,_=load_model(path,"cpu")
    n=torch.tensor([0,1,3,6]);c=torch.zeros(4,4);c[:,0]=n/6;c[:,1]=1
    z=torch.randn(4,latent);uniforms=torch.rand(4,6)
    expected=model.sample(c,n,z=z,uniforms=uniforms)
    assert torch.equal(expected,loaded.sample(c,n,z=z,uniforms=uniforms))
    assert torch.equal(expected.sum(1),n.to(torch.float32))
    damaged={k:v.clone() for k,v in model.state_dict().items()};damaged["graph_context.kernels"][0,0,0]+=.1
    torch.save(dict(state_dict=damaged,configuration=cfg),path)
    with pytest.raises(ValueError,match="buffers disagree"):
        load_model(path,"cpu")


def test_endpoint_label_protocol_does_not_claim_generated_geometry_is_relaxed(tmp_path):
    parent=tmp_path/"parent.npz";np.savez(parent,permutations=np.arange(4)[None])
    occupancy=np.array([[1,1,0,0],[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1],[0,0,1,1]],np.uint8)
    data=tmp_path/"labels.npz";np.savez(data,occupancy=occupancy,n_target=np.full(6,2),energy_eV_atom=np.arange(6.),
        split=np.array(["train"]*3+["val"]*2+["test"]),weight=np.ones(6),structure_ids=np.array([f"s{i}" for i in range(6)]),parent_sha256=np.array(sha256(parent)))
    fit=tmp_path/"fit"
    train(["--data",str(data),"--parent",str(parent),"--output-dir",str(fit),"--condition-mode","multibin",
        "--kind","ar_cvae","--latent-dim","4","--energy-target","endpoint","--label-energy-protocol","synthetic fixed-cell endpoint",
        "--beta-schedule","reference_updates","--max-updates","2","--epochs","10","--minimum-epochs","10","--patience","10",
        "--batch-size","2","--validation-samples","1","--device","cpu","--threads","1","--max-seconds","60"])
    cfg=json.loads((fit/"configuration.json").read_text())
    assert cfg["label_energy_protocol"]=="synthetic fixed-cell endpoint"
    output=tmp_path/"draws.npz"
    generate(["--checkpoint",str(fit/"best.pt"),"--parent",str(parent),"--n-target","2","--requested-bin","low",
        "--count","3","--device","cpu","--output",str(output)])
    with np.load(output,allow_pickle=False) as z:
        assert str(z["geometry_protocol"].item()).startswith("pristine-derived deletion")
        assert str(z["label_energy_protocol"].item())=="synthetic fixed-cell endpoint"
        assert str(z["energy_target"].item())=="endpoint"


def test_cyclic_schedule_changes_only_after_first_matching_reference_cycle():
    common=dict(final=.1,epoch=1,warmup_epochs=20,reference_updates_per_epoch=79)
    for u in range(2000):
        assert beta_at_update(mode="reference_updates",update=u,**common)==beta_at_update(mode="cyclic_updates",update=u,cycle_updates=2000,**common)
    assert beta_at_update(mode="cyclic_updates",update=2000,cycle_updates=2000,**common)==.1*(1/20)
    assert beta_at_update(mode="reference_updates",update=2000,**common)==.1
    assert schedule_metadata("cyclic_updates",.1,20,79,2000)["cycle_updates"]==2000
    with pytest.raises(ValueError,match="entire reference warmup"):
        beta_at_update(mode="cyclic_updates",update=0,cycle_updates=1000,**common)


@pytest.mark.parametrize("remaining,kernel_mode",[(True,"legacy"),(False,"radial_shells"),(True,"radial_shells")])
def test_optional_prefix_features_count_causality_teacher_sampling_and_reload(tmp_path,remaining,kernel_mode):
    torch.set_num_threads(1);graph=graph_fixture()
    context=CausalGraphContext(graph,kernel_mode=kernel_mode);spec=context.to_spec()
    if kernel_mode=="radial_shells":
        assert spec["feature_dim"]==3*len(spec["shell_centers_A"])
        assert spec["shell_centers_A"]==sorted(spec["shell_centers_A"])
        assert spec["shell_centers_A"][-1]>spec["shell_centers_A"][0]
    model=make_model("ar_cvae",n_sites=6,condition_dim=4,latent_dim=4,graph_context=True,
        graph_data=graph.to_dict(),context_spec=spec,remaining_count_features=remaining)
    c=torch.tensor([[.5,1.,0.,0.],[.5,0.,0.,1.]])
    z=torch.randn(2,4);uniforms=torch.rand(2,6);step_logits=[]
    hook=model.output.register_forward_hook(lambda _m,_a,out:step_logits.append(out.detach().reshape(2)))
    sample=model.sample(c,3,z=z,uniforms=uniforms);hook.remove()
    assert torch.equal(sample.sum(1),torch.tensor([3.,3.]))
    teacher=model.decode_logits(z,c,sample)
    assert torch.allclose(torch.stack(step_logits,dim=1),teacher,atol=1e-6,rtol=2e-6)
    xx=sample.clone().requires_grad_()
    for i in (0,2,5):
        gradient=torch.autograd.grad(model.decode_logits(z,c,xx)[0,i],xx,retain_graph=True)[0]
        assert torch.count_nonzero(gradient[:,i:])==0
    cfg=dict(model_format_version=4,kind="ar_cvae",n_sites=6,condition_dim=4,latent_dim=4,hidden_dim=64,
        encoder_kind="dense",graph_context=True,graph=graph.to_dict(),decoder_context_spec=spec,
        remaining_count_features=remaining,energy_target="deletion",label_energy_protocol="synthetic",
        proposal_geometry_protocol="synthetic",beta_schedule=schedule_metadata("reference_updates",.1,20,79))
    path=tmp_path/"features.pt";torch.save(dict(configuration=cfg,state_dict=model.state_dict()),path)
    loaded,_=load_model(path,"cpu")
    assert torch.equal(sample,loaded.sample(c,3,z=z,uniforms=uniforms))
    if remaining:
        with pytest.raises(ValueError,match="Condition count disagrees"):
            loaded.sample(c,2,z=z,uniforms=uniforms)

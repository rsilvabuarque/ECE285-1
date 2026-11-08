"""Exact local-edit probabilities and inference boundaries on synthetic masks."""
import inspect
import itertools

import numpy as np
import pytest
import torch

from cathode_cvae.learning import edit_model
from cathode_cvae.learning.edit_model import SeedEditCVAE, edit_targets, encode_condition, uniform_edits
from cathode_cvae.learning.generate_edit import digest, load_edit_model


def test_exact_edit_distribution_normalizes_over_all_allowed_children():
    torch.set_num_threads(1);torch.manual_seed(9)
    model=SeedEditCVAE(n_sites=4,latent_dim=2,hidden_dim=8).double()
    children=[]
    for remove,add in itertools.product([0,1],[2,3]):
        child=[1.,1.,0.,0.];child[remove]=0.;child[add]=1.;children.append(child)
    seed=torch.tensor([[1.,1.,0.,0.]]).double().repeat(4,1)
    c=torch.tensor(encode_condition(2,"low",1,4)).double().repeat(4,1)
    loss=model.reconstruction(seed,torch.tensor(children).double(),c,torch.zeros((4,2)).double(),torch.ones(4,dtype=torch.long))
    assert torch.allclose(torch.exp(-loss).sum(),torch.tensor(1.).double(),atol=1e-12,rtol=0)
    forced=torch.tensor([[0.,0.,1.,1.]]).double()
    c2=torch.tensor(encode_condition(2,"high",2,4)).double()[None]
    assert abs(model.reconstruction(seed[:1],forced,c2,torch.zeros((1,2)).double(),2).item())<1e-12


def test_sampling_is_exact_and_has_no_target_child_or_posterior_input():
    torch.set_num_threads(1);torch.manual_seed(17)
    model=SeedEditCVAE(n_sites=48,latent_dim=4,hidden_dim=16)
    counts=torch.tensor([n for n in [12,24,36] for k in [1,2,3] for _ in range(2)])
    k=torch.tensor([k for n in [12,24,36] for k in [1,2,3] for _ in range(2)])
    seed=(torch.arange(48)[None]<counts[:,None]).float()
    c=torch.tensor(encode_condition(counts.numpy(),"middle",k.numpy()))
    def forbidden(*args,**kwargs):raise AssertionError("Posterior entered inference")
    model.encode=forbidden;model.encoder.forward=forbidden
    assert "child" not in inspect.signature(model.sample).parameters
    u=torch.rand(seed.shape);v=torch.rand(seed.shape);z=torch.randn((len(seed),4))
    first=model.sample(seed,c,k,z=z,uniforms_remove=u,uniforms_add=v)
    again=model.sample(seed,c,k,z=z,uniforms_remove=u,uniforms_add=v)
    for result in [first,uniform_edits(seed,k,uniforms_remove=u,uniforms_add=v)]:
        assert torch.equal(result["child"].sum(1),counts.float())
        assert torch.equal((result["child"]!=seed).sum(1),2*k)
        assert torch.equal(result["removed"].sum(1),k.float())
        assert torch.equal(result["added"].sum(1),k.float())
        assert not bool((result["removed"]*(1-seed)).any())
        assert not bool((result["added"]*seed).any())
        assert not bool((result["removed"]*result["added"]).any())
    assert torch.equal(first["child"],again["child"])


def test_variational_encoder_and_exact_set_likelihood_have_finite_gradients():
    torch.set_num_threads(1);torch.manual_seed(8)
    model=SeedEditCVAE(n_sites=6,latent_dim=2,hidden_dim=12)
    seed=torch.tensor([[1.,1.,1.,0.,0.,0.]])
    child=torch.tensor([[1.,1.,0.,1.,0.,0.]])
    c=torch.tensor(encode_condition(3,"low",1,6))[None]
    mu,lv=model.encode(seed,child,c);z=mu+torch.exp(.5*lv)*torch.ones_like(mu)
    kl=.5*(mu.square()+lv.exp()-lv-1).sum(1)
    loss=(model.reconstruction(seed,child,c,z,1)+.1*kl).mean();loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad is not None and bool((p.grad!=0).any()) for p in model.encoder.parameters())
    assert bool((model.mu.weight.grad!=0).any())


def test_versioned_editor_reload_is_strict(tmp_path):
    model=SeedEditCVAE(n_sites=6,latent_dim=2,hidden_dim=8)
    cfg=dict(kind="seed_edit_cvae",edit_format_version=1,n_sites=6,condition_dim=5,latent_dim=2,hidden_dim=8,
        model_source_sha256=digest(edit_model.__file__))
    path=tmp_path/"editor.pt";torch.save(dict(configuration=cfg,state_dict=model.state_dict()),path)
    loaded,actual=load_edit_model(path)
    assert actual==cfg and loaded.latent_dim==2
    cfg["latent_dim"]=4;torch.save(dict(configuration=cfg,state_dict=model.state_dict()),path)
    with pytest.raises(RuntimeError):load_edit_model(path)


def test_joint_transform_preserves_edit_alignment_and_rejects_copied_seed():
    seed=torch.tensor([[1.,1.,0.,0.]])
    child=torch.tensor([[1.,0.,1.,0.]])
    removed,added,k=edit_targets(seed,child,1);p=torch.tensor([1,2,3,0])
    r,a,_=edit_targets(seed[:,p],child[:,p],1)
    assert torch.equal(r,removed[:,p]) and torch.equal(a,added[:,p])
    with pytest.raises(ValueError,match="copied child"):edit_targets(seed,seed,1)

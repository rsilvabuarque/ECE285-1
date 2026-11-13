"""Synthetic checks for probe fitting and latent-only guidance."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


path=Path(__file__).resolve().parents[1]/'scripts/adaptive_20261112/latent_physical_diagnostics.py'
spec=importlib.util.spec_from_file_location('latent_physical_diagnostics',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_probe_learns_training_linear_signal_and_reports_constant_target():
    train=np.arange(20,dtype=float).reshape(-1,1);target=2+3*train[:,0]
    coef=module.fit_ridge(train,target,alpha=.001)
    heldout=np.array([[21.],[22.]])
    assert module.predict_ridge(coef,heldout)==pytest.approx([65.,68.],abs=.001)
    result=module.regression_summary(np.ones(4),np.zeros(4))
    assert result['R2_vs_VAL_mean'] is None and result['Spearman'] is None


def test_guide_changes_only_latent_and_is_independent_of_batch_size():
    head=torch.nn.Linear(4,1,bias=False)
    with torch.no_grad():head.weight[:]=torch.tensor([[1.,0.,0.,0.]])
    for p in head.parameters():p.requires_grad_(False)
    before=head.weight.detach().clone();z=torch.tensor([[0.,.5],[0.,-.5]])
    condition=torch.zeros(2,2)
    guided,details=module.guide_latent(head,z,condition,'latent',1.)
    one,_=module.guide_latent(head,z[:1],condition[:1],'latent',1.)
    assert torch.allclose(guided[:1],one)
    assert guided[:,0].min()>.98 and torch.equal(guided[:,1],z[:,1])
    assert torch.equal(head.weight,before) and torch.equal(z[:,0],torch.zeros(2))
    assert details['guide_capable']
    bounded,_=module.guide_latent(head,z,condition,'latent',100.,steps=2,learning_rate=.1)
    assert torch.all(bounded.abs()<=3)


def test_condition_and_no_head_guides_are_no_ops():
    z=torch.tensor([[4.,.5],[0.,-.5]]);condition=torch.zeros(2,2)
    head=torch.nn.Linear(2,1)
    out,details=module.guide_latent(head,z,condition,'condition',1.)
    assert torch.equal(out,z) and not details['guide_capable']
    out,details=module.guide_latent(None,z,condition,None,1.)
    assert torch.equal(out,z) and not details['guide_capable']

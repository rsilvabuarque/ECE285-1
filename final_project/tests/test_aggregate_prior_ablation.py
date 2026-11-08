"""Analytic Gaussian proposal checks, without assessment labels or inference."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


path=Path(__file__).resolve().parents[1]/"scripts/conditioning_20261107/aggregate_prior_ablation.py"
spec=importlib.util.spec_from_file_location("aggregate_sampling",path)
ablation=importlib.util.module_from_spec(spec);spec.loader.exec_module(ablation)


def test_total_covariance_uses_population_denominator_and_posterior_variance():
    mu=np.asarray([[-1.,-2.],[1.,2.]])
    lv=np.log(np.asarray([[.25,1.],[.25,1.]]))
    m,S,L,eigen=ablation.gaussian_moments(mu,lv)
    assert np.array_equal(m,[0.,0.])
    assert np.allclose(S,[[1.25,2.],[2.,5.]],atol=1e-14,rtol=0)
    assert np.allclose(L@L.T,S,atol=1e-14,rtol=0) and (eigen>0).all()
    # With an unbiased N-1 denominator this between-mean term would double.
    assert S[0,0]!=2.25


def test_nonfinite_or_zero_variances_are_rejected_without_jitter():
    with pytest.raises(ValueError,match="strictly positive"):
        ablation.gaussian_moments(np.zeros((2,2)),np.full((2,2),-10000.))
    with pytest.raises(ValueError,match="Finite"):
        ablation.gaussian_moments(np.asarray([[np.nan,0.]]),np.zeros((1,2)))


def test_train_selection_cannot_use_heldout_masks_or_labels():
    keys=np.asarray(["0"*12+"1"*36,"0"*24+"1"*24,"0"*36+"1"*12])
    source=dict(split=np.asarray(["train","val","test"]),canonical_masks=keys,n_Li=np.asarray([12,24,36]),
        energy_eV_atom=np.asarray([-1.,-2.,-3.]),structure_ids=np.asarray(["TRAIN","VAL","TEST"]))
    a=ablation.training_examples(source)
    source["energy_eV_atom"][1:]=[1e20,-1e20]
    source["canonical_masks"][1:]=["not-valid-and-must-not-be-encoded","not-valid-and-must-not-be-encoded"]
    b=ablation.training_examples(source)
    assert all(np.array_equal(a[k],b[k]) for k in a)
    assert a["structure_ids"].tolist()==["TRAIN"]


def test_shared_epsilon_is_distinct_from_same_actual_latent():
    epsilon=np.asarray([[1.,-2.],[0.,1.]],np.float32)
    m=np.asarray([2.,3.],np.float32);L=np.asarray([[2.,0.],[.5,1.]],np.float32)
    aggregate=ablation.transformed_latents(epsilon,m,L)
    assert np.allclose(aggregate,[[4.,1.5],[2.,4.]])
    assert not np.array_equal(aggregate,epsilon)
    assert np.array_equal(ablation.transformed_latents(epsilon,np.zeros(2,np.float32),np.eye(2,dtype=np.float32)),epsilon)

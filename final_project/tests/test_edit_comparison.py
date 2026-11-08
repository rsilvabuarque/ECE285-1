"""Independent checks of matched edit information and symmetry interpretation."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from cathode_cvae.acquisition import canonical_keys
from cathode_cvae.learning.edit_model import SeedEditCVAE, encode_condition, uniform_edits


PATH=Path(__file__).resolve().parents[1]/"scripts/conditioning_20261107/generate_edit_comparison.py"
SPEC=importlib.util.spec_from_file_location("edit_comparison",PATH)
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def bank():
    x=np.zeros((6,48),np.uint8)
    for i,n in enumerate([12,12,24,24,36,36]):x[i,np.roll(np.arange(48),i)[:n]]=1
    return dict(occupancy=x,n_Li=x.sum(1),seed_bin=np.array(["low","low","middle","middle","high","high"]))


def test_paired_seed_and_budget_law_is_reproducible_and_ignores_energy():
    b=bank();first=M.paired_trials(b,20261106,24,"middle",512)
    b["seed_energy_eV_atom"]=np.arange(6)*1.e6
    second=M.paired_trials(b,20261106,24,"middle",512)
    assert all(np.array_equal(a,c) for a,c in zip(first,second))
    rows,k=first
    assert set(rows)=={2,3} and set(k)=={1,2,3}
    assert not np.array_equal(rows,M.paired_trials(b,20261107,24,"middle",512)[0])
    with pytest.raises(ValueError,match="Unsupported"):M.paired_trials(b,20261106,12,"high",8)


def test_actual_raw_edit_sites_and_two_k_are_checked():
    seed=np.array([[1,1,0,0]],np.uint8);child=np.array([[0,1,1,0]],np.uint8)
    removed=np.array([[1,0,0,0]],np.uint8);added=np.array([[0,0,1,0]],np.uint8)
    M.verify_edit(seed,child,removed,added,np.array([1]))
    with pytest.raises(ValueError,match="actual raw"):M.verify_edit(seed,child,added,removed,np.array([1]))
    with pytest.raises(ValueError,match="count"):M.verify_edit(seed,child,removed,added,np.array([2]))


def test_independent_canonicalization_does_not_preserve_local_edit():
    seed=np.array([[1,1,0,0]],np.uint8);child=np.array([[0,1,1,0]],np.uint8)
    rotations=np.array([np.roll(np.arange(4),i) for i in range(4)])
    # A nontrivial one-swap edit joins symmetry-equivalent masks. Its raw edit
    # remains real even though independently canonicalized representatives agree.
    assert np.count_nonzero(seed!=child)==2
    assert canonical_keys(seed,rotations)[0]==canonical_keys(child,rotations)[0]
    operation=rotations[1]
    removed=((seed==1)&(child==0)).astype(np.uint8)
    added=((seed==0)&(child==1)).astype(np.uint8)
    M.verify_edit(seed[:,operation],child[:,operation],removed[:,operation],added[:,operation],np.array([1]))


def test_condition_support_is_checked_per_trial_not_manufactured():
    cfg=dict(supported_requests=[dict(n_Li=12,child_bin="low",swap_count=1,train_rows=10),
                                 dict(n_Li=12,child_bin="low",swap_count=3,train_rows=0)])
    assert M.supported_trials(cfg,12,"low",np.array([1,2,3])).tolist()==[True,False,False]
    with pytest.raises(ValueError,match="explicit TRAIN"):M.supported_trials({},12,"low",np.array([1]))


def test_model_and_uniform_baseline_accept_identical_seed_budget_uniform_streams():
    seed=torch.tensor(bank()["occupancy"][:3],dtype=torch.float32)
    k=torch.tensor([1,2,3]);c=torch.tensor(encode_condition(seed.sum(1).numpy(),"low",k.numpy()))
    model=SeedEditCVAE(n_sites=48,latent_dim=2,hidden_dim=8)
    def forbidden(*args,**kwargs):raise AssertionError("Generation consulted the target encoder")
    model.encode=forbidden
    g=torch.Generator().manual_seed(9281)
    ur=torch.rand((3,48),generator=g);ua=torch.rand((3,48),generator=g);z=torch.randn((3,2),generator=g)
    learned=model.sample(seed,c,k,z=z,uniforms_remove=ur,uniforms_add=ua)
    baseline=uniform_edits(seed,k,uniforms_remove=ur,uniforms_add=ua)
    repeated=uniform_edits(seed,k,uniforms_remove=ur,uniforms_add=ua)
    assert torch.equal(baseline["child"],repeated["child"])
    for result in [learned,baseline]:
        M.verify_edit(seed.numpy(),result["child"].numpy(),result["removed"].numpy(),result["added"].numpy(),k.numpy())


def test_useful_yield_keeps_failed_known_and_duplicate_attempts(monkeypatch):
    monkeypatch.syspath_prepend(str(PATH.parent))
    import evaluate_edit_comparison as evaluator
    good=dict(canonical_mask="0011",energy_known=True,novel_both=True,novel_H0TRAIN=True,novel_history=True,
        measured_bin="low",requested_bin="low",passes_original_q20=True,delta_energy_meV_atom=-2.,
        raw_Hamming=2,symmetry_min_Hamming=0,descriptor_change=.1)
    known={**good,"canonical_mask":"0101","novel_both":False,"novel_history":False}
    failed={**good,"canonical_mask":"","energy_known":False}
    summary=evaluator.summarize_cell([good,dict(good),known,failed])
    assert summary["attempts"]==4 and summary["failed_or_unmeasured_attempts"]==1
    assert summary["distinct_novel_requested_orbits"]==1
    assert summary["novel_inbin_yield"]==.25 and summary["bin_hit_rate"]==.75
    assert summary["novel_original_q20_yield"]==.25

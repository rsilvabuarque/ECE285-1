"""Synthetic guards for the read-only editor archive-alias audit."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/conditioning_20261107"))
from check_editor_checkpoint_aliases import exact_equal,normalized_members,sha


def test_identical_torch_save_alias_payload_but_different_archive_roots(tmp_path):
    value=dict(state_dict={"weight":torch.tensor([1.,-0.])},configuration=dict(kind="seed_edit_cvae"),
        optimizer_updates=9,latent_statistics=dict(KL=np.float64(1.25)))
    left=tmp_path/"best_validation_beta_0p1.pt";right=tmp_path/"best.pt"
    torch.save(value,left);torch.save(value,right)
    assert sha(left)!=sha(right)
    assert normalized_members(left)[0]!=normalized_members(right)[0]
    assert normalized_members(left)[1]==normalized_members(right)[1]
    assert exact_equal(value,copy.deepcopy(value),torch)["tensor_elements"]==2


def test_alias_comparison_rejects_tensor_bits_metadata_and_numpy_scalar_changes():
    value=dict(state_dict={"weight":torch.tensor([1.,-0.])},optimizer_updates=9,KL=np.float64(-0.))
    changed=copy.deepcopy(value);changed["state_dict"]["weight"][1]=0.
    with pytest.raises(ValueError,match="Tensor byte mismatch"):exact_equal(value,changed,torch)
    changed=copy.deepcopy(value);changed["optimizer_updates"]=10
    with pytest.raises(ValueError,match="Scalar mismatch"):exact_equal(value,changed,torch)
    changed=copy.deepcopy(value);changed["KL"]=np.float64(0.)
    with pytest.raises(ValueError,match="NumPy scalar dtype/byte mismatch"):exact_equal(value,changed,torch)

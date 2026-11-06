"""Numerical likelihood/symmetry/gauge checks, independent of a GPU or MLIP."""
import itertools

import numpy as np
import pytest
import torch

from cathode_cvae.learning.generate_geometry import remove_translation, load_occupancy
from cathode_cvae.learning.generate_mask import canonical_masks
from cathode_cvae.learning.models import exact_count_nll, make_model, sample_exact_count
from cathode_cvae.learning.train_geometry import prepare_data, validate_parent_binding
from cathode_cvae.learning.runtime import digest


def test_conditional_partition_and_gradient_against_enumeration():
    torch.set_num_threads(1)
    logits=torch.tensor([[.2,-.4,1.3,.1]],dtype=torch.float64,requires_grad=True)
    all_binary=torch.tensor(list(itertools.product([0.,1.],repeat=4)),dtype=torch.float64)
    x=torch.tensor([[1.,0.,1.,0.]],dtype=torch.float64)
    support=all_binary[all_binary.sum(1)==2]
    expected=torch.logsumexp(support@logits[0],0)-(logits*x).sum()
    actual=exact_count_nll(logits,x)[0]
    assert torch.allclose(actual,expected,atol=1e-11,rtol=0)
    actual.backward()
    expected_grad=(torch.softmax(support@logits.detach()[0],0)[:,None]*support).sum(0)-x[0]
    assert torch.allclose(logits.grad[0],expected_grad,atol=1e-11,rtol=0)
    for n in range(5):
        support=all_binary[all_binary.sum(1)==n]
        nll=exact_count_nll(logits.detach().expand(len(support),-1),support)
        assert abs(torch.exp(-nll).sum().item()-1)<1e-11
        samples=sample_exact_count(logits.detach().expand(64,-1),n)
        assert torch.all(samples.sum(1)==n)


@pytest.mark.parametrize('kind',['ar_cvae','ar_baseline'])
def test_autoregressive_distribution_normalizes_on_exact_support(kind):
    torch.set_num_threads(1)
    torch.manual_seed(782)
    model=make_model(kind,n_sites=4).double()
    all_binary=torch.tensor(list(itertools.product([0.,1.],repeat=4)),dtype=torch.float64)
    for n in range(5):
        support=all_binary[all_binary.sum(1)==n]
        c=torch.tensor([n/4,1.],dtype=torch.float64).expand(len(support),-1)
        latent=torch.zeros(len(support),model.latent_dim,dtype=torch.float64)
        loss=model.reconstruction(support,c,latent,exact_count=True)
        assert abs(torch.exp(-loss).sum().item()-1)<1e-10
        loss.sum().backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    c=torch.tensor([[n/4,1.] for n in range(5)],dtype=torch.float64)
    samples=model.sample(c,torch.arange(5))
    assert torch.equal(samples.sum(1),torch.arange(5,dtype=torch.float64))


def test_canonical_orbit_witness_and_count():
    permutations=np.asarray([[0,1,2,3],[1,2,3,0],[2,3,0,1],[3,0,1,2]])
    occupancy=np.asarray([[1,1,0,0],[0,1,1,0],[1,0,1,0]],np.uint8)
    keys,witness=canonical_masks(occupancy,permutations)
    assert keys[0]==keys[1]
    assert keys[0]!=keys[2]
    inv=np.argsort(permutations,axis=1)
    for row,key,j in zip(occupancy,keys,witness):
        assert ''.join(map(str,(1-row)[inv[j]]))==key
        assert key.count('0')==int(row.sum())


def test_translation_gauge_ignores_absent_atoms():
    from ase.data import atomic_masses
    numbers=np.asarray([3,27,8])
    presence=np.asarray([[False,True,True],[True,True,True]])
    u=np.asarray([[[100,0,0],[1,2,3],[2,1,0]],[[1,2,3],[1,1,1],[0,2,4]]],float)
    fixed=remove_translation(u,presence,numbers)
    assert (fixed[~presence]==0).all()
    masses=atomic_masses[numbers][None,:]*presence
    assert np.allclose((fixed*masses[:,:,None]).sum(1),0,atol=1e-12)
    assert np.allclose(fixed[0,1]-fixed[0,2],u[0,1]-u[0,2])


def _geometry_archive(path,perturb_test=False,leak=False):
    # Synthetic preprocessing fixture, not an LCO result.
    cohorts=np.tile(['deletion','fixed_intermediate','fixed_endpoint'],4)
    sid=np.repeat(['parent_A','parent_B','parent_C','parent_D'],3)
    split=np.repeat(['train','train','val','test'],3)
    if leak:
        sid[-1]='parent_A'
    energy=-np.arange(12,dtype=float)
    if perturb_test:
        energy[split=='test']-=10000
    displacement=np.arange(12*9,dtype=float).reshape(12,3,3)*.01
    np.savez(path,localized=np.ones(12,bool),split=split,cohort=cohorts,structure_ids=sid,
             presence=np.ones((12,3),bool),displacement_A=displacement,
             frame_weight=np.ones(12),energy_eV=energy,
             force_component_rms_eV_A=np.arange(12)*.01,
             configurational_pressure_GPa=np.arange(12)*.1,
             deviatoric_stress_GPa=np.arange(12)*.02,n_Li=np.ones(12,int),
             occupancy=np.tile([1,0],(12,1)),deformation=np.tile(np.eye(3),(12,1,1)),
             cell_A=np.tile(np.eye(3),(12,1,1)))


def test_geometry_conditions_and_scale_exclude_test(tmp_path):
    first=tmp_path/'a.npz';second=tmp_path/'b.npz'
    _geometry_archive(first);_geometry_archive(second,perturb_test=True)
    a=prepare_data(first);b=prepare_data(second)
    assert a[7]==b[7]  # displacement scale
    assert a[8]==b[8]  # condition training means/stds
    assert a[9]==b[9]  # requested attained endpoint targets
    leak=tmp_path/'leak.npz';_geometry_archive(leak,leak=True)
    with pytest.raises(ValueError,match='cross protected splits'):
        prepare_data(leak)


def test_geometry_input_rejects_float_truncation_and_preserves_orientation(tmp_path):
    path=tmp_path/'bad.npz'
    np.savez(path,occupancy=np.asarray([[.2,.9]]))
    with pytest.raises(ValueError,match='binary before integer conversion'):
        load_occupancy(path)
    np.savez(path,occupancy=np.asarray([[0,1]]),canonical_occupancy=np.asarray([[1,0]]),
             parent_sha256=np.asarray('expected'))
    occupancy,_=load_occupancy(path,'expected')
    assert occupancy.tolist()==[[0,1]]
    with pytest.raises(ValueError,match='parent hash'):
        load_occupancy(path,'other')


def test_aligned_archive_requires_parent_provenance(tmp_path):
    parent=tmp_path/'parent.npz';other=tmp_path/'other.npz'
    np.savez(parent,numbers=[3,27,8]);np.savez(other,numbers=[11,27,8])
    source=tmp_path/'source.npz'
    np.savez(source,parent_sha256=np.asarray(digest(parent)))
    assert 'Archive' in validate_parent_binding(source,parent)
    with pytest.raises(ValueError,match='parent hash'):
        validate_parent_binding(source,other)
    np.savez(source,placeholder=[1])
    with pytest.raises(ValueError,match='need a parent_sha256'):
        validate_parent_binding(source,parent)
    assert 'Explicit producer' in validate_parent_binding(source,parent,digest(parent))


def test_mask_archive_requires_the_correct_parent(tmp_path):
    from cathode_cvae.learning.train_mask import verify_parent_binding
    parent = tmp_path / 'parent.npz'
    other = tmp_path / 'other.npz'
    source = tmp_path / 'labels.npz'
    np.savez(parent, numbers=[3, 27, 8])
    np.savez(other, numbers=[11, 27, 8])
    np.savez(source, placeholder=[1])
    verify_parent_binding(source, parent, embedded=np.asarray(digest(parent)))
    with pytest.raises(ValueError, match='different parent'):
        verify_parent_binding(source, other, embedded=np.asarray(digest(parent)))
    with pytest.raises(ValueError, match='require parent_sha256'):
        verify_parent_binding(source, parent)

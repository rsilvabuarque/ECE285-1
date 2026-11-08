"""Independent finite-model arithmetic and continuous-path preservation checks."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/'overnight_20261108'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_raw_enumeration_never_deduplicates_physical_masks():
    m=script('smallcell_reference');x=m.enumerate_occupancies(12,[3,6,9])
    assert x.shape==(1364,12)
    assert len({row.tobytes() for row in x})==1364
    for n in [3,6,9]:assert int(np.sum(x.sum(1)==n))==math.comb(12,n)
    assert np.array_equal((1-(1-x)).sum(1),x.sum(1))


def test_exact_finite_entropy_and_energy_shift():
    m=script('smallcell_reference');temperature=433.15
    e=np.array([-525.01,-525.01,-524.99,-524.97])
    r=m.raw_thermodynamics(e,temperature);s=m.raw_thermodynamics(e+73.,temperature)
    assert np.isclose(r['probability'].sum(),1.)
    np.testing.assert_allclose(r['probability'],s['probability'],rtol=1e-11)
    assert r['entropy_eV_K']==pytest.approx(s['entropy_eV_K'],rel=1e-11)
    assert s['free_energy_eV']-r['free_energy_eV']==pytest.approx(73.)
    assert r['free_energy_eV']==pytest.approx(r['mean_energy_eV']-temperature*r['entropy_eV_K'])


def test_grouped_entropy_exact_symmetric_null_and_distorted_limit():
    m=script('smallcell_reference');T=433.15
    # Two orbits, multiplicities two and three. Within each orbit energies match.
    for e in [np.array([0.,0.,.1,.1,.1]),np.array([0.,.02,.1,.11,.12])]:
        r=m.raw_thermodynamics(e,T);p=r['probability'];P=np.array([p[:2].sum(),p[2:].sum()]);g=np.array([2,3])
        uniform_conditional=-m.KB_EV_K*np.sum(P*np.log(P/g))
        if e[0]==e[1]:assert uniform_conditional==pytest.approx(r['entropy_eV_K'])
        else:assert uniform_conditional>r['entropy_eV_K']
    equal=m.raw_thermodynamics(np.zeros(12),T)
    assert equal['entropy_eV_K']==pytest.approx(m.KB_EV_K*np.log(12))


def test_invalid_nonfinite_reference_rejected():
    m=script('smallcell_reference')
    with pytest.raises(ValueError):m.raw_thermodynamics([1,np.nan],433.15)


def test_route_resampling_keeps_exact_endpoints_atom_ids_and_unwrapped_branch():
    m=script('neb_stability_run')
    base=np.array([[0.,0.,0.],[2.,0.,0.]])
    xyz=np.array([base,base+np.array([[0.,1.,0.],[0.,0.,0.]]),base+np.array([[0.,2.,0.],[0.,0.,0.]])])
    data=dict(positions_A=xyz,numbers=np.array([3,27]),cell_A=np.eye(3)*8,pbc=np.ones(3,bool))
    images,hashes=m.prepare_band(data,7)
    assert len(images)==9
    np.testing.assert_array_equal(images[0].positions,xyz[0]);np.testing.assert_array_equal(images[-1].positions,xyz[-1])
    assert [m.geometry_hash(images[0]),m.geometry_hash(images[-1])]==hashes
    assert all(np.array_equal(im.numbers,data['numbers']) for im in images)
    assert images[4].positions[0,1]==pytest.approx(1.)


def test_collapsed_route_rejected():
    m=script('neb_stability_run');xyz=np.zeros((3,2,3))
    with pytest.raises(ValueError):m.prepare_band(dict(positions_A=xyz,numbers=np.array([3,27]),cell_A=np.eye(3)*8,pbc=np.ones(3,bool)),7)

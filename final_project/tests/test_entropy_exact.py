"""Small independent analytical checks of the finite-lattice estimators."""
from math import comb,log
import numpy as np
import pytest
from cathode_cvae.entropy import (KB_EV_K,enumerate_orbits,thermodynamics,
                                   importance_benchmark,wang_landau_benchmark)


def test_orbit_multiplicity_equals_explicit_four_site_ring():
    permutations=np.array([(np.arange(4)+i)%4 for i in range(4)])
    c=enumerate_orbits(permutations)
    assert len(c['occupancy'])==6
    assert [r['symmetry_orbits'] for r in c['counts']]==[1,1,2,1,1]
    assert [int(c['multiplicity'][c['n_target']==i].sum()) for i in range(5)]==[comb(4,i) for i in range(5)]
    assert sorted(c['multiplicity'][c['n_target']==2])==[2,4]


def test_partition_entropy_uses_degeneracy_and_total_energy_shift():
    t=300.; e=np.array([0.,KB_EV_K*t*log(2)])
    r=thermodynamics(e,[1,2],t)
    np.testing.assert_allclose(r['orbit_probability'],[.5,.5])
    assert r['entropy_kB']==pytest.approx(1.5*log(2))
    assert r['heat_capacity_kB']==pytest.approx(log(2)**2/4)
    shifted=thermodynamics(e-200.,[1,2],t)
    assert shifted['entropy_kB']==pytest.approx(r['entropy_kB'],abs=1e-10)
    assert shifted['mean_energy_eV']==pytest.approx(r['mean_energy_eV']-200)


def test_equal_energy_recovers_log_binomial_not_log_orbits():
    c=enumerate_orbits(np.array([(np.arange(4)+i)%4 for i in range(4)]))
    g=c['multiplicity'][c['n_target']==2]
    assert thermodynamics([3.,3.],g,300)['entropy_kB']==pytest.approx(log(6))
    assert thermodynamics([3.,3.],[1,1],300)['entropy_kB']==pytest.approx(log(2))


def test_complete_exploration_has_exact_constant_importance_weights():
    # Equal energies and exploration=1 imply uniform probability per microstate.
    r=importance_benchmark([0.,0.],[2,4],300,draws=40,seed=9,exploration=1.)
    assert r['importance_entropy_kB']==pytest.approx(log(6))
    assert r['importance_ESS']==pytest.approx(40)


def test_known_bias_is_correctable_but_naive_frequencies_differ():
    e=np.array([0.,.04,.08]);g=np.array([2,4,2])
    r=importance_benchmark(e,g,10000,draws=50000,seed=7,tilt_temperature_K=100.,exploration=.2)
    assert abs(r['importance_entropy_kB']-r['exact_entropy_kB'])<.02
    assert abs(r['naive_sample_entropy_kB']-r['exact_entropy_kB'])>.3


def test_wang_landau_constant_energy_has_one_bin_and_correct_support():
    c=enumerate_orbits(np.arange(4)[None])
    r=wang_landau_benchmark(np.zeros(16),c['raw_occupancy'],n_target=2,n_bins=8,
                            steps=200,check_every=10,minimum_logf=.01,seed=3)
    assert r['active_bins']==1 and r['density_total_variation']==0 and r['converged']


@pytest.mark.parametrize('temperature',[0,-1,np.nan])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError):thermodynamics([0.],[1],temperature)


def test_no_latent_AR_probability_normalizes_over_fixed_count_microstates():
    import torch
    from cathode_cvae.learning.models import ARBaseline
    torch.manual_seed(28);torch.set_num_threads(1)
    model=ARBaseline(n_sites=4,condition_dim=1,hidden_dim=8)
    raw=enumerate_orbits(np.arange(4)[None])['raw_occupancy']
    for n in (1,2,3):
        x=torch.as_tensor(raw[raw.sum(1)==n],dtype=torch.float32)
        c=torch.full((len(x),1),n/4)
        q=torch.exp(-model.reconstruction(x,c,None,exact_count=True))
        assert float(q.sum().detach())==pytest.approx(1.,abs=2e-6)

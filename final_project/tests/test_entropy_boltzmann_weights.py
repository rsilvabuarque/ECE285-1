"""Check the change of training measure, independent of neural implementation."""
import numpy as np
import pytest
from cathode_cvae.entropy import KB_EV_K,boltzmann_training_weights


def test_within_composition_constant_energy_leaves_multiplicity_measure():
    g=np.array([1.,3.,5.,2.,7.,9.]);n=np.array([2,2,2,3,3,3])
    roles=np.array(['train','train','val','train','train','test'])
    out,_=boltzmann_training_weights([12.,12.,np.nan,-99.,-99.,np.nan],n,roles,g,1000.)
    np.testing.assert_allclose(out,g,rtol=1e-14,atol=1e-14)


def test_known_energy_difference_and_mass_preserving_normalization():
    kT=KB_EV_K*1000.;g=np.array([2.,3.,7.])
    out,groups=boltzmann_training_weights([0.,kT*np.log(2),np.nan],[2,2,2],['train','train','test'],g,1000.)
    assert out[0]/out[1]==pytest.approx(4/3)
    assert out[:2].sum()==pytest.approx(5.)
    assert out[2]==7.
    assert groups[0]['train_energy_reference_eV']==0.


def test_heldout_energies_never_change_weights_and_count_offsets_cancel():
    g=np.array([1.,3.,5.,2.,7.,9.]);n=np.array([2,2,2,3,3,3])
    roles=np.array(['train','train','val','train','train','test']);e=np.array([0.,.1,np.nan,-1.,-.8,np.nan])
    a,_=boltzmann_training_weights(e,n,roles,g,1000.)
    changed=e.copy();changed[n==2]+=50.;changed[n==3]-=100.
    changed[roles!='train']=[1e9,-1e9]
    b,_=boltzmann_training_weights(changed,n,roles,g,1000.)
    np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(a[roles!='train'],g[roles!='train'])


def test_invalid_training_inputs_rejected():
    for e,g,T in [([np.nan],[1.],1000.),([0.],[0.],1000.),([0.],[1.],0.)]:
        with pytest.raises(ValueError):
            boltzmann_training_weights(e,[2],['train'],g,T)

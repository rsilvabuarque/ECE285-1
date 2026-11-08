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


def test_literal_hu_raw_enumeration_orbit_multiplicities_and_stored_representatives():
    m=script('smallcell_reference')
    parent=m.load_npz(ROOT/'artifacts/current/extras/learning_all_parents/parents/Hu_O3_12.npz')
    x=m.enumerate_occupancies(12,[3,6,9]);keys=m.canonical_keys(x,parent['permutations'])
    for n,orbits,raw in [(3,6,220),(6,17,924),(9,6,220)]:
        selected=x.sum(1)==n
        assert len(set(keys[selected]))==orbits
        assert len(keys[selected])==raw
        stored=parent['masks'][(1-parent['masks']).sum(1)==n]
        assert len(stored)==orbits
        assert set(m.canonical_keys(1-stored,parent['permutations']))==set(keys[selected])
        for mask in stored:assert np.any(np.all(x==1-mask,axis=1))


def test_sampled_partition_reduces_to_exact_when_all_raw_states_once():
    m=script('smallcell_analyze');e=np.array([-500.,-499.98,-499.96,-499.9]);T=433.15
    exact=m.reference.raw_thermodynamics(e,T);sample=m.sampled_pool(e,np.arange(4),T)
    for key in ['mean_energy_eV','free_energy_eV','entropy_eV_K']:
        assert sample[key]==pytest.approx(exact[key],rel=1e-9,abs=1e-12)


def test_representative_entropy_exact_for_equal_orbit_energies_and_finite_when_underflow():
    m=script('smallcell_analyze');keys=np.array(['a','a','b','b','b']);rep={'a':0,'b':2}
    for energy in [np.array([0.,0.,.1,.1,.1]),np.array([0.,0.,100.,100.,100.])]:
        r=m.orbit_statistics(energy,keys,rep,433.15)
        assert r['representative_entropy_eV_K']==pytest.approx(r['exact']['entropy_eV_K'])
        assert np.isfinite(r['representative_entropy_eV_K'])
        assert r['orbit_mass_total_variation']==pytest.approx(0,abs=1e-15)


def test_actual_mode_null_velocity_units_energy_and_COM():
    m=script('research_harmonic_null');masses=np.array([6.94,58.933]);translation=np.zeros((6,3))
    for axis in range(3):translation[axis::3,axis]=np.sqrt(masses)
    q=np.linalg.qr(translation,mode='complete')[0][:,3:]
    nu=np.array([1.,2.,3.]);times=np.arange(10000)*.001;T=433.15
    velocity=m.generate_velocities(nu,q,masses,times,np.array([.3,.7,1.1]),T)
    com=(velocity*masses[None,:,None]).sum(1)/masses.sum()
    assert np.max(np.linalg.norm(com,axis=1))<1e-12
    kinetic=.5*m.xpt.KINETIC_EV_PER_AMU_A2_PS2*np.sum(velocity**2*masses[None,:,None],axis=(1,2))
    assert kinetic.mean()==pytest.approx(1.5*m.xpt.KB*T,rel=1e-12)


def test_harmonic_native_anchor_reference_reduces_to_exact_partition():
    m=script('research_harmonic_null');nu=np.array([1.,2.,3.]);T=433.15;emin=-100.
    anchor=emin+len(nu)*m.xpt.KB*T;r=m.matched_references(nu,T,emin,anchor)
    assert r['classical_anchor_matched_F_eV']==pytest.approx(r['exact']['F_classical_eV'])
    assert r['quantum_anchor_matched_F_eV']==pytest.approx(r['exact']['F_quantum_eV'])


def test_profile_and_summary_modules_load_without_model_imports():
    assert callable(script('neb_stability_profile').main)
    assert callable(script('neb_stability_summarize').main)


def test_retain_coordinate_initialization_is_exact_and_rejects_image_count_change():
    m=script('neb_stability_run');xyz=np.zeros((9,2,3));xyz[:,0,1]=np.linspace(0,1,9)**2
    data=dict(positions_A=xyz,numbers=np.array([3,27]),cell_A=np.eye(3)*8,pbc=np.ones(3,bool))
    images,_=m.prepare_band(data,7,'retain_coordinates')
    np.testing.assert_array_equal(np.asarray([im.positions for im in images]),xyz)
    with pytest.raises(ValueError):m.prepare_band(data,5,'retain_coordinates')


def test_interruption_reserves_inflight_position_update_and_original_budget(tmp_path):
    m=script('neb_stability_recover');run=tmp_path/'interrupted';run.mkdir()
    protocol=dict(maximum_total_optimizer_steps=500,maximum_EFS_calls=5000,maximum_seconds_including_imports_loading=2700)
    (run/'protocol.json').write_text(__import__('json').dumps(protocol))
    progress=dict(step=430,energy_eV=[0.,1.,0.],actual_EFS_attempts=3019,elapsed_seconds=557.93)
    (run/'progress.jsonl').write_text(__import__('json').dumps(progress)+'\n')
    calls=[dict(call_index=i,status='success',elapsed_seconds=.1) for i in range(1,3021)]
    (run/'calculator_calls.jsonl').write_text(''.join(__import__('json').dumps(r)+'\n' for r in calls))
    np.savez(run/'latest_band.npz',energy_eV=np.array([0.,1.,0.]),labels_match_geometry=np.ones(3,bool))
    facts=m.interruption_facts(run,'Synthetic external failure')
    assert facts['remaining_steps']==69
    assert facts['remaining_EFS_calls']==1980
    assert facts['conservative_charged_optimizer_steps']==431
    assert facts['previous_elapsed_seconds_reserved']==560
    assert facts['remaining_seconds']==2140


def test_projected_saddle_hessian_removes_only_translations_and_retains_negative_mode():
    m=script('research_saddle_modes');masses=np.array([1.,2.]);translation=np.zeros((6,3))
    for a in range(3):translation[a::3,a]=np.sqrt(masses)
    q=np.linalg.qr(translation,mode='complete')[0][:,3:];expected=np.array([-.5,1.,2.])
    dynamical=q@np.diag(expected)@q.T;rootmass=np.repeat(np.sqrt(masses),3)
    hessian=dynamical*rootmass[:,None]*rootmass[None,:]
    result=m.internal_modes(hessian,masses)
    np.testing.assert_allclose(result['internal_eigenvalues'],expected,atol=1e-12)
    assert len(result['internal_eigenvalues'])==3
    assert np.sum(result['frequencies_signed_THz']<0)==1
    assert result['eigenpair_projected_residuals'].max()<1e-12
    np.testing.assert_allclose(result['translation_basis'].T@result['eigenvectors_mass_weighted'],0,atol=1e-12)


def test_saddle_repeat_noise_units_and_identical_call_count():
    m=script('research_saddle_modes');force=np.zeros((3,2,3));force[1,0,0]=2e-6;force[2,0,0]=-2e-6
    noise=m.repeat_statistics(force,[-100.,-100.+1e-6,-100.-1e-6],.01)
    assert noise['actual_identical_geometry_EFS_calls']==3
    assert noise['diagnostic_component_curvature_noise_scale_eV_A2']==pytest.approx(2e-4)
    assert noise['max_abs_component_pair_difference_eV_A']==pytest.approx(4e-6)


def test_endpoint_taper_keeps_route_nodes_and_exact_new_endpoints():
    m=script('research_endpoint_tightening');xyz=np.zeros((3,2,3));xyz[:,0,0]=[0.,.5,2.]
    data=dict(positions_A=xyz,numbers=np.array([3,27]),cell_A=np.eye(3)*8,pbc=np.ones(3,bool))
    first=m.atoms_from(data,xyz[0]);last=m.atoms_from(data,xyz[-1]);first.positions[:,1]+=.1;last.positions[:,1]+=.3
    images,s=m.tapered_band(data,first,last)
    np.testing.assert_array_equal(images[0].positions,first.positions);np.testing.assert_array_equal(images[-1].positions,last.positions)
    assert s[1]==pytest.approx(.25)
    np.testing.assert_allclose(images[1].positions,xyz[1]+np.array([0.,.15,0.]))
    assert len(images)==len(xyz)
    assert all(np.array_equal(im.numbers,data['numbers']) for im in images)


def test_zero_endpoint_displacements_leave_every_old_route_image_exact():
    m=script('research_endpoint_tightening');xyz=np.zeros((3,2,3));xyz[:,0,0]=[0.,.5,2.]
    data=dict(positions_A=xyz,numbers=np.array([3,27]),cell_A=np.eye(3)*8,pbc=np.ones(3,bool))
    images,_=m.tapered_band(data,m.atoms_from(data,xyz[0]),m.atoms_from(data,xyz[-1]))
    np.testing.assert_array_equal(np.asarray([im.positions for im in images]),xyz)

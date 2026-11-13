"""Small-cell learning roles are frozen without labels and TEST stays sealed."""
import importlib.util
from pathlib import Path
import numpy as np
from cathode_cvae.entropy import enumerate_orbits
from cathode_cvae.data import sha256


def script():
    p=Path(__file__).parents[1]/'scripts/entropy_20261112/prepare_ar.py'
    s=importlib.util.spec_from_file_location('prepare_entropy_ar',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_roles_orbit_disjoint_and_test_energies_do_not_enter_export(tmp_path):
    c=enumerate_orbits(np.arange(4)[None]);ids=np.array([f'id{i}' for i in range(len(c['occupancy']))])
    parent=tmp_path/'parent.npz';np.savez(parent,numbers=np.array([3]*4+[8]),target_indices=np.arange(4))
    catalog=tmp_path/'catalog.npz';np.savez(catalog,structure_ids=ids,occupancy=c['occupancy'],n_target=c['n_target'],multiplicity=c['multiplicity'])
    m=script();roles=tmp_path/'roles';m.freeze(catalog,roles,9)
    with np.load(roles/'orbit_roles.npz') as z:split=z['split']
    assert set(split)=={'train','val','test'}
    outputs=[]
    for label,test_value in [('a',1000.),('b',np.nan)]:
        energy=np.arange(len(ids),dtype=float);energy[split=='test']=test_value
        scores=tmp_path/(label+'.npz');np.savez(scores,structure_ids=ids,energy_eV=energy,status=np.full(len(ids),'success'))
        out=tmp_path/label;m.export(catalog,roles,scores,parent,out)
        with np.load(out/'learning.npz') as z:outputs.append({k:z[k] for k in z.files})
    for key in outputs[0]:
        if key=='energy_eV_atom':
            assert np.isnan(outputs[0][key][split=='test']).all()
            np.testing.assert_array_equal(outputs[0][key],outputs[1][key])
        else:np.testing.assert_array_equal(outputs[0][key],outputs[1][key])
    assert sha256(roles/'orbit_roles.npz')

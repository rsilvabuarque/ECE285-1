"""Paired-response estimators retain latent sensitivity despite signed cancellation."""
import importlib.util
from pathlib import Path
import numpy as np

p=Path(__file__).resolve().parents[1]/'scripts/overnight_20261108/summarize_paired_sensitivity.py'
s=importlib.util.spec_from_file_location('paired_sensitivity_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def row(n,delta):
    return dict(teacher='synthetic',method='cvae',replicate_id='1',n_Li=n,contrast='alternate_latent_minus_low',
        both_energies_known='True',energy_shift_meV_atom=delta,raw_Hamming='2',same_orbit='False')

def test_zero_signed_mean_is_not_zero_absolute_or_RMS_response():
    composition,fits,summary=m.summarize([row(12,-1),row(12,1),row(24,-2),row(24,2)])
    r=fits[0]
    assert r['macro_mean_signed_delta_E_meV_atom']==0
    assert r['macro_mean_absolute_delta_E_meV_atom']==1.5
    assert r['RMS_from_macro_mean_squared_delta_E_meV_atom']==np.sqrt(2.5)
    assert r['macro_fraction_positive_delta']==.5
    assert len(composition)==2 and len(fits)==len(summary)==1
    assert summary[0]['fit_or_sampler_replicates']==1

def test_unknown_pairs_do_not_become_complete_sensitivity_estimates():
    r=row(12,1);r['both_energies_known']='False';r['energy_shift_meV_atom']=''
    _,fits,summary=m.summarize([r])
    assert not fits[0]['complete_pairs'] and fits[0]['macro_mean_absolute_delta_E_meV_atom']==''
    assert not summary[0]['all_pairs_complete']

"""Exact-key orbit-cache regression; all masks and groups are synthetic."""
import numpy as np
import pytest
import cathode_cvae.iterative_assessment as assessment
from cathode_cvae.iterative_assessment import orbit_set,_orbit_set_cached


def test_hit_misses_and_returned_set_are_isolated(monkeypatch):
    _orbit_set_cached.cache_clear()
    calls=[];original=assessment._canonical_vacancy
    def tracked(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(assessment,'_canonical_vacancy',tracked)
    identity=np.arange(48,dtype=np.int64)[None,:]
    keys={'1'+'0'*47,'01'+'0'*46}
    first=orbit_set(keys,identity)
    assert len(calls)==1
    result=orbit_set(set(reversed(sorted(keys))),identity.copy())
    assert len(calls)==1 and result==first and result is not first
    result.add('tampered')
    assert 'tampered' not in orbit_set(keys,identity) and len(calls)==1
    changed=set(keys);changed.add('001'+'0'*45)
    orbit_set(changed,identity);assert len(calls)==2
    swap=identity.copy();swap[0,0],swap[0,1]=swap[0,1],swap[0,0]
    group=np.concatenate([identity,swap])
    orbit_set(keys,group);assert len(calls)==3
    orbit_set(keys,identity.astype(np.int32));assert len(calls)==4
    # The returned result must match a direct uncached canonicalization.
    a=np.asarray([[int(c) for c in key] for key in sorted(keys)],dtype=np.uint8)
    assert orbit_set(keys,group)==set(original(1-a,group).tolist())


def test_invalid_mutated_key_not_hidden_by_previous_cache():
    _orbit_set_cached.cache_clear();group=np.arange(48,dtype=np.int64)[None,:]
    orbit_set({'0'*48},group)
    with pytest.raises(ValueError):orbit_set({'0'*47+'x'},group)

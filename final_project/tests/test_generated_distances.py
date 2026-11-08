"""Independent direct-bit checks and a label-forbidden portable distance audit."""
import csv,importlib.util,itertools,json,time
from pathlib import Path
import numpy as np
import pytest

p=Path(__file__).resolve().parents[1]/'scripts/overnight_20261108/audit_generated_distances.py'
s=importlib.util.spec_from_file_location('generated_distances_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_packed_orbit_distances_agree_with_direct_bit_geometry():
    x=np.asarray(list(itertools.product([0,1],repeat=4)),np.uint8)
    inverse=np.asarray([np.roll(np.arange(4),j) for j in range(4)])
    packed=m.popcount(m.pack(x)[:,None,None]^m.orbit_codes(x,inverse)[None]).min(2)
    direct=np.asarray([[min(np.count_nonzero(a!=b[p]) for p in inverse) for b in x] for a in x])
    assert np.array_equal(packed,direct)
    ref=x[[1,3,5]];fast,_=m.bounded_nearest(x,ref,inverse,deadline=time.monotonic()+30)
    assert np.array_equal(fast,direct[:,[1,3,5]].min(1))
    assert m.popcount(m.pack(np.ones((1,62),np.uint8)))[0]==62

def test_canonical_hamming_can_exceed_true_orbit_minimum():
    x=np.asarray(list(itertools.product([0,1],repeat=6)),np.uint8)
    inverse=np.asarray([np.roll(np.arange(6),j) for j in range(6)])
    keys=m.code_keys(m.canonical_codes(x,inverse),6)
    representatives=1-np.asarray([[int(c) for c in key] for key in keys],np.uint8)
    ordinary=np.count_nonzero(representatives[:,None]!=representatives[None],axis=2)
    exact=np.asarray([[min(np.count_nonzero(a!=b[p]) for p in inverse) for b in representatives] for a in representatives])
    assert np.all(exact<=ordinary) and np.any(exact<ordinary)

def test_popcount_fallback_and_time_bound(monkeypatch):
    x=np.array([0,1,3,2**63,2**64-1],np.uint64)
    if hasattr(np,'bitwise_count'):monkeypatch.delattr(np,'bitwise_count')
    assert m.popcount(x).tolist()==[0,1,2,1,64]
    with pytest.raises(TimeoutError):m.bounded_nearest(np.array([[1,0]],np.uint8),np.array([[0,1]],np.uint8),np.arange(2)[None],deadline=time.monotonic()-1)

def fixture(tmp_path):
    parent=tmp_path/'parent.npz';perms=np.asarray([np.roll(np.arange(6),j) for j in range(6)]);np.savez(parent,permutations=perms)
    allx=np.asarray([v for v in itertools.product([0,1],repeat=6) if sum(v)==3],np.uint8)
    keys=m.code_keys(m.canonical_codes(allx,np.argsort(perms,axis=1)),6)
    unique={}
    for x,key in zip(allx,keys):unique.setdefault(key,x)
    values=list(unique.values());seed=np.array([1,1,1,0,0,0],np.uint8)
    seedkey=m.code_keys(m.canonical_codes(seed[None],np.argsort(perms,axis=1)),6)[0]
    values=[seed]+[x for key,x in unique.items() if key!=seedkey]
    data=tmp_path/'data.npz';np.savez(data,occupancy=np.array(values),split=np.array(['train','train','val','test']),
        structure_ids=np.array(['train0','train1','val0','test0']),n_target=np.full(4,3),
        energy_eV_atom=np.array([object()],dtype=object)) # Opening this outcome member with allow_pickle=False would fail.
    child=seed.copy();child[2]=0;child[5]=1
    x=np.array([values[1],child,child,np.full(6,np.nan)],np.float32);seeds=np.full_like(x,np.nan);seeds[1]=seed
    validkeys=m.code_keys(m.canonical_codes(x[:3],np.argsort(perms,axis=1)),6)
    rows=[]
    for i in range(4):
        rows.append(dict(attempt_id=f'a{i}',method=['uniform','mutation','cvae','cvae'][i],replicate_id='1',fit_seed='',n_target=3,requested_bin='low',
            diagnostic='primary',raw_row=i,status='generated' if i<3 else 'failed_request',canonical_mask=validkeys[i] if i<3 else '',
            seed_source_row='0' if i==1 else '',seed_source_sha256=m.sha(data) if i==1 else '',seed_structure_id='train0' if i==1 else '',
            swap_count=1 if i==1 else '',removed_sites='2' if i==1 else '',added_sites='5' if i==1 else ''))
    gen=tmp_path/'generated';gen.mkdir();m.write(gen/'attempts.csv',rows);np.savez(gen/'raw_attempts.npz',occupancy=x,seed_occupancy=seeds)
    protocol=dict(inputs={'parent':{'sha256':m.sha(parent)},'data':{'sha256':m.sha(data)}});m.save(gen/'protocol.json',protocol)
    m.save(gen/'receipt.json',dict(status='FROZEN_BEFORE_SCORING',protocol_sha256=m.sha(gen/'protocol.json'),primary_attempts=4,total_attempts=4,
        outputs={name:m.sha(gen/name) for name in ['attempts.csv','raw_attempts.npz']}))
    return gen,parent,data

def test_portable_main_preserves_invalid_repeats_and_raw_vs_orbit_edits_without_labels(tmp_path):
    gen,parent,data=fixture(tmp_path);out=tmp_path/'audit'
    result=m.main(['--generated',str(gen),'--parent',str(parent),'--training-data',str(data),'--output-dir',str(out),'--max-seconds','30'])
    assert result['status']=='PASS' and result['energy_fields_read']==0
    assert result['attempts']==4 and result['valid_attempts']==3 and result['invalid_or_failed_attempts']==1
    assert result['mutation_TRAIN_source_checks']==1 and result['reference_orbits_per_count']=={3:2}
    rows=m.read(out/'attempt_distances.csv');assert len(rows)==4
    mutation=next(r for r in rows if r['method']=='mutation')
    assert mutation['raw_seed_child_Hamming']=='2' and mutation['symmetry_minimum_seed_child_Hamming']=='0'
    assert rows[3]['nearest_TRAIN_reference_orbit_Hamming']==''
    assert m.sha(gen/'attempts.csv')==json.loads((gen/'receipt.json').read_text())['outputs']['attempts.csv']

def test_changed_bound_input_is_rejected_with_preserved_failure(tmp_path):
    gen,parent,data=fixture(tmp_path);out=tmp_path/'failed';np.savez(parent,permutations=np.arange(6)[None])
    with pytest.raises(ValueError):m.main(['--generated',str(gen),'--parent',str(parent),'--training-data',str(data),'--output-dir',str(out)])
    assert json.loads((out/'failure_receipt.json').read_text())['status']=='FAILED_PRESERVED'

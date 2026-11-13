"""Independent checks of mapping, canonicalization and coordinate preservation."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import csv
import json
from pathlib import Path
import numpy as np
from ase.geometry import find_mic
import map_frames as m

HERE=Path(__file__).resolve().parent

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use compute allocation')
    m.initialize();d=m.DATA;p=m.PARENTS
    with (HERE/'frame_mapping.csv').open() as f:rows=list(csv.DictReader(f))
    with np.load(HERE/'site_displacements.npz') as z:encoded={k:z[k] for k in z.files}
    maximum=0.
    for i,r in enumerate(rows):
        sl=slice(d['offsets'][i],d['offsets'][i+1]);h=d['cell'][i]
        pp=p[r['parent_id']];q=encoded['orientation'][i];t=encoded['origin_shift'][i]
        f=np.asarray(pp['fractional_sites'])@q.T+t
        reconstructed=f[encoded['atom_parent_slot'][sl]]@h+encoded['displacement_A'][sl]
        frac_error=(reconstructed-d['positions'][sl])@np.linalg.inv(h)
        maximum=max(maximum,float(np.linalg.norm((frac_error-np.rint(frac_error))@h,axis=1).max()))
        assert r['geometry_key']
        assert r['vacancy_mask'].count('0')==int(r['n_Li'])
        assert r['canonical_vacancy_mask'].count('1')==len(r['vacancy_mask'])-int(r['n_Li'])
        assert int(r['mask_orbit_size'])*int(r['mask_stabilizer_size'])==len(pp['li_permutations'])
    assert maximum<1e-10
    # Compare MIC norms with ASE's Minkowski-reduction-based implementation,
    # including cell-shape extremes and high-displacement mapping cases.
    rng=np.random.default_rng(20261104)
    selected=set(rng.choice(len(rows),128,replace=False).tolist())
    selected.update(np.argsort(np.linalg.cond(d['cell']))[-12:].tolist())
    selected.update(sorted(range(len(rows)),key=lambda i:float(rows[i]['host_half_spacing_ratio_max']))[-12:])
    for phase in ('O3','O1','spinel','migration'):
        ix=[i for i,r in enumerate(rows) if r['phase']==phase]
        selected.update(ix[:2]+ix[-2:])
    mic_error=0.;n_vectors=0
    for i in sorted(selected):
        h=d['cell'][i];delta=rng.uniform(-2,2,(96,3))
        ours=m.periodic_vectors(delta,h)
        _,ase_norm=find_mic(delta@h,h,pbc=True)
        mic_error=max(mic_error,float(np.max(np.abs(np.linalg.norm(ours,axis=1)-ase_norm))))
        n_vectors+=len(delta)
    assert mic_error<1e-10
    # Independent integer encoding: bit i in strings has weight 2^(M-1-i).
    seen=set();canonical_checks=0
    for r in rows:
        key=(r['parent_id'],r['vacancy_mask'])
        if key in seen:continue
        seen.add(key);mask=r['vacancy_mask'];n=len(mask)
        empty=[i for i,b in enumerate(mask) if b=='1']
        values={sum(1<<(n-1-perm[i]) for i in empty) for perm in p[r['parent_id']]['li_permutations']}
        assert format(min(values),f'0{n}b')==r['canonical_vacancy_mask']
        assert len(values)==int(r['mask_orbit_size'])
        canonical_checks+=1
    # Translation + rigid rotation + atom reordering must preserve the orbit.
    transformations=[]
    for parent in p:
        candidates=[r for r in rows if r['parent_id']==parent and r['joint_template_eligible']=='True'
                    and 0<int(r['n_Li'])<int(r['n_parent_Li_sites'])]
        r=candidates[len(candidates)//2];i=int(r['index']);sl=slice(d['offsets'][i],d['offsets'][i+1])
        oldn=d['numbers'][sl].copy();oldpos=d['positions'][sl].copy();oldh=d['cell'][i].copy()
        rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
        if np.linalg.det(rotation)<0:rotation[:,0]*=-1
        permutation=rng.permutation(len(oldn));newh=oldh@rotation
        d['numbers'][sl]=oldn[permutation]
        d['positions'][sl]=(oldpos@rotation+rng.uniform(-.4,.4,3)@newh)[permutation]
        d['cell'][i]=newh
        result=m.map_source((parent,[r]))[0][1]
        assert result['canonical_vacancy_mask']==r['canonical_vacancy_mask'],(parent,r['index'])
        assert result['joint_template_eligible']
        transformations.append(dict(parent_id=parent,index=i,orbit_unchanged=True,localized=True))
        d['numbers'][sl]=oldn;d['positions'][sl]=oldpos;d['cell'][i]=oldh
    m.dump(HERE/'independent_validation.json',dict(status='PASS',n_coordinate_roundtrips=len(rows),
           maximum_roundtrip_error_A=maximum,n_MIC_vectors=n_vectors,max_MIC_norm_error_A=mic_error,
           n_independent_mask_checks=canonical_checks,metamorphic_checks=transformations,
           numerical_check_tolerance_A=1e-10,
           tolerance_purpose='Floating-point implementation consistency only; not geometry-equivalence or physical acceptance tolerance.'))
    print('INDEPENDENT_VALIDATION_PASS',maximum,mic_error,canonical_checks,flush=True)

if __name__=='__main__':main()

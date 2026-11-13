"""Construct a composition-balanced, symmetry-unique fixed-parent O3 timing set."""
import csv, hashlib, json, os
from pathlib import Path
import numpy as np
import spglib
from ase import Atoms
from ase.io import read, write
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
SOURCE = Path('/global/cfs/cdirs/m5056/rsb/2026/cathode_relithiation/cathode_relithiation_project/results/publication/default_systems/deepmd_collaborator_electrode/full_study_4x6x2/structure_selection/pristine_4x6x2.cif')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d): Path(p).write_text(json.dumps(d,indent=2)+'\n')

def main():
    assert os.environ.get('SLURMD_NODENAME','').startswith('nid'), 'Use allocation'
    assert not (HERE/'manifest.csv').exists(), 'Preserve existing dataset'
    (HERE/'inputs').mkdir(exist_ok=True)
    old=read(SOURCE); repeat=np.array([4,6,2]); cell=old.cell.array/repeat[:,None]
    frac=np.mod(old.get_scaled_positions()*repeat,1.)
    frac[np.isclose(frac,1.,atol=1e-10,rtol=0)]=0.
    seen=set(); nums=[]; fs=[]
    for z,f in zip(old.numbers,frac):
        key=(int(z),*np.round(f,10))
        if key not in seen: seen.add(key);nums.append(z);fs.append(f)
    base=Atoms(numbers=nums,scaled_positions=fs,cell=cell,pbc=True)
    assert base.get_chemical_formula()=='Co3Li3O6' and len(base)==12
    # Save/reload before assigning site IDs, so POSCAR sorting is explicit.
    write(HERE/'parent_POSCAR',base.repeat((2,4,2)),format='vasp',sort=True,direct=True)
    parent=read(HERE/'parent_POSCAR',format='vasp')
    li=np.flatnonzero(parent.numbers==3); assert len(li)==48 and len(parent)==192
    f=parent.get_scaled_positions(); h=parent.cell.array
    sy=spglib.get_symmetry((h,f,parent.numbers),symprec=1e-5)
    perms=[]; maxres=0.
    for rot,t in zip(sy['rotations'],sy['translations']):
        transformed=(f@rot.T+t)%1.; whole=np.zeros(len(parent),int)
        for z in np.unique(parent.numbers):
            ids=np.flatnonzero(parent.numbers==z)
            d=transformed[ids,None,:]-f[None,ids,:]; d-=np.rint(d)
            dist=np.linalg.norm(d@h,axis=-1); a,b=linear_sum_assignment(dist)
            maxres=max(maxres,float(dist[a,b].max())); whole[ids[a]]=ids[b]
        assert maxres<1e-8
        lookup={a:i for i,a in enumerate(li)}
        perms.append([lookup[whole[a]] for a in li])
    perms=np.unique(np.array(perms),axis=0); identity=np.arange(48)
    keys={tuple(p) for p in perms}; assert tuple(identity) in keys
    assert all(tuple(p[q]) in keys for p in perms for q in perms)
    inv=np.argsort(perms,axis=1)
    def canon(v):
        strings=[''.join(map(str,a)) for a in v[inv]]
        j=min(range(len(strings)),key=strings.__getitem__)
        return strings[j],j
    # Independent endpoint/one-defect checks catch incorrect site permutations.
    assert len({canon(np.eye(48,dtype=int)[i])[0] for i in range(48)})==1
    rng=np.random.default_rng(20261105); manifest=[]; unique=set()
    extra={6,12,18,30,36,42}
    for n in range(49):
        target=1 if n in [0,1,47,48] else 2+int(n in extra)
        count=0; tries=0
        while count<target:
            tries+=1; assert tries<10000
            v=np.ones(48,dtype=int); v[rng.choice(48,n,replace=False)]=0
            c,w=canon(v)
            if c in unique: continue
            unique.add(c); idx=len(manifest); sid=f'structure_{idx:03d}'
            keep=np.ones(len(parent),bool);keep[li[v==1]]=False
            atoms=parent[keep];atoms.arrays['parent_slot']=np.flatnonzero(keep)
            atoms.info.update(structure_id=sid,n_Li=n,x=n/48)
            path=HERE/'inputs'/f'{sid}.extxyz'; write(path,atoms)
            restored=read(path)
            assert np.array_equal(restored.numbers,parent.numbers[keep])
            assert np.max(abs(restored.positions-parent.positions[keep]))<1e-7
            write(HERE/'inputs'/f'{sid}.vasp',atoms,format='vasp',sort=True,direct=True)
            manifest.append(dict(structure_id=sid,n_Li=n,x=n/48,n_atoms=len(atoms),
                vacancy_mask=''.join(map(str,v)),canonical_mask=c,witness=w,
                input_file=str(path),input_sha256=sha(path)))
            count+=1
    assert len(manifest)==len(unique)==100
    assert sum(a['n_Li'] for a in manifest)==2400
    with (HERE/'manifest.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
    dump(HERE/'parent_symmetry.json',dict(n_Li_sites=48,spacegroup=spglib.get_spacegroup((h,f,parent.numbers),symprec=1e-5),
        parent_sha256=sha(HERE/'parent_POSCAR'),source=str(SOURCE),source_sha256=sha(SOURCE),
        conventional_repeat=[2,4,2],Li_atom_indices=li.tolist(),permutations=perms.tolist(),
        rotations=sy['rotations'].tolist(),translations=sy['translations'].tolist(),max_mapping_residual_A=maxres,
        symprec_A=1e-5,tolerance_role='Identify ideal parent symmetry; not an endpoint equivalence threshold',
        seed=20261105,mask_convention='1=vacancy',sampling='Uniform raw fixed-count masks with symmetry rejection; not uniform over orbits',
        closure_identity_checks='PASS',one_vacancy_orbit_check='PASS',unique_masks=100))
    print('PREPARED',len(manifest),'structures',len(perms),'Li permutations',flush=True)

if __name__=='__main__':main()

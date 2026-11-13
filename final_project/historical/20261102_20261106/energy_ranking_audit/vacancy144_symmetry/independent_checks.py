"""Independent witness/composition checks for all catalog entries."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import csv
import json
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from ase.io import read
from scipy.spatial import cKDTree
import analyze_catalog as a

def header_check(row):
    path=a.SOURCE/'uma_omat_top500_candidates'/row['candidate_relpath']/'POSCAR'
    with path.open() as f:lines=[next(f) for _ in range(7)]
    symbols=lines[5].split();counts=list(map(int,lines[6].split()))
    composition=dict(zip(symbols,counts))
    assert composition=={'Co':144,'Li':int(row['n_li_keep']),'O':288},path
    assert sum(counts)==int(row['atom_count'])==432+int(row['n_li_keep']),path
    return True

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use compute allocation')
    completion=json.loads((a.HERE/'completion.json').read_text())
    assert completion['status']=='COMPLETE'
    p=json.loads((a.HERE/'parent_symmetry.json').read_text())
    perms=np.asarray(p['Li_site_permutations_source_to_destination'])
    with (a.HERE/'candidate_orbit_mapping.csv').open() as f:rows=list(csv.DictReader(f))
    with (a.HERE/'unique_representatives.csv').open() as f:representatives=list(csv.DictReader(f))
    groups=defaultdict(list)
    for row in rows:
        empty=json.loads(row['remove_local_li_indices']);perm=perms[int(row['canonicalizing_operation_index'])]
        transformed=sum(1<<(143-int(perm[i])) for i in empty)
        assert format(transformed,'036x')==row['canonical_vacancy_mask_hex']
        assert int(row['orbit_size'])*int(row['stabilizer_size'])==len(perms)
        groups[row['mask_orbit_id']].append(row)
    assert len(groups)==len(representatives)==completion['unique_symmetry_orbits']
    assert len({r['mask_orbit_id'] for r in representatives})==len(representatives)
    for key,rr in groups.items():
        assert len(rr)<=int(rr[0]['orbit_size'])
        assert len(rr)==int(rr[0]['available_orbit_members'])
        rep=min(r['candidate_relpath'] for r in rr)
        assert all(r['representative_candidate_relpath']==rep for r in rr)
        assert sum(r['is_representative']=='True' for r in rr)==1
    # A complete independent composition check closes the custom-parser check:
    # a subset of correct host coordinates must not pass as a complete crystal.
    with ThreadPoolExecutor(max_workers=8) as pool:
        n_headers=sum(pool.map(header_check,rows))
    parent=read(a.PARENT,format='vasp');fp=parent.get_scaled_positions()%1
    trees={z:cKDTree(fp[parent.numbers==z],boxsize=1.) for z in set(parent.numbers)}
    rng=np.random.default_rng(20261105)
    selected=set(rng.choice(len(rows),16,replace=False).tolist())
    selected.update(i for i,r in enumerate(rows) if int(r['n_li_keep'])>=143 and int(r['candidate_id'])==0)
    max_error=0.
    for i in sorted(selected):
        row=rows[i];atoms=read(a.SOURCE/'uma_omat_top500_candidates'/row['candidate_relpath']/'POSCAR',format='vasp')
        assert Counter(atoms.get_chemical_symbols())=={'Co':144,'Li':int(row['n_li_keep']),'O':288}
        assert np.array_equal(atoms.cell.array,parent.cell.array)
        f=atoms.get_scaled_positions()%1
        for z in set(parent.numbers):
            _,assigned=trees[z].query(f[atoms.numbers==z]);assert len(set(assigned))==sum(atoms.numbers==z)
            delta=f[atoms.numbers==z]-fp[parent.numbers==z][assigned];delta-=np.rint(delta)
            max_error=max(max_error,float(np.linalg.norm(delta@parent.cell.array,axis=1).max()))
            if z==3:assert set(assigned)==set(range(144))-set(json.loads(row['remove_local_li_indices']))
    assert max_error<=256*np.finfo(float).eps*np.linalg.norm(parent.cell.array,axis=1).sum()
    a.dump(a.HERE/'independent_validation.json',dict(status='PASS',n_all_witness_checks=len(rows),
           n_all_composition_checks=n_headers,n_representative_groups_checked=len(groups),
           n_ASE_independent_POSCAR_reads=len(selected),maximum_ASE_geometry_error_A=max_error,
           checks=['witness maps raw mask to canonical mask','orbit-stabilizer identity',
                   'one actual deterministic representative per orbit','orbit membership count',
                   'all saved atom counts/compositions','ASE parser/geometry cross-check']))
    print('INDEPENDENT_CHECKS_PASS',len(rows),len(groups),flush=True)

if __name__=='__main__':main()

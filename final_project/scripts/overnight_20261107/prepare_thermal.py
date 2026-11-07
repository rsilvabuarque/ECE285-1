"""Select three low-deletion-energy masks for a same-UMA thermal method pilot.

The six-site orbit-Hamming separation is a declared diversity design heuristic
(three Li/vacancy exchanges), not a convergence or thermodynamic threshold.
Selection uses existing training rows, never a claim of a new CVAE discovery.
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write
from cathode_cvae.runtime import require_compute_allocation
from cathode_cvae.data import validate_permutation_group, deletion_atoms

p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
require_compute_allocation()
a.output.mkdir(parents=True,exist_ok=False)
parent_file=a.snapshot/'core/data/parents/O3_N48_pilot.npz'
data_file=a.snapshot/'improvement_v1/training_round0.npz'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
with np.load(parent_file) as z: parent={k:z[k] for k in z.files}
with np.load(data_file) as z: data={k:z[k] for k in z.files}
validate_permutation_group(parent['permutations'])
eligible=np.flatnonzero((data['n_Li']==36)&(data['split']=='train'))
order=eligible[np.lexsort((data['canonical_masks'][eligible],data['energy_eV_atom'][eligible]))]
pool=order[:1000]
occ=data['occupancy'][pool].astype(np.uint8)
def distances(index):
    target=data['occupancy'][index].astype(np.uint8)
    return np.count_nonzero(occ[:,None,:]!=target[parent['permutations']][None,:,:],axis=2).min(axis=1)
selected=[];nearest=np.full(len(pool),49)
for _ in range(3):
    allowed=np.flatnonzero(nearest>=6)
    if not len(allowed):raise RuntimeError('Prespecified pool has no sufficiently separated next arrangement')
    chosen=int(pool[allowed[0]]);selected.append(chosen)
    nearest=np.minimum(nearest,distances(chosen))
rows=[]
for j,index in enumerate(selected):
    mask=str(data['canonical_masks'][index]);assert len(mask)==48 and mask.count('0')==36
    occupancy=np.fromiter((c=='0' for c in mask),np.uint8)
    atoms,_=deletion_atoms(parent,occupancy)
    name=f'basin_{j+1}';target=a.output/f'{name}_deletion.vasp';write(target,atoms,format='vasp',direct=True,sort=False)
    rows.append(dict(basin_id=name,source_structure_id=str(data['structure_ids'][index]),source_row=int(index),vacancy_mask=mask,n_Li=36,n_Co=48,n_atoms=len(atoms),source_energy_eV_atom=float(data['energy_eV_atom'][index]),source_total_energy_eV=float(data['energy_eV_atom'][index])*len(atoms),deletion_POSCAR=target.name,POSCAR_sha256=sha(target)))
receipt=dict(status='PREPARED_NOT_RELAXED',parent_sha256=sha(parent_file),data_sha256=sha(data_file),selection='Lowest existing training UMA deletion energy sequentially, requiring >=6 site orbit-Hamming distance from every previous selection, within the first1000 lowest-energy n36 training rows',pool_size=len(pool),temperature_K=433.15,common_cell_A=parent['cell_A'].tolist(),geometry_protocol='pristine-derived canonical deletion; fixed parent cell; new tight relaxation required',rows=rows)
(a.output/'selection.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(rows,indent=2))

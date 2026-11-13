"""VESTA examples selected after final scoring; no new optimization or performance test."""
import csv,json,os
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import read,write
from ase.geometry import find_mic
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
rows=[r for r in csv.DictReader((HERE/'reports/final/quality/fresh_best_candidates.csv').open()) if r['teacher']=='uma' and r['method']=='ar_adapt_cvae']
with np.load(HERE.parent/'data/parents/O3_N48_pilot.npz') as z:p={k:z[k] for k in z.files}
out=HERE/'vesta_new_examples';out.mkdir(exist_ok=False);selected=[]
for n in [12,24,36]:
 r=min([r for r in rows if int(r['n_Li'])==n],key=lambda r:float(r['energy_eV_atom']))
 v=np.array(list(r['canonical_mask']),int);keep=np.ones(192,bool);keep[p['li_indices']]=~v.astype(bool)
 a=Atoms(numbers=p['numbers'][keep],positions=p['positions_A'][keep],cell=p['cell_A'],pbc=True)
 order=np.argsort([{3:0,27:1,8:2}[int(z)] for z in a.numbers],kind='stable');a=a[order];a.wrap();stem=f'AR_CVAE_Li{n}_{r["scoring_structure_id"]}_unrelaxed'
 for suffix,fmt in [('.vasp','vasp'),('.cif','cif')]:
  file=out/(stem+suffix)
  if fmt=='vasp':write(file,a,format=fmt,direct=True,vasp5=True,sort=False)
  else:write(file,a,format=fmt)
  b=read(file,format=fmt);assert np.array_equal(a.numbers,b.numbers)
  _,dist=find_mic(b.positions-a.positions,a.cell,pbc=True);assert max(dist)<1e-6
 selected.append(r)
(out/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
(out/'README.txt').write_text('Three examples from the round0 autoregressive CVAE, selected for illustration after final scoring: the lowest UMA-scored fresh proposal across its3seeds at each Li count. They are symmetry-unique and novel relative to all earlier history. Coordinates remain pristine-derived; no geometry or cell relaxation, DFT optimization, or ground-state certification is implied. POSCAR and CIF versions describe the same explicit atoms. This favorable selection is not an additional performance benchmark. See selection.json for measured energies/IDs.\n')
print('VESTA_EXPORT_PASS',len(selected),'lineages',flush=True)

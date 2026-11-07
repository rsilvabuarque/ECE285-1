"""Three exact-geometry MACE checks for DFT prioritization, not a second MD survey."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from ase import Atoms
from cathode_cvae.mlip.calculators import make_calculator,evaluate
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True)
p.add_argument('--checkpoint',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
calculator,provenance=make_calculator('mace',checkpoint=a.checkpoint,device='cuda',threads=4)
rows=[]
for basin in ['basin_1','basin_2','basin_3']:
 source=a.results/'relax'/basin/'final.npz'
 with np.load(source) as z:d={k:z[k] for k in z.files}
 atoms=Atoms(numbers=d['numbers'],positions=d['positions_A'],cell=d['cell_A'],pbc=True)
 energy,force,stress=evaluate(calculator,atoms)
 np.savez_compressed(a.output/f'{basin}.npz',energy_eV=energy,forces_eV_A=force,stress_eV_A3=stress,positions_A=atoms.positions,numbers=atoms.numbers,cell_A=atoms.cell.array)
 rows.append(dict(basin_id=basin,UMA_minimum_E_eV=float(d['energy_eV']),MACE_E_eV=energy,MACE_Fmax_eV_A=float(np.linalg.norm(force,axis=1).max()),force_teacher_difference_component_RMSE_eV_A=float(np.sqrt(np.mean((force-d['forces_eV_A'])**2))),stress_teacher_difference_max_eV_A3=float(np.max(np.abs(stress-d['stress_eV_A3']))),input_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
for row in rows:
 row['UMA_gap_eV']=row['UMA_minimum_E_eV']-min(x['UMA_minimum_E_eV'] for x in rows)
 row['MACE_gap_eV']=row['MACE_E_eV']-min(x['MACE_E_eV'] for x in rows)
with (a.output/'comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(a.output/'receipt.json').write_text(json.dumps(dict(status='COMPLETE',logical_single_point_EFS_calls=3,teacher=provenance,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,interpretation='MACE predictions on UMA minima. Teacher disagreement is not a DFT error or a MACE-relaxed minimum.'),indent=2)+'\n')
print(json.dumps(rows,indent=2))

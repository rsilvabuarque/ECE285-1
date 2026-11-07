"""Traceable existing examples and finite-pool counterexample, not new samples."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import write

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent_path=ROOT/'artifacts/current/core/data/parents/O3_N48_pilot.npz'
learn_path=ROOT/'artifacts/current/improvement_v1/training_original.npz'
with np.load(parent_path) as z:p={k:z[k] for k in ['numbers','positions_A','cell_A','li_indices','permutations','orbit_capacities']}
with np.load(learn_path) as z:
 ids=np.flatnonzero((z['n_Li']==24)&(z['split']=='train'));order=ids[np.argsort(z['energy_eV_atom'][ids])]
 examples=[dict(occupancy=z['occupancy'][j],source_id=str(z['structure_ids'][j]),energy_eV_atom=float(z['energy_eV_atom'][j]),canonical_mask=str(z['canonical_masks'][j]),description=desc) for j,desc in [(order[0],'low training energy'),(order[-1],'high training energy')]]
frac=p['positions_A']@np.linalg.inv(p['cell_A']);levels=np.unique(np.round(frac[p['li_indices'],2],7));layers=[np.flatnonzero(np.isclose(frac[p['li_indices'],2],zz,atol=1e-7)) for zz in levels]
assert len(layers)==6 and {len(x) for x in layers}=={8}
# Exact Burnside counts from cycle-generating polynomials, independent of saved capacities.
N=len(p['li_indices']);fixed=np.zeros(N+1,dtype=object)
for perm in p['permutations']:
 seen=set();cycles=[]
 for i in range(N):
  if i in seen:continue
  j=i;n=0
  while j not in seen:seen.add(j);n+=1;j=int(perm[j])
  cycles.append(n)
 dp=[0]*(N+1);dp[0]=1
 for size in cycles:
  for n in range(N,size-1,-1):dp[n]+=dp[n-size]
 fixed+=np.array(dp,dtype=object)
assert all(int(x)%len(p['permutations'])==0 for x in fixed)
orbits=np.array([int(x)//len(p['permutations']) for x in fixed],dtype=object)
assert all(int(x)==int(y) for x,y in zip(orbits,p['orbit_capacities']))
fig,axs=plt.subplots(1,2,figsize=(7,2.4),sharey=True)
for ax,ex in zip(axs,examples):
 for layer,ii in enumerate(layers):
  present=ex['occupancy'][ii].astype(bool)
  ax.scatter(np.arange(8)[~present],np.full((~present).sum(),layer),facecolors='white',edgecolors='#999999',s=60,linewidths=1)
  ax.scatter(np.arange(8)[present],np.full(present.sum(),layer),c='#1f77b4',s=60)
 ax.set(xticks=[],yticks=range(6),yticklabels=[str(i+1) for i in range(6)],ylim=(-.6,5.6),xlim=(-.6,7.6),title=ex['description']+'\n'+f"{ex['energy_eV_atom']:.5f} eV/atom")
 ax.set_xlabel('Eight indexed parent Li sites per plane')
axs[0].set_ylabel('Li plane');fig.suptitle('Same Li24Co48O96 composition, different vacancy arrangements',fontsize=10);fig.tight_layout();fig.savefig(OUT/'occupation_examples.pdf');fig.savefig(OUT/'occupation_examples.png',dpi=180);plt.close(fig)
export=OUT/'examples';export.mkdir(exist_ok=True);metadata=[]
for i,ex in enumerate(examples):
 keep=np.ones(len(p['numbers']),bool);keep[p['li_indices'][ex['occupancy']==0]]=False
 atoms=Atoms(numbers=p['numbers'][keep],positions=p['positions_A'][keep],cell=p['cell_A'],pbc=True)
 name=['low_training_deletion','high_training_deletion'][i];write(export/f'{name}.vasp',atoms,format='vasp',direct=True,sort=False);write(export/f'{name}.cif',atoms)
 metadata.append({k:v for k,v in ex.items() if k!='occupancy'}|dict(occupancy=ex['occupancy'].tolist(),n_Li=24,n_atoms=len(atoms),geometry='Pristine-derived unrelaxed deletion',teacher='UMA-S-1.2.1/omat FP32',split='train',files={f.name:sha(f) for f in export.glob(name+'.*')}))
(export/'manifest.json').write_text(json.dumps(dict(parent_sha256=sha(parent_path),learning_sha256=sha(learn_path),examples=metadata,half_lithiation_orbits=int(orbits[24]),group_operations=len(p['permutations']),scope='Existing training examples; neither newly generated nor relaxed'),indent=2)+'\n')
rows=list(csv.DictReader((OUT/'Hu_O3_Li10_optimized_counterexample.csv').open()));dft={r['structure_id']:r for r in rows if r['model']=='uma'};ordered=sorted(dft,key=lambda x:float(dft[x]['DFT_rank']))
fig,axs=plt.subplots(1,2,figsize=(7.5,2.8));colors={'uma':'#1f77b4','mace':'#d95f02','m3gnet':'#777777'}
axs[0].plot([1,5],[1,5],'k--',lw=1,label='Identical order')
for model in ['uma','mace','m3gnet']:
 r={v['structure_id']:v for v in rows if v['model']==model};rank=[float(r[s]['MLIP_rank']) for s in ordered];gap=[float(r[s]['MLIP_above_min_meV_atom']) for s in ordered]
 axs[0].plot(range(1,6),rank,'o-',color=colors[model],label=model,ms=4);axs[1].plot(range(1,6),gap,'o-',color=colors[model],ms=4)
axs[1].plot(range(1,6),[float(dft[s]['DFT_above_min_meV_atom']) for s in ordered],'ko--',label='DFT',ms=4)
for ax in axs:ax.set_xticks(range(1,6));ax.set_xlabel('Rank in the five-member DFT pool');ax.grid(alpha=.2)
axs[0].set(yticks=range(1,6),ylabel='MLIP rank (1 = lowest)');axs[0].legend(fontsize=7);axs[1].set_ylabel('Energy above own minimum (meV/atom)');axs[1].legend(fontsize=7)
fig.suptitle('Hu optimized O3 Li10Co12O24: complete pool contains five structures',fontsize=10);fig.tight_layout();fig.savefig(OUT/'five_member_counterexample.pdf');fig.savefig(OUT/'five_member_counterexample.png',dpi=180);plt.close(fig)
print('EXAMPLES_AND_BURNSIDE_PASS',int(orbits[24]))

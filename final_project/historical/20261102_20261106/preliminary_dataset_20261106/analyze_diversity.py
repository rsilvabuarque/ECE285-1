"""Exact symmetry-minimized mask distances separate novelty from local exploration."""
import csv,json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
with np.load(HERE/'data/parents/O3_N48_pilot.npz') as z:p={k:z[k] for k in z.files}
with np.load(HERE/'data/generated_candidates.npz') as z:g={k:z[k] for k in z.files}
with np.load(HERE/'data/learning_O3_N48_pilot_uma.npz') as z:d={k:z[k] for k in z.files}
splits={r['structure_id']:r['split'] for r in csv.DictReader((HERE/'splits/lineage_splits.csv').open())};train=np.array([splits[s]=='train' for s in d['structure_ids']]);E=d['energy_eV']/d['n_atoms'];bits=np.left_shift(np.uint64(1),np.arange(47,-1,-1,dtype=np.uint64));inv=np.argsort(p['permutations'],axis=1)
words=(g['masks'].astype(np.uint64)*bits).sum(1);nli=48-g['masks'].sum(1);distance=np.full(len(words),-1,np.int16);lowdistance=distance.copy()
for n in [12,24,36]:
    tr=np.flatnonzero(train&(d['n_Li']==n));q=np.sort(E[tr])[int(np.ceil(.2*len(tr)))-1];lo=tr[E[tr]<=q]
    expanded=np.unique((p['masks'][tr][:,inv].astype(np.uint64)*bits).sum(2).ravel());lowexpanded=np.unique((p['masks'][lo][:,inv].astype(np.uint64)*bits).sum(2).ravel())
    ids=np.flatnonzero(nli==n)
    for start in range(0,len(ids),32):
        ii=ids[start:start+32];distance[ii]=np.bitwise_count(words[ii,None]^expanded[None,:]).min(1);lowdistance[ii]=np.bitwise_count(words[ii,None]^lowexpanded[None,:]).min(1)
    assert (distance[ids]>0).all() and (distance[ids]%2==0).all();print('DIVERSITY_DISTANCE',n,len(ids),len(expanded),flush=True)
np.savez_compressed(HERE/'data/generated_training_distances.npz',structure_ids=g['structure_ids'],minimum_train_Hamming=distance,minimum_UMA_lowtrain_Hamming=lowdistance)
lookup={s:i for i,s in enumerate(g['structure_ids'])};groups={}
for r in csv.DictReader((HERE/'cvae/proposal_attempts.csv').open()):
    if int(r['n_Li']) in [12,24,36]:groups.setdefault((r['method'],int(r['n_Li']),int(r['seed'])),[]).append(r)
rows=[]
for (method,n,seed),rr in sorted(groups.items()):
    ids=np.unique([lookup[r['scoring_structure_id']] for r in rr]);rows.append(dict(method=method,n_Li=n,seed=seed,unique_orbits=len(ids),mean_minimum_train_swaps=float(distance[ids].mean()/2),median_minimum_train_swaps=float(np.median(distance[ids])/2),mean_minimum_UMA_lowtrain_swaps=float(lowdistance[ids].mean()/2),fraction_within3_training_swaps=float((distance[ids]<=6).mean())))
with (HERE/'reports/generation_diversity.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
metrics={(r['method'],int(r['n_Li']),int(r['seed'])):float(r['novel_low_yield_per_attempt']) for r in csv.DictReader((HERE/'reports/generation_metrics.csv').open()) if r['teacher']=='uma' and r['threshold_scope']=='train'}
fig,ax=plt.subplots(figsize=(9,5))
for method in sorted({r['method'] for r in rows}):
    rr=[r for r in rows if r['method']==method];xx=[r['mean_minimum_train_swaps'] for r in rr];yy=[metrics[(method,r['n_Li'],r['seed'])] for r in rr];ax.scatter(np.mean(xx),np.mean(yy),label=method.replace('_high','_not-low'),s=45)
ax.set_xlabel('Mean minimum Li/vacancy swaps to any training orbit\n(exact minimum over all 192 parent permutations)');ax.set_ylabel('Novel low-tail orbits / 512 attempts (UMA)');ax.legend(fontsize=7,bbox_to_anchor=(1.02,1),loc='upper left');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(HERE/'reports/generation_quality_diversity.pdf');fig.savefig(HERE/'reports/generation_quality_diversity.png',dpi=160)
(HERE/'validation/diversity.json').write_text(json.dumps(dict(status='PASS',primary_candidates=int((distance>=0).sum()),groups=len(rows),definition='Exactminimumfixed-countHammingdistanceoveralltrainingorbitsandallparentpermutations;dividedby2isminimumLi/vacancyswaps. LowtrainingreferenceusesUMA20%class. Heldcompositionsnotassignedsame-counttrainingdistance.',scope='Noveltylocalitydiagnostic, notthermodynamiccoverageorstatisticalindependence.'),indent=2)+'\n');print('DIVERSITY_PASS',flush=True)

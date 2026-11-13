"""Freeze lineage splits before fitting conditions or sampling correlated frames."""
import csv,hashlib,json,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
path=HERE/'splits/lineage_splits.csv';assert not path.exists()
rows=list(csv.DictReader((HERE/'data/mask_manifest.csv').open()));rng=np.random.default_rng(20261106)
parents=sorted({r['parent'] for r in rows});output=[]
for parent in parents:
    for n in sorted({int(r['n_Li']) for r in rows if r['parent']==parent}):
        group=[r for r in rows if r['parent']==parent and int(r['n_Li'])==n]
        available=[r for r in group if r['pilot_lineage_holdout']!='True'];rng.shuffle(available)
        diagnostic=len(group)<12
        for i,r in enumerate(available):
            if diagnostic:s='diagnostic'
            elif i<int(.8*len(available)):s='train'
            elif i<int(.9*len(available)):s='val'
            else:s='test'
            r['split']=s
        for r in group:
            if r['pilot_lineage_holdout']=='True':r['split']='pilot_holdout'
            r['composition_split']='composition_test' if parent=='O3_N48_pilot' and n in [9,15,21,27,33,39] else r['split']
            output.append({k:r[k] for k in ['structure_id','parent','canonical_mask','n_Li','split','composition_split']})
assert len({r['structure_id'] for r in output})==len(rows)
with path.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
counts={}
for r in output:counts[(r['parent'],r['split'])]=counts.get((r['parent'],r['split']),0)+1
(HERE/'splits/receipt.json').write_text(json.dumps({'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(rows),'counts':{str(k):v for k,v in counts.items()},'group':'parent+canonical mask, shared across all calculators/cohorts/temperatures/frames','small_support':'fewerthan12orbitspercomposition isdiagnostic; no train/val/test performanceclaim','cross_size':'Auxiliary cells are separate diagnostic/transfer assets; dense48CVAE does notclaimcrosssizegeneralization. Periodically reducible cross-parent aliases mustbeexcluded beforefuturemixedsizegraphbenchmark.'},indent=2)+'\n')
print('SPLITS_COMPLETE',len(rows),flush=True)

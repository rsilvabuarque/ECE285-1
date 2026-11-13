"""Independent forward-permutation canonicalization and exhaustive 12-site check."""
import csv,hashlib,itertools,json,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
rows=list(csv.DictReader((HERE/'data/mask_manifest.csv').open()));rec=json.loads((HERE/'data/preparation_receipt.json').read_text());checked=[]
for parent in rec['parents']:
    path=HERE/'data/parents'/f"{parent['parent']}.npz";assert hashlib.sha256(path.read_bytes()).hexdigest()==parent['sha256']
    with np.load(path) as z:d={k:z[k] for k in z.files}
    n=d['masks'].shape[1];perms=d['permutations'];powers=1<<np.arange(n-1,-1,-1,dtype=np.int64)
    assert all(np.array_equal(np.sort(p),np.arange(n)) for p in perms)
    # Forward site maps: input site i contributes bit weight at transformed site p[i].
    weights=powers[perms].T;canonical=[]
    for start in range(0,len(d['masks']),128):canonical.extend((d['masks'][start:start+128].astype(np.int64)@weights).min(axis=1).tolist())
    group=sorted([r for r in rows if r['parent']==parent['parent']],key=lambda r:int(r['parent_index']))
    assert len(set(canonical))==len(group)==len(d['masks'])
    assert canonical==[int(r['canonical_mask'],2) for r in group]
    assert np.array_equal(d['structure_ids'],[r['structure_id'] for r in group])
    assert np.array_equal(n-d['masks'].sum(axis=1),[int(r['n_Li']) for r in group])
    if n==12:
        raw=np.array(list(itertools.product([0,1],repeat=12)),np.int64);codes=(raw@weights).min(axis=1)
        assert len(set(codes))==87 and set(codes)==set(canonical)
    checked.append(dict(parent=parent['parent'],masks=len(group),canonicalization='forward permutation bitweights',passed=True))
(HERE/'validation/masks.json').write_text(json.dumps(dict(status='PASS',parents=checked,total=sum(r['masks'] for r in checked),scope='Within-parent lattice orbits; no assertion of cross-cell primitive equivalence'),indent=2)+'\n')
print('MASK_VALIDATION_PASS',checked,flush=True)

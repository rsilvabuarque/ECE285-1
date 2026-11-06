"""Independently verify parent geometry, canonical identities and the novelty boundary."""
import argparse,csv,hashlib,json,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pool',required=True);ap.add_argument('--known-keys',required=True);a=ap.parse_args();assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
 prefix=HERE/'data'/a.pool
 with np.load(HERE.parent/'data/parents/O3_N48_pilot.npz') as z:par={k:z[k] for k in z.files}
 known=set(json.loads(Path(a.known_keys).read_text())['canonical_masks']);allkeys=set();freshkeys=set();powers=1<<np.arange(47,-1,-1,dtype=np.int64);inv=np.argsort(par['permutations'],axis=1)
 for suffix,target in [('.npz',allkeys),('_fresh.npz',freshkeys)]:
  with np.load(str(prefix)+suffix) as z:
   for key in ['numbers','positions_A','cell_A','li_indices']:assert np.array_equal(z[key],par[key])
   masks=z['masks'];ids=z['structure_ids'];assert masks.shape==(len(ids),48) and np.isin(masks,[0,1]).all()
   codes=masks.astype(np.int64)@powers
   for start in range(0,len(ids),64):
    candidates=masks[start:start+64,inv];minimum=(candidates.astype(np.int64)*powers).sum(-1).min(1)
    assert np.array_equal(minimum,codes[start:start+64])
   for code,sid in zip(codes,ids):
    assert str(sid)==f'mask48_{int(code):012x}';target.add(format(int(code),'048b'))
   assert len(target)==len(ids)
 assert freshkeys==allkeys-known
 attempts=list(csv.DictReader(Path(str(prefix)+'_attempts.csv').open()))
 assert {r['canonical_mask'] for r in attempts}==allkeys
 for r in attempts:
  assert r['canonical_mask'].count('0')==int(r['n_Li'])
  assert (r['known_before']=='True')==(r['canonical_mask'] in known)
 receipt=json.loads(Path(str(prefix)+'_receipt.json').read_text())
 for f in receipt['frozen_models']:assert sha(f['checkpoint'])==f['checkpoint_sha256']
 result={'status':'PASS','pool':a.pool,'attempts':len(attempts),'all_orbits':len(allkeys),'fresh_orbits':len(freshkeys),'known_keys':len(known),'known_sha256':sha(a.known_keys),'checks':['NPZ parent species/coordinates/cell/Li rows equal evaluated parent','Every vacancy mask canonical under all192permutations','ID equals canonicalmaskhex and exactcomposition','Fresh set exactly all minus frozen history','Every frozen generator checkpoint unchanged']}
 Path(str(prefix)+'_identity_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()

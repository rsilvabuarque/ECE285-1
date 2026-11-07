"""Freeze existing train data and an independent raw-uniform allocation reference."""
import argparse,csv,json,hashlib,math
from pathlib import Path
import numpy as np
from cathode_cvae.conditions import fit_schema,save_schema
from cathode_cvae.data import sha256,validate_permutation_group
from cathode_cvae.mlip.registry import MODELS
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--reference-count',type=int,default=8192);p.add_argument('--seed',type=int,default=20261107);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
s=a.snapshot;parentfile=s/'core/data/parents/O3_N48_pilot.npz'
with np.load(parentfile) as z:parent={k:z[k] for k in z.files}
validate_permutation_group(parent['permutations']);assert parent['li_indices'].tolist()==list(range(48,96))
with np.load(s/'improvement_v1/training_round0.npz') as z:train={k:z[k] for k in z.files}
provenance=dict(parent_sha256=sha256(parentfile),data_sha256=sha256(s/'improvement_v1/training_round0.npz'),split_sha256=hashlib.sha256('\n'.join(f'{sid}:{sp}' for sid,sp in zip(train['structure_ids'],train['split'])).encode()).hexdigest(),teacher='uma',teacher_head='omat',teacher_checkpoint_name=MODELS['uma']['name'],teacher_checkpoint_hashes=MODELS['uma']['files'],geometry_protocol='pristine-derived deletion; unrelaxed; exact canonical parent coordinates',energy_label_origin='Existing round0 training corpus, same protected splits; quantiles describe this training corpus, not DOS')
schema=fit_schema(train['n_Li'],train['energy_eV_atom'],train['split'],n_sites=48,provenance=provenance);save_schema(schema,a.output/'condition_schema_uma.json')
# Exact canonical codes of raw-uniform draws. Frequencies retain the raw-mask measure.
perms=np.argsort(parent['permutations'],axis=1);powers=1<<np.arange(47,-1,-1,dtype=np.int64)
all_codes=[];all_n=[];all_raw=[]
for n in [12,24,36]:
 rng=np.random.default_rng(a.seed+n);raw=np.zeros((a.reference_count,48),np.uint8)
 for row in raw:row[rng.choice(48,n,replace=False)]=1
 codes=[]
 for start in range(0,len(raw),128):
  v=1-raw[start:start+128];codes.extend((v[:,perms]*powers).sum(-1).min(1).tolist())
 all_codes.extend(codes);all_n.extend([n]*len(raw));all_raw.extend(raw)
codes=np.array(all_codes,dtype=np.int64);unique,inverse,freq=np.unique(codes,return_inverse=True,return_counts=True);masks=((unique[:,None]>>np.arange(47,-1,-1))&1).astype(np.uint8);occupancy=1-masks;ids=np.array([f'mask48_{int(c):012x}' for c in unique]);n=occupancy.sum(1)
np.savez_compressed(a.output/'reference.npz',occupancy=occupancy,masks=masks,structure_ids=ids,n_Li=n,n_target=n,n_atoms=144+n,raw_frequency=freq,raw_sample_indices=inverse,raw_occupancy=np.array(all_raw),raw_n_Li=np.array(all_n),parent_sha256=np.array(sha256(parentfile)))
old=json.loads((s/'improvement_v1/known_keys_final.json').read_text());old=old.get('keys',old.get('canonical_masks',old)) if isinstance(old,dict) else old
history=set(old)
with np.load(s/'improvement_v1/data/final.npz') as z:
 history.update(''.join(map(str,row)) for row in z['masks'])
before=len(history);history.update(format(int(c),'048b') for c in unique)
(a.output/'history_before_reference.json').write_text(json.dumps(sorted(set(old)|set(''.join(map(str,row)) for row in np.load(s/'improvement_v1/data/final.npz')['masks'])))+'\n')
(a.output/'history_assessment.json').write_text(json.dumps(sorted(history))+'\n')
(a.output/'training_keys.json').write_text(json.dumps(train['canonical_masks'][train['split']=='train'].tolist())+'\n')
receipt=dict(status='PREPARED',parent_sha256=sha256(parentfile),reference_sha256=sha256(a.output/'reference.npz'),raw_draws=len(codes),unique_geometries=len(unique),per_composition_raw=a.reference_count,seed=a.seed,counts=[12,24,36],prior_history=before,assessment_history=len(history),reference_measure='Independent uniform raw fixed-count masks; canonical physical scoring; raw frequencies retained; no multiplicity double count',condition_schema_id=schema['identity'],provenance=provenance)
(a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

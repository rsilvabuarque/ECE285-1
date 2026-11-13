"""Compare existing fresh seed rescoring with its original H0 TRAIN labels; no new queries."""
from pathlib import Path
import csv,hashlib,json,time
import numpy as np
F=Path(__file__).parent;BASE=F.parent/'study_v2'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=F/'seed_reference_check';out.mkdir(exist_ok=False)
 protocol=json.loads((F/'protocol.json').read_text());h0=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project/artifacts/current/improvement_v1/training_round0.npz');bank=BASE/'seed_bank/bank.npz';fresh=F/'seed_rescoring/collected_uma/uma.npz';verifier=json.loads((F/'seed_rescoring/verified_uma.json').read_text());assert verifier['status']=='VERIFIED_FRESH_FRONTIER_TRAIN_SEED_RESCORING' and sha(fresh)==verifier['oracle_evidence']['score']['sha256']
 with np.load(h0,allow_pickle=False) as z:
  ii=np.flatnonzero(z['split'].astype(str)=='train');ids=z['structure_ids'][ii].astype(str);ee=z['energy_eV_atom'][ii];old=dict(zip(ids,ee))
 with np.load(bank,allow_pickle=False) as z:reference={str(k):float(old[str(s)]) for k,s in zip(z['canonical_masks'],z['structure_ids'])}
 with np.load(fresh,allow_pickle=False) as z:
  ids=z['structure_ids'].astype(str);energy=z['energy_eV']/z['n_atoms'];nn=z['n_Li']
 rr=[]
 for s,e,n in zip(ids,energy,nn):
  key=format(int(s.split('_')[1],16),'048b');r=reference[key];rr.append(dict(structure_id=s,n_target=int(n),H0_TRAIN_energy_eV_atom=r,fresh_UMA_energy_eV_atom=float(e),difference_meV_atom=1000*(float(e)-r)))
 with (out/'rows.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 summary=[]
 for count in [12,24,36]:
  v=np.asarray([r['difference_meV_atom'] for r in rr if r['n_target']==count]);summary.append(dict(n_target=count,seeds=len(v),mean_difference_meV_atom=float(v.mean()),rms_difference_meV_atom=float(np.sqrt(np.mean(v*v))),maximum_absolute_difference_meV_atom=float(np.abs(v).max())))
 with (out/'summary.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 (out/'receipt.json').write_text(json.dumps(dict(status='EXISTING_TRAIN_SEED_REFERENCE_CONSISTENCY_MEASURED',created_unix=time.time(),source_sha256=sha(Path(__file__)),inputs={str(p):sha(p) for p in [h0,bank,fresh,F/'seed_rescoring/verified_uma.json']},outputs={p.name:sha(p) for p in out.iterdir()},no_new_queries_or_tolerance=True,scope='TRAIN-only H0 labels; reference and newly predicted energies already existed. Numerical differences reported, no physical validation criterion chosen.'),indent=2)+'\n')
 print(summary,flush=True)
if __name__=='__main__':main()

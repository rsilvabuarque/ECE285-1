"""One declared adaptive update from development labels, before a new final pool."""
import csv,hashlib,json,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
 with np.load(HERE/'training_round0.npz') as z:d={k:z[k] for k in z.files}
 with np.load(HERE/'data/development_fresh.npz') as z:fresh={k:z[k] for k in z.files}
 with np.load(HERE/'data/development_scores_uma.npz') as z:energies={str(s):float(e/n) for s,e,n in zip(z['structure_ids'],z['energy_eV'],z['n_atoms'])}
 keys=[''.join(map(str,v)) for v in fresh['masks']];splits=[]
 for k in keys:
  value=int(hashlib.sha256(('round1_split_v1:'+k).encode()).hexdigest()[:12],16)/16**12
  splits.append('train' if value<.8 else 'val' if value<.9 else 'test')
 extra={'occupancy':1-fresh['masks'],'n_Li':48-fresh['masks'].sum(1),'energy_eV_atom':[energies[str(s)] for s in fresh['structure_ids']],'split':splits,'weight':np.ones(len(keys),np.float32),'structure_ids':fresh['structure_ids'],'canonical_masks':keys,'source':['development_now_adaptation']*len(keys)}
 assert not(set(keys)&set(d['canonical_masks']))
 merged={k:np.concatenate([d[k],np.asarray(extra[k])]) for k in d};path=HERE/'training_round1.npz';assert not path.exists();np.savez_compressed(path,**merged)
 known=set(json.loads((HERE/'known_keys_round0.json').read_text())['canonical_masks'])
 with np.load(HERE/'data/development.npz') as z:known.update(''.join(map(str,v)) for v in z['masks'])
 (HERE/'known_keys_final.json').write_text(json.dumps({'canonical_masks':sorted(known),'scope':'Original+prior adaptation+ALL development proposals, including adaptation validation/test; frozen before final proposals.'})+'\n')
 rows=list(csv.DictReader((HERE/'reports/development/summary.csv').open()));candidates=[r for r in rows if r['teacher']=='uma' and r['method'] in ['dense_adapt_count','ar_adapt_cvae']]
 chosen=max(candidates,key=lambda r:float(r['mean_low_yield']));family=chosen['method'];first=json.loads((HERE/'runs'/f'{family}_20261106'/'configuration.json').read_text());kind=first['kind']
 tasks=[]
 for beta,label in [(.1,'beta01'),(1.,'beta1')]:
  for seed in [20261106,20261107,20261108]:
   variant=f'adaptive_{family}_{label}';name=f'{variant}_{seed}'
   args=['--kind',kind,'--data',str(path),'--parent',str(HERE.parent/'data/parents/O3_N48_pilot.npz'),'--output-dir',str(HERE/'runs'/name),'--condition-mode','energy','--seed',str(seed),'--beta-final',str(beta),'--orientation','canonical','--epochs','100','--batch-size','512','--max-seconds','600','--class-balance','--warm-start',str(HERE/'runs'/f'{family}_{seed}'/'best.pt')]
   tasks.append({'name':name,'variant':variant,'kind':kind,'seed':seed,'source':'training_round1.npz','args':args,'trainer':'train_round1.py'})
 (HERE/'data/adaptive_fit_tasks.json').write_text(json.dumps(tasks,indent=2)+'\n')
 original=json.loads((HERE/'data/fit_tasks.json').read_text());(HERE/'data/final_run_list.json').write_text(json.dumps(original+tasks,indent=2)+'\n')
 receipt={'status':'ADAPTATION_PREPARED','selected_family':family,'selection':'Highest development UMA mean novel low yield among expanded-data energy-conditioned dense and AR CVAEs; no final scores seen.','selected_development_yield':float(chosen['mean_low_yield']),'added_development_orbits':len(keys),'rows':len(merged['n_Li']),'split_counts':{s:int((merged['split']==s).sum()) for s in ['train','val','test']},'training_sha256':sha(path),'final_known_keys':len(known),'final_known_keys_sha256':sha(HERE/'known_keys_final.json'),'changes':'One adaptive data round, train-only energy classes recomputed; class-balanced loss; beta0.1 versus beta1. Coupled exploration, not pure sample-size effect.','final_protocol':'All24 newfits and frozen legacy/uniform/mutation controls get512 independent final attempts percomposition/seed. No model changes after final scores.'}
 (HERE/'data/adaptation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()

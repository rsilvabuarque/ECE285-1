"""Verify fresh physical labels and join cached scalar energies for known hits."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prefix',required=True);ap.add_argument('--wait',action='store_true');a=ap.parse_args();assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
 prefix=Path(a.prefix).resolve();tag=prefix.name
 with np.load(str(prefix)+'.npz') as z:all_data={k:z[k] for k in z.files}
 with np.load(str(prefix)+'_fresh.npz') as z:fresh={k:z[k] for k in z.files}
 tasks=[json.loads(s) for s in (HERE/'data/score_tasks.jsonl').read_text().splitlines() if json.loads(s)['pool']==tag]
 while True:
  missing=[t for t in tasks if not (HERE/'scores'/t['model']/(t['task_id']+'.json')).exists()]
  if not missing:break
  print('COLLECT_WAIT',tag,len(missing),'of',len(tasks),flush=True)
  if not a.wait:raise RuntimeError('Unfinished shards')
  time.sleep(10)
 receipts={};workers=set()
 for model in ['uma','mace']:
  ids=all_data['structure_ids'];N=len(ids);index={str(s):i for i,s in enumerate(ids)};E=np.full(N,np.nan);provenance=np.full(N,'',dtype='<U16');fr=np.full(N,np.nan);fm=np.full(N,np.nan);stress=np.full((N,6),np.nan)
  # Known geometries are scalar cache hits, never fresh-novel successes.
  caches=[HERE/f'data/cache_{model}.npz']
  for old in ['development','final']:
   path=HERE/f'data/{old}_scores_{model}.npz'
   if path.exists() and not path.name.startswith(tag+'_scores'):caches.append(path)
  for path in caches:
   with np.load(path) as z:
    for sid,e in zip(z['structure_ids'],z['energy_eV']):
     i=index.get(str(sid))
     if i is not None:E[i]=e;provenance[i]='cached'
  seen=np.zeros(len(fresh['structure_ids']),int)
  for task in [t for t in tasks if t['model']==model]:
   base=HERE/'scores'/model/task['task_id'];r=json.loads(base.with_suffix('.json').read_text());assert r['status']=='complete'
   for key,value in task.items():assert r[key]==value
   assert sha(base.with_suffix('.npz'))==r['output_sha256'];workers.add(r['worker'])
   with np.load(base.with_suffix('.npz')) as z:
    ii=z['indices'];assert np.array_equal(ii,np.arange(task['start'],task['stop'])) and not seen[ii].any();seen[ii]+=1
    assert np.array_equal(z['structure_ids'],fresh['structure_ids'][ii]);ff=z['forces_parent_eV_A'];ss=z['stress_eV_A3'];ee=z['energy_eV'];assert ff.shape==(len(ii),192,3) and ss.shape==(len(ii),6)
    assert np.isfinite(ff).all() and np.isfinite(ss).all() and np.isfinite(ee).all()
    with np.load(HERE.parent/'data/parents/O3_N48_pilot.npz') as par:li=par['li_indices']
    presence=np.ones((len(ii),192),bool);presence[:,li]=~fresh['masks'][ii].astype(bool);assert (ff[~presence]==0).all()
    for j,sid in enumerate(z['structure_ids']):
     i=index[str(sid)];E[i]=ee[j];provenance[i]='fresh';stress[i]=ss[j];fr[i]=np.sqrt((ff[j]**2).sum()/(3*presence[j].sum()));fm[i]=np.linalg.norm(ff[j],axis=1).max()
  assert (seen==1).all() and np.isfinite(E).all()
  out=Path(str(prefix)+f'_scores_{model}.npz');n_atoms=192-all_data['masks'].sum(1)
  np.savez_compressed(out,structure_ids=ids,energy_eV=E,n_atoms=n_atoms,n_Li=n_atoms-144,provenance=provenance,force_component_rms_eV_A=fr,force_max_eV_A=fm,stress_eV_A3=stress)
  receipts[model]={'all_scored':N,'fresh':int((provenance=='fresh').sum()),'cached':int((provenance=='cached').sum()),'sha256':sha(out)}
 receipt={'status':'PASS','pool':tag,'fresh_unique_orbits':len(fresh['structure_ids']),'shards':len(tasks),'scores':receipts,'workers':sorted(workers),'fresh_input_sha256':sha(str(prefix)+'_fresh.npz'),'scope':'All fresh shards identity/shape/finite EFS/padding/hash checked. Cached scalar energies retain old orientation/model provenance; cached hits do not count as novel.'}
 Path(str(prefix)+'_collection.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()

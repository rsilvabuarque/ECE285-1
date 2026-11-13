"""Independent scope/count/hash/metric audit before the collaborator handoff."""
import csv,hashlib,json,os,time
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
 while not (HERE/'data/portable_relocated.json').exists():time.sleep(2)
 receipts={}
 for rel in ['data/model_tests.json','data/portable_relocated.json','data/development_identity_audit.json','data/development_collection.json','reports/development/evaluation_receipt.json','data/final_identity_audit.json','data/final_collection.json','reports/final/evaluation_receipt.json','reports/final/quality/quality_receipt.json','reports/final_diversity64_receipt.json']:
  v=read(HERE/rel);assert v['status']=='PASS',(rel,v['status']);receipts[rel]=sha(HERE/rel)
 fits=json.loads((HERE/'data/final_run_list.json').read_text());assert len(fits)==24
 for task in fits:
  folder=HERE/'runs'/task['name'];c=read(folder/'configuration.json');s=read(folder/'summary.json')
  assert s['status']=='FIT_COMPLETE' and s['heldout_test_evaluated'] is False
  assert sha(folder/'best.pt')==s['checkpoint_sha256'] and sha(c['data'])==c['data_sha256']
  assert c['model_source_sha256']==sha(HERE/'models.py')
  assert c['trainer_sha256'] in {sha(HERE/'train.py'),sha(HERE/'train_round1.py')}
  for kind in ['dense_bce','dense_count','ar_cvae','ar_baseline']:assert read(HERE/'smoke'/kind/'sampling_check.json')['status']=='PASS'
 assert read(HERE/'smoke/portable_ar_cpu/summary.json')['status']=='FIT_COMPLETE'
 sets={};splitmaps={}
 for name in ['training_original','training_round0','training_round1']:
  with np.load(HERE/f'{name}.npz') as z:
   keys=z['canonical_masks'].astype(str);assert len(set(keys))==len(keys)
   assert np.array_equal(z['occupancy'].sum(1),z['n_Li']) and np.isfinite(z['energy_eV_atom']).all()
   sets[name]=set(keys);splitmaps[name]=dict(zip(keys,z['split']))
 assert sets['training_original']<=sets['training_round0']<=sets['training_round1']
 for k,v in splitmaps['training_round0'].items():assert splitmaps['training_round1'][k]==v
 for k,v in splitmaps['training_original'].items():assert splitmaps['training_round0'][k]==v
 known=set(read(HERE/'known_keys_final.json')['canonical_masks']);assert sets['training_round1']<=known
 with np.load(HERE/'data/final_fresh.npz') as z:assert not({''.join(map(str,v)) for v in z['masks']} & known)
 gen=read(HERE/'data/final_receipt.json')
 assert gen['model_source_sha256']==sha(HERE/'models.py') and gen['generator_source_sha256']==sha(HERE/'generate_final.py')
 assert gen['run_list_sha256']==sha(HERE/'data/final_run_list.json')
 for item in gen['outputs'].values():assert sha(item['path'])==item['sha256']
 for item in gen['frozen_models']:assert sha(item['checkpoint'])==item['checkpoint_sha256']
 assert gen['known_keys_sha256']==sha(HERE/'known_keys_final.json')
 tasks=[json.loads(s) for s in (HERE/'data/score_tasks.jsonl').read_text().splitlines() if json.loads(s)['pool']=='final']
 first_claim=min(read(HERE/'claims'/t['task_id']/'owner.json')['started'] for t in tasks)
 proposal_freeze=(HERE/'data/final_receipt.json').stat().st_mtime
 checkpoint_latest=max(Path(item['checkpoint']).stat().st_mtime for item in gen['frozen_models'])
 assert checkpoint_latest<=proposal_freeze<=first_claim
 collection=read(HERE/'data/final_collection.json')
 for worker in collection['workers']:
  w=read(HERE/'workers'/f'{worker}.json')
  assert w['source_sha256']==sha(HERE/'score_service.py') and w['calculator_source_sha256']==sha(ROOT/'calculators.py')
 proposal=list(csv.DictReader((HERE/'data/final_attempts.csv').open()));groups=defaultdict(list)
 for r in proposal:groups[r['method'],r['seed'],int(r['n_Li'])].append(r)
 thresholds={(r['teacher'],int(r['n_Li'])):float(r['q20_eV_atom']) for r in csv.DictReader((ROOT/'reports/generation_thresholds.csv').open()) if r['source']=='main_train'}
 measured={(r['teacher'],r['method'],r['seed'],int(r['n_Li'])):r for r in csv.DictReader((HERE/'reports/final/metrics.csv').open())}
 verified=0
 for model in ['uma','mace']:
  with np.load(HERE/f'data/final_scores_{model}.npz') as z:energy=dict(zip(z['structure_ids'],z['energy_eV']/z['n_atoms']))
  for (method,seed,n),rr in groups.items():
   assert len(rr)==512
   unique={r['scoring_structure_id']:r['canonical_mask'] for r in rr};low=sum(key not in known and energy[sid]<=thresholds[model,n] for sid,key in unique.items())
   row=measured[model,method,seed,n];assert int(row['distinct_novel_low_orbits'])==low and abs(float(row['novel_low_yield_per_attempt'])-low/512)<1e-14;verified+=1
 assert verified==252
 summary=list(csv.DictReader((HERE/'reports/final/summary.csv').open()));lines=['Lunch-hour CVAE improvement results — 6 November 2026','', 'Final fresh assessment: 512 attempts per method/seed/count; 3 seeds, Li12/24/36.','Thresholds remain the original training q20; novelty excludes all adaptation/development.','', 'Method                                      UMA yield    MACE yield']
 methods=sorted({r['method'] for r in summary})
 for method in methods:
  vv={r['teacher']:100*float(r['mean_low_yield']) for r in summary if r['method']==method}
  lines.append(f'{method:43s} {vv["uma"]:8.2f}% {vv["mace"]:10.2f}%')
 lines+=['','All methods are reported; do not select a new model on these results and reuse this final pool as an untouched test.','The round0 AR CVAE family was selected using development results before adaptive fitting.','Adaptation changes both data distribution and training low-class thresholds; this is not a pure sample-size effect.','Architecture, initialization, and training protocol are described by each configuration, not isolated causal claims.','The new corpus is MLIP-labeled; no DFT accuracy or global minimum claim.','No end-to-end cost advantage established: count adaptation scoring and training before amortizing generation.','Distance-to-training results use a declared64-orbit/group subset, not every final proposal.','The original10-page report is background; use these final tables for current sampling results.']
 (HERE/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
 receipt={'status':'PASS','utc_seconds':time.time(),'scope':'Scientific computation and data audit; archive integrity and final terminal allocation accounting are separate delivery checks.','new_fits':24,'fit_seeds':[20261106,20261107,20261108],'training_round1_rows':len(sets['training_round1']),'development_new_orbits':25000,'final_proposals':len(proposal),'final_new_orbits':gen['fresh_orbits'],'new_model_structure_labels':2*(25000+gen['fresh_orbits']),'independently_recomputed_final_metric_rows':verified,'receipts':receipts,'RESULTS_sha256':sha(HERE/'RESULTS.txt'),'all_final_checkpoints_unchanged':True,'chronology_utc_seconds':{'latest_checkpoint':checkpoint_latest,'proposal_freeze':proposal_freeze,'first_final_oracle_claim':first_claim},'original_and_adaptation_splits_preserved':True,'CPU_and_GPU_operability':True,'no_DFT':True,'limitations':['MLIP teacher consistency, not DFT ground truth','Three fitted seeds; sampling groups are not independent fit replications','Fixed48site parent and three compositions, no material/size transfer','No demonstrated end-to-end speedup','Final energy outcomes not used for further fitting']}
 (HERE/'science_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()

"""Bind completed result/cost receipts and audit roles without opening protected energies."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from cathode_cvae.adaptive_iteration import load_boundary
from cathode_cvae.iterative_acquisition import _parent,_canonical_vacancy
from cathode_cvae.runtime import require_compute_allocation
I=Path(__file__).parent;R=I/'study_v2';C=I/'causal_range';F=I/'frontier'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def oracle(p):
 r=json.loads(p.read_text());assert r['status'] in ['VERIFIED_FRESH_ADAPTIVE_ORACLE','VERIFIED_FRESH_FRONTIER_TRAIN_SEED_RESCORING'];v=r.get('oracle_evidence');assert v is not None
 return dict(receipt=bind(p),unique_structures=r['unique_structures'],inference_wall_seconds=v['summed_inference_seconds'],raw_scoring_tasks=v['scoring_tasks'])
def taskstats(root):
 rr=[]
 for p in sorted((root/'taskfiles').glob('*.json')):
  ts=json.loads(p.read_text())['tasks'];claims=p.parent/(p.stem+'.claims');receipts=[]
  for t in ts:
   q=claims/t['id']/'receipt.json';v=json.loads(q.read_text());assert v['status']=='COMPLETE';receipts.append(v)
  rr.append(dict(taskfile=bind(p),tasks=len(ts),sum_one_GPU_child_wall_seconds=sum(v['finished_unix']-v['started_unix'] for v in receipts),all_complete=True))
 return rr

def main():
 require_compute_allocation()
 for p in [R/'completion.json',R/'analysis/receipt.json',C/'completion.json',C/'analysis/receipt.json',F/'completion.json',F/'analysis/receipt.json']:assert p.exists(),str(p)
 core={phase:{m:oracle(R/phase/f'verified_{m}.json') for m in ['uma','mace']} for phase in ['round1_acquisition','round1_evaluation','round2_acquisition','round2_evaluation']}
 causal={m:oracle(C/'evaluation'/f'verified_{m}.json') for m in ['uma','mace']}
 frontier={name:{m:oracle(F/name/f'verified_{m}.json') for m in ['uma','mace']} for name in ['seed_rescoring','step1','step2','step3','step4']}
 old,k0,p0=load_boundary(I/'boundary_base');last,known,protected=load_boundary(F/'boundary_after_step4');perms,_=_parent(old['parent']['path'])
 h0=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project/artifacts/current/improvement_v1/training_round0.npz')
 with np.load(h0,allow_pickle=False) as z:roles=z['split'].astype(str);x=z['occupancy'][roles=='train']
 train=set(_canonical_vacancy(x,perms).tolist());new_train=set();train_sources=[]
 for phase in ['round1_acquisition','round2_acquisition']:
  models=json.loads((R/phase/'models.json').read_text())
  for model in models:
   if model['family'] not in ['ar','d2']:continue
   stage=Path(model['stage'])
   with np.load(stage/'candidates.npz',allow_pickle=False) as z:keys={''.join(map(str,m)) for m in z['masks']}
   new_train.update(keys);train_sources.append(bind(stage/'receipt.json'))
 assert protected<=known and train<=known and not (new_train&protected) and not ((protected-p0)&train)
 role=dict(status='FINAL_PROTECTED_ROLE_AUDIT_PASS',created_unix=time.time(),base_boundary=bind(I/'boundary_base/receipt.json'),final_boundary=bind(F/'boundary_after_step4/receipt.json'),h0=bind(h0),h0_train_orbits=len(train),inherited_protected_H0_TRAIN_orbits=len(p0&train),final_protected_H0_TRAIN_orbits=len(protected&train),newly_protected_H0_TRAIN_orbits=len((protected-p0)&train),new_training_orbits_across_alternative_lineages=len(new_train),new_training_protected_overlap=len(new_train&protected),all_protected_known=protected<=known,known_orbits=len(known),protected_orbits=len(protected),acquisition_stage_receipts=train_sources,no_energy_members_read=True,source=bind(__file__))
 write(I/'final_protected_role_audit.json',role)
 cdone=json.loads((C/'completion.json').read_text());fdone=json.loads((F/'completion.json').read_text());assert cdone['elapsed_seconds']<=40*60 and fdone['elapsed_seconds']<=50*60
 cnode=4.5*cdone['elapsed_seconds']/3600;fnode=4.5*fdone['elapsed_seconds']/3600;assert cnode<=3 and fnode<=4
 counts={'core_geometry_teacher_pairs':sum(v['unique_structures'] for p in core.values() for v in p.values()),'causal_geometry_teacher_pairs':sum(v['unique_structures'] for v in causal.values()),'frontier_new_geometry_teacher_pairs':sum(v['unique_structures'] for name,p in frontier.items() if name!='seed_rescoring' for v in p.values()),'frontier_TRAIN_seed_rescoring_pairs':sum(v['unique_structures'] for v in frontier['seed_rescoring'].values())}
 assert counts['frontier_new_geometry_teacher_pairs']<=55296 and counts['frontier_TRAIN_seed_rescoring_pairs']<=576
 fits=[]
 for p in [*sorted((R/'round1/fits').glob('*/summary.json')),*sorted((R/'round2/fits').glob('*/summary.json')),*sorted((C/'fits').glob('*/summary.json'))]:
  s=json.loads(p.read_text());assert s['status']=='FIT_COMPLETE' and s['optimizer_updates']==2000;fits.append(bind(p))
 assert len(fits)==72
 completed={str(p):sha(p) for p in [R/'completion.json',R/'analysis/receipt.json',R/'interpretation_diagnostics/receipt.json',R/'presentation_v2/receipt.json',R/'acquisition_coverage/receipt.json',C/'completion.json',C/'analysis/receipt.json',F/'completion.json',F/'analysis/receipt.json',F/'first_step_paired_control_check.json',F/'seed_reference_check/receipt.json']}
 receipt=dict(status='ADAPTIVE_SUBAGENT_ALL_EXPERIMENTS_AND_ANALYSES_COMPLETE',created_unix=time.time(),source=bind(__file__),counts=counts,new_generative_fits=72,new_optimizer_updates=144000,new_DFT_calculations=0,new_MLIP_fine_tuning_updates=0,core_oracles=core,causal_oracles=causal,frontier_oracles=frontier,fit_summaries=fits,completed_artifacts=completed,protected_role_audit=bind(I/'final_protected_role_audit.json'),causal_elapsed_seconds=cdone['elapsed_seconds'],frontier_elapsed_seconds=fdone['elapsed_seconds'],causal_18_slot_window_node_equivalent_hours=cnode,frontier_18_slot_window_node_equivalent_hours=fnode,local_followup_caps_pass=True,cost_definition='18GPU slots/4GPU per node times experiment elapsed; conservative local window attribution, NOT billed NERSC accounting. Per-call inference wall and one-GPU child lifetimes are separate. Root allocation/resource receipts remain billing authority.',GPU_task_receipts={'core':taskstats(R),'causal':taskstats(C),'frontier':taskstats(F)},physical_scope='Fixed pristine-derived N48 vacancy masks, no relaxation or DFT/equilibrium/global-extrema validation.',geometry_models='UMAomat andMACEomat_pbe, pinnedcheckpoints/float32/TF32off in rawworkerreceipts')
 receipt['prior_failed_path_helper']=bind(I/'final_receipt.py');receipt['prior_path_failure_log']=bind(I/'final_receipt.log')
 write(I/'RESULT_AND_COST_RECEIPT.json',receipt);print(json.dumps(dict(status=receipt['status'],counts=counts,causal_window_node_hours=cnode,frontier_window_node_hours=fnode,protected=role['protected_orbits'],known=role['known_orbits']),indent=2),flush=True)
if __name__=='__main__':main()

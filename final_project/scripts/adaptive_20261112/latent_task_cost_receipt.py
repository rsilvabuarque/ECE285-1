"""Task-attributed timing evidence, separate from the root SLURM allocation ledger."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
from cathode_cvae.runtime import require_compute_allocation
require_compute_allocation();import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args();p=args.run_root.resolve();b=p.parent;r=b/'latent_diagnostics';p2=b/'latent_phase2';sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest();rows=[];oracle=[]
for stage,root,name,kind,prefix in [('initial',b,'initial_fit_tasks','fit',None),('initial',r,'diagnostics_and_entropy_tasks','diagnostic','diagnostic_'),('strength',p2,'fit_tasks','fit',None),('strength',p2,'diagnostic_tasks','diagnostic',None),('mean',p,'fit_tasks','fit',None),('mean',p,'diagnostic_tasks','diagnostic',None)]+[(stage,root,name,'oracle_worker',None) for stage,root in [('initial',r),('strength',p2),('mean',p)] for name in ['latent_oracle_tasks','latent_uma_extra_tasks']]:
 taskfile=root/(name+'.json')
 if not taskfile.exists():continue
 for task in json.loads(taskfile.read_text())['tasks']:
  if prefix and not task['id'].startswith(prefix):continue
  receipt=root/(name+'.claims')/task['id']/'receipt.json';q=json.loads(receipt.read_text())
  if q['status']!='COMPLETE' or q['task']!=task:raise ValueError('Noncomplete ormutatedtask')
  rows.append(dict(stage=stage,kind=kind,task_id=task['id'],visible_GPU_count=1,task_GPU_wall_seconds=q['finished_unix']-q['started_unix'],receipt=str(receipt),receipt_sha256=sha(receipt),task_file_sha256=sha(taskfile)))
for stage,root in [('initial',r),('strength',p2),('mean',p)]:
 for teacher in ['uma','mace']:
  receipt=root/f'collected_{teacher}/receipt.json';q=json.loads(receipt.read_text())
  if q['status']!='PASS':raise ValueError('NonPASSoracle')
  with np.load(root/f'collected_{teacher}/{teacher}.npz',allow_pickle=False) as z:
   seconds=z['seconds'];count=len(z['structure_ids'])
   if count!=q['model_counts'][teacher] or not np.isfinite(seconds).all() or np.any(seconds<0):raise ValueError('Badtiming/coverage')
  oracle.append(dict(stage=stage,teacher=teacher,new_geometry_evaluations=count,summed_model_evaluation_seconds=float(seconds.sum()),collection_receipt=str(receipt),collection_receipt_sha256=sha(receipt)))
with (p/'latent_task_cost_rows.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary={}
for row in rows:
 key=row['stage']+':'+row['kind'];summary.setdefault(key,dict(tasks=0,GPU_wall_seconds=0));summary[key]['tasks']+=1;summary[key]['GPU_wall_seconds']+=row['task_GPU_wall_seconds']
result=dict(status='PASS',scope='Latent initial40+strength30+mean15scientificfits,120fullguideddiagnostics,anddedicatedUMA/MACEworkers. Initialentropyworkers/retryandalladaptive/entropyactivityexcluded toavoidcrossattribution.',one_GPU_per_task='RootlauncherassignsonevisibleGPUperworker; taskwallincludesmodelinitialization,queueI/O,idleandSTOPlatency',task_GPU_wall_seconds=sum(x['task_GPU_wall_seconds'] for x in rows),task_GPU_hours=sum(x['task_GPU_wall_seconds'] for x in rows)/3600,task_node_hour_equivalent_at4GPUs=sum(x['task_GPU_wall_seconds'] for x in rows)/14400,not_allocation_accounting='Task-attributedtimingonly; rootSLURMledgerremainsauthoritativeforallocated/chargednodehours. Model-evaluationsecondsareoverlappingsubsetandmustnotbeaddedagain.',task_summary=summary,oracle_counts=oracle,total_new_geometry_teacher_evaluations=sum(x['new_geometry_evaluations'] for x in oracle),new_DFT_calculations=0,new_MD_trajectories=0,new_MLIP_finetunes=0,source_sha256=sha(__file__),task_rows_sha256=sha(p/'latent_task_cost_rows.csv'))
with (p/'latent_cost_receipt.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['oracle_counts','task_summary']},indent=2),flush=True)

"""Collect complete immutable queues; transfer freed MACE workers to UMA."""
from pathlib import Path
import hashlib,json,time
from cathode_cvae.runtime import require_compute_allocation
from cathode_cvae.adaptive_oracle import collect
require_compute_allocation()
import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args();r=args.run_root.resolve();complete=set();last={};sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
while len(complete)<2:
 for model in ('uma','mace'):
  if model in complete:continue
  q=r/f'oracle_{model}';registry=json.loads((q/'registry.json').read_text());tasks=registry['tasks'];done=list((q/'scores'/model).glob('*.json'));failed=list((q/'claims').glob('*/failure.json'))
  count=len(done)
  if count!=last.get(model):print(json.dumps(dict(model=model,complete_tasks=count,total_tasks=len(tasks),failed_tasks=len(failed),unix=time.time())),flush=True);last[model]=count
  if failed:raise RuntimeError(f'{model} failed task receipts found; preserve and inspect')
  if count==len(tasks):
   result=collect(q,r/f'collected_{model}')
   if result['status']!='PASS':raise RuntimeError(f'{model} collection incomplete: {result}')
   (q/'STOP').write_text('All registered batches collected with strict PASS; stop resident workers.\n')
   complete.add(model);print('COLLECTED',model,result,flush=True)
   if model=='mace' and 'uma' not in complete:
    old=json.loads((r/'latent_oracle_tasks.json').read_text())['tasks'];base=[t for t in old if t['id'].endswith('latent_uma_worker_00')][0];tasks=[]
    for i in range(9):
     t=json.loads(json.dumps(base));t['id']=f'latent_uma_extra_{i:02}';tasks.append(t)
    path=r/'latent_uma_extra_tasks.json'
    with path.open('x') as f:json.dump({'tasks':tasks},f,indent=2);f.write('\n')
    c=r.parent/'controls/mace.json';tmp=c.with_suffix('.latent.tmp');tmp.write_text(json.dumps({'task_file':str(path)},indent=2)+'\n');tmp.replace(c)
    print('PUBLISHED9_EXTRA_UMA_WORKERS',sha(path),flush=True)
 time.sleep(10)
result=dict(status='PASS',source_sha256=sha(__file__),collections={m:sha(r/f'collected_{m}/receipt.json') for m in complete},scope='No scoring retries/cache aliases; teacher STOP only after complete successful registry collection')
(r/'oracle_monitor_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)

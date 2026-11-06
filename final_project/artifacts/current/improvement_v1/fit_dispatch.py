"""One fitting process per scheduler-bound GPU; next task claimed on completion."""
import json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
rank=int(os.environ['SLURM_PROCID']);tasks=json.loads((HERE/'data/fit_tasks.json').read_text())
for task in tasks[rank:]+tasks[:rank]:
 claim=HERE/'claims'/('fit_'+task['name'])
 try:claim.mkdir()
 except FileExistsError:continue
 (claim/'owner.json').write_text(json.dumps({'job_id':os.environ['SLURM_JOB_ID'],'host':os.environ['SLURMD_NODENAME'],'rank':rank,'started':time.time()})+'\n')
 with (HERE/'logs'/f'fit_{task["name"]}.log').open('w') as f:
  rc=subprocess.call([sys.executable,str(HERE/'train.py'),*task['args']],stdout=f,stderr=subprocess.STDOUT)
 if rc:raise RuntimeError(f'fit failed {task["name"]}: {rc}')
 print('FIT_DONE',task['name'],flush=True)
print('FITS_DISPATCH_DONE',rank,flush=True)

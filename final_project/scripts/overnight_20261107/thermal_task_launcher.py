"""Map one explicitly prepared thermal command to each allocated GPU task."""
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True)
p.add_argument('--runner',type=Path,required=True);a=p.parse_args()
rank=int(os.environ.get('SLURM_PROCID','0'))
tasks=json.loads(a.tasks.read_text())
if not 0<=rank<len(tasks):raise ValueError('No prespecified command for this task rank')
cmd=[sys.executable,str(a.runner),*tasks[rank]]
print('THERMAL_TASK_START',rank,tasks[rank][0],flush=True)
os.execv(sys.executable,cmd)

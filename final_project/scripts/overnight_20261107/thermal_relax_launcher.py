"""Actual three-task Slurm launch; scheduler/account selection stays outside Python."""
import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--results',type=Path,required=True);p.add_argument('--checkpoint',required=True)
a=p.parse_args();rank=int(os.environ['SLURM_PROCID'])
row=json.loads((a.root/'runs/thermal_pilot_20261107/selection/selection.json').read_text())['rows'][rank]
cmd=[sys.executable,str(a.root/'scripts/overnight_20261107/thermal_dynamics.py'),'relax',
     '--parent',str(a.root/'artifacts/current/core/data/parents/O3_N48_pilot.npz'),
     '--vacancy-mask',row['vacancy_mask'],'--basin-id',row['basin_id'],
     '--checkpoint',a.checkpoint,'--output',str(a.results/'relax'/row['basin_id'])]
os.execv(sys.executable,cmd)

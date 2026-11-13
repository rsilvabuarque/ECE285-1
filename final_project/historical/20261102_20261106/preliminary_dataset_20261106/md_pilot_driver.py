"""Short debug allocation for numerical and operating-point pilots, not production."""
import os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def run(stage):
    with (HERE/'logs'/f"{os.environ['SLURM_JOB_ID']}_{stage}.log").open('w') as f:
        subprocess.run(['srun','--overlap','-N2','-n8','--ntasks-per-node=4','-c8','--gpus-per-task=1','--gpu-bind=single:1',sys.executable,str(HERE/'md_dispatch_v2.py'),'--stage',stage],stdout=f,stderr=subprocess.STDOUT,check=True)
run('md_extra_gate');run('md_pilot');print('SHORT_MD_PILOTS_DISPATCH_RETURNED',flush=True)

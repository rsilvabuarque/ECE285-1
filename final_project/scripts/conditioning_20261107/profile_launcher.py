"""Select one installed backend per explicitly GPU-bound Slurm task."""
import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
c=json.loads(a.config.read_text());rank=int(os.environ.get('SLURM_PROCID','0'))
model=['uma','mace'][rank];env=os.environ.copy()
env.update(PYTHONPATH=c['source_path'],OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
cmd=[c['interpreters'][model],str(Path(c['root'])/'scripts/conditioning_20261107/pilot.py'),
     '--root',c['root'],'--model',model,'--checkpoint',c['checkpoints'][model]]
os.execve(cmd[0],cmd,env)

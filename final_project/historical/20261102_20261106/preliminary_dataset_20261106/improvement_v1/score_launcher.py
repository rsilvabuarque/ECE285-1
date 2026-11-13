"""Respect scheduler GPU binding while using each teacher's existing environment."""
import os,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent.parent/'energy_ranking_audit/multimodel_20261104'
rank=int(os.environ['SLURM_PROCID']);model=['uma','mace'][rank%2]
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1')
with (HERE/'logs'/f'oracle_{os.environ["SLURM_JOB_ID"]}_{os.environ["SLURMD_NODENAME"]}_{rank}.log').open('w') as f:
 raise SystemExit(subprocess.call([str(BASE/'envs'/model/'bin/python'),str(HERE/'score_service.py'),'--model',model],env=env,stdout=f,stderr=subprocess.STDOUT))

"""Reuse completed geometry checkpoints after repairing metadata serialization."""
import subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;TRAIN=HERE.parent/'energy_ranking_audit/multimodel_20261104/envs/uma/bin/python'
with (HERE/'logs/generate_geometry.log').open('w') as f:subprocess.run(['srun','--overlap','-N1','-n1','-c8','--gpus-per-task=1','--gpu-bind=single:1',str(TRAIN),str(HERE/'cvae/generate_geometry.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
with (HERE/'logs/decoded_scoring.log').open('w') as f:subprocess.run(['srun','--overlap','-N1','-n2','--ntasks-per-node=2','-c8','--gpus-per-task=1','--gpu-bind=single:1',sys.executable,str(HERE/'explicit_dispatch.py'),'--stage','decoded_score'],stdout=f,stderr=subprocess.STDOUT,check=True)
print('GEOMETRY_RESUME_COMPLETE',flush=True)

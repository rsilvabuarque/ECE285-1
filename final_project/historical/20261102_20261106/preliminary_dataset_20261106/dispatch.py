"""Launch isolated model environments, respecting the scheduler's single-GPU binding."""
import argparse,json,os,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'energy_ranking_audit/multimodel_20261104'
PILOT=HERE.parent/'energy_ranking_audit/relaxation_pilot48_20261105'
ENVS={'uma':OLD/'envs/uma/bin/python','mace':OLD/'envs/mace/bin/python','sevennet':PILOT/'envs/sevennet/bin/python','chgnet':OLD/'envs/matgl/bin/python','m3gnet':OLD/'envs/matgl/bin/python'}
ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);ap.add_argument('--model');a=ap.parse_args()
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',MPLBACKEND='Agg')
rank=int(env.get('SLURM_PROCID',0));models=['uma','mace','sevennet','chgnet','m3gnet'];models=models[rank%5:]+models[:rank%5]
if a.model:models=[a.model]
path=HERE/'data'/f'{a.stage}_tasks.jsonl'
while not path.exists():time.sleep(10)
while True:
    tasks=[json.loads(s) for s in path.read_text().splitlines() if s]
    counts={m:sum(t['model']==m and not (HERE/'claims'/t['task_id']).exists() for t in tasks) for m in models}
    if max(counts.values(),default=0)==0:break
    model=max(models,key=lambda m:counts[m]);log=HERE/'logs'/f"{env['SLURM_JOB_ID']}_{env.get('SLURMD_NODENAME')}_{rank}_{a.stage}_{model}_{time.time_ns()}.log"
    with log.open('w') as f:rc=subprocess.call([str(ENVS[model]),str(HERE/'worker.py'),'--model',model,'--stage',a.stage],env=env,stdout=f,stderr=subprocess.STDOUT)
    if rc:raise RuntimeError(f'Worker failed; inspect {log}; return{rc}')
print('DISPATCH_DONE',a.stage,rank,flush=True)

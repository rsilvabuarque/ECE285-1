"""One supervisor per visible GPU, switching model environments as queues finish."""
import argparse,json,os,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'multimodel_20261104'
ENVS={'uma':OLD/'envs/uma/bin/python','mace':OLD/'envs/mace/bin/python','m3gnet':OLD/'envs/matgl/bin/python','chgnet':OLD/'envs/matgl/bin/python','sevennet':HERE/'envs/sevennet/bin/python'}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model');ap.add_argument('--validate-only',action='store_true');ap.add_argument('--gpu');a=ap.parse_args()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MPLBACKEND='Agg',PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',HF_HUB_OFFLINE='1')
    if a.gpu is not None:env['CUDA_VISIBLE_DEVICES']=a.gpu
    order=['uma','mace','sevennet','chgnet','m3gnet'];rank=int(os.environ.get('SLURM_PROCID',a.gpu or 0));order=order[rank%5:]+order[:rank%5]
    waiting_since=time.monotonic()
    while True:
        ready=[]
        for model in order:
            if a.model and model!=a.model:continue
            v=HERE/'validation'/f'{model}.json'
            if not a.validate_only and (not v.exists() or not json.loads(v.read_text()).get('passed')):continue
            remaining=100-sum(1 for p in (HERE/'claims').iterdir() if p.name.startswith(model+'__'))
            if remaining>0:ready.append((remaining,model))
        if not ready:
            if a.validate_only or a.model:break
            if len(list((HERE/'claims').iterdir()))>=500:break
            if time.monotonic()-waiting_since>1200:
                print('WAIT_FOR_VALIDATION_TIMEOUT',flush=True);break
            time.sleep(10);continue
        waiting_since=time.monotonic()
        model=max(ready,key=lambda p:p[0])[1]
        log=HERE/'logs'/f"{os.environ.get('SLURM_JOB_ID')}_{os.environ.get('SLURMD_NODENAME')}_{rank}_{model}_{time.time_ns()}.log"
        cmd=[str(ENVS[model]),str(HERE/'worker.py'),'--model',model]
        if a.validate_only:cmd.append('--validate-only')
        with log.open('w') as f:rc=subprocess.call(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        if rc:print('WORKER_FAILED',model,rc,str(log),flush=True);break
        if a.model or a.validate_only:break
    print('SUPERVISOR_DONE',rank,flush=True)
if __name__=='__main__':main()

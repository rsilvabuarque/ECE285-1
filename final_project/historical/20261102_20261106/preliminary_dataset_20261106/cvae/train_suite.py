"""Claim independent small fits on one scheduler-assigned GPU; preserve failures."""
import json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
tasks=[]
for seed in [20261106,20261107,20261108]:
    tasks.extend((teacher,kind,seed,'main') for teacher,kind in [('uma','cvae'),('uma','composition'),('uma','vae'),('mace','cvae')])
tasks.extend([('uma','cvae',20261106,'composition')])
claims=HERE/'run_claims';claims.mkdir(exist_ok=True)
rank=int(os.environ.get('SLURM_PROCID',0));tasks=tasks[rank%len(tasks):]+tasks[:rank%len(tasks)]
for teacher,kind,seed,split in tasks:
    ready=ROOT/'validation'/f'scf_{teacher}.json'
    while not ready.exists():time.sleep(20)
    assert json.loads(ready.read_text())['status']=='PASS'
    name=f'{teacher}_{kind}_{split}_{seed}'
    try:(claims/name).mkdir()
    except FileExistsError:continue
    log=ROOT/'logs'/f'train_{name}.log'
    with log.open('w') as f:
        subprocess.run([sys.executable,str(HERE/'train.py'),'--teacher',teacher,'--kind',kind,'--seed',str(seed),'--split',split],stdout=f,stderr=subprocess.STDOUT,check=True)
    print('FIT_COMPLETE',name,flush=True)

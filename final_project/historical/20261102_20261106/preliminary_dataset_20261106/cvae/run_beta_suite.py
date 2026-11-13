import os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
rank=int(os.environ.get('SLURM_PROCID',0));seeds=[20261106,20261107,20261108];seeds=seeds[rank%3:]+seeds[:rank%3]
claims=HERE/'run_claims';claims.mkdir(exist_ok=True)
for seed in seeds:
    name=f'uma_cvae_beta01_main_{seed}'
    try:(claims/name).mkdir()
    except FileExistsError:continue
    with (ROOT/'logs'/f'train_{name}.log').open('w') as f:subprocess.run([sys.executable,str(HERE/'train_beta.py'),'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,check=True)
    print('BETA_FIT_COMPLETE',name,flush=True)

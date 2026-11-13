"""Allocation-only short model fit, reload and exact-count sampling checks."""
import json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
rank=int(os.environ['SLURM_PROCID'])
kind=['dense_bce','dense_count','ar_cvae','ar_baseline'][rank]
out=HERE/'smoke'/kind
cmd=[sys.executable,str(HERE/'train.py'),'--data',str(HERE/'training_original.npz'),'--parent',str(HERE.parent/'data/parents/O3_N48_pilot.npz'),'--output-dir',str(out),'--kind',kind,'--epochs','2','--minimum-epochs','2','--max-seconds','120','--batch-size','256']
subprocess.run(cmd,check=True)
import torch
from models import make_model
torch.set_num_threads(4);ck=torch.load(out/'best.pt',map_location='cuda',weights_only=False);m=make_model(kind).cuda();m.load_state_dict(ck['state_dict']);m.eval()
for n in [0,12,24,36,48]:
 c=torch.tensor([[n/48,1.]]*32,device='cuda');x=m.sample(c,n)
 assert torch.isfinite(x).all() and (x.sum(1)==n).all()
(out/'sampling_check.json').write_text(json.dumps({'status':'PASS','kind':kind,'counts':[0,12,24,36,48],'samples_per_count':32})+'\n')
print('SMOKE_PASS',kind,flush=True)

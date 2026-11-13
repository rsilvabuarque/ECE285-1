"""Exercise collaborator CPU training and sampling in an allocation, not login."""
import json,os,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;PACK=HERE/'collaborator_package';PY=HERE.parent/'energy_ranking_audit/multimodel_20261104/envs/uma/bin/python'
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
out=HERE/'validation/portable_cpu_smoke';out.mkdir(exist_ok=True)
checks=[]
for name,args in [('mask',['train.py','--teacher','uma','--kind','cvae']),('geometry',['train_geometry.py','--kind','cvae'])]:
    target=out/name
    with (out/(name+'.log')).open('w') as f:subprocess.run([str(PY),str(PACK/'cvae'/args[0]),*args[1:],'--data-root',str(PACK),'--output-dir',str(target),'--device','cpu','--threads','4','--epochs','2'],stdout=f,stderr=subprocess.STDOUT,check=True)
    assert (target/'best.pt').exists() and (target/'summary.json').exists();checks.append(name+' CPUfit+checkpoint')
with (out/'sample.log').open('w') as f:subprocess.run([str(PY),str(HERE/'cvae/portable_sample.py'),'--checkpoint',str(out/'mask/best.pt'),'--n-li','24','--count','32','--sampling','conditional','--device','cpu','--output',str(out/'samples.npz')],stdout=f,stderr=subprocess.STDOUT,check=True)
import numpy as np
with np.load(out/'samples.npz') as z:assert z['occupancy'].shape==(32,48) and (z['occupancy'].sum(1)==24).all()
checks.append('CPUcheckpoint reload + exactcount sample')
(HERE/'validation/portable.json').write_text(json.dumps(dict(status='PASS',checks=checks,scope='Two-epochoperabilitytest, notconvergenceor scientificperformance. PyTorchCPUtestedusinginstalled2.13; otherplatformversionsnotverified.'),indent=2)+'\n');print('PORTABLE_CPU_SMOKE_PASS',flush=True)

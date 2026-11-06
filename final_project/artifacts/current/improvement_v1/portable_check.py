"""Fit/reload from a relocated learning-only copy inside the compute allocation."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
dest=HERE/'validation_export_copy';dest.mkdir(exist_ok=False);(dest/'improvement_v1').mkdir();(dest/'data/parents').mkdir(parents=True)
source={HERE/'models.py':dest/'improvement_v1/models.py',HERE/'train.py':dest/'improvement_v1/train.py',HERE/'training_original.npz':dest/'improvement_v1/training_original.npz',HERE.parent/'data/parents/O3_N48_pilot.npz':dest/'data/parents/O3_N48_pilot.npz'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for src,target in source.items():shutil.copy2(src,target);assert sha(src)==sha(target)
subprocess.run([sys.executable,'improvement_v1/train.py','--data','improvement_v1/training_original.npz','--kind','ar_cvae','--device','cpu','--epochs','2','--minimum-epochs','2','--output-dir','runs/cpu_smoke'],cwd=dest,check=True)
code="""import json,torch
from models import make_model
ck=torch.load('../runs/cpu_smoke/best.pt',map_location='cpu',weights_only=False)
m=make_model('ar_cvae');m.load_state_dict(ck['state_dict']);m.eval()
for n in [12,24,36]:
 c=torch.tensor([[n/48,1.]]*100);x=m.sample(c,n)
 assert torch.isfinite(x).all() and (x.sum(1)==n).all()
print('RELOCATED_RELOAD_SAMPLE_PASS')
"""
subprocess.run([sys.executable,'-c',code],cwd=dest/'improvement_v1',check=True)
result={'status':'PASS','relocated_root':str(dest),'checks':['Learning-only source/data relocation','CPU two-epoch AR-CVAE fit','Checkpoint CPU reload','100exact-count samples each atLi12/24/36'],'copied_hashes':{str(src.relative_to(HERE.parent)):sha(src) for src in source},'scope':'Portable operation, not scientific model convergence.'}
(HERE/'data/portable_relocated.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

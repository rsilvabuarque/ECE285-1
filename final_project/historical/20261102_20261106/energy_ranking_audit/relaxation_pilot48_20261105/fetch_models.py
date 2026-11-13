"""Pin new checkpoint files; reuse the three already audited checkpoints."""
from pathlib import Path
import hashlib,json,urllib.request
from huggingface_hub import HfApi,snapshot_download
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'multimodel_20261104'
paths=json.loads((OLD/'checkpoints/paths.json').read_text())
repo='materialyze/CHGNet-PES-MatPES-PBE-1M-2026.9'
revision=HfApi().model_info(repo).sha
paths['chgnet']=snapshot_download(repo_id=repo,revision=revision,local_dir=HERE/'checkpoints/chgnet',allow_patterns=['*.json','*.pt'])
url='https://github.com/MDIL-SNU/SevenNet/releases/download/v0.12.0.cp/checkpoint_sevennet_omni.pth'
dest=HERE/'checkpoints/checkpoint_sevennet_omni.pth'
if not dest.exists():
    tmp=dest.with_suffix('.download');urllib.request.urlretrieve(url,tmp);tmp.replace(dest)
paths['sevennet']=str(dest)
(HERE/'checkpoints/paths.json').write_text(json.dumps(paths,indent=2)+'\n')
checks={}
for name,p in paths.items():
    p=Path(p);fs=[p] if p.is_file() else sorted(q for q in p.iterdir() if q.is_file() and q.suffix in ['.pt','.json'])
    checks[name]={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in fs}
(HERE/'checkpoints/receipt.json').write_text(json.dumps(dict(files=checks,chgnet_revision=revision,sevennet_url=url),indent=2)+'\n')
print('CHECKPOINTS_READY',list(paths),flush=True)

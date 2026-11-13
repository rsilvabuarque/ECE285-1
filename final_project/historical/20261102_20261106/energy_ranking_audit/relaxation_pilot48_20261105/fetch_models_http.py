"""Pinned direct HTTPS downloads after the Hub client's stalled local-cache attempt."""
from pathlib import Path
import hashlib,json,requests
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'multimodel_20261104'
paths=json.loads((OLD/'checkpoints/paths.json').read_text())
repo='materialyze/CHGNet-PES-MatPES-PBE-1M-2026.9'
r=requests.get('https://huggingface.co/api/models/'+repo,timeout=30);r.raise_for_status();revision=r.json()['sha']
def fetch(url,dest):
    dest=Path(dest);dest.parent.mkdir(exist_ok=True,parents=True);tmp=dest.with_suffix(dest.suffix+'.partial')
    with requests.get(url,stream=True,timeout=(20,60)) as r:
        r.raise_for_status()
        with tmp.open('wb') as out:
            for chunk in r.iter_content(1024*1024):out.write(chunk)
    tmp.replace(dest);print('DOWNLOADED',dest.name,dest.stat().st_size,flush=True)
for name in ['model.json','model.pt','state.pt']:
    fetch(f'https://huggingface.co/{repo}/resolve/{revision}/{name}',HERE/'checkpoints/chgnet'/name)
paths['chgnet']=str(HERE/'checkpoints/chgnet')
url='https://github.com/MDIL-SNU/SevenNet/releases/download/v0.12.0.cp/checkpoint_sevennet_omni.pth'
dest=HERE/'checkpoints/checkpoint_sevennet_omni.pth';fetch(url,dest);paths['sevennet']=str(dest)
checks={}
for name,p in paths.items():
    p=Path(p);fs=[p] if p.is_file() else sorted(q for q in p.iterdir() if q.is_file() and q.suffix in ['.pt','.json'])
    checks[name]={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in fs}
(HERE/'checkpoints/paths.json').write_text(json.dumps(paths,indent=2)+'\n')
(HERE/'checkpoints/receipt.json').write_text(json.dumps(dict(files=checks,chgnet_revision=revision,sevennet_url=url),indent=2)+'\n')
print('CHECKPOINTS_READY',list(paths),flush=True)

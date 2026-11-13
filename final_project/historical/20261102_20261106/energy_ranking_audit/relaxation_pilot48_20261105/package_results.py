"""Bundle scientific inputs, outputs and reproducibility metadata; omit environments/weights."""
from pathlib import Path
import json,os,zipfile
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
assert json.loads((HERE/'independent_validation.json').read_text())['status']=='PASS'
assert json.loads((HERE/'completion.json').read_text())['terminal_tasks']==500
dest=HERE/'relaxation_pilot48_results.zip'
files=[p for p in HERE.iterdir() if p.is_file() and p.suffix in ['.py','.json','.csv','.md','.pdf','.png'] and p.name not in ['resources.json','DELIVERY_RECEIPT.json']]
files.append(HERE/'parent_POSCAR')
for sub in ['inputs','results','validation','workers','executed_sources','logs']:
    files.extend(p for p in (HERE/sub).rglob('*') if p.is_file())
files.extend([HERE/'checkpoints/paths.json',HERE/'checkpoints/receipt.json'])
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as archive:
    for p in sorted(set(files)):archive.write(p,p.relative_to(HERE))
with zipfile.ZipFile(dest) as archive:assert archive.testzip() is None
print('PACKAGED',len(set(files)),dest.stat().st_size,flush=True)

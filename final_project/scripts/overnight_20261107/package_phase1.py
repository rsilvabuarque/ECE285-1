"""Curate reusable run inputs; large/raw outputs and new weights stay outside Git."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
from cathode_cvae.data import sha256
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;run=root/'runs/overnight_20261107';out=root/'artifacts/overnight_20261107';out.mkdir(parents=True,exist_ok=True)
def cp(source,relative):
 target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
for name in ['condition_schema_uma.json','condition_schema_mace.json','training_keys.json','history_assessment.json','receipt.json']:
 cp(run/'data'/name,'data/'+name)
cp(run/'data/reference_scored.npz','data/reference.npz')
for part in ['benchmark','demo']:
 for name in ['attempts.csv','candidates.npz']:
  cp(run/part/name,f'{part}/{name}')
 for model in ['uma','mace']:
  cp(run/'collected_final'/f'{model}_with_provenance.npz',f'{part}/scores_{model}.npz')
 receipt='freeze_receipt.json' if part=='benchmark' else 'generation_receipt.json';cp(run/part/receipt,f'{part}/{receipt}')
for fit in [20261106,20261107,20261108]:
 for name in ['best.pt','configuration.json','history.csv','summary.json','condition_schema.json']:
  cp(run/f'fits/multibin_v2_{fit}'/name,f'fits/multibin_v2_{fit}/{name}')
for p in (run/'allocation').iterdir():
 if p.is_file():cp(p,'allocation/'+p.name)
for part in ['benchmark','demo']:
 for p in (run/'evaluation'/part).rglob('*'):
  if p.is_file():cp(p,'evaluation/'+part+'/'+str(p.relative_to(run/'evaluation'/part)))
cp(run/'generation_attempt0_failure.json','generation_attempt0_failure.json')
protocol=json.loads((run/'execution.json').read_text())
for field in ['status','phase2_started','owned_jobs','phase1_gate']:
 protocol.pop(field,None)
protocol['note']='Frozen experiment design. Completion and final accounting are recorded in the Git-tracked phase1_gate.json and cost_receipt.json beside this packet.'
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
# A previous packaging attempt may have contained a live status snapshot.
# Preserve that snapshot in the raw run, rather than publishing stale state.
if (out/'execution.json').exists():
 cp_source=out/'execution.json'
 shutil.copy2(cp_source,run/'packaged_execution_snapshot_attempt0.json')
 cp_source.unlink()
for name in ['pilot_uma','pilot_mace']:
 for p in (run/name).glob('*.json'):cp(p,f'pilot/{name}/{p.name}')
files={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['manifest.json','download.json']}
manifest=dict(schema_version=1,base_commit='7f7485bda6aaa441b8bd88eebc6cbd64210dccc3',branch='overnight/multibin-acquisition-20261107',files=files,immutable_historical_snapshot='artifacts/current/SNAPSHOT_MANIFEST.json',scope='Phase1 measured multibin/acquisition run; new CVAE checkpoints delivered as external release assets, not Git blobs; raw EFS shards remain HPC',raw_EFS_root=str(run/'oracle'),source_hashes={str(p.relative_to(root)):sha256(p) for p in sorted((root/'src').rglob('*.py'))})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=run/'ECE285_multibin_20261107.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for name in files:z.write(out/name,name)
 z.write(out/'manifest.json','manifest.json')
checksum=sha256(archive);(archive.with_suffix('.zip.sha256')).write_text(checksum+'  '+archive.name+'\n')
url='https://github.com/rsilvabuarque/ECE285-1/releases/download/ece285-midterm-20261107/'+archive.name
(out/'download.json').write_text(json.dumps(dict(url=url,sha256=checksum,extract_to='artifacts/overnight_20261107',bytes=archive.stat().st_size,files=len(files)+1),indent=2)+'\n')
print(json.dumps(dict(archive=str(archive),sha256=checksum,bytes=archive.stat().st_size,files=len(files)+1),indent=2))

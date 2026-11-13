"""Freeze a full source snapshot with only a claim-enumeration repair."""
from pathlib import Path
import argparse,hashlib,json,shutil
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True);p.add_argument('--helper',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
shutil.copytree(a.source_root,a.output_root,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
target=a.output_root/'src/cathode_cvae/adaptive_oracle.py';old=target.read_text();before="candidates=[t for t in tasks if t['model']==model and not (root/'claims'/t['task_id']).exists()]";after="candidates=unclaimed_tasks(tasks,root/'claims',model)";imp='from .oracle_queue import unclaimed_tasks\n'
if old.count(before)!=1 or imp in old:raise ValueError('Unexpected source candidate filter; refusing adaptive edit')
new=old.replace('from .mlip.registry import MODELS\n','from .mlip.registry import MODELS\n'+imp).replace(before,after)
if new.replace(imp,'').replace(after,before)!=old:raise ValueError('Changes extend beyond import and queue enumeration')
target.write_text(new);shutil.copyfile(a.helper,a.output_root/'src/cathode_cvae/oracle_queue.py')
oldfiles={str(q.relative_to(a.source_root)):sha(q) for q in a.source_root.rglob('*.py')};newfiles={str(q.relative_to(a.output_root)):sha(q) for q in a.output_root.rglob('*.py')};changed=[q for q in newfiles if oldfiles.get(q)!=newfiles[q]]
if sorted(changed)!=['src/cathode_cvae/adaptive_oracle.py','src/cathode_cvae/oracle_queue.py']:raise ValueError('Unexpected source changes')
receipt=dict(status='FROZEN_OPERATIONAL_QUEUE_ENUMERATION_REPAIR',old_source=str(a.source_root),new_source=str(a.output_root),changed_source_paths=changed,old_adaptive_oracle_sha256=sha(a.source_root/'src/cathode_cvae/adaptive_oracle.py'),new_adaptive_oracle_sha256=sha(target),helper_sha256=sha(a.helper),inference_body_unchanged=True,registry_inputs_checkpoints_unchanged=True,policy='Snapshot all claim names once per loop; same model/task filter, randomized taskchoice, atomicmkdir and FileExistsError handling; no failedclaim reclamation. No energybased selection.',source_files=newfiles,freezer_source_sha256=sha(__file__))
with a.manifest.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='source_files'},indent=2))

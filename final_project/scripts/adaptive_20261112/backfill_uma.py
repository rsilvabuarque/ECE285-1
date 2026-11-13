"""Administrative backfill; immutable scientific registry and existing claims remain unchanged."""
from pathlib import Path
import argparse,copy,importlib.util,json,time
r=Path(__file__).parent
spec=importlib.util.spec_from_file_location('frozen_driver',r/'orchestrate.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
p=argparse.ArgumentParser();p.add_argument('--phase',required=True);a=p.parse_args();d.require_compute_allocation()
root=r/a.phase/'oracle_mace';reg=json.loads((root/'registry.json').read_text())
assert (root/'STOP').exists(),'MACE workers must be told complete before their slots are backfilled'
assert all((root/'scores/mace'/(t['task_id']+'.json')).exists() for t in reg['tasks'])
collection=r/a.phase/'backfill_mace_collection';v=d.oracle.collect(root,collection)
assert v['status']=='PASS','Strict collector gate failed'
old=r/'taskfiles'/(a.phase+'_oracle.json');tasks=json.loads(old.read_text())['tasks'];original=next(t for t in tasks if t['id']=='uma_00');supp=[]
for i in range(9):
 t=copy.deepcopy(original);t['id']=f'uma_backfill_{i:02d}';t['bindings']+=[d.binding(__file__),d.binding(collection/'receipt.json')];supp.append(t)
new=d.freeze_tasks(a.phase+'_uma_backfill',supp)
d.write(r/a.phase/'backfill_receipt.json',dict(status='FROZEN_ADMINISTRATIVE_UMA_BACKFILL',created_unix=time.time(),reason='Completed MACE slots now process existing pending UMA physical keys; no new scientific inputs or query budget',original_tasks=d.binding(old),supplemental_tasks=d.binding(new),mace_collector_pass=d.binding(collection/'receipt.json'),source=d.binding(__file__),registry_uma=d.binding(r/a.phase/'oracle_uma/registry.json'),registry_mace=d.binding(root/'registry.json')))
d.start_tasks(new)

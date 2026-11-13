"""Publish existing frozen core tasks to the explicitly handed-off second GPU group."""
from pathlib import Path
import argparse,copy,importlib.util,json,time
r=Path(__file__).parent
spec=importlib.util.spec_from_file_location('frozen_core_driver',r/'orchestrate.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
p=argparse.ArgumentParser();p.add_argument('--task-file',type=Path,required=True);a=p.parse_args();d.require_compute_allocation()
acceptance=r/'mace_group_handoff_acceptance.json';v=json.loads(acceptance.read_text());assert v['status']=='SOLE_MACE_GROUP_PUBLISHER_ACCEPTED'
source=a.task_file
while not source.exists() or json.loads((d.RUN/'controls/uma.json').read_text())['task_file']!=str(source):
    d.assert_deadline();time.sleep(2)
original=json.loads(source.read_text())
assert source.parent==r/'taskfiles' and source.name.startswith('round2_')
if source.name.endswith('_oracle.json'):
    tasks=[]
    for task in original['tasks']:
        t=copy.deepcopy(task);t['id']='lent_'+t['id'];t['bindings']+=[d.binding(acceptance),d.binding(__file__)];tasks.append(t)
    target=d.freeze_tasks(source.stem+'_lent',tasks)
else:
    assert source.name.endswith('_generation.json') or source.name.endswith('_fits.json')
    target=source
control=d.RUN/'controls/mace.json';previous=json.loads(control.read_text())
receipt=r/'lent_publications'/(source.stem+'.json')
d.write(receipt,dict(status='FROZEN_LENT_RESOURCE_PUBLICATION',created_unix=time.time(),ownership_acceptance=d.binding(acceptance),source=d.binding(__file__),original_tasks=d.binding(source),target_tasks=d.binding(target),previous_control_value=previous,new_control_value={'task_file':str(target)},scientific_input_or_budget_changes=False,claim_semantics='Original taskfile claims shared for fits/generation; distinct modelworker claims share exact physicaloracle atomicclaims forSCFs'))
d.atomic(control,{'task_file':str(target)});d.log('PUBLISHED_LENT_MACE',str(target))

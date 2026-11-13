"""Gate the selected cell methods on new parents before producing geometry labels."""
import json,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
choices=json.loads((HERE/'data/optimizer_selection.json').read_text())['choices']
while True:
    rows=[json.loads(f.read_text()) for f in (HERE/'data/confirm').glob('*/*.json')]
    if len(rows)==100:break
    print('CONFIRMATION_WAIT',len(rows),flush=True);time.sleep(20)
counts={m:sum(r['model']==m and r['converged'] for r in rows) for m in choices};assert all(v>=18 for v in counts.values()),counts
fixed=[json.loads(s) for s in (HERE/'data/fixed_tasks.jsonl').read_text().splitlines()];tasks=[]
for t in fixed:
    a=choices[t['model']]['selected'];t=t.copy();t.update(task_id=t['task_id'].replace('fixed__','variable__',1),stage='variable',cell=True,optimizer=a['optimizer'],factor=a['factor']);tasks.append(t)
path=HERE/'data/variable_tasks.jsonl';tmp=path.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(t)+'\n' for t in tasks));tmp.replace(path)
(HERE/'data/variable_gate.json').write_text(json.dumps({'status':'PASS','new_parent_successes_out_of20':counts,'criterion':'Atleast18/20newparentsphysicalforce/stressconvergedwithin500; heuristicproductionreadinessgate, notphysicalaccuracyvalidation.','selected_methods':{m:v['selected'] for m,v in choices.items()},'tasks':len(tasks),'independent_parents':1000,'failures':'Anyproductionstep-limited/exceptioncasesstayflagged; noautomaticcriterionrelaxation.'},indent=2)+'\n')
print('VARIABLE_PREPARED',len(tasks),counts,flush=True)

"""Prepare operating-point pilots only for models with completed numerical checks."""
import json,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
cfg=json.loads((HERE/'config.json').read_text());source=[json.loads(s) for s in (HERE/'data/md_gate_tasks.jsonl').read_text().splitlines()]
while True:
    path=HERE/'validation/md_gate_partial.json'
    if path.exists():
        gate=json.loads(path.read_text());tasks=[]
        for t in source:
            q=gate['qualifications'].get(t['model'])
            if not q or q['status']!='NUMERICALLY_QUALIFIED':continue
            for i,condition in enumerate(cfg['MD_targets']):
                task=dict(t);task.update(task_id=f"md_pilot__{t['model']}__{t['structure_id']}__condition{i}",stage='md_pilot',protocol='npt',T_K=condition['T_K'],P_MPa=condition['P_MPa'],dt_fs=q['dt_fs'],warm_ps=1.,duration_ps=1.,seed=20261106+100*t['n_Li']+i)
                tasks.append(task)
        if tasks:
            out=HERE/'data/md_pilot_tasks.jsonl';tmp=out.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(t)+'\n' for t in tasks));tmp.replace(out)
            print('PILOTS_READY',len(tasks),flush=True)
        if gate['status']=='COMPLETE':break
    time.sleep(30)

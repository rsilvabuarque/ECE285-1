"""Audit logical relaxations, resolving infrastructure retries without energy selection."""
import csv,json,os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from ase.stress import voigt_6_to_full_3x3_stress
HERE=Path(__file__).resolve().parent

def audit(item):
    stage,t,retry=item
    options=[(stage,t)]+([('fixed_resume1',retry[t['task_id']])] if t['task_id'] in retry else [])
    for folder,attempt in options:
        base=HERE/'data'/folder/t['model']/attempt['task_id']
        if not base.with_suffix('.json').exists():continue
        r=json.loads(base.with_suffix('.json').read_text())
        if r['status']=='exception':continue
        break
    else:raise RuntimeError('Missing logical terminal: '+t['task_id'])
    for key in ['task_id','model','input_file','input_sha256','optimizer','factor']:
        assert r[key]==attempt[key],(key,t['task_id'])
    assert r['structure_id']==t['structure_id']
    with np.load(base.with_suffix('.npz')) as z:
        for k in ['positions_A','forces_eV_A','stress_eV_A3','cell_A','energy_eV','trace']:assert np.isfinite(z[k]).all(),(k,t['task_id'])
        assert (np.linalg.det(z['cell_A'])>0).all()
        assert z['steps'][0]==0 and z['steps'][-1]==r['steps'] and (np.diff(z['steps'])>0).all()
        assert z['trace'].shape==(r['steps']+1,7) and np.array_equal(z['trace'][:,0],np.arange(r['steps']+1))
        fm=float(np.linalg.norm(z['forces_eV_A'][-1],axis=1).max())
        sm=float(abs(np.linalg.eigvalsh(voigt_6_to_full_3x3_stress(z['stress_eV_A3'][-1]))).max()*160.21766208)
        expected=fm<.05 and (not t['cell'] or sm<.1)
        assert r['converged']==expected and abs(fm-r['final_fmax_eV_A'])<1e-6 and abs(sm-r['final_stress_max_GPa'])<1e-5
        if not expected:assert r['steps']==t['max_steps']
        assert np.array_equal(z['positions_A'][0],z['initial_positions_A']) and np.array_equal(z['cell_A'][0],z['initial_cell_A'])
        if not t['cell']:assert np.array_equal(z['cell_A'][-1],z['initial_cell_A'])
        assert len(np.unique(z['parent_slot']))==len(z['numbers'])
        snapshots=len(z['steps'])
    return dict(cohort=stage,model=t['model'],structure_id=t['structure_id'],n_Li=t['n_Li'],split=t['split'],logical_task_id=t['task_id'],selected_attempt=attempt['task_id'],status=r['status'],steps=r['steps'],calculator_calls=r['calculator_calls'],wall_seconds=r['wall_seconds'],calculator_seconds=r['calculator_seconds'],final_fmax_eV_A=fm,final_stress_max_GPa=sm,energy_change_eV=r['final_energy_eV']-r['initial_energy_eV'],volume_ratio=r['volume_ratio'],snapshots=snapshots)

def main():
    assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
    retry={t['logical_task_id']:t for t in map(json.loads,(HERE/'data/fixed_resume1_tasks.jsonl').read_text().splitlines())}
    items=[]
    for stage in ['fixed','variable']:
        items.extend((stage,t,retry if stage=='fixed' else {}) for t in map(json.loads,(HERE/'data'/f'{stage}_tasks.jsonl').read_text().splitlines()))
    with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(audit,items))
    with (HERE/'reports/relaxation_attempts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for stage in ['fixed','variable']:
        for m in sorted({r['model'] for r in rows}):
            rr=[r for r in rows if r['cohort']==stage and r['model']==m]
            summary.append(dict(cohort=stage,model=m,attempts=len(rr),converged=sum(r['status']=='converged' for r in rr),step_limit=sum(r['status']=='step_limit' for r in rr),median_steps=float(np.median([r['steps'] for r in rr])),median_wall_seconds=float(np.median([r['wall_seconds'] for r in rr])),median_final_fmax_eV_A=float(np.median([r['final_fmax_eV_A'] for r in rr])),median_final_stress_GPa=float(np.median([r['final_stress_max_GPa'] for r in rr]))))
    with (HERE/'reports/relaxation_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    receipt=dict(status='PASS',logical_tasks=len(rows),snapshots=sum(r['snapshots'] for r in rows),counts=dict(Counter(r['cohort']+':'+r['model']+':'+r['status'] for r in rows)),criteria={'force_eV_A':.05,'variable_max_abs_principal_stress_GPa':.1,'step_cap':500},selection='First valid original terminal else explicitly linked infrastructure retry; no energy-based attempt selection.',scope='Numerical and trajectory-bookkeeping audit, not DFT accuracy or phase validation.')
    (HERE/'validation/relaxations_logical.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()

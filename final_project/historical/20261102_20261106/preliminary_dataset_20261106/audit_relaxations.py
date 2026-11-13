"""Independently verify physical stopping and sampled trajectory bookkeeping."""
import argparse,json,os
from pathlib import Path
import numpy as np
from ase.stress import voigt_6_to_full_3x3_stress
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);a=ap.parse_args()
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
tasks=[json.loads(s) for s in (HERE/'data'/f'{a.stage}_tasks.jsonl').read_text().splitlines()];counts={};frames=0
for t in tasks:
    base=HERE/'data'/a.stage/t['model']/t['task_id'];r=json.loads(Path(str(base)+'.json').read_text());assert r['status']!='exception'
    for key in ['task_id','model','input_file','optimizer','factor']:assert r[key]==t[key]
    with np.load(Path(str(base)+'.npz')) as z:
        assert np.isfinite(z['positions_A']).all() and np.isfinite(z['forces_eV_A']).all() and np.isfinite(z['stress_eV_A3']).all()
        assert (np.linalg.det(z['cell_A'])>0).all();assert z['steps'][0]==0 and z['steps'][-1]==r['steps'] and (np.diff(z['steps'])>0).all()
        assert z['trace'].shape==(r['steps']+1,7);assert np.array_equal(z['trace'][:,0],np.arange(r['steps']+1))
        fm=float(np.linalg.norm(z['forces_eV_A'][-1],axis=1).max());sm=float(abs(np.linalg.eigvalsh(voigt_6_to_full_3x3_stress(z['stress_eV_A3'][-1]))).max()*160.21766208)
        expected=fm<.05 and (not t['cell'] or sm<.1);assert r['converged']==expected
        assert abs(fm-r['final_fmax_eV_A'])<1e-6 and abs(sm-r['final_stress_max_GPa'])<1e-5
        if not expected:assert r['steps']==t['max_steps']
        assert np.array_equal(z['positions_A'][0],z['initial_positions_A']);assert np.array_equal(z['cell_A'][0],z['initial_cell_A'])
        if not t['cell']:assert np.array_equal(z['cell_A'][-1],z['initial_cell_A'])
        assert len(np.unique(z['parent_slot']))==len(z['numbers']);frames+=len(z['steps'])
    key=f"{t['model']}:{r['status']}";counts[key]=counts.get(key,0)+1
(HERE/'validation'/f'{a.stage}.json').write_text(json.dumps({'status':'PASS','tasks':len(tasks),'snapshots':frames,'status_counts':counts,'physicalcriteria':{'force_eV_A':.05,'stress_GPa':.1},'scope':'Numerical/provenancevalidation; notDFTaccuracy,phasestabilityorMDadequacy'},indent=2)+'\n')
print('RELAXATION_AUDIT_PASS',a.stage,len(tasks),counts,flush=True)

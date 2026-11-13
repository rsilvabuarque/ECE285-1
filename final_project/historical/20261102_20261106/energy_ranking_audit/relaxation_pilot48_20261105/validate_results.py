"""Independent completeness, exact input, trajectory and stopping-record checks."""
import csv,hashlib,json,os
from pathlib import Path
import numpy as np
from ase.io import read
from ase.io.trajectory import Trajectory
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
cfg=json.loads((HERE/'config.json').read_text());rows=list(csv.DictReader((HERE/'manifest.csv').open()))
assert len(rows)==100 and len({r['canonical_mask'] for r in rows})==100
sy=json.loads((HERE/'parent_symmetry.json').read_text());perms=sy['permutations'];ids=set();composition={}
for r in rows:
    raw=r['vacancy_mask'];images=[]
    for p in perms:
        bits=['0']*48
        for i,j in enumerate(p):bits[j]=raw[i]
        images.append(''.join(bits))
    assert min(images)==r['canonical_mask'];ids.add(min(images));composition[int(r['n_Li'])]=composition.get(int(r['n_Li']),0)+1
assert len(ids)==100 and set(composition)==set(range(49))
checked=0;statuses={};trajframes=0
for model in cfg['models']:
    v=json.loads((HERE/'validation'/f'{model}.json').read_text());assert v['passed']
    for r in rows:
        d=HERE/'results'/model/r['structure_id'];result=json.loads((d/'result.json').read_text())
        assert result['model']==model and result['structure_id']==r['structure_id']
        assert result['input_sha256']==r['input_sha256']==hashlib.sha256(Path(r['input_file']).read_bytes()).hexdigest()
        statuses[model+':'+result['status']]=statuses.get(model+':'+result['status'],0)+1
        if result['status']=='exception':continue
        a=read(r['input_file'])
        with np.load(d/'initial.npz') as initial,np.load(d/'final.npz') as final:
            assert np.array_equal(initial['numbers'],a.numbers)
            assert np.array_equal(initial['positions_A'],a.positions)
            assert np.array_equal(initial['cell_A'],a.cell.array)
            assert np.array_equal(final['numbers'],initial['numbers'])
            if not cfg['relax_cell']:assert np.array_equal(final['cell_A'],initial['cell_A'])
            assert all(np.isfinite(final[k]).all() for k in final.files)
            fmax=float(np.linalg.norm(final['forces_eV_A'],axis=1).max())
            assert abs(fmax-result['final_fmax_eV_A'])<1e-7
            if result['converged']:assert fmax<cfg['fmax_eV_A']
            with Trajectory(str(d/'relaxation.traj')) as trajectory:
                assert len(trajectory)==result['optimizer_steps']+1
                assert np.array_equal(trajectory[0].positions,initial['positions_A'])
                assert np.array_equal(trajectory[-1].positions,final['positions_A'])
                trajframes+=len(trajectory)
        checked+=1
receipt=dict(status='PASS',terminal_records=500,successful_EFS_trajectories=checked,trajectory_frames=trajframes,status_counts=statuses,
    unique_input_orbits=100,composition_counts=composition,checks=['independent integer-mask orbit recomputation','all input hashes','exact saved initial geometry','atom conservation','fixed cell preservation','finite final arrays','force stop records','all trajectory endpoints and lengths'])
(HERE/'independent_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('VALIDATED',receipt,flush=True)

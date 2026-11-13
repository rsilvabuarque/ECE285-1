"""Discriminate image/gauge errors from actual thermal site motion, without MD reruns."""
import csv,json,os
from pathlib import Path
import numpy as np
from ase.data import atomic_masses
from ase.geometry import find_mic
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
with np.load(HERE/'data/parents/O3_N48_pilot.npz') as z:p={k:z[k] for k in z.files}
frac=p['positions_A']@np.linalg.inv(p['cell_A']);rows=[]
for m in ['uma','mace']:
    with np.load(HERE/'data'/f'geometry_{m}_md.npz') as z:d={k:z[k] for k in z.files}
    host=p['numbers']!=3;li=p['li_indices'];norm=np.linalg.norm(d['displacement_A'],axis=2)
    for n in [12,24,36,42,46]:
        for T in np.unique(d['target_temperature_K']):
            ii=(d['n_Li']==n)&(d['target_temperature_K']==T);valid_li=d['presence'][ii][:,li]
            rows.append(dict(model=m,n_Li=n,T_K=float(T),frames=int(ii.sum()),localized_fraction=float(d['localized'][ii].mean()),changed_fraction=float(d['occupancy_changed'][ii].mean()),host_displacement_RMS_A=float(np.sqrt((norm[ii][:,host]**2).mean())),Li_displacement_RMS_A=float(np.sqrt((norm[ii][:,li][valid_li]**2).mean())),maximum_host_displacement_A=float(norm[ii][:,host].max()),maximum_Li_displacement_A=float(norm[ii][:,li][valid_li].max()),mean_translation_A=float(np.linalg.norm(d['global_translation_A'][ii],axis=1).mean())))
with (HERE/'reports/MD_mapping_groups.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
tasks=list(map(json.loads,(HERE/'data/md_prod_tasks.jsonl').read_text().splitlines()));selected=[]
for m in ['uma','mace']:
    for n in [12,24,46]:
        for T in [298.15,493.15]:selected.append(next(t for t in tasks if t['model']==m and t['n_Li']==n and t['T_K']==T))
diag=[]
for t in selected:
    path=HERE/'data/md_prod'/t['model']/(t['task_id']+'.npz')
    with np.load(path) as z:d={k:z[k] for k in z.files}
    with np.load(t['seed_npz']) as z:seed={k:z[k] for k in z.files}
    slots=d['npt_parent_slot'].astype(int);nums=p['numbers'][slots];mass=atomic_masses[nums];host=nums!=3
    assert np.array_equal(slots,seed['parent_slot']) and np.array_equal(d['npt_numbers'],p['numbers'][slots])
    assert np.array_equal(d['warm_positions_A'][0],seed['positions_A'][-1])
    assert np.array_equal(d['npt_positions_A'][0],d['warm_positions_A'][-1])
    fseries=np.array([r@np.linalg.inv(H) for r,H in zip(d['npt_positions_A'],d['npt_cell_A'])]);jump=int((abs(np.diff(fseries,axis=0))>.5).sum())
    for seg,index,label in [('warm',0,'seed'),('warm',-1,'warm_end'),('npt',0,'npt_start'),('npt',-1,'npt_end')]:
        H=d[seg+'_cell_A'][index];R=d[seg+'_positions_A'][index];D=R-frac[slots]@H;t0=np.average(D,axis=0,weights=mass);old,_=find_mic(D-t0,H,pbc=True)
        df=R@np.linalg.inv(H)-frac[slots];phase=np.average(np.exp(2j*np.pi*df[host]),axis=0,weights=mass[host]);circular=np.angle(phase)/(2*np.pi);aligned,_=find_mic(D-circular@H,H,pbc=True)
        motion=R-fseries[0]@H if seg=='npt' else np.zeros_like(R);motion-=np.average(motion[host],axis=0,weights=mass[host]);motion,_=find_mic(motion,H,pbc=True)
        diag.append(dict(task_id=t['task_id'],model=t['model'],n_Li=t['n_Li'],T_K=t['T_K'],stage=label,volume_A3=float(abs(np.linalg.det(H))),volume_over_parent=float(abs(np.linalg.det(H))/abs(np.linalg.det(p['cell_A']))),old_host_RMS_A=float(np.sqrt((old[host]**2).sum(1).mean())),circular_host_RMS_A=float(np.sqrt((aligned[host]**2).sum(1).mean())),old_Li_RMS_A=float(np.sqrt((old[~host]**2).sum(1).mean())),circular_Li_RMS_A=float(np.sqrt((aligned[~host]**2).sum(1).mean())),Li_motion_since_NPT_start_RMS_A=float(np.sqrt((motion[~host]**2).sum(1).mean())),old_translation_A=t0.tolist(),circular_translation_A=(circular@H).tolist(),host_phase_concentration=abs(phase).tolist(),large_fractional_snapshot_jumps=jump))
    print('MD_MAPPING_DIAG',t['task_id'],flush=True)
with (HERE/'reports/MD_mapping_diagnostic.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(diag[0]));w.writeheader();w.writerows(diag)
(HERE/'validation/MD_mapping_review.json').write_text(json.dumps(dict(status='DIAGNOSTIC_COMPLETE_REQUIRES_INTERPRETATION',raw_cases=len(selected),stage_rows=len(diag),verified=['Raw warm initial positions exactly equal immutable relaxed seed','NPT starts at exactly warm final positions','Ancestral atom slots and species unchanged'],scope='Compare mass-COM gauge with independent periodic host registration; no coordinates/labels changed and no simulations rerun.'),indent=2)+'\n');print('MD_MAPPING_REVIEW_COMPLETE',flush=True)

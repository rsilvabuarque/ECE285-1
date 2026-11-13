"""Audit and export thermal frames without confusing ancestry with occupied sites."""
import csv,hashlib,json,os,time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from ase import Atoms
from ase.data import atomic_masses
from ase.geometry import find_mic
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
    queue=HERE/'data/md_prod_tasks.jsonl'
    while not queue.exists():time.sleep(30)
    tasks=list(map(json.loads,queue.read_text().splitlines()))
    while True:
        missing=[t['task_id'] for t in tasks if not (HERE/'data/md_prod'/t['model']/(t['task_id']+'.json')).exists()]
        if not missing:break
        print('MD_EXPORT_WAIT',len(tasks)-len(missing),len(tasks),flush=True);time.sleep(30)
    with np.load(HERE/'data/parents/O3_N48_pilot.npz') as z:p={k:z[k] for k in z.files}
    H0=p['cell_A'];frac=p['positions_A']@np.linalg.inv(H0);numbers=p['numbers'];N=len(numbers);li=p['li_indices'];inv=np.argsort(p['permutations'],axis=1);bits=np.left_shift(np.uint64(1),np.arange(47,-1,-1,dtype=np.uint64))
    host=Atoms(numbers=numbers,positions=p['positions_A'],cell=H0,pbc=True);spacing={}
    for el in np.unique(numbers):
        dd=host[numbers==el].get_all_distances(mic=True);np.fill_diagonal(dd,np.inf);spacing[int(el)]=float(dd.min())
    allout={m:{} for m in ['uma','mace']};metadata=[];summary=[];workers=set();statuses=Counter()
    for t in tasks:
        base=HERE/'data/md_prod'/t['model']/t['task_id'];r=json.loads(base.with_suffix('.json').read_text());statuses[r['status']]+=1
        assert r['status']=='complete',('PreserveandreviewterminalMDfailure',t['task_id'],r['status'])
        for k in ['task_id','model','seed_sha256','T_K','P_MPa','split']:assert r[k]==t[k]
        workers.add((t['model'],r['worker']));digest=sha(base.with_suffix('.npz'))
        with np.load(base.with_suffix('.npz')) as z:
            for seg in ['warm','npt']:
                tr=z[seg+'_trace'];expected_steps=round(t['warm_ps' if seg=='warm' else 'duration_ps']*1000/t['dt_fs']);assert r['segments'][seg]['completed_steps']==expected_steps
                assert tr.shape==(expected_steps+1,13) and np.array_equal(tr[:,0],np.arange(expected_steps+1))
                assert np.isfinite(tr[:,[0,1,2,3,4,5,6,7,8,9,11,12]]).all()
                if seg=='npt':assert np.isfinite(tr[:,10]).all()
                for k in ['positions_A','forces_eV_A','cell_A','velocities_A_fs','configurational_stress_eV_A3','total_stress_eV_A3']:assert np.isfinite(z[seg+'_'+k]).all()
            tr=z['npt_trace'];slots=z['npt_parent_slot'].astype(int);assert np.array_equal(z['npt_numbers'],numbers[slots])
            present=np.zeros(N,bool);present[slots]=True;ancestral=present[li].astype(np.uint8);assert ancestral.sum()==t['n_Li']
            last_half=tr[len(tr)//2:];summary.append(dict(task_id=t['task_id'],model=t['model'],structure_id=t['structure_id'],n_Li=t['n_Li'],split=t['split'],target_temperature_K=t['T_K'],target_pressure_MPa=t['P_MPa'],duration_ps=t['duration_ps'],mean_temperature_K=float(last_half[:,5].mean()),mean_total_pressure_MPa=float(last_half[:,6].mean()),std_total_pressure_MPa=float(last_half[:,6].std()),mean_volume_A3=float(last_half[:,8].mean()),extended_energy_range_meV_atom=float(np.ptp(tr[:,10])/len(slots)*1000),volume_min_over_seed=float(tr[:,8].min()/tr[0,8]),volume_max_over_seed=float(tr[:,8].max()/tr[0,8]),scope='Last2.5psexploratoryblock;notindependentorequilibratedthermodynamicestimate'))
            indices=np.unique(np.rint(np.linspace(1,len(z['npt_snapshot_steps'])-1,20)).astype(int))
            for idx in indices:
                H=z['npt_cell_A'][idx];step=int(z['npt_snapshot_steps'][idx]);assert step==tr[step,0]
                D=z['npt_positions_A'][idx]-frac[slots]@H;translation=np.average(D,axis=0,weights=atomic_masses[numbers[slots]]);u,_=find_mic(D-translation,H,pbc=True);deform=np.linalg.solve(H0,H);stretch=np.linalg.svd(deform,compute_uv=False).min()
                localized=all(np.linalg.norm(u[numbers[slots]==el],axis=1).max(initial=0)<.5*spacing[int(el)]*stretch for el in spacing)
                pp=np.zeros((N,3),np.float32);pp[slots]=u;ff=np.zeros((N,3),np.float32);ff[slots]=z['npt_forces_eV_A'][idx]
                lirows=np.flatnonzero(numbers[slots]==3);delta=z['npt_positions_A'][idx][lirows,None,:]-translation-frac[li][None,:,:]@H
                _,dist=find_mic(delta.reshape(-1,3),H,pbc=True);dist=dist.reshape(len(lirows),48);aa,bb=linear_sum_assignment(dist*dist);observed=np.zeros(48,np.uint8);observed[bb]=1
                assigned=dist[aa,bb];other=dist.copy();other[aa,bb]=np.inf;margin=(other.min(1)-assigned).min() if len(aa) else np.nan
                mask=1-observed;canon=int((mask[inv].astype(np.uint64)*bits).sum(1).min());changed=bool(not np.array_equal(observed,ancestral))
                S=z['npt_configurational_stress_eV_A3'][idx];P=-float(S[:3].mean())*160.21766208;tensor=np.array([[S[0],S[5],S[4]],[S[5],S[1],S[3]],[S[4],S[3],S[2]]])*160.21766208;dev=float(np.linalg.norm(tensor-np.eye(3)*np.trace(tensor)/3))
                row=dict(displacement_A=pp,forces_parent_eV_A=ff,presence=present,ancestral_occupancy=ancestral,observed_occupancy=observed,occupancy_changed=changed,assignment_max_distance_A=float(assigned.max(initial=0)),assignment_min_local_margin_A=float(margin),localized=localized,cell_A=H,deformation=deform,global_translation_A=translation,energy_eV=float(tr[step,2]),stress_eV_A3=S,total_stress_eV_A3=z['npt_total_stress_eV_A3'][idx],n_Li=t['n_Li'],target_temperature_K=t['T_K'],target_pressure_MPa=t['P_MPa'],actual_temperature_K=float(tr[step,5]),actual_total_pressure_MPa=float(tr[step,6]),force_component_rms_eV_A=float(np.sqrt((ff*ff).sum()/(3*present.sum()))),force_max_eV_A=float(np.linalg.norm(ff,axis=1).max()),configurational_pressure_GPa=P,deviatoric_stress_GPa=dev,frame_weight=1/(5*len(indices)),structure_ids=t['structure_id'],split=t['split'],cohort='NPT_exploration',step=step)
                for k,v in row.items():allout[t['model']].setdefault(k,[]).append(v)
                metadata.append(dict(model=t['model'],task_id=t['task_id'],structure_id=t['structure_id'],split=t['split'],source_npz=str(base.with_suffix('.npz')),source_sha256=digest,source_snapshot_index=int(idx),step=step,observed_canonical_vacancy_mask=f'{canon:048b}',ancestral_canonical_mask=t['canonical_mask'],occupancy_changed=changed,localized=localized))
    exports=[]
    for m,arrays in allout.items():
        path=HERE/'data'/f'geometry_{m}_md.npz';np.savez_compressed(path,**{k:np.array(v) for k,v in arrays.items()});exports.append(dict(model=m,path=str(path),frames=len(arrays['step']),sha256=sha(path),localized_frames=int(sum(arrays['localized'])),observed_mask_changed_frames=int(sum(arrays['occupancy_changed']))))
    for name,rows in [('MD_trajectory_summary.csv',summary),('MD_frame_mapping.csv',metadata)]:
        with (HERE/'reports'/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    cfg=json.loads((HERE/'config.json').read_text())
    for m,w in workers:
        r=json.loads((HERE/'workers'/f'{w}.json').read_text());prior=json.loads((Path(cfg['fixed_pilot'])/'validation'/f'{m}.json').read_text())['receipt']
        for k in ['checkpoint_hashes','dtype','tf32','package_version','torch']:assert r[k]==prior[k]
        for fn,h in r['source_hashes'].items():assert sha(HERE/fn)==h
    receipt=dict(status='PASS',tasks=len(tasks),terminal_status_counts=dict(statuses),parent_lineages=len({t['structure_id'] for t in tasks}),exports=exports,alignment='pristinefractionalsites@instantaneouscell minus globalmassCOM, thenMIC; rawcoordinatespreserved',assignment='Rectangularone-to-oneLi-to48site minimumsquaredperiodicdistance. Storelocalalternativemarginsandancestral/currentmaskdifference; negativeorzeromarginsflagambiguity, notcertifiedhopping/phaseclassifier.',weight='Eachparent/modeltotalweight1across5conditions;20framespercondition. Splitsfrozenbyancestralmasklineage.',scope='5ps explorations; noequilibrium, phasestability, optimalT/P, orDFTaccuracyclaim. Thermalframesnotfilteredbyoptimizationforcethreshold.')
    (HERE/'validation/md_export.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()

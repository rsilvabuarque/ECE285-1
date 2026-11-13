"""Export parent-aligned displacements with one lineage across all saved frames."""
import argparse,csv,hashlib,json,os,time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.data import atomic_masses
from ase.geometry import find_mic
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='uma');ap.add_argument('--cohort',choices=['fixed','variable'],default='fixed');ap.add_argument('--wait',action='store_true');a=ap.parse_args()
    assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
    tasks=[json.loads(s) for s in (HERE/'data'/f'{a.cohort}_tasks.jsonl').read_text().splitlines()];tasks=[t for t in tasks if t['model']==a.model]
    retries={}
    if a.cohort=='fixed':
        for line in (HERE/'data/fixed_resume1_tasks.jsonl').read_text().splitlines():
            t=json.loads(line);retries[t['logical_task_id']]=t
    selected=[]
    while True:
        selected=[];missing=[]
        for t in tasks:
            candidates=[(a.cohort,t)]
            if t['task_id'] in retries:candidates.append(('fixed_resume1',retries[t['task_id']]))
            found=None
            for folder,attempt in candidates:
                base=HERE/'data'/folder/a.model/attempt['task_id'];j=Path(str(base)+'.json')
                if not j.exists():continue
                r=json.loads(j.read_text())
                if r['status']=='exception':continue
                assert r['model']==t['model'] and r['structure_id']==t['structure_id'] and r['input_sha256']==t['input_sha256']
                found=(t,attempt,base,r);break
            if found is None:missing.append(t['task_id'])
            else:selected.append(found)
        if not missing:break
        print('GEOMETRY_WAIT',a.model,a.cohort,len(selected),len(tasks),flush=True)
        if not a.wait:raise RuntimeError(f'{len(missing)}logical tasks missing')
        time.sleep(30)
    with np.load(HERE/'data/parents/O3_N48_pilot.npz') as z:parent={k:z[k] for k in z.files}
    h0=parent['cell_A'];frac=parent['positions_A']@np.linalg.inv(h0);numbers=parent['numbers'];N=len(numbers)
    host=Atoms(numbers=numbers,positions=parent['positions_A'],cell=h0,pbc=True);min_reference={}
    for element in np.unique(numbers):
        d=host[numbers==element].get_all_distances(mic=True);np.fill_diagonal(d,np.inf);min_reference[int(element)]=float(d.min())
    output={k:[] for k in ['displacement_A','forces_parent_eV_A','presence','occupancy','cell_A','deformation','global_translation_A','energy_eV','stress_eV_A3','n_Li','force_component_rms_eV_A','force_max_eV_A','configurational_pressure_GPa','deviatoric_stress_GPa','localized','endpoint_converged','frame_weight','structure_ids','split','cohort','step']}
    metadata=[];logical=[]
    for task,attempt,base,r in selected:
        path=Path(str(base)+'.npz');digest=sha(path)
        logical.append(dict(logical_task_id=task['task_id'],selected_attempt=attempt['task_id'],source_npz=str(path),source_sha256=digest,status=r['status']))
        with np.load(path) as z:
            slots=z['parent_slot'].astype(int);assert len(np.unique(slots))==len(slots) and np.array_equal(z['numbers'],numbers[slots])
            presence=np.zeros(N,bool);presence[slots]=True
            occupancy=presence[parent['li_indices']].astype(np.uint8);assert occupancy.sum()==task['n_Li']
            available=np.arange(1,len(z['steps']));frames=np.unique(np.rint(np.linspace(1,len(z['steps'])-1,min(5,len(available)))).astype(int)) if len(available) else np.array([],int)
            # Fixed and variable share step0. Export deletion only once, with fixed.
            if a.cohort=='fixed':frames=np.r_[0,frames]
            if not len(frames):frames=np.array([0])
            for idx in frames:
                H=z['cell_A'][idx];F=np.linalg.solve(h0,H);D=z['positions_A'][idx]-frac[slots]@H
                translation=np.average(D,axis=0,weights=atomic_masses[numbers[slots]]);u,_=find_mic(D-translation,H,pbc=True)
                padded=np.zeros((N,3),np.float32);padded[slots]=u;forces=np.zeros((N,3),np.float32);forces[slots]=z['forces_eV_A'][idx]
                smallest_stretch=float(np.linalg.svd(F,compute_uv=False).min());localized=True
                for element,spacing in min_reference.items():
                    ids=numbers[slots]==element
                    if ids.any() and np.linalg.norm(u[ids],axis=1).max()>=.5*smallest_stretch*spacing:localized=False
                s=z['stress_eV_A3'][idx];xx,yy,zz,yz,xz,xy=s;tensor=np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])*160.21766208;hydro=tensor.trace()/3;dev=tensor-np.eye(3)*hydro
                step=int(z['steps'][idx]);cohort='deletion' if step==0 else f'{a.cohort}_endpoint' if idx==len(z['steps'])-1 else f'{a.cohort}_intermediate'
                values=dict(displacement_A=padded,forces_parent_eV_A=forces,presence=presence,occupancy=occupancy,cell_A=H,deformation=F,global_translation_A=translation,energy_eV=float(z['energy_eV'][idx]),stress_eV_A3=s,n_Li=task['n_Li'],force_component_rms_eV_A=float(np.sqrt((forces*forces).sum()/(3*presence.sum()))),force_max_eV_A=float(np.linalg.norm(forces,axis=1).max()),configurational_pressure_GPa=float(-hydro),deviatoric_stress_GPa=float(np.linalg.norm(dev)),localized=localized,endpoint_converged=r['converged'],frame_weight=1/len(frames),structure_ids=task['structure_id'],split=task['split'],cohort=cohort,step=step)
                for key,value in values.items():output[key].append(value)
                metadata.append(dict(structure_id=task['structure_id'],canonical_mask=task['canonical_mask'],split=task['split'],model=a.model,parent=task['parent'],logical_task_id=task['task_id'],selected_attempt=attempt['task_id'],source_npz=str(path),source_sha256=digest,source_frame_index=int(idx),step=step,physical_cohort=cohort,localized=localized,endpoint_converged=r['converged']))
    out=HERE/'data'/f'geometry_{a.model}_{a.cohort}.npz';tmp=out.with_suffix('.tmp.npz');np.savez_compressed(tmp,**{k:np.array(v) for k,v in output.items()});tmp.replace(out)
    for name,rows in [(f'geometry_{a.model}_{a.cohort}_frames.csv',metadata),(f'logical_completion_{a.model}_{a.cohort}.csv',logical)]:
        with (HERE/'data'/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    receipt=dict(status='PASS',model=a.model,physical_cohort=a.cohort,logical_tasks=len(selected),independent_parents=len({t[0]['structure_id'] for t in selected}),frames=len(metadata),localized_frames=int(sum(output['localized'])),sha256=sha(out),alignment='Pristinefullparentfractionalsites@framecell; subtractmassweightedglobaltranslationbeforeMIC; no rawcoordinateschanged.',localization_rule='Sufficientnearest-siteguarantee maxspeciesdisplacement <halfminimumsame-speciesparentseparation*minprincipalstretch; notphaseclassifier.',weight='Eachparentcohort totalrowweight1; cap5nonzerosteps+fixeddeletion; samefrozenmasklineagesplits.',presence='Full192parentrows, Liindicesreadfromparent, absentforces/Uzerosmaskednotlabels.',source_selection='Firstvalidoriginalterminalelseexplicitinfrastructureretry; neverbestenergyattempt.')
    (HERE/'validation'/f'geometry_{a.model}_{a.cohort}.json').write_text(json.dumps(receipt,indent=2)+'\n');print('GEOMETRY_EXPORTED',receipt,flush=True)
if __name__=='__main__':main()

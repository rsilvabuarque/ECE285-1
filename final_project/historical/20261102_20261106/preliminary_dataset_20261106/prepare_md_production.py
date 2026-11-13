"""Freeze MD parent lineages and require physical seed and integration gates."""
import csv,hashlib,json,os,time
from pathlib import Path
import numpy as np
from ase import units
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
cfg=json.loads((HERE/'config.json').read_text());manifest=list(csv.DictReader((HERE/'data/relaxation_parent_manifest.csv').open()));rng=np.random.default_rng(20261110)
selection=HERE/'data/md_parent_selection.json'
if selection.exists():parents=json.loads(selection.read_text())['parents']
else:
    parents=[]
    for n in [12,24,36,42,46]:
        for split,count in [('train',2),('val',1),('test',1)]:
            pool=[r for r in manifest if int(r['n_Li'])==n and r['split']==split];assert len(pool)>=count,(n,split,len(pool))
            parents.extend(pool[i] for i in rng.choice(len(pool),count,replace=False))
    selection.write_text(json.dumps({'parents':parents,'rule':'Fivecompositions;2train+1val+1testparentseach; uniformselectionwithinfrozensplit, beforeinspectingMDoutcomes; allT/P/modelcopieskeepthissplit.','seed':20261110},indent=2)+'\n')
extra_tasks=[json.loads(s) for s in (HERE/'data/md_extra_gate_tasks.jsonl').read_text().splitlines()];qualification={};extra_records=[]
while True:
    missing=[t['task_id'] for t in extra_tasks if not (HERE/'data/md_extra_gate'/t['model']/(t['task_id']+'.json')).exists()]
    if not missing:break
    print('EXTRA_GATE_TERMINAL_WAIT',len(missing),flush=True);time.sleep(20)
for model in ['uma','mace']:
    path=HERE/'validation/md_gate_partial.json';core=json.loads(path.read_text())['qualifications'][model];assert core['status']=='NUMERICALLY_QUALIFIED'
    all_full=core['dt_fs']==1.;all_half=True
    for t in [t for t in extra_tasks if t['model']==model]:
        base=HERE/'data/md_extra_gate'/model/t['task_id'];r=json.loads(Path(str(base)+'.json').read_text());assert r['status']=='complete'
        branches={}
        with np.load(Path(str(base)+'.npz')) as z:
            for key in ['positions_A','cell_A','velocities_A_fs']:assert np.array_equal(z['dt1_'+key][0],z['dt0p5_'+key][0])
            for name in ['dt1','dt0p5']:
                trace=z[name+'_trace'];N=len(z[name+'_numbers']);dof=r['segments'][name]['physical_degrees_of_freedom'];bound=float(.01*dof/(2*N)*units.kB*493.15*1000);E=trace[:,4]/N*1000
                ok=bool(np.ptp(E)<=bound and abs(E[-1]-E[0])<=bound);branches[name]=dict(passed=ok,range_meV_atom=float(np.ptp(E)),bound_meV_atom=bound)
            all_full=all_full and branches['dt1']['passed'];all_half=all_half and branches['dt0p5']['passed']
        extra_records.append(dict(model=model,n_Li=t['n_Li'],branches=branches))
    assert all_half,model;qualification[model]=dict(dt_fs=1. if all_full else .5,scope='Fourrepresentativecompositionsat493.15K; notglobalMD/DFTvalidation')
# NPT pilot qualification: actual total pressure need not equal its target on 1 ps.
pilot_tasks=[json.loads(s) for s in (HERE/'data/md_pilot_tasks.jsonl').read_text().splitlines()];pilot_checks=[]
while True:
    missing=[t['task_id'] for t in pilot_tasks if t['model'] in qualification and not (HERE/'data/md_pilot'/t['model']/(t['task_id']+'.json')).exists()]
    if not missing:break
    print('NPT_PILOT_TERMINAL_WAIT',len(missing),flush=True);time.sleep(20)
for model in qualification:
    tasks=[t for t in pilot_tasks if t['model']==model];assert len(tasks)==10
    for t in tasks:
        base=HERE/'data/md_pilot'/model/t['task_id'];r=json.loads(Path(str(base)+'.json').read_text());assert r['status']=='complete',r
        with np.load(Path(str(base)+'.npz')) as z:
            tr=z['npt_trace'];N=len(z['npt_numbers']);bound=float(.01*1.5*units.kB*t['T_K']*1000);extended=tr[:,10]/N*1000
            assert np.isfinite(tr).all();spread=float(np.ptp(extended));passed=spread<=bound
            pilot_checks.append(dict(model=model,task_id=t['task_id'],extended_range_meV_atom=spread,bound_meV_atom=bound,passed=bool(passed)))
            assert passed,('NPT_extended_energy_gate',model,t['task_id'],spread,bound)
tasks=[];pending=[]
while True:
    tasks=[];pending=[]
    for row in parents:
        for model,qual in qualification.items():
            sid=row['structure_id'];name=f'variable__{model}__{sid}';base=HERE/'data/variable'/model/name;j=Path(str(base)+'.json')
            if not j.exists():pending.append(name);continue
            r=json.loads(j.read_text());assert r['converged'],('SelectedMDseedneedsreview/rescue',name,r['status'])
            path=Path(str(base)+'.npz');digest=hashlib.sha256(path.read_bytes()).hexdigest()
            for i,tp in enumerate(cfg['MD_targets']):
                tasks.append(dict(task_id=f'md_prod__{model}__{sid}__condition{i}',stage='md_prod',protocol='npt',model=model,parent='O3_N48_pilot',structure_id=sid,n_Li=int(row['n_Li']),split=row['split'],canonical_mask=row['canonical_mask'],seed_npz=str(path),seed_sha256=digest,seed=20261306+100*int(row['n_Li'])+i+int(row['parent_index']),T_K=tp['T_K'],P_MPa=tp['P_MPa'],dt_fs=qual['dt_fs'],warm_ps=1.,duration_ps=5.))
    if not pending:break
    print('MD_SEEDS_WAIT',len(pending),flush=True);time.sleep(30)
assert len(tasks)==200
out=HERE/'data/md_prod_tasks.jsonl';tmp=out.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(t)+'\n' for t in tasks));tmp.replace(out)
(HERE/'validation/md_production_gate.json').write_text(json.dumps({'status':'PASS','qualifications':qualification,'extreme_NVE_cases':extra_records,'NPT_pilot_checks':pilot_checks,'tasks':len(tasks),'parent_lineages':20,'sampling':'FivecoupledT/Poperatingpoints;5psNPTafter1psNVT, exploratorygeometrydataset, notequilibriumvalidation.','splits':'10train+5val+5testparents, sharedacrossallmodels/conditions/frames.'},indent=2)+'\n')
print('MD_PRODUCTION_PREPARED',len(tasks),qualification,flush=True)

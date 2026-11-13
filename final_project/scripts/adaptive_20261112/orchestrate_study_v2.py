"""Frozen two-round adaptive study driver; only new iteration outputs and UMA pointer are written."""
from pathlib import Path
import argparse, csv, hashlib, json, os, sys, time
import numpy as np

RUN=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285/final_project/runs/adaptive_20261112')
ROOTITER=RUN/'iteration'; R=ROOTITER/'study_v2'; SRC=ROOTITER/'source/src'; sys.path.insert(0,str(SRC))
from cathode_cvae import adaptive_iteration as ai
from cathode_cvae import adaptive_oracle as oracle
from cathode_cvae.data import sha256
from cathode_cvae.iterative_boundary import binding
from cathode_cvae.conditions import load_schema
from cathode_cvae.iterative_acquisition import _parent, _canonical_vacancy
from cathode_cvae.runtime import require_compute_allocation
PROTOCOL=json.loads((RUN/'protocol.json').read_text())
PARENT=Path(PROTOCOL['parent']['path']); H0=Path(PROTOCOL['data']['path']); SCHEMA=Path(PROTOCOL['schema']['path'])
MACE_SCHEMA=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project/artifacts/overnight_20261107/data/condition_schema_mace.json')
UMA_PY='/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285/final_project/energy_ranking_audit/multimodel_20261104/envs/uma/bin/python'
MACE_PY='/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285/final_project/energy_ranking_audit/multimodel_20261104/envs/mace/bin/python'
CHECKPOINTS={'uma':'/global/cfs/cdirs/m4537/rsb/hf/hub/models--facebook--UMA/snapshots/ce3b2744ff57c0f49a010b763efcbfe9ba82e2e1/checkpoints/uma-s-1p2p1.pt', 'mace':'/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285/final_project/energy_ranking_audit/multimodel_20261104/checkpoints/mace-mh-1.model'}
SEEDS=PROTOCOL['iteration']['fit_seeds']; FAMILIES=PROTOCOL['iteration']['families']; MODES=PROTOCOL['iteration']['schema_modes']
ENV={'PYTHONPATH':str(SRC),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}


def log(*args): print(time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),*args,flush=True)
def write(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); ai.write_json(path,obj)
def atomic(path,obj):
    path=Path(path);tmp=path.with_name(path.name+'.iteration.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def assert_deadline():
    if time.time()>1794578068: raise RuntimeError('Goal deadline reached; preserve partial outputs without further tasks')
def template(taskid,argv,bindings=(),checks=(),python=UMA_PY):
    return dict(id=taskid,argv=[python,*map(str,argv)],cwd=str(R),env=ENV,
                bindings=[binding(x) for x in bindings],checks=list(checks))
def freeze_tasks(name,tasks):
    path=R/'taskfiles'/(name+'.json'); write(path,dict(status='FROZEN_TASKS_BEFORE_DISPATCH',created_unix=time.time(),driver=binding(__file__),protocol=binding(R/'design.json'),source=binding(ROOTITER/'source/receipt.json'),tasks=tasks));return path

def start_tasks(path):
    assert_deadline(); atomic(RUN/'controls/uma.json',dict(task_file=str(path)));log('DISPATCH',path.name)
def wait_tasks(path):
    tasks=json.loads(path.read_text())['tasks']; claims=path.parent/(path.stem+'.claims');last=None
    while True:
        assert_deadline();done=[];failed=[]
        for t in tasks:
            p=claims/t['id']/'receipt.json'
            if p.exists():
                v=json.loads(p.read_text());(done if v['status']=='COMPLETE' else failed).append(v)
        state=(len(done),len(failed))
        if state!=last:log('TASK_PROGRESS',path.name,state,'of',len(tasks));last=state
        if failed:raise RuntimeError('Frozen task failure(s), inspect '+str(claims)+': '+','.join(v['task']['id'] for v in failed))
        if len(done)==len(tasks):return done
        time.sleep(10)
def run_tasks(name,tasks):
    p=freeze_tasks(name,tasks);start_tasks(p);return wait_tasks(p)

def protect_latent():
    base=ROOTITER/'boundary_base'
    while not (base/'receipt.json').exists():assert_deadline();time.sleep(10)
    boundary,known,protected=ai.load_boundary(base)
    pool=RUN/'latent_diagnostics/raw_valid_canonical_union.npz'; receipt=RUN/'latent_diagnostics/raw_union_receipt.json'
    log('WAIT_LATENT_UNION',str(receipt))
    while not receipt.exists():assert_deadline();time.sleep(10)
    v=json.loads(receipt.read_text())
    if v['status']!='ALL40_RAW_DIAGNOSTICS_FROZEN_BEFORE_ORACLE' or v['pool_sha256']!=sha256(pool) or v['parent_sha256']!=sha256(PARENT):raise ValueError('Latent raw union receipt mismatch')
    perms,_=_parent(PARENT)
    with np.load(pool,allow_pickle=False) as z:
        if str(z['parent_sha256'].item())!=sha256(PARENT):raise ValueError('Latent pool parent mismatch')
        occupancy=z['occupancy']; masks=z['masks']; ids=z['structure_ids'].astype(str)
    keys=_canonical_vacancy(occupancy,perms)
    if not np.array_equal(masks,1-occupancy) or len(set(keys.tolist()))!=len(keys) or ids.tolist()!=[ai.identity(k) for k in keys]:raise ValueError('Latent union is not unique canonical masks')
    if len(keys)!=v['unique_valid_orbits']:raise ValueError('Latent union count differs')
    # Late parent clarification: existing TRAIN identities stay TRAIN, not newly protected.
    base_known=set(known);base_protected=set(protected)
    with np.load(H0,allow_pickle=False) as z:
        roles=z['split'].astype(str);train_x=z['occupancy'][roles=='train']
    train_keys=set(_canonical_vacancy(train_x,perms).tolist())
    new_protected=set(keys.tolist())-base_known-train_keys
    oldtask=ROOTITER/'taskfiles/round1_acquisition_generation.json'
    oldtasks=json.loads(oldtask.read_text())['tasks'];oldclaims=oldtask.parent/(oldtask.stem+'.claims')
    while not all((oldclaims/t['id']/'receipt.json').exists() for t in oldtasks):assert_deadline();time.sleep(10)
    old_attempts=0;old_keys=set();old_receipts=[]
    for t in oldtasks:
        rr=json.loads((oldclaims/t['id']/'receipt.json').read_text())
        if rr['status']!='COMPLETE':raise ValueError('Pre-correction generation did not complete; inspect preserved task')
        out0=Path(t['checks'][0]['path']).parent;r0=ai.validate_stage(out0)
        old_attempts+=r0['attempts'];old_keys.update(json.loads((out0/'all_generated_keys.json').read_text()));old_receipts.append(binding(out0/'receipt.json'))
    if list(ROOTITER.glob('round*/oracle*/registry.json')):raise ValueError('Pre-correction driver unexpectedly registered oracle')
    out=R/'boundary_initial';out.mkdir(exist_ok=False);known.update(keys.tolist());known.update(old_keys);protected.update(new_protected)
    if (protected-base_protected)&train_keys:raise ValueError('Newly protected H0TRAIN overlap')
    write(out/'known_history.json',sorted(known));write(out/'protected.json',sorted(protected))
    write(out/'receipt.json',dict(status='FROZEN_ADAPTIVE_CANONICAL_BOUNDARY',created_unix=time.time(),parent=boundary['parent'],history=binding(out/'known_history.json'),protected=binding(out/'protected.json'),known_orbits=len(known),protected_orbits=len(protected),previous_boundary=binding(base/'receipt.json'),latent_raw_union=binding(pool),latent_receipt=binding(receipt),latent_raw_unique_orbits=len(keys),newly_latent_protected_orbits=len(new_protected),newly_protected_h0_train_orbits=0,inherited_h0_train_protected_orbits=len(base_protected&train_keys),precorrection_raw_attempts=old_attempts,precorrection_raw_unique_orbits=len(old_keys),precorrection_stage_receipts=old_receipts,precorrection_driver=binding(ROOTITER/'orchestrate.py'),precorrection_design=binding(ROOTITER/'design.json'),correction_reason='Root clarified overbroad raw diagnostic protection: all raw known, only new eligible protected; no labels inspected. Prior12generation tasks finished, no oracle registered, every completed raw orbit added to known.',no_energy_members_read=True,source=binding(__file__)))
    log('INITIAL_BOUNDARY',len(known),len(protected),len(keys));return out

def make_stage_task(phase,id,boundary,namespace,seed,method,schema,checkpoint=None,sampler=None,attempts=512,queries=256):
    out=R/phase/'stages'/id
    args=['-m','cathode_cvae.adaptive_iteration','propose','--parent',PARENT,'--schema',schema,'--boundary-dir',boundary,'--output-dir',out,'--namespace',namespace,'--seed',seed,'--method',method,'--attempts-per-cell',attempts,'--queries-per-cell',queries,'--device','cuda','--threads','4']
    inputs=[PARENT,schema,Path(boundary)/'receipt.json',SRC/'cathode_cvae/adaptive_iteration.py']
    if checkpoint:args+=['--checkpoint',checkpoint];inputs.append(checkpoint)
    else:
        args+=['--sampler',sampler]
        if sampler=='mutation':args+=['--seed-bank',R/'seed_bank'];inputs.append(R/'seed_bank/receipt.json')
    return out,template(id,args,inputs,[dict(path=str(out/'receipt.json'),equals={'status':'FROZEN_ADAPTIVE_STAGE_BEFORE_SCORING'})])

def stages_for(phase,models,boundary,evaluation=False):
    # A common namespace pairs all neural ablations and samplers at each fit seed.
    namespace=f'adaptive_20261112_study_v2_{phase}';tasks=[];stages=[];meta=[]
    for item in models:
        out,t=make_stage_task(phase,item['id'],boundary,namespace,item['seed'],item['method'],item['schema'],checkpoint=item['checkpoint'],attempts=128 if evaluation else 512,queries=128 if evaluation else 256)
        tasks.append(t);stages.append(out);meta.append(dict(item,stage=str(out),checkpoint=str(item['checkpoint']),schema=str(item['schema'])))
    for seed in SEEDS:
        for sampler in ['uniform','mutation']:
            id=f'{sampler}_{seed}';out,t=make_stage_task(phase,id,boundary,namespace,seed,sampler,SCHEMA,sampler=sampler,attempts=128 if evaluation else 512,queries=128 if evaluation else 256)
            tasks.append(t);stages.append(out);meta.append(dict(id=id,family=sampler,seed=seed,method=sampler,mode='baseline',stage=str(out),checkpoint=None,schema=str(SCHEMA)))
    meta=[{k:str(v) if isinstance(v,Path) else v for k,v in item.items()} for item in meta]
    write(R/phase/'models.json',meta);run_tasks(phase+'_generation',tasks);return stages

def score_phase(phase,stages):
    # All stage receipts must predate all score tasks, including later stage registrations.
    for stage in stages:ai.validate_stage(stage)
    roots={m:R/phase/('oracle_'+m) for m in ['uma','mace']}
    for m,root in roots.items():
        for stage in stages:oracle.register_pool(root,stage/'candidates.npz',PARENT,models=[m],batch_size=64)
    jobs=[]
    for m,root in roots.items():
        for i in range(9):
            jobs.append(template(f'{m}_{i:02d}',['-m','cathode_cvae.adaptive_oracle','worker','--root',root,'--model',m,'--checkpoint',CHECKPOINTS[m],'--threads','4'],[root/'registry.json',SRC/'cathode_cvae/adaptive_oracle.py',SRC/'cathode_cvae/mlip/registry.py'],python=UMA_PY if m=='uma' else MACE_PY))
    path=freeze_tasks(phase+'_oracle',jobs);start_tasks(path);pending=set(roots);last={}
    while pending:
        assert_deadline()
        for m in list(pending):
            root=roots[m];reg=json.loads((root/'registry.json').read_text());tasks=reg['tasks'];expected=len(tasks)
            failures=list((root/'claims').glob('*/failure.json')) if (root/'claims').exists() else []
            if failures:raise RuntimeError('Oracle task failed '+str(failures[0]))
            receipts=list((root/'scores'/m).glob('*.json')) if (root/'scores'/m).exists() else []
            if last.get(m)!=len(receipts):log('ORACLE_PROGRESS',phase,m,len(receipts),expected);last[m]=len(receipts)
            for p in receipts:
                if json.loads(p.read_text())['status']!='complete':raise RuntimeError('Oracle structure failed '+str(p))
            if len(receipts)==expected:
                (root/'STOP').write_text('All frozen registry tasks complete; phase-specific workers may exit.\n');pending.remove(m)
        time.sleep(10)
    wait_tasks(path)
    verifiers={}
    for m,root in roots.items():
        collection=R/phase/('collected_'+m);receipt=oracle.collect(root,collection)
        if receipt['status']!='PASS':raise ValueError('Incomplete oracle '+m)
        verifier=R/phase/('verified_'+m+'.json');ai.verify_oracle(stages,root,collection/'receipt.json',verifier,m);verifiers[m]=verifier
    log('VERIFIED_PHASE',phase,{m:json.loads(v.read_text())['unique_structures'] for m,v in verifiers.items()});return verifiers

def summarize_phase(phase,stages,verifiers):
    rows=[];metadata={str(Path(m['stage'])):m for m in json.loads((R/phase/'models.json').read_text())}
    for stage in stages:
        for m,v in verifiers.items():
            out=R/phase/'metrics'/m/stage.name;ai.summarize(stage,v,SCHEMA if m=='uma' else MACE_SCHEMA,out)
            with (out/'metrics.csv').open() as f:
                for r in csv.DictReader(f):
                    info=metadata[str(stage)];rows.append(dict(phase=phase,stage_id=stage.name,family=info['family'],schema_mode=info.get('mode','anchor'),**r))
    ai.write_csv(R/phase/'all_cell_metrics.csv',rows)
    aggregate=[]
    for id in sorted({r['stage_id'] for r in rows}):
        for m in ['uma','mace']:
            rr=[r for r in rows if r['stage_id']==id and r['teacher']==m]
            if len(rr)!=9:raise ValueError('Unexpected conditional cells')
            lows=[float(r['fixed_low_yield']) for r in rr if r['requested_bin']=='low'];highs=[float(r['fixed_high_yield']) for r in rr if r['requested_bin']=='high']
            aggregate.append(dict(phase=phase,stage_id=id,family=rr[0]['family'],schema_mode=rr[0]['schema_mode'],seed=rr[0]['seed'],teacher=m,balanced_fixed_yield=float(np.mean([float(r['fixed_inbin_yield']) for r in rr])),balanced_moving_yield=float(np.mean([float(r['moving_inbin_yield']) for r in rr])),low_tail_yield=float(np.mean(lows)),high_tail_yield=float(np.mean(highs)),attempts=sum(int(r['attempts']) for r in rr),scored_cell_incidences=sum(int(r['scored_unique']) for r in rr)))
    ai.write_csv(R/phase/'aggregate_metrics.csv',aggregate);return aggregate

def initial_models():
    return [dict(id=f'initial_{f}_{s}',family=f,seed=s,method='initial_'+f,mode='anchor',schema=SCHEMA,checkpoint=RUN/'initial_fits'/f'{f}_{s}'/'last_update.pt') for f in FAMILIES for s in SEEDS]

def fit_task(id,family,seed,data,schema,warmstart,out):
    originals=json.loads((RUN/'initial_fit_tasks.json').read_text())['tasks'];orig=next(t for t in originals if t['id']==f'{family}_{seed}')
    args=orig['argv'][1:];replacements={'--data':data,'--condition-schema-in':schema,'--max-updates':'2000','--max-seconds':'2400','--output-dir':out}
    for flag,value in replacements.items():args[args.index(flag)+1]=str(value)
    args+=['--warm-start',str(warmstart),'--allow-rebinned-warm-start']
    inputs=[data,schema,warmstart,PARENT,R/'design.json',SRC/'cathode_cvae/learning/train_mask.py',SRC/'cathode_cvae/learning/physical_auxiliary.py']
    return template(id,args,inputs,[dict(path=str(out/'summary.json'),equals={'status':'FIT_COMPLETE','optimizer_updates':2000})])

def train_round(round_index,acq_models,verifiers):
    phase=f'round{round_index}';tasks=[];models=[];controls=[];train_records=[]
    for src in acq_models:
        if src['family'] not in FAMILIES:continue
        f=src['family'];s=src['seed'];modes=MODES if round_index==1 else [src['mode']]
        data_before=H0 if round_index==1 else Path(src['data'])
        merge=R/phase/'training'/src['id']
        ai.merge_training(data_before,PARENT,src['schema'],src['stage'],verifiers['uma'],merge,round_index=round_index)
        for mode in modes:
            id=f'r{round_index}_{f}_{mode}_{s}';schema_out=R/phase/'schemas'/id
            ai.fit_round_schema(merge/'training.npz',PARENT,SCHEMA,mode,schema_out);schema=schema_out/'condition_schema.json';out=R/phase/'fits'/id
            tasks.append(fit_task(id,f,s,merge/'training.npz',schema,src['checkpoint'],out))
            item=dict(id=id,family=f,seed=s,method=f'{f}_{mode}_r{round_index}',mode=mode,schema=schema,checkpoint=out/'last_update.pt',data=merge/'training.npz')
            models.append(item);train_records.append(dict(id=id,base_data=str(data_before),new_data=str(merge/'training.npz'),schema=str(schema),warm_start=str(src['checkpoint']),new_rows=json.loads((merge/'receipt.json').read_text())['added_train_rows']))
        # In round1 all schema alternatives share one no-new H0 continuation.
        control_id=f'r{round_index}_no_new_{f}_{src.get("mode","anchor")}_{s}';out=R/phase/'fits'/control_id
        tasks.append(fit_task(control_id,f,s,data_before,src['schema'],src['checkpoint'],out))
        controls.append(dict(id=control_id,family=f,seed=s,method=f'{f}_no_new_r{round_index}',mode=src.get('mode','anchor'),schema=src['schema'],checkpoint=out/'last_update.pt',data=data_before))
    write(R/phase/'training_design.json',dict(status='FROZEN_TRAINING_BEFORE_FITS',source=binding(__file__),training_records=train_records,new_models=[{k:str(v) if isinstance(v,Path) else v for k,v in d.items()} for d in models],controls=[{k:str(v) if isinstance(v,Path) else v for k,v in d.items()} for d in controls]));run_tasks(phase+'_fits',tasks);return models,controls

def freeze_design():
    design=dict(status='FROZEN_BEFORE_FIRST_ADAPTIVE_ACQUISITION',created_unix=time.time(),parent_protocol=binding(RUN/'protocol.json'),source_snapshot=binding(ROOTITER/'source/receipt.json'),driver=binding(__file__),seeds=SEEDS,families=FAMILIES,schema_modes=MODES,rounds=2,
        acquisition=dict(attempts_per_cell=512,query_cap_per_cell=256,counts=[12,24,36],bins=['low','middle','high'],selection='First eligible novel canonical orbit, no energy filtering',exploration_baselines=['uniform','TRAIN-fixed-bin seed mutation, 1-3 fixed-count swaps']),
        evaluation=dict(attempts_per_cell=128,query_cap_per_cell=128,counts=[12,24,36],bins=['low','middle','high'],teachers=['uma','mace'],fixed_schemas={'uma':binding(SCHEMA),'mace':binding(MACE_SCHEMA)},primary='Macro-average fixed-schema in-bin novel unique yield per all128 attempts, 9 conditional cells',secondary=['LOW requested novel fixed-low yield','HIGH requested novel fixed-high yield','moving label fidelity','composition-specific min/max'],no_true_equilibrium_claim=True),
        controls='R1 identical new data across3labels and one shared2000update H0no-new per family/seed; R2 eachpolicy2000new-data updates vs same-lineage2000no-new continuation',
        schedules='Equal2000 optimizer updates, fresh Adam/RNG/schedule, last_update checkpoint; warmstart explicit rebin guard; no validation selection',
        assessed_models='R1 original6 + R1updated18 + no-new6 +baseline6; R2 original6 + R2updated18 + R2no-new18 +baseline6',
        acquisition_baselines='Uniform and mutation have equal attempts/query caps; do not enter neural training; all raw attempted identities added to next history',
        protected='Historical canonical history/protected plus all initial latent raw valid identities and pre-correction unscored raw proposals as known. Only newly eligible latent identities become protected; zero new H0TRAINprotected. Each phase raw attempts become known; selected fresh assessment orbits protected before nextacquisition.',
        iteration_semantics='R1same checkpoint sharedquerybatch acrossmodes; R2divergent conditioned policies. Adaptive empirical q20/q80 vs observed-range20/80vsfixedschema. Only measuredUMA labels enter TRAIN.',
        endpoint_decision='Report all paired family/seed differences with mean/sampleSD; do not choose policies using assessment and retrain them in thisstudy',
        stop='Frozen parentdeadline and resource limits; any stage/worker/verification failure stops driver without overwriting evidence')
    write(R/'design.json',design)

def main():
    require_compute_allocation();freeze_design();boundary=protect_latent();ai.freeze_seed_bank(H0,PARENT,SCHEMA,boundary,R/'seed_bank')
    anchors=initial_models();acq1=stages_for('round1_acquisition',anchors,boundary)
    v1=score_phase('round1_acquisition',acq1);summarize_phase('round1_acquisition',acq1,v1)
    meta1=json.loads((R/'round1_acquisition/models.json').read_text());models1,controls1=train_round(1,meta1,v1)
    b1=R/'boundary_after_r1_acquisition';ai.extend_boundary(boundary,acq1,[],b1)
    ev1=stages_for('round1_evaluation',anchors+models1+controls1,b1,evaluation=True)
    ve1=score_phase('round1_evaluation',ev1);summarize_phase('round1_evaluation',ev1,ve1)
    b2=R/'boundary_after_r1_evaluation';ai.extend_boundary(b1,[],ev1,b2);log('ROUND1_COMPLETE')
    acq2=stages_for('round2_acquisition',models1,b2);v2=score_phase('round2_acquisition',acq2);summarize_phase('round2_acquisition',acq2,v2)
    meta2=json.loads((R/'round2_acquisition/models.json').read_text());models2,controls2=train_round(2,meta2,v2)
    b3=R/'boundary_after_r2_acquisition';ai.extend_boundary(b2,acq2,[],b3)
    ev2=stages_for('round2_evaluation',anchors+models2+controls2,b3,evaluation=True);ve2=score_phase('round2_evaluation',ev2);summarize_phase('round2_evaluation',ev2,ve2)
    b4=R/'boundary_final';ai.extend_boundary(b3,[],ev2,b4)
    write(R/'completion.json',dict(status='TWO_ROUND_ADAPTIVE_STUDY_COMPLETE',created_unix=time.time(),design=binding(R/'design.json'),driver=binding(__file__),boundary=binding(b4/'receipt.json'),phases={phase:binding(R/phase/'aggregate_metrics.csv') for phase in ['round1_acquisition','round1_evaluation','round2_acquisition','round2_evaluation']},physical_scope='Unrelaxed fixed-parent vacancy configurations evaluated using pinned UMA/MACE, not DFT or equilibrium.'))
    log('COMPLETE')
if __name__=='__main__':main()

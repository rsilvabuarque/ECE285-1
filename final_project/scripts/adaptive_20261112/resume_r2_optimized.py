"""Complete frozen R2 with validated queue-only worker optimization; all science code stays original."""
from pathlib import Path
import importlib.util,json,time
r=Path(__file__).parent
spec=importlib.util.spec_from_file_location('original_frozen_driver',r/'orchestrate.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
config=json.loads((r/'oracle_worker_override.json').read_text());W=Path(config['source'])


def check_override():
    assert config['status']=='VALIDATED_QUEUE_ONLY_WORKER_OVERRIDE'
    for item in [config['snapshot_receipt'],config['validation']]:d.binding(item['path'],item['sha256'])
    receipt=json.loads(Path(config['snapshot_receipt']['path']).read_text())
    assert receipt['status']=='FROZEN_VALIDATED_QUEUE_ONLY_WORKER_SNAPSHOT' and receipt['byte_reversal_recovers_original_oracle']
    for name,sha in receipt['files'].items():d.binding(W/name,sha)


def score_phase(phase,stages):
    check_override()
    for stage in stages:d.ai.validate_stage(stage)
    roots={m:d.R/phase/('oracle_'+m) for m in ['uma','mace']}
    for m,root in roots.items():
        for stage in stages:d.oracle.register_pool(root,stage/'candidates.npz',d.PARENT,models=[m],batch_size=64)
    jobs=[]
    for m,root in roots.items():
        for i in range(9):
            t=d.template(f'{m}_{i:02d}',['-m','cathode_cvae.adaptive_oracle','worker','--root',root,'--model',m,'--checkpoint',d.CHECKPOINTS[m],'--threads','4'],[root/'registry.json',W/'cathode_cvae/adaptive_oracle.py',W/'cathode_cvae/oracle_queue.py',W/'cathode_cvae/mlip/registry.py',W/'cathode_cvae/mlip/calculators.py',W/'cathode_cvae/data.py',r/'r2_queue_operational_amendment.json',r/'oracle_worker_override.json'],python=d.UMA_PY if m=='uma' else d.MACE_PY)
            # template shares its ENV dictionary; copy to keep all trainers/generators on original source.
            t['env']=dict(t['env'],PYTHONPATH=str(W));jobs.append(t)
    assert d.ENV['PYTHONPATH']==str(d.SRC),'Worker override leaked into science-code environment'
    path=d.freeze_tasks(phase+'_oracle',jobs);d.start_tasks(path);pending=set(roots);last={}
    while pending:
        d.assert_deadline()
        for m in list(pending):
            root=roots[m];expected=len(json.loads((root/'registry.json').read_text())['tasks'])
            failures=list((root/'claims').glob('*/failure.json'))
            if failures:raise RuntimeError('Oracle task failed '+str(failures[0]))
            rr=list((root/'scores'/m).glob('*.json'))
            if last.get(m)!=len(rr):d.log('OPTIMIZED_ORACLE_PROGRESS',phase,m,len(rr),expected);last[m]=len(rr)
            if any(json.loads(p.read_text())['status']!='complete' for p in rr):raise RuntimeError('Oracle structure failed')
            if len(rr)==expected:(root/'STOP').write_text('All frozen registry tasks complete; stop only phase-specific workers.\n');pending.remove(m)
        time.sleep(10)
    d.wait_tasks(path);verified={}
    for m,root in roots.items():
        out=d.R/phase/('collected_'+m);v=d.oracle.collect(root,out)
        if v['status']!='PASS':raise ValueError('Original strict collector incomplete')
        p=d.R/phase/('verified_'+m+'.json');d.ai.verify_oracle(stages,root,out/'receipt.json',p,m);verified[m]=p
    d.log('VERIFIED_OPTIMIZED_PHASE',phase,{m:json.loads(p.read_text())['unique_structures'] for m,p in verified.items()});return verified


def main():
    d.require_compute_allocation();check_override();gen=r/'taskfiles/round2_acquisition_generation.json'
    assert gen.exists(),'Controlled handoff requires frozen R2 generation taskfile'
    assert not list((r/'round2_acquisition').glob('oracle_*/registry.json')),'R2 scoring already started; cannot mix source silently'
    d.write(r/'r2_queue_operational_amendment.json',dict(status='FROZEN_QUEUE_ONLY_AMENDMENT_BEFORE_FIRST_R2_SCF',created_unix=time.time(),source=d.binding(__file__),worker_override=d.binding(r/'oracle_worker_override.json'),original_driver=d.binding(r/'orchestrate.py'),previous_driver=d.binding(r/'resume_with_phase2.py'),frozen_acquisition_generation_tasks=d.binding(gen),scientific_design=d.binding(r/'design.json'),phase2_identity_amendment=d.binding(r/'phase2_boundary_amendment.json'),change='Only score-worker PYTHONPATH and its bound queue-source files; original generator, trainer, registration, collector, strict verifier, schemas, raw stages and query budgets remain byte-identical. All R2 workers use optimized source from outset.',validation=config['validation'],no_failed_claim_reclamation=True))
    d.wait_tasks(gen);meta=json.loads((r/'round2_acquisition/models.json').read_text());stages=[Path(m['stage']) for m in meta]
    v2=score_phase('round2_acquisition',stages);d.summarize_phase('round2_acquisition',stages,v2)
    models2,controls2=d.train_round(2,meta,v2)
    boundary=r/'boundary_before_r2_with_phase2';b3=r/'boundary_after_r2_acquisition';d.ai.extend_boundary(boundary,stages,[],b3)
    ev2=d.stages_for('round2_evaluation',d.initial_models()+models2+controls2,b3,evaluation=True);ve2=score_phase('round2_evaluation',ev2);d.summarize_phase('round2_evaluation',ev2,ve2)
    b4=r/'boundary_final';d.ai.extend_boundary(b3,[],ev2,b4)
    d.write(r/'completion.json',dict(status='TWO_ROUND_ADAPTIVE_STUDY_COMPLETE',created_unix=time.time(),design=d.binding(r/'design.json'),phase2_boundary_amendment=d.binding(r/'phase2_boundary_amendment.json'),queue_operational_amendment=d.binding(r/'r2_queue_operational_amendment.json'),driver=d.binding(r/'orchestrate.py'),first_resume_driver=d.binding(r/'resume_with_phase2.py'),r2_resume_driver=d.binding(__file__),boundary=d.binding(b4/'receipt.json'),phases={p:d.binding(r/p/'aggregate_metrics.csv') for p in ['round1_acquisition','round1_evaluation','round2_acquisition','round2_evaluation']},physical_scope='Unrelaxed fixed-parent vacancy configurations evaluated with pinnedUMA/MACE; noDFT or equilibrium validation.'))
    d.log('COMPLETE_CORE_WITH_QUEUE_OPTIMIZATION')
if __name__=='__main__':main()

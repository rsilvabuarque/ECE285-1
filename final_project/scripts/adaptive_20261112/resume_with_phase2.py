"""Resume the frozen core at its R1 assessment, adding independent phase2 raw identity protection."""
from pathlib import Path
import importlib.util,json,time
import numpy as np
p=Path(__file__).parent
spec=importlib.util.spec_from_file_location('frozen_iteration_driver',p/'orchestrate.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


def finish_running_oracle(phase,stages):
    roots={m:d.R/phase/('oracle_'+m) for m in ['uma','mace']};pending=set(roots);last={}
    while pending:
        d.assert_deadline()
        for m in list(pending):
            root=roots[m];reg=json.loads((root/'registry.json').read_text());expected=len(reg['tasks'])
            failures=list((root/'claims').glob('*/failure.json'))
            if failures:raise RuntimeError('Oracle task failure '+str(failures[0]))
            rr=list((root/'scores'/m).glob('*.json'))
            if last.get(m)!=len(rr):d.log('RESUMED_ORACLE_PROGRESS',phase,m,len(rr),expected);last[m]=len(rr)
            if any(json.loads(q.read_text())['status']!='complete' for q in rr):raise RuntimeError('Failed oracle structure')
            if len(rr)==expected:
                if not (root/'STOP').exists():(root/'STOP').write_text('Frozen phase complete; stop only phase-specific oracle workers.\n')
                pending.remove(m)
        time.sleep(10)
    d.wait_tasks(d.R/'taskfiles'/(phase+'_oracle.json'));verified={}
    for m,root in roots.items():
        collection=d.R/phase/('collected_'+m)
        if collection.exists():raise ValueError('Old driver unexpectedly collected phase; preserve and review ownership')
        receipt=d.oracle.collect(root,collection)
        if receipt['status']!='PASS':raise ValueError('Oracle collection incomplete')
        v=d.R/phase/('verified_'+m+'.json');d.ai.verify_oracle(stages,root,collection/'receipt.json',v,m);verified[m]=v
    return verified


def phase2_boundary(base):
    receipt_path=d.RUN/'latent_phase2/raw_union_receipt.json';pool=d.RUN/'latent_phase2/raw_valid_canonical_union.npz'
    if not receipt_path.exists():raise ValueError('Phase2 rawunion not frozen; do not delay primary study indefinitely')
    rr=json.loads(receipt_path.read_text())
    if rr['status']!='ALL40_RAW_DIAGNOSTICS_FROZEN_BEFORE_ORACLE' or not rr['no_energy_filtering'] or rr['pool_sha256']!=d.sha256(pool) or rr['parent_sha256']!=d.sha256(d.PARENT):raise ValueError('Phase2 rawunion binding mismatch')
    boundary,known,protected=d.ai.load_boundary(base);base_known=set(known);base_protected=set(protected);perms,_=d._parent(d.PARENT)
    with np.load(pool,allow_pickle=False) as z:
        x=z['occupancy'];m=z['masks'];ids=z['structure_ids'].astype(str)
        if str(z['parent_sha256'].item())!=d.sha256(d.PARENT):raise ValueError('Phase2 parent mismatch')
    keys=d._canonical_vacancy(x,perms)
    if not np.array_equal(m,1-x) or ids.tolist()!=[d.ai.identity(k) for k in keys] or len(set(keys.tolist()))!=rr['unique_valid_orbits']:raise ValueError('Phase2 union invalid canonical masks')
    with np.load(d.H0,allow_pickle=False) as z:roles=z['split'].astype(str);tx=z['occupancy'][roles=='train']
    tk=set(d._canonical_vacancy(tx,perms).tolist());new=set(keys.tolist())-known-tk;known.update(keys.tolist());protected.update(new)
    if (protected-base_protected)&tk:raise ValueError('New H0TRAIN protected overlap')
    out=d.R/'boundary_before_r2_with_phase2';out.mkdir(exist_ok=False)
    d.write(out/'known_history.json',sorted(known));d.write(out/'protected.json',sorted(protected))
    d.write(out/'receipt.json',dict(status='FROZEN_ADAPTIVE_CANONICAL_BOUNDARY',created_unix=time.time(),parent=boundary['parent'],history=d.binding(out/'known_history.json'),protected=d.binding(out/'protected.json'),known_orbits=len(known),protected_orbits=len(protected),previous_boundary=d.binding(base/'receipt.json'),latent_phase2_raw_union=d.binding(pool),latent_phase2_raw_receipt=d.binding(receipt_path),new_phase2_protected_orbits=len(new),newly_protected_h0_train_orbits=0,inherited_h0_train_protected_orbits=len(base_protected&tk),source=d.binding(__file__),amendment=d.binding(d.R/'phase2_boundary_amendment.json'),no_energy_members_read=True))
    d.log('PHASE2_BOUNDARY_READY',len(known),len(protected),len(new));return out


def main():
    d.require_compute_allocation()
    if list((d.R/'round2_acquisition').glob('stages/*')):raise ValueError('R2 already started; preserve original prospective boundary')
    if not (d.R/'taskfiles/round1_evaluation_oracle.json').exists():raise ValueError('R1 assessment oracle not yet frozen/dispatched')
    if not (d.RUN/'latent_phase2/raw_union_receipt.json').exists():raise ValueError('No phase2 rawunion yet')
    d.write(d.R/'phase2_boundary_amendment.json',dict(status='FROZEN_BEFORE_R2_ACQUISITION',created_unix=time.time(),reason='Independent phase2 latent diagnostic raw identities must not enter later adaptive TRAIN. Root authorized adding newly eligible identities to protection, allraw to known; no hyperparameters or assessment-derived policy changes.',parent_design=d.binding(d.R/'design.json'),original_driver=d.binding(d.R/'orchestrate.py'),resume_source=d.binding(__file__),phase2_union_receipt=d.binding(d.RUN/'latent_phase2/raw_union_receipt.json'),r1_assessment_taskfile=d.binding(d.R/'taskfiles/round1_evaluation_oracle.json'),no_model_or_budget_changes=True))
    phase='round1_evaluation';meta=json.loads((d.R/phase/'models.json').read_text());stages=[Path(m['stage']) for m in meta]
    verifiers=finish_running_oracle(phase,stages);d.summarize_phase(phase,stages,verifiers)
    b2=d.R/'boundary_after_r1_evaluation';d.ai.extend_boundary(d.R/'boundary_after_r1_acquisition',[],stages,b2)
    boundary=phase2_boundary(b2);d.log('ROUND1_COMPLETE_WITH_PHASE2_BOUNDARY')
    td=json.loads((d.R/'round1/training_design.json').read_text());models1=td['new_models'];anchors=d.initial_models()
    acq2=d.stages_for('round2_acquisition',models1,boundary);v2=d.score_phase('round2_acquisition',acq2);d.summarize_phase('round2_acquisition',acq2,v2)
    meta2=json.loads((d.R/'round2_acquisition/models.json').read_text());models2,controls2=d.train_round(2,meta2,v2)
    b3=d.R/'boundary_after_r2_acquisition';d.ai.extend_boundary(boundary,acq2,[],b3)
    ev2=d.stages_for('round2_evaluation',anchors+models2+controls2,b3,evaluation=True);ve2=d.score_phase('round2_evaluation',ev2);d.summarize_phase('round2_evaluation',ev2,ve2)
    b4=d.R/'boundary_final';d.ai.extend_boundary(b3,[],ev2,b4)
    d.write(d.R/'completion.json',dict(status='TWO_ROUND_ADAPTIVE_STUDY_COMPLETE',created_unix=time.time(),design=d.binding(d.R/'design.json'),phase2_boundary_amendment=d.binding(d.R/'phase2_boundary_amendment.json'),driver=d.binding(d.R/'orchestrate.py'),resume_driver=d.binding(__file__),boundary=d.binding(b4/'receipt.json'),phases={p:d.binding(d.R/p/'aggregate_metrics.csv') for p in ['round1_acquisition','round1_evaluation','round2_acquisition','round2_evaluation']},physical_scope='Unrelaxed fixed-parent vacancy configurations evaluated using pinnedUMA/MACE, notDFTorequilibrium.'))
    d.log('COMPLETE_RESUMED_STUDY')
if __name__=='__main__':main()

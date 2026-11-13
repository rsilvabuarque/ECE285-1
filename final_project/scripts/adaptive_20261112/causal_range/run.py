"""Post-hoc 2x2 rows/conditioning control, isolated from the frozen two-round study."""
from pathlib import Path
import csv,importlib.util,json,time
import numpy as np
C=Path(__file__).parent;BASE=C.parent/'study_v2'
spec=importlib.util.spec_from_file_location('frozen_core_driver',BASE/'orchestrate.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
CFG=json.loads((BASE/'oracle_worker_override.json').read_text());W=Path(CFG['source'])


def budget():
 d.assert_deadline()
 if (C/'protocol.json').exists() and time.time()-json.loads((C/'protocol.json').read_text())['started_unix']>38*60:raise TimeoutError('Causal active38minute budget exhausted; preserve outputs within40minute allocation cap')
def freeze_tasks(name,tasks):
 p=C/'taskfiles'/(name+'.json');d.write(p,dict(status='FROZEN_CAUSAL_TASKS',created_unix=time.time(),source=d.binding(__file__),protocol=d.binding(C/'protocol.json'),tasks=tasks));return p
def publish(p):budget();d.atomic(d.RUN/'controls/mace.json',{'task_file':str(p)});d.log('CAUSAL_DISPATCH',p.name)
def wait(p):
 tasks=json.loads(p.read_text())['tasks'];claims=p.parent/(p.stem+'.claims');last=None
 while True:
  budget();rr=[]
  for t in tasks:
   q=claims/t['id']/'receipt.json'
   if q.exists():
    v=json.loads(q.read_text())
    if v['status']!='COMPLETE':raise RuntimeError('Causal task failed '+str(q))
    rr.append(v)
  if len(rr)!=last:d.log('CAUSAL_TASKS',p.name,len(rr),len(tasks));last=len(rr)
  if len(rr)==len(tasks):return
  time.sleep(5)
def run_tasks(name,tasks):p=freeze_tasks(name,tasks);publish(p);wait(p)


def models_and_tasks(extra):
 models=[];tasks=[];inputs=[]
 for family in d.FAMILIES:
  for i,seed in enumerate(d.SEEDS):
   draw=2026111701+i;initial=d.RUN/'initial_fits'/f'{family}_{seed}'/'last_update.pt'
   common=dict(family=family,fit_seed=seed,draw_seed=draw)
   fixed_control=BASE/'round1/fits'/f'r1_no_new_{family}_anchor_{seed}'/'last_update.pt'
   models.append(dict(common,id=f'h0_fixed_{family}_{seed}',cell='h0_fixed',checkpoint=str(fixed_control),schema=str(d.SCHEMA)))
   for mode in ['fixed','range']:
    path=BASE/'round1/fits'/f'r1_{family}_{mode}_{seed}'/'last_update.pt';schema=BASE/'round1/schemas'/f'r1_{family}_{mode}_{seed}'/'condition_schema.json'
    models.append(dict(common,id=f'aug_{mode}_{family}_{seed}',cell='aug_'+mode,checkpoint=str(path),schema=str(schema)))
   variants=[('h0_range_h0',C/'h0_range_schema/condition_schema.json')]
   if extra:variants.append(('h0_range_r1',BASE/'round1/schemas'/f'r1_{family}_range_{seed}'/'condition_schema.json'))
   for cell,schema in variants:
    id=f'{cell}_{family}_{seed}';out=C/'fits'/id;task=d.fit_task(id,family,seed,d.H0,schema,initial,out)
    task['argv'][task['argv'].index('--max-seconds')+1]='600';task['bindings']+=[d.binding(C/'plan.json'),d.binding(C/'preparation_receipt.json'),d.binding(__file__)]
    tasks.append(task);models.append(dict(common,id=id,cell=cell,checkpoint=str(out/'last_update.pt'),schema=str(schema)))
   inputs.append(dict(family=family,fit_seed=seed,draw_seed=draw,initial=d.binding(initial),h0_fixed=d.binding(fixed_control),aug_fixed=d.binding(BASE/'round1/fits'/f'r1_{family}_fixed_{seed}'/'last_update.pt'),aug_range=d.binding(BASE/'round1/fits'/f'r1_{family}_range_{seed}'/'last_update.pt')))
 return models,tasks,inputs


def score(stages):
 for p in stages:d.ai.validate_stage(p)
 roots={m:C/'evaluation'/('oracle_'+m) for m in ['uma','mace']}
 for m,root in roots.items():
  for p in stages:d.oracle.register_pool(root,p/'candidates.npz',d.PARENT,models=[m],batch_size=64)
 jobs=[]
 for m,root in roots.items():
  for i in range(9):
   t=d.template(f'{m}_{i:02d}',['-m','cathode_cvae.adaptive_oracle','worker','--root',root,'--model',m,'--checkpoint',d.CHECKPOINTS[m],'--threads','4'],[C/'protocol.json',root/'registry.json',W/'cathode_cvae/adaptive_oracle.py',W/'cathode_cvae/oracle_queue.py',BASE/'oracle_worker_override.json'],python=d.UMA_PY if m=='uma' else d.MACE_PY)
   t['env']=dict(t['env'],PYTHONPATH=str(W));jobs.append(t)
 assert d.ENV['PYTHONPATH']==str(d.SRC)
 p=freeze_tasks('oracle',jobs);publish(p);pending=set(roots);last={}
 try:
  while pending:
   budget()
   for m in list(pending):
    root=roots[m];expected=len(json.loads((root/'registry.json').read_text())['tasks']);rr=list((root/'scores'/m).glob('*.json'))
    if list((root/'claims').glob('*/failure.json')):raise RuntimeError('Causal oracle task failed')
    if any(json.loads(q.read_text())['status']!='complete' for q in rr):raise RuntimeError('Causal structure evaluation failed')
    if last.get(m)!=len(rr):d.log('CAUSAL_ORACLE',m,len(rr),expected);last[m]=len(rr)
    if len(rr)==expected:(root/'STOP').write_text('All frozen causal registry tasks complete.\n');pending.remove(m)
   time.sleep(5)
 except BaseException:
  for root in roots.values():(root/'STOP').write_text('Bounded causal follow-up stopped; preserve incomplete outputs.\n')
  raise
 wait(p);verified={}
 for m,root in roots.items():
  out=C/'evaluation'/('collected_'+m);v=d.oracle.collect(root,out)
  if v['status']!='PASS':raise ValueError('Causal collection incomplete')
  p=C/'evaluation'/('verified_'+m+'.json');d.ai.verify_oracle(stages,root,out/'receipt.json',p,m);verified[m]=p
 return verified


def summarize(models,verified,extra):
 from cathode_cvae.conditions import classify
 cells=[];aggregates=[]
 for model in models:
  budget();stage=Path(model['stage'])
  for teacher,verifier in verified.items():
   _,scores,_=d.ai.scored_rows(stage,verifier);schema=d.load_schema(d.SCHEMA if teacher=='uma' else d.MACE_SCHEMA);rows=d.ai.read_csv(stage/'attempts.csv');by=[]
   for count in [12,24,36]:
    for requested in ['low','middle','high']:
     rr=[r for r in rows if int(r['n_target'])==count and r['requested_bin']==requested];selected={r['structure_id'] for r in rr if r['status']=='selected'};energy=np.array([scores[s][0]/scores[s][1] for s in sorted(selected)]);labels=classify(schema,np.full(len(energy),count),energy)
     row=dict(model_id=model['id'],cell=model['cell'],family=model['family'],fit_seed=model['fit_seed'],draw_seed=model['draw_seed'],teacher=teacher,n_target=count,requested_bin=requested,attempts=len(rr),scored_unique=len(selected),fixed_inbin_yield=float(np.sum(labels==requested)/len(rr)),fixed_low_yield=float(np.sum(labels=='low')/len(rr)),fixed_high_yield=float(np.sum(labels=='high')/len(rr)),known_excluded=sum(r['status']=='known_boundary' for r in rr),protected_excluded=sum(r['status']=='protected_boundary' for r in rr))
     cells.append(row);by.append(row)
   aggregates.append(dict(model_id=model['id'],cell=model['cell'],family=model['family'],fit_seed=model['fit_seed'],draw_seed=model['draw_seed'],teacher=teacher,balanced_fixed_yield=float(np.mean([r['fixed_inbin_yield'] for r in by])),low_tail_yield=float(np.mean([r['fixed_low_yield'] for r in by if r['requested_bin']=='low'])),high_tail_yield=float(np.mean([r['fixed_high_yield'] for r in by if r['requested_bin']=='high'])),attempts=sum(r['attempts'] for r in by),scored_cell_incidences=sum(r['scored_unique'] for r in by)))
 d.ai.write_csv(C/'all_cell_metrics.csv',cells);d.ai.write_csv(C/'aggregate_metrics.csv',aggregates)
 lookup={(r['family'],r['fit_seed'],r['teacher'],r['cell']):r for r in aggregates};pairs=[];summary=[]
 same='h0_range_r1' if extra else 'h0_range_h0';contrasts=[('H0_relabeling','h0_range_h0','h0_fixed'),('augmented_relabeling','aug_range','aug_fixed'),('added_rows_fixed','aug_fixed','h0_fixed'),('added_rows_identical_range_edges','aug_range',same)]
 for family in d.FAMILIES:
  for teacher in ['uma','mace']:
   for name,a,b in contrasts:
    for metric in ['balanced_fixed_yield','low_tail_yield','high_tail_yield']:
     values=[]
     for seed in d.SEEDS:
      aa=lookup[(family,seed,teacher,a)][metric];bb=lookup[(family,seed,teacher,b)][metric];gain=100*(aa-bb);values.append(gain);pairs.append(dict(family=family,teacher=teacher,contrast=name,metric={'low_tail_yield':'H0_fixed_low_yield','high_tail_yield':'H0_fixed_high_yield'}.get(metric,metric),fit_seed=seed,policy_cell=a,reference_cell=b,policy_yield=float(aa),reference_yield=float(bb),gain_pp=gain))
     summary.append(dict(family=family,teacher=teacher,contrast=name,metric={'low_tail_yield':'H0_fixed_low_yield','high_tail_yield':'H0_fixed_high_yield'}.get(metric,metric),fit_count=len(values),gain_pp_mean=float(np.mean(values)),gain_pp_sample_sd=float(np.std(values,ddof=1))))
   for metric in ['balanced_fixed_yield','low_tail_yield','high_tail_yield']:
    values=[]
    for seed in d.SEEDS:
     get=lambda c:lookup[(family,seed,teacher,c)][metric]
     values.append(100*((get('aug_range')-get(same))-(get('aug_fixed')-get('h0_fixed'))))
    summary.append(dict(family=family,teacher=teacher,contrast='rows_by_schema_interaction',metric={'low_tail_yield':'H0_fixed_low_yield','high_tail_yield':'H0_fixed_high_yield'}.get(metric,metric),fit_count=len(values),gain_pp_mean=float(np.mean(values)),gain_pp_sample_sd=float(np.std(values,ddof=1))))
 d.ai.write_csv(C/'paired_results.csv',pairs);d.ai.write_csv(C/'paired_summary.csv',summary)


def main():
 d.require_compute_allocation()
 if not (BASE/'completion.json').exists() or not (BASE/'analysis/receipt.json').exists():raise ValueError('Frozen core and derived analysis must be ready before follow-up')
 prep=json.loads((C/'preparation_receipt.json').read_text());support=json.loads((C/'H0_support_receipt.json').read_text());assert support['all_cells_nonempty'],'Unsupported H0 condition; preserve diagnostic, do not retune edges'
 extra=prep['extra_H0_using_R1edges'];models,tasks,inputs=models_and_tasks(extra)
 d.write(C/'protocol.json',dict(status='FROZEN_POSTHOC_CAUSAL_FOLLOWUP_BEFORE_FITS',started_unix=time.time(),plan=d.binding(C/'plan.json'),preparation=d.binding(C/'preparation_receipt.json'),support=d.binding(C/'H0_support_receipt.json'),core_completion=d.binding(BASE/'completion.json'),core_analysis=d.binding(BASE/'analysis/receipt.json'),boundary=d.binding(BASE/'boundary_final/receipt.json'),worker_override=d.binding(BASE/'oracle_worker_override.json'),source=d.binding(__file__),inputs=inputs,new_fits=len(tasks),extra_H0_using_R1_edges=extra,models=models,active_guard_minutes=38,hard_wall_minutes=40,max_gpu_node_hours=3,fit_wall_seconds_cap=600,fit_wall_cap_provenance='Administrative10minute fit cap reserves rest of40minute follow-up budget; prior2000updatefits completedin~2minutes. No convergence tolerance changed; a fit that misses2000updates fails.',group='MACE18 only',moving_MACE_metrics='Not computed; only independent fixed teacher schemas are used.'))
 for task in tasks:task['bindings'].append(d.binding(C/'protocol.json'))
 run_tasks('fits',tasks)
 gen=[];stages=[]
 for model in models:
  out=C/'evaluation/stages'/model['id'];model['stage']=str(out);args=['-m','cathode_cvae.adaptive_iteration','propose','--parent',d.PARENT,'--schema',model['schema'],'--boundary-dir',BASE/'boundary_final','--output-dir',out,'--namespace','causal_range_20261113_v1','--seed',model['draw_seed'],'--method',model['id'],'--attempts-per-cell','128','--queries-per-cell','128','--checkpoint',model['checkpoint'],'--device','cuda','--threads','4']
  gen.append(d.template(model['id'],args,[d.PARENT,model['schema'],model['checkpoint'],C/'protocol.json',BASE/'boundary_final/receipt.json'],[dict(path=str(out/'receipt.json'),equals={'status':'FROZEN_ADAPTIVE_STAGE_BEFORE_SCORING'})]));stages.append(out)
 d.write(C/'models.json',models);run_tasks('generation',gen)
 reserved=C/'reserved_boundary';d.ai.extend_boundary(BASE/'boundary_final',[],stages,reserved)
 d.write(C/'RAW_RESERVATION_READY.json',dict(status='CAUSAL_RAW_RESERVED_BEFORE_FRONTIER',created_unix=time.time(),protocol=d.binding(C/'protocol.json'),boundary=d.binding(reserved/'receipt.json'),models=d.binding(C/'models.json'),all_attempts=sum(json.loads((p/'receipt.json').read_text())['attempts'] for p in stages),source=d.binding(__file__)))
 d.write(C.parent/'frontier/input_boundary.json',d.binding(reserved/'receipt.json'));d.log('CAUSAL_RAW_RESERVATION_READY')
 verified=score(stages)
 d.write(C/'GPU_IDLE_NO_FUTURE_TASKS.json',dict(status='CAUSAL_ALL_GPU_WORK_COMPLETE_STRICT_ORACLES_PASS',created_unix=time.time(),verifiers={m:d.binding(p) for m,p in verified.items()},source=d.binding(__file__),no_future_GPU_tasks=True,remaining_work='CPU metrics and figures run under existingUMAallocation, notMACEjobs. RootmayreleaseMACEallocations after checkingotherCPUsteps.'))
 d.log('CAUSAL_GPU_IDLE_NO_FUTURE_TASKS')
 summarize(models,verified,extra)
 d.write(C/'completion.json',dict(status='POSTHOC_CAUSAL_FOLLOWUP_COMPLETE',created_unix=time.time(),protocol=d.binding(C/'protocol.json'),source=d.binding(__file__),summary=d.binding(C/'paired_summary.csv'),metrics=d.binding(C/'all_cell_metrics.csv'),boundary=d.binding(reserved/'receipt.json'),elapsed_seconds=time.time()-json.loads((C/'protocol.json').read_text())['started_unix']))
 d.log('CAUSAL_COMPLETE')
if __name__=='__main__':
 try:main()
 except BaseException as e:
  if not (C/'failure.json').exists():d.write(C/'failure.json',dict(status='CAUSAL_FOLLOWUP_STOPPED',created_unix=time.time(),error_type=type(e).__name__,error=str(e)))
  raise

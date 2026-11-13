from pathlib import Path
import json,hashlib,time
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--diagnostic-root',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args();r=args.diagnostic_root.resolve();b=r.parent;p=args.output_dir.resolve();p.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=json.loads((b/'initial_fit_tasks.json').read_text())['tasks'];alltasks=[]
protocol=dict(status='PREDECLARED_BEFORE_PHASE1_ORACLE_ANALYSIS',created_unix=time.time(),hypothesis='lambda1 may produce insufficient physical-descriptor gradient; increasing strength must improve genuine heldout target prediction and actual generated physical control rather than KL alone',families=[dict(weight=w,mode=m) for w in [10,100] for m in ['latent','condition','shuffled']],fit_seeds=list(range(2026111301,2026111306)),diagnostic_seeds=list(range(2026111401,2026111406)),optimizer_updates=4000,unchanged='All initial D2 physical auxiliary flags/source/data/precision/Adam/beta schedule and head structure unchanged except lambda',controls='Matched condition-only and within-cell shuffled heads at each lambda; original AR and original D2 checkpoints sampled on fresh same streams; no model selection from oracle',diagnostics=dict(draws=128,guided=True,guide_steps=20,learning_rate=.1,clamp=[-3,3],interventions=11),primary='Terminal checkpoint; VAL genuine standardized residual descriptor MSE, actual guide-vs-clamp C_s response, fixed original teacher-bin novel yields',boundary='Freeze common history containing phase1 raw diagnostic identities plus all prior/current acquisition attempts before phase2 scoring; exclude known/protected and any newTRAIN overlaps',resource_limit='At most10 extra GPU-node-hours and root overall deadline; existing MACEgroup18GPUs only; no allocation submission',no_new_DFT=True,source_manifest=dict(path=str(r/'source_snapshot.json'),sha256=sha(r/'source_snapshot.json')))
with (p/'protocol.json').open('x') as f:json.dump(protocol,f,indent=2);f.write('\n')
for w in [10,100]:
 for mode in ['latent','condition','shuffled']:
  for seed in range(2026111301,2026111306):
   t=json.loads(json.dumps(next(x for x in base if x['id']==f'physical_{mode}_d2_{seed}')));t['id']=f'lambda{w}_{mode}_d2_{seed}';argv=t['argv'];argv[argv.index('--latent-physical-weight')+1]=str(w);out=p/'fits'/t['id'];argv[argv.index('--output-dir')+1]=str(out);t['cwd']=str(p)
   # Retain exact original frozen trainer source and source bindings.
   t['bindings'].append({'path':str(p/'protocol.json'),'sha256':sha(p/'protocol.json')});t['checks']=[{'path':str(out/'summary.json'),'equals':{'status':'FIT_COMPLETE','optimizer_updates':4000}}];alltasks.append(t)
with (p/'fit_tasks.json').open('x') as f:json.dump({'tasks':alltasks},f,indent=2);f.write('\n')
print(json.dumps(dict(protocol=str(p/'protocol.json'),protocol_sha256=sha(p/'protocol.json'),tasks=30,taskfile=str(p/'fit_tasks.json'),taskfile_sha256=sha(p/'fit_tasks.json')),indent=2))

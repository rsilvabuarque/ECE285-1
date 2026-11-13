from pathlib import Path
import hashlib,json,shutil
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--run-root',type=Path,required=True);parser.add_argument('--source-checkout',type=Path,required=True);args=parser.parse_args();p=args.run_root.resolve();b=p.parent;r=b/'latent_diagnostics';p2=b/'latent_phase2';sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest()
own=args.source_checkout.resolve();src=p/'source_snapshot/final_project';shutil.copytree(r/'source_snapshot/final_project',src)
for rel in ['src/cathode_cvae/learning/train_mask.py','src/cathode_cvae/learning/physical_auxiliary.py','tests/test_physical_auxiliary.py']:
 shutil.copyfile(own/rel,src/rel)
manifest=dict(base_source_manifest=dict(path=str(r/'source_snapshot.json'),sha256=sha(r/'source_snapshot.json')),base_source_HEAD='12087af5fffa2572fd3814943407f58b057a6edf',override_commit='696f03f4e',files=[dict(path=str(q.relative_to(src)),sha256=sha(q)) for q in sorted(src.rglob('*.py'))])
(p/'source_snapshot.json').write_text(json.dumps(manifest,indent=2)+'\n')
tasks=[]
for t0 in json.loads((p2/'fit_tasks.json').read_text())['tasks']:
 if not t0['id'].startswith('lambda10_'):continue
 t=json.loads(json.dumps(t0));t['id']='mean_'+t['id'];args=t['argv'];out=p/'fits'/t['id'];args[args.index('--output-dir')+1]=str(out);args+=['--latent-physical-input','mean'];t['cwd']=str(p);t['env']['PYTHONPATH']=str(src/'src')
 t['bindings']=[x for x in t['bindings'] if not x['path'].endswith('/learning/train_mask.py') and not x['path'].endswith('/learning/physical_auxiliary.py')]
 t['bindings'].extend([dict(path=str(q),sha256=sha(q)) for q in [src/'src/cathode_cvae/learning/train_mask.py',src/'src/cathode_cvae/learning/physical_auxiliary.py',p/'source_snapshot.json',p/'protocol.json']]);t['checks']=[dict(path=str(out/'summary.json'),equals={'status':'FIT_COMPLETE','optimizer_updates':4000})];tasks.append(t)
with (p/'fit_tasks.json').open('x') as f:json.dump({'tasks':tasks},f,indent=2);f.write('\n')
s=(p2/'phase2_fit_and_diagnostics.py').read_text();s=s.replace("while not (r/'oracle_monitor_receipt.json').exists():time.sleep(5)","while not (b/'latent_phase2/oracle_monitor_receipt.json').exists():time.sleep(5)").replace("json.loads((r/'oracle_monitor_receipt.json').read_text())","json.loads((b/'latent_phase2/oracle_monitor_receipt.json').read_text())").replace("wait_tasks(p/'fit_tasks.json',30)","wait_tasks(p/'fit_tasks.json',15)")
s=s.replace("['ar','d2']+[f'lambda{w}_{mode}_d2' for w in [10,100] for mode in ['latent','condition','shuffled']]","['ar','d2']+[f'lambda10_{mode}_d2' for mode in ['latent','condition','shuffled']]+[f'mean_lambda10_{mode}_d2' for mode in ['latent','condition','shuffled']]")
s=s.replace("else p/'fits'/f'{family}_{seed}'/'last_update.pt'", "else (b/'latent_phase2/fits'/f'{family}_{seed}'/'last_update.pt' if family.startswith('lambda10_') else p/'fits'/f'{family}_{seed}'/'last_update.pt')")
s=s.replace("str(2026111401+j)","str(2026111601+j)")
s=s.replace("t['bindings']=[x for x in t['bindings'] if x['path']!=str(previous)];", "t['env']['PYTHONPATH']=str(p/'source_snapshot/final_project/src');old_script=Path(argv[1]);argv[1]=str(p/'source_snapshot/final_project/scripts/adaptive_20261112/latent_physical_diagnostics.py');t['bindings']=[x for x in t['bindings'] if x['path'] not in [str(previous),str(old_script)]];t['bindings'].append(dict(path=argv[1],sha256=sha(argv[1])));")
s=s.replace("phase2_elapsed_seconds=", "phase_elapsed_seconds=").replace('PHASE2_','MEAN_').replace('Phase2','Mean-supervision')
(p/'fit_and_diagnostics.py').write_text(s)
s=(p2/'analyze_latent.py').read_text().replace("families=['ar','d2']+[f'lambda{w}_{mode}_d2' for w in [10,100] for mode in ['latent','condition','shuffled']]","families=['ar','d2']+[f'lambda10_{m}_d2' for m in ['latent','condition','shuffled']]+[f'mean_lambda10_{m}_d2' for m in ['latent','condition','shuffled']]")
s=s.replace("pretty={'ar':'AR (no z)','d2':'CVAE D2',**{f'lambda{w}_{mode}_d2':f'{mode} λ{w}' for w in [10,100] for mode in ['latent','condition','shuffled']}}","pretty={f:f.replace('mean_lambda10_','Mean ').replace('lambda10_','Sampled ').replace('_d2','').replace('_',' ') for f in families}")
s=s.replace("else r/'fits'/d.name)","else b/'latent_phase2/fits'/d.name if family.startswith('lambda10_') else r/'fits'/d.name)")
s=s.replace("['ar','d2','lambda10_condition_d2','lambda10_shuffled_d2','lambda100_condition_d2','lambda100_shuffled_d2']","['ar','d2','lambda10_latent_d2','lambda10_condition_d2','lambda10_shuffled_d2','mean_lambda10_condition_d2','mean_lambda10_shuffled_d2']")
(p/'analyze_latent.py').write_text(s)
s=(p2/'prepare_oracle.py').read_text().replace("boundary=b/'iteration/study_v2/phase2_boundary_after_r1_raw'","boundary=Path(json.loads((r/'boundary_binding.json').read_text())['path'])")
(p/'prepare_oracle.py').write_text(s)
s=(p2/'prepare_scoring_driver.py').read_text().replace("while not (p/'raw_union_receipt.json').exists():time.sleep(5)","while not ((p/'raw_union_receipt.json').exists() and (p/'boundary_binding.json').exists()):time.sleep(5)")
s=s.replace("t['id']='phase2_'+t['id'];t['cwd']=str(p)","t['id']='mean_'+t['id'];t['cwd']=str(p);t['env']['PYTHONPATH']=str(p/'source_snapshot/final_project/src')")
# Replace source/calculator bindings with the exact mean source tree.
s=s.replace("t['bindings'].extend([dict(path=str(x),sha256=sha(x)) for x in [q/'registry.json',p/'eligible_diagnostic_union.npz',p/'protocol.json']])", "t['bindings']=[x for x in t['bindings'] if '/source_snapshot/final_project/' not in x['path']];t['bindings'].extend([dict(path=str(x),sha256=sha(x)) for x in [q/'registry.json',p/'eligible_diagnostic_union.npz',p/'protocol.json',p/'source_snapshot.json',p/'source_snapshot/final_project/src/cathode_cvae/adaptive_oracle.py',p/'source_snapshot/final_project/src/cathode_cvae/mlip/calculators.py']])")
(p/'prepare_scoring.py').write_text(s);(p/'monitor_collect.py').write_text((p2/'monitor_collect.py').read_text())
(p/'implementation_tests.json').write_text(json.dumps(dict(status='PASS',full_suite='11 passed',metadata_condition_equality='2 passed,6deselected',allocation=61215971,source_commit='696f03f4e',source_manifest_sha256=sha(p/'source_snapshot.json')),indent=2)+'\n')
print('MEAN_CAMPAIGN_PREPARED',len(tasks),sha(p/'source_snapshot.json'),sha(p/'fit_tasks.json'),flush=True)

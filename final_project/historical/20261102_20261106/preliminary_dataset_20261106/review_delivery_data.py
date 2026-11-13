"""Check broad delivery claims against immutable outputs, not only success flags."""
import csv,hashlib,json,math,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
audit=json.loads((HERE/'validation/requirement_audit.json').read_text());assert audit['status']=='PASS'
for c in audit['checks']:
    if 'receipt' in c:assert sha(HERE/c['receipt'])==c['sha256'],c['check']
scf=[]
for m in ['uma','mace','sevennet','chgnet','m3gnet']:
    p=HERE/'validation'/f'scf_{m}.json';r=json.loads(p.read_text());assert r['status']=='PASS' and r['rows']==32855
    assert sum(x['rows'] for x in r['exports'])==32855
    for x in r['exports']:assert sha(x['path'])==x['sha256']
    scf.append(dict(model=m,structures=r['rows'],receipt_sha256=sha(p)))
gate=json.loads((HERE/'data/variable_gate.json').read_text());assert gate['status']=='PASS' and set(gate['new_parent_successes_out_of20'].values())=={20}
splits={r['structure_id']:r for r in csv.DictReader((HERE/'splits/lineage_splits.csv').open())};cache={};frozen=json.loads((HERE/'cvae/frozen_generation_models.json').read_text());assert len(frozen['runs'])==16
for r in frozen['runs']:
    folder=HERE/'cvae/runs'/r['run']
    for name,key in [('best.pt','checkpoint_sha256'),('configuration.json','configuration_sha256'),('summary.json','summary_sha256')]:assert sha(folder/name)==r[key],(r['run'],name)
    cfg=json.loads((folder/'configuration.json').read_text());teacher=cfg['teacher'];source=HERE/'data'/f'learning_O3_N48_pilot_{teacher}.npz'
    assert sha(source)==cfg['source_sha256'] and sha(HERE/'splits/lineage_splits.csv')==cfg['split_sha256']
    if teacher not in cache:
        with np.load(source) as z:cache[teacher]={k:z[k] for k in z.files}
    d=cache[teacher];field='split' if cfg['split']=='main' else 'composition_split';train=np.array([splits[s][field]=='train' for s in d['structure_ids']]);E=d['energy_eV']/d['n_atoms']
    assert set(map(int,cfg['condition_stats']))==set(map(int,np.unique(d['n_Li'][train])))
    for n,stats in cfg['condition_stats'].items():
        values=E[train&(d['n_Li']==int(n))];q=float(np.sort(values)[math.ceil(.2*len(values))-1]);assert q==stats['low_quantile_eV_atom'] and len(values)==stats['n_train']
assert frozen['attempt_manifest_sha256']==sha(HERE/'cvae/proposal_attempts.csv')
pilots={}
for t in map(json.loads,(HERE/'data/md_pilot_tasks.jsonl').read_text().splitlines()):
    r=json.loads((HERE/'data/md_pilot'/t['model']/(t['task_id']+'.json')).read_text());assert r['status']=='complete' and r['segments']['warm']['completed_duration_ps']==1. and r['segments']['npt']['completed_duration_ps']==1.
    pilots[t['model']]=pilots.get(t['model'],0)+1
assert len(pilots)==5 and set(pilots.values())=={10}
tasks=list(map(json.loads,(HERE/'data/md_prod_tasks.jsonl').read_text().splitlines()));assert len(tasks)==200
for t in tasks:
    r=json.loads((HERE/'data/md_prod'/t['model']/(t['task_id']+'.json')).read_text());assert r['status']=='complete' and r['segments']['npt']['completed_duration_ps']==5. and r['seed_sha256']==t['seed_sha256']
md=list(csv.DictReader((HERE/'reports/MD_trajectory_summary.csv').open()));assert len(md)==200
with np.load(HERE/'data/learning_O3_N48_pilot_uma.npz') as z:assert set(map(int,np.unique(z['n_Li'])))==set(range(49))
r=dict(status='PASS',SCF_models=scf,SCF_model_structure_pairs=164275,confirmed_new_variable_parents=100,frozen_mask_fits=16,train_only_energy_thresholds_verified=True,NPT_pilots_by_model=pilots,production_trajectories=200,production_duration_ps_each=5.,production_max_extended_energy_range_meV_atom=max(float(r['extended_energy_range_meV_atom']) for r in md),all49_main_compositions_present=True,scope='Direct counts, hashes and train-only condition reconstruction extend the independent 19-check dataset audit. No global-minimum, phase or DFT-accuracy claim.')
(HERE/'validation/delivery_data_review.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)

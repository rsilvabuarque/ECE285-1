"""Join every score shard to its task, immutable parent and calculator receipt."""
import csv,hashlib,json,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
cfg=json.loads((HERE/'config.json').read_text());config_hash=hashlib.sha256((HERE/'config.json').read_bytes()).hexdigest()
parents={r['parent']:r for r in json.loads((HERE/'data/preparation_receipt.json').read_text())['parents']};cache={};checked=0
for line in (HERE/'data/scf_tasks.jsonl').read_text().splitlines():
    t=json.loads(line);base=HERE/'data/labels'/t['model']/t['task_id'];r=json.loads(base.with_suffix('.json').read_text());assert r['status']=='complete'
    for key in ['task_id','model','parent','start','stop']:assert r[key]==t[key]
    worker=json.loads((HERE/'workers'/f"{r['worker']}.json").read_text());old=json.loads((Path(cfg['fixed_pilot'])/'validation'/f"{t['model']}.json").read_text())['receipt']
    assert worker['model']==t['model'] and worker['stage']=='scf' and worker['config_sha256']==config_hash
    for key in ['checkpoint_hashes','dtype','tf32','package_version','torch']:assert worker[key]==old[key]
    name=t['parent']
    if name not in cache:
        p=HERE/'data/parents'/f'{name}.npz';assert hashlib.sha256(p.read_bytes()).hexdigest()==parents[name]['sha256']
        with np.load(p) as z:cache[name]={k:z[k] for k in ['numbers','structure_ids']}
    p=cache[name];n=t['stop']-t['start']
    with np.load(base.with_suffix('.npz')) as z:
        assert z['energy_eV'].shape==(n,) and z['stress_eV_A3'].shape==(n,6)
        assert z['forces_parent_eV_A'].shape==(n,len(p['numbers']),3)
        assert np.array_equal(z['structure_ids'],p['structure_ids'][t['start']:t['stop']])
    checked+=n
rows=list(csv.DictReader((HERE/'splits/lineage_splits.csv').open()));groups={}
for r in rows:
    key=(r['parent'],r['canonical_mask']);assert key not in groups;groups[key]=r['split']
assert len(groups)==32855 and checked==5*32855
(HERE/'validation/scf_identity.json').write_text(json.dumps({'status':'PASS','model_structure_labels':checked,'unique_parent_masks':len(groups),'checks':['alltask/resultidentityfields','allworker/calculatormodelandcheckpointidentities','frozenconfigandparenthashes','explicitEFSshapes','allstructureIDs','onecanonicalorbitpersplit']},indent=2)+'\n')
print('SCF_IDENTITY_PASS',checked,flush=True)

"""Model-specific dynamics worker; preserve all diagnostic and failed segments."""
import argparse,hashlib,json,os,socket,time,traceback
from pathlib import Path
import numpy as np
from ase import Atoms
from calculators import load
from dynamics_v2 import gate_protocol,npt_protocol
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,(float,np.floating)) and not np.isfinite(x):return None
        return x
    p=Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(clean(d),indent=2,allow_nan=False)+'\n');t.replace(p)
def save(p,arrays):
    p=Path(p);t=p.with_suffix('.tmp.npz')
    with t.open('wb') as f:np.savez_compressed(f,**arrays)
    t.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);ap.add_argument('--model',required=True);a=ap.parse_args();assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
    hashes={p.name:sha(p) for p in [Path(__file__),HERE/'dynamics_v2.py',HERE/'calculators.py',HERE/'config.json']};calc,receipt=load(a.model)
    identity=f"{os.environ['SLURM_JOB_ID']}_{socket.gethostname()}_{os.getpid()}";receipt.update(worker=identity,stage=a.stage,source_hashes=hashes);dump(HERE/'workers'/f'{identity}.json',receipt)
    cfg=json.loads((HERE/'config.json').read_text());previous=json.loads((Path(cfg['fixed_pilot'])/'validation'/f'{a.model}.json').read_text())['receipt']
    for key in ['checkpoint_hashes','dtype','tf32','package_version','torch']:assert receipt[key]==previous[key]
    tasks=[json.loads(s) for s in (HERE/'data'/f'{a.stage}_tasks.jsonl').read_text().splitlines()];tasks=[t for t in tasks if t['model']==a.model]
    np.random.default_rng(sum(map(ord,identity))).shuffle(tasks)
    for t in tasks:
        claim=HERE/'claims'/t['task_id']
        try:claim.mkdir()
        except FileExistsError:continue
        dump(claim/'owner.json',dict(worker=identity,started=time.time(),job_id=os.environ['SLURM_JOB_ID']))
        assert sha(t['seed_npz'])==t['seed_sha256']
        with np.load(t['seed_npz']) as z:
            atoms=Atoms(numbers=z['numbers'],positions=z['positions_A'][-1],cell=z['cell_A'][-1],pbc=True);atoms.arrays['parent_slot']=z['parent_slot']
        out=HERE/'data'/a.stage/a.model;out.mkdir(parents=True,exist_ok=True);base=out/t['task_id']
        def checkpoint(results,arrays):
            save(Path(str(base)+'.partial.npz'),arrays);dump(Path(str(base)+'.progress.json'),dict(task_id=t['task_id'],worker=identity,segments=results,updated=time.time()))
        if t['protocol']=='gate':results,arrays=gate_protocol(calc,atoms,t,checkpoint)
        else:results,arrays=npt_protocol(calc,atoms,t,checkpoint)
        save(Path(str(base)+'.npz'),arrays)
        status='complete' if all(r['status']=='complete' for r in results.values()) else 'needs_review'
        dump(Path(str(base)+'.json'),dict(t,worker=identity,status=status,segments=results,finished=time.time(),restart_note='PhysicalNPZpreservesframes; itisnotanexactMTKextended-statecheckpoint.'))
        print('MD_TASK',t['task_id'],status,flush=True)
if __name__=='__main__':main()

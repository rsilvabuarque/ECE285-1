"""Evaluate explicit, common geometries while retaining parent-padded force labels."""
import argparse,hashlib,json,os,socket,time
from pathlib import Path
import numpy as np
from ase import Atoms
from calculators import load,evaluate
from simulation import sync
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);ap.add_argument('--model',required=True);a=ap.parse_args();assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
    hashes={p.name:sha(p) for p in [Path(__file__),HERE/'calculators.py',HERE/'config.json']};calc,receipt=load(a.model);identity=f"{os.environ['SLURM_JOB_ID']}_{socket.gethostname()}_{os.getpid()}";receipt.update(worker=identity,stage=a.stage,source_hashes=hashes);dump(HERE/'workers'/f'{identity}.json',receipt)
    cfg=json.loads((HERE/'config.json').read_text());previous=json.loads((Path(cfg['fixed_pilot'])/'validation'/f'{a.model}.json').read_text())['receipt']
    for key in ['checkpoint_hashes','dtype','tf32','package_version','torch']:assert receipt[key]==previous[key]
    tasks=[json.loads(s) for s in (HERE/'data'/f'{a.stage}_tasks.jsonl').read_text().splitlines()];tasks=[t for t in tasks if t['model']==a.model];cache={}
    np.random.default_rng(sum(map(ord,identity))).shuffle(tasks);out=HERE/'data'/a.stage/a.model;out.mkdir(parents=True,exist_ok=True)
    for t in tasks:
        claim=HERE/'claims'/t['task_id']
        try:claim.mkdir()
        except FileExistsError:continue
        dump(claim/'owner.json',dict(worker=identity,job_id=os.environ['SLURM_JOB_ID'],started=time.time()))
        if t['input_npz'] not in cache:
            assert sha(t['input_npz'])==t['input_sha256']
            with np.load(t['input_npz']) as z:cache[t['input_npz']]={k:z[k] for k in z.files}
        p=cache[t['input_npz']];energy=[];force=[];stress=[];indices=[];statuses=[]
        for i in range(t['start'],t['stop']):
            keep=p['atom_presence'][i].astype(bool)
            if 'valid' in p and not p['valid'][i]:energy.append(np.nan);force.append(np.full((len(keep),3),np.nan));stress.append(np.full(6,np.nan));statuses.append('geometric_guard');indices.append(i);continue
            numbers=p['numbers_parent'][i] if p['numbers_parent'].ndim==2 else p['numbers_parent']
            atoms=Atoms(numbers=numbers[keep],positions=p['positions_parent_A'][i][keep],cell=p['cell_A'][i],pbc=True);calc.reset();e,f,s=evaluate(calc,atoms);ff=np.zeros((len(keep),3),np.float32);ff[keep]=f
            energy.append(e);force.append(ff);stress.append(s);statuses.append('complete');indices.append(i)
        base=out/t['task_id'];tmp=Path(str(base)+'.tmp.npz');np.savez_compressed(tmp,energy_eV=energy,forces_parent_eV_A=force,stress_eV_A3=stress,indices=indices,statuses=statuses,structure_ids=p['structure_ids'][t['start']:t['stop']]);tmp.replace(Path(str(base)+'.npz'))
        dump(Path(str(base)+'.json'),dict(t,worker=identity,status='complete',finished=time.time()));print('EXPLICIT_SCORED',t['task_id'],flush=True)
if __name__=='__main__':main()

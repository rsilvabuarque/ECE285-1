"""Score frozen unique generated masks under all five unchanged teacher calculators."""
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
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--stage',default='generated');a=ap.parse_args();assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
    hashes={p.name:sha(p) for p in [Path(__file__),HERE/'calculators.py',HERE/'config.json']};calc,receipt=load(a.model)
    identity=f"{os.environ['SLURM_JOB_ID']}_{socket.gethostname()}_{os.getpid()}";receipt.update(worker=identity,stage='generated',source_hashes=hashes);dump(HERE/'workers'/f'{identity}.json',receipt)
    cfg=json.loads((HERE/'config.json').read_text());old=json.loads((Path(cfg['fixed_pilot'])/'validation'/f'{a.model}.json').read_text())['receipt']
    for k in ['checkpoint_hashes','dtype','tf32','package_version','torch']:assert receipt[k]==old[k]
    tasks=[json.loads(s) for s in (HERE/'data/generated_tasks.jsonl').read_text().splitlines()];tasks=[t for t in tasks if t['model']==a.model]
    assert sha(tasks[0]['input_npz'])==tasks[0]['input_sha256']
    with np.load(tasks[0]['input_npz']) as z:p={k:z[k] for k in z.files}
    np.random.default_rng(sum(map(ord,identity))).shuffle(tasks);out=HERE/'data/generated'/a.model;out.mkdir(parents=True,exist_ok=True)
    for t in tasks:
        claim=HERE/'claims'/t['task_id']
        try:claim.mkdir()
        except FileExistsError:continue
        dump(claim/'owner.json',dict(worker=identity,job_id=os.environ['SLURM_JOB_ID'],started=time.time()))
        energy=[];forces=[];stress=[];presence=[];seconds=[]
        for v in p['masks'][t['start']:t['stop']]:
            keep=np.ones(len(p['numbers']),bool);keep[p['li_indices'][v.astype(bool)]]=False
            atoms=Atoms(numbers=p['numbers'][keep],positions=p['positions_A'][keep],cell=p['cell_A'],pbc=True);calc.reset();sync();started=time.perf_counter();e,f,s=evaluate(calc,atoms);sync()
            seconds.append(time.perf_counter()-started);full=np.zeros((len(keep),3),np.float32);full[keep]=f
            energy.append(e);forces.append(full);stress.append(s);presence.append(keep)
        base=out/t['task_id'];tmp=Path(str(base)+'.tmp.npz')
        np.savez_compressed(tmp,energy_eV=energy,forces_parent_eV_A=forces,stress_eV_A3=stress,atom_presence=presence,structure_ids=p['structure_ids'][t['start']:t['stop']],seconds=seconds);tmp.replace(Path(str(base)+'.npz'))
        dump(Path(str(base)+'.json'),dict(t,worker=identity,status='complete',seconds=float(sum(seconds)),finished=time.time()))
        print('GENERATED_SCORE',t['task_id'],flush=True)
if __name__=='__main__':main()

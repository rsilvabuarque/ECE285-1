"""One persistent calculator per GPU; atomic claims distribute independent tasks."""
import argparse,csv,hashlib,json,os,socket,time,traceback
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import read
from calculators import load,evaluate
from simulation import relax,sync
HERE=Path(__file__).resolve().parent
def dump(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def save_npz(path,arrays):
    path=Path(path);tmp=path.with_suffix('.tmp.npz')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--stage',required=True);args=ap.parse_args()
    assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
    cfg=json.loads((HERE/'config.json').read_text());calc,receipt=load(args.model)
    worker=f"{os.environ['SLURM_JOB_ID']}_{socket.gethostname()}_{os.getpid()}"
    receipt.update(worker=worker,stage=args.stage,worker_sha256=sha(__file__),simulation_sha256=sha(HERE/'simulation.py'),config_sha256=sha(HERE/'config.json'))
    dump(HERE/'workers'/f'{worker}.json',receipt)
    prior=json.loads((Path(cfg['fixed_pilot'])/'validation'/f'{args.model}.json').read_text());assert prior['passed']
    for key in ['checkpoint_hashes','dtype','tf32','package_version','torch']:
        assert receipt[key]==prior['receipt'][key],f'Calculator provenance changed:{key}'
    assert sha(HERE/'calculators.py')==prior['receipt']['calculator_source_sha256']
    warm=read(Path(cfg['fixed_pilot'])/'inputs/structure_049.extxyz');sync();started=time.perf_counter();evaluate(calc,warm);sync()
    receipt['warmup_seconds']=time.perf_counter()-started;dump(HERE/'workers'/f'{worker}.json',receipt)
    tasks=[json.loads(s) for s in (HERE/'data'/f'{args.stage}_tasks.jsonl').read_text().splitlines() if s]
    tasks=[t for t in tasks if t['model']==args.model];rng=np.random.default_rng(sum(map(ord,worker)));rng.shuffle(tasks)
    parent_cache={}
    for task in tasks:
        claim=HERE/'claims'/task['task_id']
        try:claim.mkdir()
        except FileExistsError:continue
        dump(claim/'owner.json',dict(worker=worker,started=time.time(),job_id=os.environ['SLURM_JOB_ID']))
        out=HERE/'data'/('labels' if args.stage=='scf' else args.stage)/args.model;out.mkdir(parents=True,exist_ok=True)
        result=dict(task,worker=worker,job_id=os.environ['SLURM_JOB_ID'])
        try:
            if args.stage=='scf':
                name=task['parent']
                if name not in parent_cache:
                    with np.load(HERE/'data/parents'/f'{name}.npz') as z:parent_cache[name]={k:z[k] for k in z.files}
                p=parent_cache[name];lo,hi=task['start'],task['stop'];energies=[];forces=[];stresses=[];times=[];presence=[]
                for v in p['masks'][lo:hi]:
                    keep=np.ones(len(p['numbers']),bool);keep[p['li_indices'][v.astype(bool)]]=False
                    a=Atoms(numbers=p['numbers'][keep],positions=p['positions_A'][keep],cell=p['cell_A'],pbc=True)
                    calc.reset();sync();started=time.perf_counter();e,f,s=evaluate(calc,a);sync();times.append(time.perf_counter()-started)
                    fp=np.zeros((len(keep),3),np.float32);fp[keep]=f;energies.append(e);forces.append(fp);stresses.append(s);presence.append(keep)
                arrays=dict(energy_eV=np.array(energies),forces_parent_eV_A=np.array(forces),stress_eV_A3=np.array(stresses),atom_presence=np.array(presence),seconds=np.array(times),structure_ids=p['structure_ids'][lo:hi],parent_indices=np.arange(lo,hi))
                result.update(status='complete',structures=hi-lo,seconds=float(sum(times)))
            else:
                a=read(task['input_file']);r,arrays=relax(calc,a,cell=task.get('cell',False),optimizer=task.get('optimizer','FIRE'),factor=task.get('factor',.25),steps=task.get('max_steps',500),stride=task.get('stride',5))
                result.update(r)
            save_npz(out/(task['task_id']+'.npz'),arrays)
        except Exception as exc:
            result.update(status='exception',error=repr(exc),traceback=traceback.format_exc());dump(out/(task['task_id']+'.json'),result);raise
        result['finished']=time.time();dump(out/(task['task_id']+'.json'),result)
        print('TASK',task['task_id'],result['status'],flush=True)
if __name__=='__main__':main()

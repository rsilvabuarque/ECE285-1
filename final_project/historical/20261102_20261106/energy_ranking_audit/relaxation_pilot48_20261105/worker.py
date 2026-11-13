"""Persistent single-GPU calculator; atomic claims dynamically distribute tasks."""
import argparse,csv,hashlib,json,os,socket,time,traceback
from pathlib import Path
import numpy as np
from ase.calculators.calculator import Calculator,all_changes
from ase.io import read,write
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE
from ase.geometry import find_mic
from calculators import load,evaluate

HERE=Path(__file__).resolve().parent
def dump(p,d):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sync():
    import torch
    torch.cuda.synchronize()

class TimedCalculator(Calculator):
    implemented_properties=['energy','free_energy','forces','stress']
    def __init__(self,base):
        super().__init__();self.base=base;self.calls=0;self.seconds=0.
    def calculate(self,atoms=None,properties=None,system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        self.base.reset();sync();t=time.perf_counter()
        e,f,s=evaluate(self.base,atoms.copy());sync()
        self.seconds+=time.perf_counter()-t;self.calls+=1
        self.results=dict(energy=e,free_energy=e,forces=f,stress=s)

def validate(calc,a):
    """Two-step FP32 derivative probes and repeated-cell invariance.

    Bounds describe floating-point/interface checks, not agreement with DFT.
    Energy cancellation uses an explicit FP32 roundoff estimate. The repeat
    force bound is 2% of the provisional 0.05 eV/A stopping target.
    """
    calc.reset();e,f,s=evaluate(calc,a.copy());i,j=np.unravel_index(abs(f).argmax(),f.shape)
    fd=[];roundoff=[]
    for h in [0.01,0.005]:
        ap,am=a.copy(),a.copy();ap.positions[i,j]+=h;am.positions[i,j]-=h
        calc.reset();ep=evaluate(calc,ap)[0];calc.reset();em=evaluate(calc,am)[0]
        fd.append(-(ep-em)/(2*h));roundoff.append(4*np.finfo(np.float32).eps*max(abs(ep),abs(em))/h)
    sd=[];sr=[]
    for h in [0.002,0.001]:
        ap,am=a.copy(),a.copy();ap.set_cell(a.cell.array*(1+h),scale_atoms=True);am.set_cell(a.cell.array*(1-h),scale_atoms=True)
        calc.reset();ep=evaluate(calc,ap)[0];calc.reset();em=evaluate(calc,am)[0]
        sd.append((ep-em)/(6*h*a.get_volume()));sr.append(4*np.finfo(np.float32).eps*max(abs(ep),abs(em))/(3*h*a.get_volume()))
    fb=max(roundoff[-1],10*abs(fd[1]-fd[0]));sb=max(sr[-1],10*abs(sd[1]-sd[0]))
    tiled=a.repeat((2,1,1));calc.reset();er,fr,ss=evaluate(calc,tiled)
    dif=dict(energy_eV_atom=abs(er/len(tiled)-e/len(a)),force_eV_A=float(abs(fr-np.tile(f,(2,1))).max()),stress_eV_A3=float(abs(ss-s).max()))
    passed=abs(fd[-1]-f[i,j])<=fb and abs(sd[-1]-np.mean(s[:3]))<=sb and dif['energy_eV_atom']<=1e-4 and dif['force_eV_A']<=1e-3 and dif['stress_eV_A3']<=1e-5
    return dict(passed=bool(passed),force_analytic=float(f[i,j]),force_FD=fd,force_bound=float(fb),
        mean_stress=float(np.mean(s[:3])),stress_FD=sd,stress_bound=float(sb),repeat_differences=dif,
        bounds_role='FP32 numerical/interface consistency; not DFT accuracy or physical convergence')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
    config=json.loads((HERE/'config.json').read_text());rows=list(csv.DictReader((HERE/'manifest.csv').open()))
    calc,receipt=load(args.model);ident=f"{os.environ['SLURM_JOB_ID']}_{socket.gethostname()}_{os.getpid()}"
    receipt.update(worker=ident,worker_source_sha256=sha(__file__),calculator_source_sha256=sha(HERE/'calculators.py'),config_sha256=sha(HERE/'config.json'))
    dump(HERE/'workers'/f'{ident}.json',receipt)
    warm=read(next(r['input_file'] for r in rows if r['n_Li']=='24'))
    sync();t=time.perf_counter();evaluate(calc,warm);sync();receipt['warmup_seconds']=time.perf_counter()-t
    dump(HERE/'workers'/f'{ident}.json',receipt)
    if args.validate_only:
        result=validate(calc,warm);result['model']=args.model;result['receipt']=receipt
        dump(HERE/'validation'/f'{args.model}.json',result)
        print('VALIDATION',args.model,result,flush=True)
        if not result['passed']:raise RuntimeError('Numerical validation failed')
        return
    validation=json.loads((HERE/'validation'/f'{args.model}.json').read_text());assert validation['passed']
    for field in ['checkpoint_hashes','dtype','tf32','package_version','torch','calculator_source_sha256','config_sha256']:
        assert receipt[field]==validation['receipt'][field], f'Validation signature mismatch: {field}'
    rng=np.random.default_rng(20261105+sum(map(ord,ident)));rng.shuffle(rows)
    for row in rows:
        sid=row['structure_id'];task=f'{args.model}__{sid}';claim=HERE/'claims'/task
        try:claim.mkdir()
        except FileExistsError:continue
        dump(claim/'owner.json',dict(worker=ident,job_id=os.environ['SLURM_JOB_ID'],started=time.time()))
        out=HERE/'results'/args.model/sid;out.mkdir(parents=True)
        result=dict(model=args.model,structure_id=sid,n_Li=int(row['n_Li']),x=float(row['x']),n_atoms=int(row['n_atoms']),worker=ident,job_id=os.environ['SLURM_JOB_ID'],input_sha256=row['input_sha256'])
        try:
            assert sha(row['input_file'])==row['input_sha256']
            atoms=read(row['input_file']);startpos=atoms.positions.copy();timed=TimedCalculator(calc);atoms.calc=timed
            latencies=[]
            for _ in range(config['single_point_repeats']):
                timed.reset();sync();t=time.perf_counter();e0,f0,s0=evaluate(timed,atoms);sync();latencies.append(time.perf_counter()-t)
            np.savez_compressed(out/'initial.npz',energy_eV=e0,forces_eV_A=f0,stress_eV_A3=s0,positions_A=startpos,cell_A=atoms.cell.array,numbers=atoms.numbers)
            call0,sec0=timed.calls,timed.seconds
            target=atoms
            if config['relax_cell']:
                from ase.filters import FrechetCellFilter
                target=FrechetCellFilter(atoms,scalar_pressure=0.)
            opt=FIRE(target,logfile=str(out/'optimizer.log'),maxstep=config['maxstep_A'])
            traj=Trajectory(str(out/'relaxation.traj'),'w',atoms)
            opt.attach(traj.write,interval=1)
            sync();t=time.perf_counter();converged=bool(opt.run(fmax=config['fmax_eV_A'],steps=config['max_steps']));sync();wall=time.perf_counter()-t
            relax_seconds=timed.seconds-sec0;relax_calls=timed.calls-call0
            traj.close();e1,f1,s1=evaluate(timed,atoms)
            np.savez_compressed(out/'final.npz',energy_eV=e1,forces_eV_A=f1,stress_eV_A3=s1,positions_A=atoms.positions,cell_A=atoms.cell.array,numbers=atoms.numbers)
            write(out/'final.extxyz',atoms);write(out/'final.vasp',atoms,format='vasp',sort=True,direct=True)
            _,displacements=find_mic(atoms.positions-startpos,atoms.cell,pbc=True)
            result.update(status='converged' if converged else 'step_limit',converged=converged,
                single_point_seconds=latencies,single_point_mean_s=float(np.mean(latencies)),relaxation_wall_s=wall,
                relaxation_calculator_s=relax_seconds,relaxation_evaluations=relax_calls,optimizer_steps=opt.nsteps,
                initial_energy_eV=e0,final_energy_eV=e1,energy_change_eV=e1-e0,
                initial_fmax_eV_A=float(np.linalg.norm(f0,axis=1).max()),final_fmax_eV_A=float(np.linalg.norm(f1,axis=1).max()),
                initial_stress_eV_A3=s0.tolist(),final_stress_eV_A3=s1.tolist(),
                rms_displacement_A=float(np.sqrt(np.mean(displacements**2))),max_displacement_A=float(displacements.max()),
                final_volume_A3=float(atoms.get_volume()),finished=time.time())
        except Exception as exc:
            result.update(status='exception',converged=False,error=repr(exc),traceback=traceback.format_exc(),finished=time.time())
            print(result['traceback'],flush=True)
        dump(out/'result.json',result);print('TASK_DONE',task,result['status'],result.get('relaxation_wall_s'),flush=True)

if __name__=='__main__':main()

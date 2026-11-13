"""Shared conservative E/F/stress evaluation and physically stopped relaxation."""
import time
import numpy as np
from ase.calculators.calculator import Calculator,all_changes
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE,LBFGS
from ase.stress import voigt_6_to_full_3x3_stress
from ase.geometry import find_mic
from calculators import evaluate
GPA_PER_EV_A3=160.21766208
def stress_max_GPa(s):return float(abs(np.linalg.eigvalsh(voigt_6_to_full_3x3_stress(s))).max()*GPA_PER_EV_A3)
def sync():
    import torch
    torch.cuda.synchronize()
class TimedCalculator(Calculator):
    implemented_properties=['energy','free_energy','forces','stress']
    def __init__(self,base):super().__init__();self.base=base;self.calls=0;self.seconds=0.
    def calculate(self,atoms=None,properties=None,system_changes=all_changes):
        super().calculate(atoms,properties,system_changes);self.base.reset();sync();t=time.perf_counter()
        e,f,s=evaluate(self.base,atoms.copy());sync();self.seconds+=time.perf_counter()-t;self.calls+=1
        self.results=dict(energy=e,free_energy=e,forces=f,stress=s)
class PhysicalConvergence:
    def __init__(self,target,physical_atoms,stress_target=None,**kwargs):
        self.physical_atoms=physical_atoms;self.stress_target=stress_target;super().__init__(target,**kwargs)
    def converged(self,gradient):
        # A common generalized-force bound would change the physical criterion
        # when cell scaling changes. Test physical force/stress directly instead.
        fmax=np.linalg.norm(self.physical_atoms.get_forces(),axis=1).max()
        return bool(fmax<self.fmax and (self.stress_target is None or stress_max_GPa(self.physical_atoms.get_stress())<self.stress_target))
class PhysicalFIRE(PhysicalConvergence,FIRE):pass
class PhysicalLBFGS(PhysicalConvergence,LBFGS):pass

def relax(calc,atoms,cell=False,optimizer='FIRE',factor=.25,steps=500,fmax=.05,stress=.1,stride=5,initial_repeats=3):
    a=atoms.copy();startpos=a.positions.copy();startcell=a.cell.array.copy();startfrac=a.get_scaled_positions(wrap=False)
    timed=TimedCalculator(calc);a.calc=timed;lat=[]
    for _ in range(initial_repeats):
        timed.reset();sync();t=time.perf_counter();e0,f0,s0=evaluate(timed,a);sync();lat.append(time.perf_counter()-t)
    target=FrechetCellFilter(a,exp_cell_factor=factor*len(a),scalar_pressure=0.) if cell else a
    cls={'FIRE':PhysicalFIRE,'LBFGS':PhysicalLBFGS}[optimizer]
    opt=cls(target,a,stress_target=stress if cell else None,logfile=None,maxstep=.15)
    snapshots=[];trace=[];started=time.perf_counter();calls0=timed.calls;seconds0=timed.seconds
    def capture(force=False):
        e,f,s=evaluate(timed,a);fm=float(np.linalg.norm(f,axis=1).max());sm=stress_max_GPa(s)
        if not trace or trace[-1][0]!=opt.nsteps:
            trace.append([opt.nsteps,e,fm,sm,a.get_volume(),time.perf_counter()-started,timed.calls-calls0])
        if force or opt.nsteps%stride==0:
            if snapshots and snapshots[-1][0]==opt.nsteps:return
            snapshots.append((opt.nsteps,e,a.positions.copy(),f.copy(),s.copy(),a.cell.array.copy()))
    opt.attach(capture,interval=1);sync();t=time.perf_counter();converged=bool(opt.run(fmax=fmax,steps=steps));sync();wall=time.perf_counter()-t
    capture(force=True);e1,f1,s1=evaluate(timed,a)
    gf=float(np.linalg.norm(target.get_forces(),axis=1).max())
    _,d=find_mic(a.positions-startfrac@a.cell.array,a.cell,pbc=True)
    arrays=dict(steps=np.array([r[0] for r in snapshots]),energy_eV=np.array([r[1] for r in snapshots]),positions_A=np.array([r[2] for r in snapshots]),forces_eV_A=np.array([r[3] for r in snapshots]),stress_eV_A3=np.array([r[4] for r in snapshots]),cell_A=np.array([r[5] for r in snapshots]),numbers=a.numbers,parent_slot=a.arrays.get('parent_slot',np.arange(len(a))),trace=np.array(trace),initial_positions_A=startpos,initial_cell_A=startcell,initial_energy_eV=e0,initial_forces_eV_A=f0,initial_stress_eV_A3=s0)
    result=dict(status='converged' if converged else 'step_limit',converged=converged,cell_relaxed=cell,optimizer=optimizer,cell_factor_over_N=factor if cell else None,steps=opt.nsteps,calculator_calls=timed.calls-calls0,calculator_seconds=timed.seconds-seconds0,wall_seconds=wall,initial_seconds=lat,initial_energy_eV=e0,final_energy_eV=e1,final_fmax_eV_A=float(np.linalg.norm(f1,axis=1).max()),final_stress_max_GPa=stress_max_GPa(s1),final_generalized_fmax=gf,volume_ratio=float(a.get_volume()/abs(np.linalg.det(startcell))),internal_rms_displacement_A=float(np.sqrt(np.mean(d*d))),internal_max_displacement_A=float(d.max()),snapshots=len(snapshots))
    assert converged==bool(result['final_fmax_eV_A']<fmax and (not cell or result['final_stress_max_GPa']<stress))
    return result,arrays

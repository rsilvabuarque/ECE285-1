"""Bounded MLIP dynamics with explicit ensemble, units and raw-force labels."""
import time
import numpy as np
from ase import units
from ase.calculators.calculator import Calculator,all_changes
from ase.constraints import FixCom
from ase.md.bussi import Bussi
from ase.md.verlet import VelocityVerlet
from ase.md.nose_hoover_chain import MTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.parallel import DummyMPI
from calculators import evaluate
from simulation import sync,stress_max_GPa

class ExplorationLimit(RuntimeError):pass
class RawMDCalculator(Calculator):
    implemented_properties=['energy','free_energy','forces','stress']
    def __init__(self,base):super().__init__();self.base=base;self.calls=0
    def calculate(self,atoms=None,properties=None,system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        raw=atoms.copy();raw.set_constraint();assert np.isfinite(raw.positions).all() and np.isfinite(raw.cell.array).all()
        self.base.reset();e,f,s=evaluate(self.base,raw);self.calls+=1
        self.results=dict(energy=e,free_energy=e,forces=f,stress=s)

def initialize(seed_atoms,base,T,seed,fix_com=False):
    a=seed_atoms.copy();a.set_constraint(FixCom() if fix_com else []);a.calc=RawMDCalculator(base)
    rng=np.random.default_rng(seed);MaxwellBoltzmannDistribution(a,temperature_K=T,rng=rng,force_temp=False,comm=DummyMPI())
    return a

def simulate(a,base,ensemble,duration_ps,dt_fs,T,P_MPa,seed,snapshot_fs=10.,checkpoint=None):
    """Callbacks read physical atoms only; MTK internal states are never modified."""
    a.calc=RawMDCalculator(base);comm=DummyMPI();rng=np.random.default_rng(seed)
    if ensemble=='NVT':dyn=Bussi(a,timestep=dt_fs*units.fs,temperature_K=T,taut=100*units.fs,rng=rng,comm=comm)
    elif ensemble=='NVE':dyn=VelocityVerlet(a,timestep=dt_fs*units.fs,comm=comm)
    elif ensemble=='NPT':
        assert not a.constraints
        dyn=MTKNPT(a,timestep=dt_fs*units.fs,temperature_K=T,pressure_au=P_MPa*1e6/1.602176634e11,tdamp=100*units.fs,pdamp=1000*units.fs,tchain=3,pchain=3,tloop=1,ploop=1,comm=comm)
    else:raise ValueError(ensemble)
    steps=int(round(duration_ps*1000/dt_fs));snap_stride=max(1,round(snapshot_fs/dt_fs));guard_stride=max(1,round(100/dt_fs))
    h0=a.cell.array.copy();v0=a.get_volume();trace=[];snapshots=[];started=time.perf_counter();status='complete';error=None
    def capture(force=False):
        e=float(a.get_potential_energy());k=float(a.get_kinetic_energy());f=a.get_forces(apply_constraint=False);s=a.get_stress(include_ideal_gas=False);st=a.get_stress(include_ideal_gas=True)
        extended=float(dyn.get_conserved_energy()) if ensemble=='NPT' else np.nan
        heat=float(dyn.transferred_energy) if ensemble=='NVT' else 0.
        com=(a.get_masses()[:,None]*a.get_velocities()).sum(axis=0)/a.get_masses().sum()*units.fs
        if not trace or trace[-1][0]!=dyn.nsteps:
            trace.append([dyn.nsteps,dyn.nsteps*dt_fs,e,k,e+k,a.get_temperature(),-float(st[:3].mean())*160217.66208,-float(s[:3].mean())*160217.66208,a.get_volume(),float(np.linalg.norm(f,axis=1).max()),extended,heat,float(np.linalg.norm(com))])
        if force or dyn.nsteps%snap_stride==0:
            if not snapshots or snapshots[-1][0]!=dyn.nsteps:snapshots.append((dyn.nsteps,a.positions.copy(),a.get_velocities().copy()*units.fs,f.copy(),s.copy(),st.copy(),a.cell.array.copy()))
        if dyn.nsteps%guard_stride==0:
            stretches=np.linalg.svd(np.linalg.solve(h0,a.cell.array),compute_uv=False)
            if not (.25<a.get_volume()/v0<4.) or stretches.max()>3. or stretches.min()<.25:
                raise ExplorationLimit('Resourceguard: volume outside[0.25,4]xseed or principal stretch outside[0.25,3]; not a phase/stability classifier')
            distances=a.get_all_distances(mic=True);np.fill_diagonal(distances,np.inf)
            if distances.min()<.6:raise ExplorationLimit('Resourceguard: nearest separation below0.6A; preserve for review')
    dyn.attach(capture,interval=1)
    try:dyn.run(steps)
    except ExplorationLimit as exc:status='exploratory_limit';error=str(exc)
    except Exception as exc:status='numerical_failure';error=repr(exc)
    try:capture(force=True)
    except Exception:pass
    arrays=dict(trace=np.array(trace),snapshot_steps=np.array([s[0] for s in snapshots]),positions_A=np.array([s[1] for s in snapshots]),velocities_A_fs=np.array([s[2] for s in snapshots]),forces_eV_A=np.array([s[3] for s in snapshots]),configurational_stress_eV_A3=np.array([s[4] for s in snapshots]),total_stress_eV_A3=np.array([s[5] for s in snapshots]),cell_A=np.array([s[6] for s in snapshots]),numbers=a.numbers,parent_slot=a.arrays['parent_slot'],last_positions_A=a.positions,last_cell_A=a.cell.array,last_velocities_A_fs=a.get_velocities()*units.fs)
    result=dict(status=status,error=error,ensemble=ensemble,target_temperature_K=T,target_pressure_MPa=P_MPa if ensemble=='NPT' else None,dt_fs=dt_fs,requested_duration_ps=duration_ps,completed_steps=dyn.nsteps,completed_duration_ps=dyn.nsteps*dt_fs/1000,physical_degrees_of_freedom=a.get_number_of_degrees_of_freedom(),COM_constraint=bool(a.constraints),wall_seconds=time.perf_counter()-started,calculator_calls=a.calc.calls,snapshots=len(snapshots),final_volume_ratio=float(a.get_volume()/v0))
    if trace:
        tr=np.array(trace);energy=tr[:,4]/len(a)*1000;times=tr[:,1]/1000
        result.update(physical_total_energy_range_meV_atom=float(np.ptp(energy)),physical_total_energy_end_drift_meV_atom=float(energy[-1]-energy[0]),physical_total_energy_slope_meV_atom_ps=float(np.polyfit(times,energy,1)[0]) if len(times)>1 else None,mean_temperature_K=float(tr[:,5].mean()),mean_total_pressure_MPa=float(tr[:,6].mean()),mean_COM_speed_A_fs=float(tr[:,12].mean()))
    if checkpoint:checkpoint(result,arrays)
    return result,arrays,a

def gate_protocol(base,seed_atoms,task,checkpoint):
    seed=task['seed'];T=493.15;dtwarm=task.get('warm_dt_fs',1.);warm_ps=task.get('warm_ps',1.);nve_ps=task.get('nve_ps',2.)
    a=initialize(seed_atoms,base,T,seed,fix_com=True);results={};arrays={}
    r,z,reference=simulate(a,base,'NVT',warm_ps,dtwarm,T,0.,seed+1,snapshot_fs=10.);results['warm']=r;arrays.update({'warm_'+k:v for k,v in z.items()});checkpoint(results,arrays)
    if r['status']!='complete':return results,arrays
    for name,dt in [('dt1',1.),('dt0p5',.5)]:
        trial=reference.copy();r,z,_=simulate(trial,base,'NVE',nve_ps,dt,T,0.,seed+2,snapshot_fs=10.)
        results[name]=r;arrays.update({name+'_'+k:v for k,v in z.items()});checkpoint(results,arrays)
    return results,arrays

def npt_protocol(base,seed_atoms,task,checkpoint):
    T=task['T_K'];P=task['P_MPa'];seed=task['seed'];dt=task['dt_fs'];a=initialize(seed_atoms,base,T,seed,fix_com=False)
    results={};arrays={}
    r,z,a=simulate(a,base,'NVT',task.get('warm_ps',1.),dt,T,P,seed+1,snapshot_fs=20.);results['warm']=r;arrays.update({'warm_'+k:v for k,v in z.items()});checkpoint(results,arrays)
    if r['status']=='complete':
        r,z,_=simulate(a,base,'NPT',task['duration_ps'],dt,T,P,seed+2,snapshot_fs=20.);results['npt']=r;arrays.update({'npt_'+k:v for k,v in z.items()});checkpoint(results,arrays)
    return results,arrays

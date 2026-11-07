"""Analyze an unchanged completed trajectory on a declared coarser velocity grid.

This calls the frozen actual-ATLAS wrapper API. Integration timestep stays in
producer metadata; observation spacing follows the retained velocity indices.
No trajectory, producer file, package source, model, or MD calculation changes.
"""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
import numpy as np

WRAPPER_SHA='1019fb739af11cf107dae2967fd39fd1ce3dbb00d7789661d4539a220707a894'
SOURCE=Path(__file__).read_bytes()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['trajectory','hessian','minimum','vendor','output-dir','raw-output-dir']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--wrapper',type=Path,default=Path('scripts/overnight_20261107/thermal_xpt.py'))
    p.add_argument('--stride',type=int,default=2);p.add_argument('--max-duration-ps',type=float,default=5.)
    a=p.parse_args()
    if os.environ.get('NERSC_HOST') and not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use the authorized compute allocation')
    if a.stride<1 or a.max_duration_ps<=0:raise ValueError('Positive stride/window required')
    if sha(a.wrapper)!=WRAPPER_SHA:raise ValueError('Frozen wrapper changed')
    spec=importlib.util.spec_from_file_location('frozen_xpt',a.wrapper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=module.audit_vendor(a.vendor);producer_path=a.trajectory.with_name('receipt.json');producer=json.loads(producer_path.read_text())
    if producer['status']!='COMPLETE':raise ValueError('Only a completed producer trajectory can be used')
    a.output_dir.mkdir(parents=True,exist_ok=False);(a.output_dir/'analysis_source.py').write_bytes(SOURCE);(a.output_dir/'frozen_wrapper_source.py').write_bytes(a.wrapper.read_bytes())
    d=module.velocity_axis(module.load_npz(a.trajectory));original_dt=module.validate_trajectory(d)
    index=np.arange(len(d['time_fs']));index=index[d['time_fs']<=d['time_fs'][0]+a.max_duration_ps*1000][::a.stride]
    np.savez_compressed(a.output_dir/'observed_indices.npz',velocity_indices=index,velocity_time_fs=d['time_fs'][index])
    h,minimum=module.bound_hessian(a.hessian,a.minimum);frequencies=h['frequencies_signed_THz'];energy=module.minimum_energy(minimum)
    if not np.allclose(d['cell_A'],minimum['cell_A'],rtol=0,atol=1e-10):raise ValueError('Common cell differs')
    rows=[]
    integration_dt=float(producer['arguments']['timestep_fs'])
    for mode in ['1PT','2PT','3PT']:
        row=module.run_package(d,index,a.raw_output_dir/mode,mode=mode,corlen=.25,target_K=433.15)
        row.update(basin_id='basin_1',repeat_id='half_dt_control',scope='half_dt_integrator025_observation05_control_native_only',integration_timestep_fs=integration_dt,original_velocity_interval_fs=original_dt,observation_stride=a.stride)
        module.attach_harmonic(row,frequencies,energy,433.15);rows.append(row)
    module.save_rows(a.output_dir/'native_results.csv',rows)
    receipt=dict(status='CALCULATED_MATCHED_OBSERVATION_CONTROL',input_trajectory=str(a.trajectory),input_trajectory_sha256=sha(a.trajectory),producer_receipt=str(producer_path),producer_receipt_sha256=sha(producer_path),hessian_sha256=sha(a.hessian),minimum_sha256=sha(a.minimum),source_sha256=hashlib.sha256(SOURCE).hexdigest(),wrapper_sha256=WRAPPER_SHA,ATLAS_source=source,integration_timestep_fs=integration_dt,original_velocity_interval_fs=original_dt,observation_stride=a.stride,observation_interval_fs=original_dt*a.stride,retained_frames=len(index),observed_indices_sha256=sha(a.output_dir/'observed_indices.npz'),first_observed_time_fs=float(d['time_fs'][index[0]]),last_observed_time_fs=float(d['time_fs'][index[-1]]),mean_total_energy_on_exact_retained_frames_eV=float(d['energy_total_eV'][index].mean()),maximum_physical_lag_ps=1.25,limitations=['No source trajectory or producer metadata rewritten','Only observation-grid sensitivity is changed relative to native half-dt analysis','Other timestep/thermostat comparisons retain temperature and warmup-path confounding','This does not establish causal timestep convergence order or physical free-energy accuracy'])
    (a.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(status=receipt['status'],frames=len(index),output=str(a.output_dir))))

if __name__=='__main__':main()

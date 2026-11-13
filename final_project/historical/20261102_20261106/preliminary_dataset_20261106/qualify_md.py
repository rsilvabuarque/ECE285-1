"""Audit paired NVE states and publish model-specific timestep qualifications."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
from ase import units
HERE=Path(__file__).resolve().parent
def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');a=ap.parse_args();assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
    cfg=json.loads((HERE/'config.json').read_text());tasks=[json.loads(s) for s in (HERE/'data/md_gate_tasks.jsonl').read_text().splitlines()]
    previous=None
    while True:
        cases=[]
        for t in tasks:
            base=HERE/'data/md_gate'/t['model']/t['task_id'];j=Path(str(base)+'.json')
            if not j.exists():continue
            r=json.loads(j.read_text());branches={};valid=r['status']=='complete'
            with np.load(Path(str(base)+'.npz')) as z:
                if valid:
                    for key in ['positions_A','cell_A','velocities_A_fs']:
                        assert np.array_equal(z['dt1_'+key][0],z['dt0p5_'+key][0])
                    for name,dt in [('dt1',1.),('dt0p5',.5)]:
                        tr=z[name+'_trace'];N=len(z[name+'_numbers']);dof=r['segments'][name]['physical_degrees_of_freedom']
                        assert np.isfinite(tr[:,[0,1,2,3,4,5,6,7,8,9,11,12]]).all()
                        assert np.allclose(tr[:,4],tr[:,2]+tr[:,3],atol=1e-9,rtol=0)
                        assert len(tr)==round(t['nve_ps']*1000/dt)+1
                        assert np.allclose(tr[:,1],np.arange(len(tr))*dt,atol=1e-9,rtol=0)
                        energy=tr[:,4]/N*1000;bound=.01*(dof/(2*N))*units.kB*493.15*1000
                        bound=float(bound)
                        spread=float(np.ptp(energy));end=float(energy[-1]-energy[0]);noise_estimate=float(4*np.finfo(np.float32).eps*float(abs(tr[:,2]).max())/N*1000)
                        branches[name]=dict(range_meV_atom=spread,end_drift_meV_atom=end,bound_meV_atom=bound,estimated_FP32_cancellation_scale_meV_atom=noise_estimate,passed=bool(spread<=bound and abs(end)<=bound))
            cases.append(dict(task_id=t['task_id'],model=t['model'],n_Li=t['n_Li'],complete=valid,branches=branches))
        qualifications={}
        for m in cfg['models']:
            rows=[r for r in cases if r['model']==m]
            if len(rows)<2:continue
            half=all(r['complete'] and r['branches']['dt0p5']['passed'] for r in rows)
            full=half and all(r['branches']['dt1']['passed'] for r in rows)
            qualifications[m]=dict(status='NUMERICALLY_QUALIFIED' if half else 'HOLD',dt_fs=1. if full else .5 if half else None,states_tested=2,compositions=[r['n_Li']/48 for r in rows],temperature_K=493.15)
        result=dict(status='COMPLETE' if len(cases)==10 else 'PARTIAL',terminal_cases=len(cases),qualifications=qualifications,cases=cases,scope='Representativeconservativeintegrationcheck; notDFTaccuracy,longrunstability,orthermodynamicvalidation. FP32scaleisanestimatedcancellationbound, notmeasuredrepeatnoise.')
        atomic(HERE/'validation/md_gate_partial.json',result)
        if len(cases)==10:atomic(HERE/'validation/md_gate.json',result)
        state=(len(cases),sorted(qualifications))
        if state!=previous:print('MD_QUALIFICATION',len(cases),qualifications,flush=True);previous=state
        if len(cases)==10 or not a.watch:break
        time.sleep(30)
if __name__=='__main__':main()

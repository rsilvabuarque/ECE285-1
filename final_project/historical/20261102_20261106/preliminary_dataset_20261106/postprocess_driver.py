"""Use the existing shared allocation for bounded diagnostics and derived exports."""
import json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE=Path(__file__).resolve().parent
PY=HERE.parent/'energy_ranking_audit/multimodel_20261104/envs/uma/bin/python'
def run(tag,args):
    with (HERE/'logs'/f'{tag}.log').open('w') as f:subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,check=True,cwd=HERE)
def export(pair):
    model,cohort=pair
    if (HERE/'validation'/f'geometry_{model}_{cohort}.json').exists():return
    run(f'export_geometry_{model}_{cohort}',[PY,HERE/'export_geometry.py','--model',model,'--cohort',cohort])
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
assert (HERE/'data/diagnostics_design.json').exists()
base=['srun','--overlap','-N1','-n2','--ntasks-per-node=2','-c8','--gpus-per-task=1','--gpu-bind=single:1',sys.executable]
run('phase_context_dispatch',base+[HERE/'explicit_dispatch.py','--stage','phase_context'])
run('decoded_relax_dispatch',base+[HERE/'dispatch_v2.py','--stage','decoded_relax'])
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(export,[(m,c) for m in ['uma','mace','sevennet','chgnet','m3gnet'] for c in ['fixed','variable']]))
run('build_portable_v2',[PY,HERE/'build_portable.py'])
while not (HERE/'validation/generated.json').exists():
    print('WAITING_GENERATED_AUDIT',flush=True);time.sleep(30)
print('POSTPROCESS_DRIVER_COMPLETE',flush=True)

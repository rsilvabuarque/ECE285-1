"""Additional identical workers; preserve one exporting owner after allocations end."""
import json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;PY=HERE.parent/'energy_ranking_audit/multimodel_20261104/envs/uma/bin/python'
with (HERE/'logs'/f"{os.environ['SLURM_JOB_ID']}_md_parallel.log").open('w') as f:
    subprocess.run(['srun','--overlap','-N4','-n16','--ntasks-per-node=4','-c8','--gpus-per-task=1','--gpu-bind=single:1',sys.executable,str(HERE/'md_dispatch_v2.py'),'--stage','md_prod'],stdout=f,stderr=subprocess.STDOUT,check=True)
while not (HERE/'validation/md_export.json').exists():
    state=subprocess.check_output(['sacct','-X','-j','60897021','-nP','--format=State'],text=True).strip().split('|')[0]
    if state in ['COMPLETED','FAILED','TIMEOUT','CANCELLED']:
        # The earlier export watcher belonged to that allocation. Its raw inputs
        # remain immutable; rerun only the derived export after terminal proof.
        with (HERE/'logs/export_md_after_allocation.log').open('w') as f:subprocess.run([str(PY),str(HERE/'export_md.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
        break
    print('WAITING_SINGLE_MD_EXPORT_OWNER',state,flush=True);time.sleep(30)
print('MD_PARALLEL_COMPLETE',flush=True)

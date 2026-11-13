"""Foreground batch MD phases, using disjoint nodes for concurrent pilot stages."""
import os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
nodes=subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']],text=True).split();assert len(nodes)==4
def args(hosts):return ['srun','--overlap','-N',str(len(hosts)),'-w',','.join(hosts),'-n',str(4*len(hosts)),'--ntasks-per-node=4','-c8','--gpus-per-task=1','--gpu-bind=single:1',sys.executable]
while not (HERE/'data/md_pilot_tasks.jsonl').exists():time.sleep(10)
procs=[]
for tag,hosts,stage in [('extra',nodes[:1],'md_extra_gate'),('pilots',nodes[1:],'md_pilot')]:
    f=(HERE/'logs'/f"{os.environ['SLURM_JOB_ID']}_{tag}.log").open('w');procs.append((subprocess.Popen(args(hosts)+[str(HERE/'md_dispatch_v2.py'),'--stage',stage],stdout=f,stderr=subprocess.STDOUT),f))
for proc,f in procs:code=proc.wait();f.close();assert code==0
with (HERE/'logs'/f"{os.environ['SLURM_JOB_ID']}_prepare_production.log").open('w') as f:subprocess.run([sys.executable,str(HERE/'prepare_md_production.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
with (HERE/'logs'/f"{os.environ['SLURM_JOB_ID']}_production.log").open('w') as f:subprocess.run(args(nodes)+[str(HERE/'md_dispatch_v2.py'),'--stage','md_prod'],stdout=f,stderr=subprocess.STDOUT,check=True)
# Complete any diagnostic pilots added when the slower models qualified.
with (HERE/'logs'/f"{os.environ['SLURM_JOB_ID']}_remaining_pilots.log").open('w') as f:subprocess.run(args(nodes)+[str(HERE/'md_dispatch_v2.py'),'--stage','md_pilot'],stdout=f,stderr=subprocess.STDOUT,check=True)
print('MD_ALLOCATION_COMPLETE',os.environ['SLURM_JOB_ID'],flush=True)

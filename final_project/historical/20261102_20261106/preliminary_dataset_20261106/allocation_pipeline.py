"""Foreground allocation driver; each resource slot advances through bounded stages."""
import json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
nodes=subprocess.check_output(['scontrol','show','hostnames',os.environ['SLURM_JOB_NODELIST']],text=True).split();assert len(nodes)==4
def step_args(hosts):return ['srun','--overlap','-N',str(len(hosts)),'-w',','.join(hosts),'-n',str(4*len(hosts)),'--ntasks-per-node=4','-c8','--gpus-per-task=1','--gpu-bind=single:1',sys.executable]
path=HERE/'data/fixed_resume1_tasks.jsonl'
while not path.exists():print('WAIT_FIXED_RECOVERY_MANIFEST',flush=True);time.sleep(10)
with (HERE/'logs'/f"{os.environ['SLURM_JOB_ID']}_fixed_resume1.log").open('w') as f:
    subprocess.run(step_args(nodes)+[str(HERE/'dispatch_v2.py'),'--stage','fixed_resume1'],stdout=f,stderr=subprocess.STDOUT,check=True)
print('RECOVERY_DISPATCH_RETURNED',flush=True)
processes=[]
for tag,hosts,command in [('generated_a',nodes[:2],[str(HERE/'score_dispatch.py'),'--stage','generated']),('md_generated_b',nodes[2:],[str(HERE/'chain_slot.py'),'md_gate','generated'])]:
    f=(HERE/'logs'/f"{os.environ['SLURM_JOB_ID']}_{tag}.log").open('w')
    processes.append((subprocess.Popen(step_args(hosts)+command,stdout=f,stderr=subprocess.STDOUT),f,tag))
codes=[]
for proc,f,tag in processes:
    code=proc.wait();f.close();codes.append(code);print('PIPELINE_STAGE',tag,code,flush=True)
assert not any(codes),codes
print('ALLOCATION_PIPELINE_COMPLETE',os.environ['SLURM_JOB_ID'],flush=True)

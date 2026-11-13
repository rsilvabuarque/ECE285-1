"""Read Slurm accounting on the host; containers need no Slurm client install."""
import json,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=HERE/'resources.json';d=json.loads(p.read_text());ids=[str(r['job_id']) for key in ['initial_allocations','subsequent_allocations'] for r in d[key]]
raw=subprocess.check_output(['sacct','-X','-j',','.join(ids),'-nP','--format=JobID,State,ElapsedRaw,AllocTRES'],text=True);rows=[]
for line in raw.splitlines():
    a=line.split('|');tres=dict(v.split('=',1) for v in a[3].split(',') if '=' in v);g=int(tres.get('gres/gpu','0'));sec=int(a[2]);rows.append(dict(job_id=int(a[0]),state=a[1],elapsed_seconds=sec,gpus=g,device_hours=g*sec/3600))
assert {r['job_id'] for r in rows}==set(map(int,ids))
d['last_observed']=dict(utc_seconds=time.time(),jobs=rows,allocated_gpu_device_hours=sum(r['device_hours'] for r in rows),allocated_four_gpu_node_equivalent_hours=sum(r['device_hours'] for r in rows)/4,raw_sacct=raw)
tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
assert d['last_observed']['allocated_four_gpu_node_equivalent_hours']<=d['gpu_node_hour_limit']
print('ACCOUNTING',d['last_observed']['allocated_four_gpu_node_equivalent_hours'],[(r['job_id'],r['state']) for r in rows],flush=True)

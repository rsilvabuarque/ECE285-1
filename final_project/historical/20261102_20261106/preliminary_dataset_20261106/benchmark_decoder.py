"""Resident decoder timing, separately from oracle/relaxation workflow cost."""
import json,os,sys,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;PACK=HERE/'collaborator_package';sys.path.insert(0,str(PACK/'cvae'))
from model import MaskCVAE
from geometry_model import DisplacementModel
from sampling import conditional_count,top_count
assert os.environ.get('SLURMD_NODENAME','').startswith('nid') and torch.cuda.is_available()
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
for name,checkpoint in [('mask','uma_cvae_main_20261106'),('geometry','cvae_20261400')]:
    ck=torch.load(PACK/'checkpoints'/checkpoint/'best.pt',map_location='cpu',weights_only=False)
    for device in ['cuda','cpu']:
        model=(MaskCVAE() if name=='mask' else DisplacementModel()).to(device);model.load_state_dict(ck['state_dict']);model.eval()
        for B in [1,100]:
            z=torch.zeros((B,8 if name=='mask' else 16),device=device);c=torch.zeros((B,2 if name=='mask' else 65),device=device);times=[]
            with torch.no_grad():
                for j in range(105):
                    if device=='cuda':torch.cuda.synchronize()
                    started=time.perf_counter();output=model.decode(z,c)
                    if device=='cuda':torch.cuda.synchronize()
                    elapsed=time.perf_counter()-started
                    if j>=5:times.append(elapsed)
            rows.append(dict(model=name,device=device,batch=B,median_batch_seconds=float(np.median(times)),p95_batch_seconds=float(np.quantile(times,.95)),repeats=100))
            if name=='mask' and device=='cpu':
                logits=output.numpy();rng=np.random.default_rng(20261106)
                for method,fn in [('top_count',top_count),('conditional_count_DP',conditional_count)]:
                    times=[]
                    for _ in range(10):
                        started=time.perf_counter();o=fn(logits,24,rng);times.append(time.perf_counter()-started);assert (o.sum(1)==24).all()
                    rows.append(dict(model=method,device='cpu',batch=B,median_batch_seconds=float(np.median(times)),p95_batch_seconds=float(np.quantile(times,.95)),repeats=10))
out=dict(status='PASS',hardware_GPU=torch.cuda.get_device_name(),torch=torch.__version__,results=rows,scope='Residentforwarddecoder andCPUcountdecoder timings;excludescheckpointload,CUDAimport,latentdraw,geometryexport,deduplication,oracleandrelaxation. Notend-to-endspeedup.')
(HERE/'validation/decoder_latency.json').write_text(json.dumps(out,indent=2)+'\n');print('DECODER_BENCHMARK_PASS',flush=True)

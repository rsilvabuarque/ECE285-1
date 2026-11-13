"""Dedicated one-GPU portable checks, then final CPU audit/report on this node."""
import json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;PY=HERE.parent/'energy_ranking_audit/multimodel_20261104/envs/uma/bin/python'
def run(tag,args):
    with (HERE/'logs'/f'{tag}.log').open('w') as f:subprocess.run(list(map(str,args)),cwd=HERE,stdout=f,stderr=subprocess.STDOUT,check=True)
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
out=HERE/'validation/portable_gpu_smoke'
for kind,trainer,args in [('mask','train.py',['--teacher','uma','--kind','cvae']),('geometry','train_geometry.py',['--kind','cvae'])]:
    run('portable_GPU_'+kind,['srun','--overlap','-N1','-n1','-c4','--gpus-per-task=1','--gpu-bind=single:1',PY,HERE/'collaborator_package/cvae'/trainer,*args,'--epochs','2','--device','cuda','--output-dir',out/kind])
run('decoder_latency',['srun','--overlap','-N1','-n1','-c4','--gpus-per-task=1','--gpu-bind=single:1',PY,HERE/'benchmark_decoder.py'])
(HERE/'validation/portable_gpu.json').write_text(json.dumps(dict(status='PASS',checks=['MaskportableCUDAfit/checkpoint','GeometryportableCUDAfit/checkpoint','Frozenresidentdecoder CPU/CUDA latency'],scope='Twoepochsoperabilityonly;notconvergence. DedicatedsingleGPU.'),indent=2)+'\n')
while not (HERE/'validation/md_export.json').exists():
    print('WAITING_MD_EXPORT_FOR_FINALIZATION',flush=True);time.sleep(30)
while not (HERE/'finalize_data.py').exists():time.sleep(10)
run('finalize_data',[PY,HERE/'finalize_data.py'])
run('report_facts_final',[PY,HERE/'prepare_report.py'])
with (HERE/'logs/build_report_final.log').open('w') as f:subprocess.run(['bash','-lc','module load texlive/2024\nlatexmk -pdf -interaction=nonstopmode -halt-on-error preliminary_report.tex'],cwd=HERE/'reports',stdout=f,stderr=subprocess.STDOUT,check=True)
print('FINAL_VALIDATION_DRIVER_COMPLETE',flush=True)

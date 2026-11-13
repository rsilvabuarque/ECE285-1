"""Resume only delivery analysis after the timed portability allocation ends."""
import json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
EXPECTED='bbc2b67e2533edd63ff1496bb1ed00a00338cdc1478af6c1a0bf9f4b369977e7'
def run(tag,args,env=None):
    with (HERE/'logs'/f'{tag}.log').open('w') as f:subprocess.run(list(map(str,args)),cwd=HERE,stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
def read_pass(path):
    try:r=json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return False
    assert r.get('status') in ['PASS'],r
    return True
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
actual=os.environ.get('SHIFTER_IMAGE');assert not actual or actual==EXPECTED,(actual,EXPECTED)
import numpy,torch,scipy,matplotlib
receipt=dict(image_id=EXPECTED,image_env=actual,python=sys.version,numpy=numpy.__version__,torch=torch.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,scope='Postprocessing and portability support only. No MLIP labels or dynamics rerun in this container.')
(HERE/'validation/postprocess_environment.json').write_text(json.dumps(receipt,indent=2)+'\n')
env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',MPLBACKEND='Agg',MPLCONFIGDIR='/tmp/ece285_final_mpl_'+env['SLURM_JOB_ID'])
while not read_pass(HERE/'validation/md_export.json'):
    try:state=json.loads((HERE/'execution.json').read_text()).get('MD',{})
    except json.JSONDecodeError:time.sleep(2);continue
    if str(state.get('export_owner_job'))==env['SLURM_JOB_ID'] and state.get('previous_export_owner_terminal'):
        # Root transfers ownership only after scheduler proof that the earlier
        # exporting allocation is terminal. No two exporters write the outputs.
        target='/tmp/ece285_postprocess_packages_'+env['SLURM_JOB_ID']
        run('install_analysis_ASE',[sys.executable,'-m','pip','install','--index-url','https://pypi.org/simple','--no-deps','--target',target,'ase==3.26.0'],env=env)
        env['PYTHONPATH']=target
        run('export_md_container',[sys.executable,HERE/'export_md.py'],env=env)
        assert read_pass(HERE/'validation/md_export.json')
        break
    print('WAITING_NATIVE_MD_EXPORT_OR_EXPLICIT_OWNER_TRANSFER',flush=True);time.sleep(15)
run('finalize_data_container',[sys.executable,HERE/'finalize_data.py'],env=env)
run('report_facts_container',[sys.executable,HERE/'prepare_report.py'],env=env)
print('FINAL_POSTPROCESS_COMPLETE',flush=True)

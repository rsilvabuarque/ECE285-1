"""Bounded CPU engineering check through the project's normal MLIP CLI."""
import argparse,csv,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def call(command,env,limit):
    process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,text=True,start_new_session=True)
    try:
        stdout,stderr=process.communicate(timeout=limit); return process.returncode,stdout,stderr,False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        try:stdout,stderr=process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);stdout,stderr=process.communicate()
        return process.returncode,stdout,stderr,True

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--checkpoint-manifest',type=Path,required=True)
    p.add_argument('--environment-map',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--report-dir',type=Path,required=True);p.add_argument('--timeout-seconds',type=float,default=300)
    a=p.parse_args();root=Path.cwd();a.output_dir.mkdir(parents=True,exist_ok=False);a.report_dir.mkdir(parents=True,exist_ok=True)
    if (a.report_dir/'receipt.json').exists():raise FileExistsError('Preserve the previous smoke receipt')
    checkpoints=json.loads(a.checkpoint_manifest.read_text());environments=json.loads(a.environment_map.read_text())
    env=os.environ.copy();env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(root/'src'),HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    env.pop('CATHODE_CVAE_IN_BACKEND',None)
    rows=[];started=time.time()
    for model in ['sevennet','chgnet','m3gnet']:
        out=a.output_dir/model;checkpoint=Path(checkpoints[model]);backend='sevennet' if model=='sevennet' else 'matgl'
        interpreter=environments['interpreters'][backend]
        query="import importlib.metadata as m,json; names=['torch','numpy','ase','sevenn','e3nn','matgl','torch-geometric','torch-cluster','torch-sparse','pyg-lib','pymatgen-core']; result={};\nfor n in names:\n try:result[n]=m.version(n)\n except m.PackageNotFoundError:result[n]=None\nprint(json.dumps(result))"
        code,stdout,stderr,timed_out=call([interpreter,'-c',query],env,30)
        versions=json.loads(stdout) if code==0 else {'query_error':stderr}
        command=[sys.executable,'-m','cathode_cvae.mlip.cli','score','--model',model,'--device','cpu','--threads','4','--input',str(a.input),'--checkpoint',str(checkpoint),'--environment-map',str(a.environment_map),'--output',str(out)]
        files=[checkpoint] if checkpoint.is_file() else [checkpoint/n for n in ['model.json','model.pt','state.pt']]
        before=time.monotonic();code,stdout,stderr,timed_out=call(command,env,a.timeout_seconds);elapsed=time.monotonic()-before
        (a.report_dir/f'{model}.stdout.log').write_text(stdout);(a.report_dir/f'{model}.stderr.log').write_text(stderr)
        row=dict(model=model,engineering_status='FAIL',exit_code=code,timed_out=timed_out,wall_seconds=elapsed,device='cpu',threads=4,checkpoint=str(checkpoint),checkpoint_hashes={f.name:sha(f) for f in files},versions=versions,command=command,output_dir=str(out))
        if code==0 and (out/'predictions.npz').exists():
            with np.load(out/'predictions.npz',allow_pickle=False) as d:
                checks=dict(one_geometry=len(d['energy_eV'])==1,atoms_180=d['n_atoms'].tolist()==[180],Li_36=d['n_Li'].tolist()==[36],force_shape=d['forces_eV_A'].shape==(180,3),stress_shape=d['stress_eV_A3'].shape==(1,6),finite=all(np.isfinite(d[k]).all() for k in ['energy_eV','forces_eV_A','stress_eV_A3']))
                row.update(checks=checks,energy_eV=float(d['energy_eV'][0]),forces_shape=list(d['forces_eV_A'].shape),stress_shape=list(d['stress_eV_A3'].shape))
                row['engineering_status']='PASS' if all(checks.values()) else 'FAIL'
            row['model_receipt']=json.loads((out/'model_receipt.json').read_text());row['run_receipt']=json.loads((out/'run_receipt.json').read_text())
            for filename in ['model_receipt.json','run_receipt.json','results.jsonl','summary.csv','predictions.npz']:
                target=a.report_dir/model/filename;target.parent.mkdir(exist_ok=True);target.write_bytes((out/filename).read_bytes())
        elif (out/'results.jsonl').exists():
            row['forward_failures']=[json.loads(line) for line in (out/'results.jsonl').read_text().splitlines()]
        rows.append(row);print(json.dumps(dict(model=model,status=row['engineering_status'],wall_seconds=elapsed,exit_code=code,timed_out=timed_out)),flush=True)
        (a.report_dir/f'{model}_receipt.json').write_text(json.dumps(row,indent=2)+'\n')
    receipt=dict(status='PASS' if all(r['engineering_status']=='PASS' for r in rows) else 'ISSUES_FOUND',scope='Three CPU single-point engineering interface cases, not a model accuracy/ranking benchmark',input=str(a.input),input_sha256=sha(a.input),checkpoint_manifest_sha256=sha(a.checkpoint_manifest),environment_map_sha256=sha(a.environment_map),source_hashes={str(p):sha(p) for p in [Path(__file__),Path('src/cathode_cvae/mlip/cli.py'),Path('src/cathode_cvae/mlip/calculators.py'),Path('src/cathode_cvae/mlip/registry.py')]},started_unix=started,finished_unix=time.time(),inference_timeout_seconds_per_model=a.timeout_seconds,new_GPU_calls=0,new_DFT_calls=0,model_weights_in_git=False,units='Declared CLI contract: eV/cell; force eV/Angstrom; tensile-positive ASE stress eV/Angstrom^3',results=rows)
    (a.report_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    with (a.report_dir/'summary.csv').open('w') as f:
        fields=['model','engineering_status','exit_code','timed_out','wall_seconds','device','threads'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    print(json.dumps(dict(status=receipt['status'],report_dir=str(a.report_dir))),flush=True)

if __name__=='__main__':main()

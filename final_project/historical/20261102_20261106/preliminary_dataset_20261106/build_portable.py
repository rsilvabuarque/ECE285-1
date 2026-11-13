"""Create separate portable trainers without changing executed experiment sources."""
import hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'collaborator_package'
def replace_once(s,old,new):
    assert s.count(old)==1,old
    return s.replace(old,new)
def main():
    OUT.mkdir(exist_ok=True)
    for folder in ['cvae','data/parents','splits','checkpoints']:(OUT/folder).mkdir(exist_ok=True,parents=True)
    files=['data/learning_O3_N48_pilot_'+m+'.npz' for m in ['uma','mace','sevennet','chgnet','m3gnet']]+['data/parents/O3_N48_pilot.npz','data/geometry_uma_fixed.npz','splits/lineage_splits.csv','splits/receipt.json']
    for name in files:shutil.copy2(HERE/name,OUT/name)
    for name in ['model.py','geometry_model.py','sampling.py','portable_sample.py']:shutil.copy2(HERE/'cvae'/name,OUT/'cvae'/name)
    for name in ['train.py','train_geometry.py']:
        s=(HERE/'cvae'/name).read_text()
        args="ap.add_argument('--data-root',type=Path,default=ROOT);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');ap.add_argument('--threads',type=int,default=4);"
        if name=='train.py':args+="ap.add_argument('--beta-final',type=float,default=1.);"
        s=replace_once(s,'a=ap.parse_args()',args+"a=ap.parse_args();global ROOT;ROOT=a.data_root.resolve();device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device;assert device!='cuda' or torch.cuda.is_available()")
        s=replace_once(s,"    assert os.environ.get('SLURMD_NODENAME','').startswith('nid') and torch.cuda.is_available()\n",'')
        s=s.replace('torch.set_num_threads(4)','torch.set_num_threads(a.threads)')
        s=s.replace('torch.cuda.manual_seed_all(a.seed)',"torch.cuda.manual_seed_all(a.seed) if device=='cuda' else None")
        s=s.replace("device='cuda';",'')
        if name=='train.py':
            s=replace_once(s,"out=HERE/'runs'/f'{a.teacher}_{a.kind}_{a.split}_{a.seed}'",'out=a.output_dir')
            s=s.replace('gpu=torch.cuda.get_device_name()',"gpu=torch.cuda.get_device_name() if device=='cuda' else 'CPU'")
            s=s.replace('beta=min(1.,(epoch+1)/20)','beta=a.beta_final*min(1.,(epoch+1)/20)')
            s=s.replace('beta_anneal_epochs=20,','beta_final=a.beta_final,beta_anneal_epochs=20,')
        else:
            s=replace_once(s,"out=HERE/'geometry_runs'/f'{a.kind}_{a.seed}'",'out=a.output_dir')
            s=s.replace('if epoch+1>=20 and vl<best-1e-4:','if epoch+1>=min(20,a.epochs) and vl<best-1e-4:')
        # ROOT is used in an argparse default before the global declaration; use a local root default.
        s=s.replace('default=ROOT);ap.add_argument',"default=Path(__file__).resolve().parents[1]);ap.add_argument")
        (OUT/'cvae'/name).write_text(s)
    for rel in ['cvae/runs/uma_cvae_main_20261106','cvae/runs/uma_cvae_beta01_main_20261106','cvae/geometry_runs/cvae_20261400','cvae/geometry_runs/regressor_20261400']:
        path=HERE/rel
        if path.exists():
            target=OUT/'checkpoints'/path.name;target.mkdir(exist_ok=True)
            for name in ['best.pt','configuration.json','summary.json']:
                if (path/name).exists():shutil.copy2(path/name,target/name)
    (OUT/'requirements.txt').write_text('numpy>=1.26\ntorch>=2.2\n')
    receipt={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='MANIFEST.json'}
    (OUT/'MANIFEST.json').write_text(json.dumps({'files':receipt,'scope':'MLIP-free learning and sampling; dense models specific to this48site parent. All original experiment sources remain unchanged.'},indent=2)+'\n');print('PORTABLE_PACKAGE',len(receipt),flush=True)
if __name__=='__main__':main()

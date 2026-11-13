"""Validate completed shards and export small MLIP-independent learning arrays."""
import argparse,csv,hashlib,json,os,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--wait',action='store_true');args=ap.parse_args()
    assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
    receipt=json.loads((HERE/'data/preparation_receipt.json').read_text())
    tasks=[json.loads(s) for s in (HERE/'data/scf_tasks.jsonl').read_text().splitlines()];tasks=[t for t in tasks if t['model']==args.model]
    while True:
        pending=[t for t in tasks if not (HERE/'data/labels'/args.model/(t['task_id']+'.json')).exists()]
        print('COLLECTION_WAIT',args.model,len(tasks)-len(pending),len(tasks),flush=True)
        if not pending:break
        if not args.wait:raise RuntimeError('Shards incomplete')
        time.sleep(30)
    exported=[]
    for parent in receipt['parents']:
        name=parent['parent'];path=HERE/'data/parents'/f'{name}.npz';assert sha(path)==parent['sha256']
        with np.load(path) as z:p={k:z[k] for k in z.files}
        n=len(p['masks']);energy=np.full(n,np.nan);stress=np.full((n,6),np.nan);force_rms=np.full(n,np.nan);fmax=np.full(n,np.nan);seen=np.zeros(n,int)
        for t in [t for t in tasks if t['parent']==name]:
            base=HERE/'data/labels'/args.model/t['task_id'];r=json.loads(base.with_suffix('.json').read_text());assert r['status']=='complete'
            with np.load(base.with_suffix('.npz')) as z:
                ids=z['parent_indices'];assert np.array_equal(ids,np.arange(t['start'],t['stop']))
                assert np.array_equal(z['structure_ids'],p['structure_ids'][ids]);assert not seen[ids].any();seen[ids]+=1
                presence=z['atom_presence'];expected=np.ones((len(ids),len(p['numbers'])),bool);expected[:,p['li_indices']]=~p['masks'][ids].astype(bool)
                assert np.array_equal(presence,expected)
                forces=z['forces_parent_eV_A'];assert forces.shape==(*presence.shape,3)
                assert np.isfinite(forces).all() and (forces[~presence]==0).all()
                energy[ids]=z['energy_eV'];stress[ids]=z['stress_eV_A3']
                force_rms[ids]=np.sqrt((forces*forces).sum(axis=(1,2))/(3*presence.sum(axis=1)))
                fmax[ids]=np.linalg.norm(forces,axis=2).max(axis=1)
        assert (seen==1).all() and np.isfinite(energy).all() and np.isfinite(stress).all()
        out=HERE/'data'/f'learning_{name}_{args.model}.npz';tmp=out.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,occupancy=1-p['masks'],structure_ids=p['structure_ids'],n_Li=(1-p['masks']).sum(axis=1),n_atoms=len(p['numbers'])-p['masks'].sum(axis=1),energy_eV=energy,stress_eV_A3=stress,force_component_rms_eV_A=force_rms,force_max_eV_A=fmax)
        tmp.replace(out);exported.append(dict(parent=name,rows=n,path=str(out),sha256=sha(out)))
    out=HERE/'validation'/f'scf_{args.model}.json';out.write_text(json.dumps(dict(status='PASS',model=args.model,rows=sum(r['rows'] for r in exported),exports=exported,checks=['immutable parent input hashes','all expectedshards terminalcomplete','one-to-one IDs/indices','exact vacancy-to-atom-presence mapping','finite EFS and absent-slot masking']),indent=2)+'\n')
    print('COLLECTION_COMPLETE',args.model,flush=True)
if __name__=='__main__':main()

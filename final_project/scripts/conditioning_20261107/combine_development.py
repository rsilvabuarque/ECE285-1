"""Join immutable development replications without pretending they are final.

Method aliases may be harmonized, but original attempt IDs/stage hashes remain.
No proposal or physical outcome is filtered by this operation.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from cathode_cvae.data import sha256


def write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,nargs='+',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--rename-method',action='append',default=[])
    a=p.parse_args();aliases=dict(v.split('=',1) for v in a.rename_method)
    a.output_dir.mkdir(parents=True,exist_ok=False)
    rows=[];raw=[];seeds=[];candidates={};sources=[];parent=None;offset=0
    for folder in a.input:
        path=folder/'attempts_all.csv'
        with path.open() as f: rr=list(csv.DictReader(f))
        with np.load(folder/'raw_attempts.npz') as z:
            x=z['occupancy'];s=z['seed_occupancy']
        if len(rr)!=len(x) or len(s)!=len(x):raise ValueError('Raw attempt alignment differs')
        for r in rr:
            r['source_method']=r['method'];r['source_stage']=folder.name
            r['method']=aliases.get(r['method'],r['method'])
            r['raw_row']=int(r['raw_row'])+offset
        rows.extend(rr);raw.append(x);seeds.append(s);offset+=len(x)
        with np.load(folder/'candidates.npz') as z:
            actual=str(z['parent_sha256'].item())
            if parent is not None and actual!=parent:raise ValueError('Different parent geometries')
            parent=actual
            for sid,mask in zip(z['structure_ids'].astype(str),z['masks']):
                if sid in candidates and not np.array_equal(candidates[sid],mask):
                    raise ValueError('One structure ID names different geometries')
                candidates[sid]=mask
        sources.append(dict(stage=folder.name,attempts=len(rr),
            files={n:sha256(folder/n) for n in ['attempts_all.csv','raw_attempts.npz','candidates.npz']}))
    if len({r['attempt_id'] for r in rows})!=len(rows):raise ValueError('Duplicate attempt IDs across input stages')
    ids=sorted(candidates);masks=np.asarray([candidates[s] for s in ids],dtype=np.uint8)
    np.savez_compressed(a.output_dir/'candidates.npz',structure_ids=np.asarray(ids),masks=masks,
        occupancy=1-masks,n_Li=48-masks.sum(1),n_atoms=192-masks.sum(1),parent_sha256=np.asarray(parent))
    np.savez_compressed(a.output_dir/'raw_attempts.npz',occupancy=np.concatenate(raw),seed_occupancy=np.concatenate(seeds))
    write(a.output_dir/'attempts_all.csv',rows)
    primary=[r for r in rows if r['diagnostic']=='primary'];write(a.output_dir/'attempts.csv',primary)
    receipt=dict(status='COMBINED_DEVELOPMENT_EVIDENCE',scope='Development pilot plus predeclared fit-seed replications; not untouched assessment',
        attempts=len(rows),primary_attempts=len(primary),unique_geometries=len(ids),
        method_aliases=aliases,sources=sources,source_sha256=sha256(__file__),
        outputs={n:sha256(a.output_dir/n) for n in ['attempts_all.csv','attempts.csv','raw_attempts.npz','candidates.npz']})
    (a.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

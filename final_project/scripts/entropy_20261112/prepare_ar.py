"""Freeze orbit roles before scores, then export sealed-TEST small-cell AR data."""
import argparse
import json
from pathlib import Path
import numpy as np
from cathode_cvae.data import sha256


def freeze(catalog,output,seed):
    if output.exists():raise FileExistsError(output)
    with np.load(catalog,allow_pickle=False) as z:
        ids=z['structure_ids'].astype(str);counts=z['n_target'].astype(int)
    rng=np.random.default_rng(seed);roles=np.full(len(ids),'train',dtype='<U5');groups=[]
    for n in sorted(set(counts)):
        ix=np.flatnonzero(counts==n);ix=rng.permutation(ix)
        hold=max(1,int(np.floor(.2*len(ix)))) if len(ix)>=3 else 0
        roles[ix[:hold]]='val';roles[ix[hold:2*hold]]='test'
        groups.append(dict(n_target=int(n),orbits=len(ix),train=int(np.sum(roles[ix]=='train')),
                           val=int(np.sum(roles[ix]=='val')),test=int(np.sum(roles[ix]=='test'))))
    output.mkdir(parents=True)
    np.savez_compressed(output/'orbit_roles.npz',structure_ids=ids,split=roles,n_target=counts)
    receipt=dict(status='FROZEN_BEFORE_ENERGY_IMPORT',catalog=str(catalog),catalog_sha256=sha256(catalog),
        seed=int(seed),source_labels_read=False,role_counts=groups,
        roles_sha256=sha256(output/'orbit_roles.npz'),
        scope='Orbit-disjoint tiny N12 learning diagnostic; no relation to protected N48 assessment; singleton compositions have TRAIN only')
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


def export(catalog,roles_dir,score,parent,output):
    if output.exists():raise FileExistsError(output)
    receipt=json.loads((roles_dir/'receipt.json').read_text())
    if receipt['catalog_sha256']!=sha256(catalog) or receipt['roles_sha256']!=sha256(roles_dir/'orbit_roles.npz'):
        raise ValueError('Frozen role or catalog hash changed')
    with np.load(catalog,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    with np.load(roles_dir/'orbit_roles.npz',allow_pickle=False) as z:
        ids=z['structure_ids'].astype(str);roles=z['split'].astype(str)
    if not np.array_equal(ids,d['structure_ids'].astype(str)):raise ValueError('Role ID order differs')
    with np.load(score,allow_pickle=False) as z:
        lookup={sid:i for i,sid in enumerate(z['structure_ids'].astype(str))};energy=z['energy_eV']
        status=z['status'].astype(str) if 'status' in z else np.full(len(energy),'success')
    active=roles!='test';e=np.full(len(ids),np.nan)
    for i in np.flatnonzero(active):
        j=lookup[ids[i]]
        if status[j] not in ('success','new') or not np.isfinite(energy[j]):raise ValueError('Missing TRAIN/VAL label')
        e[i]=energy[j]
    with np.load(parent,allow_pickle=False) as z:
        host=len(z['numbers'])-len(z['target_indices'] if 'target_indices' in z else z['li_indices'])
    output.mkdir(parents=True)
    np.savez_compressed(output/'learning.npz',structure_ids=ids,occupancy=d['occupancy'],
        n_target=d['n_target'],energy_eV_atom=e/(host+d['n_target']),
        split=roles,weight=d['multiplicity'].astype(float),parent_sha256=np.array(sha256(parent)))
    export_receipt=dict(status='TRAIN_VAL_LABEL_EXPORT_TEST_VALUES_SEALED',catalog_sha256=sha256(catalog),
        frozen_roles_sha256=sha256(roles_dir/'orbit_roles.npz'),source_scores_sha256=sha256(score),
        parent_sha256=sha256(parent),learning_sha256=sha256(output/'learning.npz'),
        train_rows=int(np.sum(roles=='train')),val_rows=int(np.sum(roles=='val')),test_rows=int(np.sum(roles=='test')),
        test_labels_exported=False,weighting='Orbit multiplicity; use train augmentation for microstate orientation',
        limitation='Compressed source score array is decompressed; TEST values are not indexed/exported or used in fitting')
    (output/'receipt.json').write_text(json.dumps(export_receipt,indent=2)+'\n');print(json.dumps(export_receipt,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('--catalog',type=Path,required=True);f.add_argument('--output-dir',type=Path,required=True);f.add_argument('--seed',type=int,default=2026111204)
    e=sub.add_parser('export');e.add_argument('--catalog',type=Path,required=True);e.add_argument('--roles-dir',type=Path,required=True);e.add_argument('--score',type=Path,required=True);e.add_argument('--parent',type=Path,required=True);e.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.command=='freeze':freeze(a.catalog,a.output_dir,a.seed)
    else:export(a.catalog,a.roles_dir,a.score,a.parent,a.output_dir)


if __name__=='__main__':main()

"""Receipt-bound H0 TRAIN seeds and paired exact-k mutation proposal streams.

The bank remains fixed across acquisition rounds. A round-specific stream must
be reused across compared arms and changed between rounds. No energies are read
from protected assessment sets and no MLIP is called by either command.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from cathode_cvae.conditions import classify
from cathode_cvae.data import sha256
from cathode_cvae.iterative_acquisition import _base_lineage, _canonical_vacancy, _keys, _parent
from cathode_cvae.iterative_boundary import binding, write
from cathode_cvae.runtime import require_compute_allocation


def freeze_bank(parent, base_data, schema, protected, counts, bins, out):
    permutations,_ = _parent(parent)
    _,sch = _base_lineage(base_data,schema,parent,1,None)
    with np.load(base_data,allow_pickle=False) as z:
        x=z['occupancy']; roles=z['split'].astype(str); ids=z['structure_ids'].astype(str)
        n=z['n_target'] if 'n_target' in z else z['n_Li']
        train=np.flatnonzero(roles=='train')
        energy=z['energy_eV_atom'][train]
    if len(set(ids))!=len(ids) or len(x)!=len(roles): raise ValueError('Invalid H0 rows')
    keys=_canonical_vacancy(x[train],permutations)
    pk=_keys(protected,48)
    pk_array=np.asarray([[int(c) for c in k] for k in sorted(pk)],dtype=np.uint8).reshape(-1,48)
    canonical_protected=set(_canonical_vacancy(1-pk_array,permutations).tolist())
    labels=classify(sch,n[train],energy)
    allowed=np.asarray([key not in canonical_protected for key in keys])
    chosen=train[allowed & np.isin(n[train],counts) & np.isin(labels,bins)]
    selected_labels=labels[allowed & np.isin(n[train],counts) & np.isin(labels,bins)]
    for c in counts:
        for b in bins:
            if not np.any((n[chosen]==c)&(selected_labels==b)):
                raise ValueError(f'No unprotected H0 TRAIN seed for {c}:{b}')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'bank.npz',occupancy=x[chosen],structure_ids=ids[chosen],
                        n_target=n[chosen],energy_bin=selected_labels,source_row=chosen,
                        parent_sha256=np.asarray(sha256(parent)))
    r={'status':'FROZEN_TRAIN_SEED_BANK','inputs':{'parent':binding(parent),'base_data':binding(base_data),
       'schema':binding(schema),'protected':binding(protected)},'bank_sha256':sha256(out/'bank.npz'),
       'counts':list(counts),'bins':list(bins),'seed_rows':len(chosen),
       'excluded_protected_train_orbits':int((~allowed).sum()),'source':binding(__file__),
       'policy':'H0 TRAIN only; exclude protected seed orbits; labels used solely to choose frozen TRAIN seed strata.'}
    write(out/'receipt.json',r);return r


def draw_seed(namespace, replicate, n, requested):
    return int.from_bytes(hashlib.sha256(f'fixed-train-edit-v1|{namespace}|{replicate}|{n}|{requested}'.encode()).digest()[:8],'little')


def generate(manifest_path,parent,base_data,schema,history,protected,bank_dir,out):
    m=json.loads(Path(manifest_path).read_text()); bank_dir=Path(bank_dir)
    if not str(m.get('status','')).startswith('FROZEN') or m.get('diagnostics'):
        raise ValueError('Need frozen nondiagnostic manifest')
    for field,path in [('parent',parent),('data',base_data),('schema',schema),('history',history)]:
        binding(path,m['inputs'][field]['sha256'])
    bound=m.get('boundary_receipt')
    if not bound: raise ValueError('Manifest lacks frozen boundary receipt')
    binding(bound['path'],bound['sha256'])
    boundary=json.loads(Path(bound['path']).read_text())
    if boundary['history']['sha256']!=sha256(history) or boundary['protected']['sha256']!=sha256(protected):
        raise ValueError('History/protected inputs differ from frozen manifest boundary')
    br=json.loads((bank_dir/'receipt.json').read_text())
    if br.get('status')!='FROZEN_TRAIN_SEED_BANK': raise ValueError('Bank not frozen')
    for field in ('parent','base_data','schema','protected'):
        binding(br['inputs'][field]['path'],br['inputs'][field]['sha256'])
    if br['inputs']['parent']['sha256']!=sha256(parent) or br['inputs']['schema']['sha256']!=sha256(schema):
        raise ValueError('Bank parent/schema differ')
    binding(bank_dir/'bank.npz',br['bank_sha256'])
    permutations,host=_parent(parent)
    with np.load(bank_dir/'bank.npz',allow_pickle=False) as z: bank={k:z[k] for k in z.files}
    pk=_keys(protected,48)
    pk_array=np.asarray([[int(c) for c in k] for k in sorted(pk)],dtype=np.uint8).reshape(-1,48)
    protected_orbits=set(_canonical_vacancy(1-pk_array,permutations).tolist())
    if set(_canonical_vacancy(bank['occupancy'],permutations).tolist()) & protected_orbits:
        raise ValueError('A fixed bank seed is newly protected; do not reuse this bank')
    with np.load(base_data,allow_pickle=False) as z: data={k:z[k] for k in ('occupancy','split','structure_ids')}
    train=np.flatnonzero(data['split'].astype(str)=='train'); lookup={str(data['structure_ids'][i]):i for i in train}
    if len(lookup)!=len(train): raise ValueError('Duplicate TRAIN seed IDs')
    indices=[]
    for sid,mask in zip(bank['structure_ids'].astype(str),bank['occupancy']):
        i=lookup.get(sid)
        if i is None or not np.array_equal(data['occupancy'][i],mask):
            raise ValueError('Fixed bank seed missing/changed in bound TRAIN data')
        indices.append(i)
    indices=np.asarray(indices)
    counts=m['counts'];bins=m['requested_bins'];attempts=m['count_per_group'];namespace=m['namespace']
    models=m['models']; expected=len(models)*len(counts)*len(bins)*attempts
    if m.get('expected_primary_attempts',expected)!=expected: raise ValueError('Manifest budget differs')
    if not models or any(e.get('sampler')!='mutation' or e.get('checkpoint') for e in models):
        raise ValueError('Only mutation entries are accepted')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    protocol={'status':'FROZEN_BEFORE_GENERATION_AND_SCORING','no_energy_filtering':True,
      'namespace':namespace,'inputs':{name:binding(path) for name,path in [('parent',parent),('data',base_data),('schema',schema),('history',history)]},
      'manifest':binding(manifest_path),'bank_receipt':binding(bank_dir/'receipt.json'),'protected':binding(protected),
      'condition_schema_id':m['expected_condition_schema_id'],'counts':counts,'requested_bins':bins,'count_per_group':attempts,
      'expected_primary_attempts':expected, 'random_array_convention':'uniforms are actual removal uniforms, not AR uniforms; latents are unused zeros',
      'geometry_protocol':'pristine-derived deletion; unrelaxed; exact canonical parent coordinates',
      'stream_policy':'Within a round, same namespace/replicate/count/bin shares literal seed, k and edit uniforms across data lineages; change namespace between rounds.'}
    write(out/'protocol.json',protocol)
    rows=[];raw=[];raw_seed=[];raw_remove=[];raw_add=[]; all_keys={}
    for e in models:
      method=e['method'];rep=str(e['replicate_id'])
      for n in counts:
       for b in bins:
        pool=np.flatnonzero((bank['n_target']==n)&(bank['energy_bin'].astype(str)==b))
        if not len(pool): raise ValueError(f'Bank lacks {n}:{b}')
        rng=np.random.default_rng(draw_seed(namespace,rep,n,b))
        chosen=rng.choice(pool,attempts,replace=True)
        swaps=rng.integers(1,min(3,n,48-n)+1,size=attempts)
        ru=rng.random((attempts,48)).astype(np.float32);au=rng.random((attempts,48)).astype(np.float32)
        seed=bank['occupancy'][chosen].copy(); child=seed.copy()
        for i in range(attempts):
            occupied=np.flatnonzero(seed[i]);vacant=np.flatnonzero(1-seed[i]);k=int(swaps[i])
            rem=np.sort(occupied[np.argsort(ru[i,occupied],kind='stable')[:k]])
            add=np.sort(vacant[np.argsort(au[i,vacant],kind='stable')[:k]])
            child[i,rem]=0;child[i,add]=1
        keys=_canonical_vacancy(child,permutations)
        for i,key in enumerate(keys):
            sid=f'mask48_{int(key,2):012x}';all_keys[key]=sid
            rem=np.flatnonzero((seed[i]==1)&(child[i]==0));add=np.flatnonzero((seed[i]==0)&(child[i]==1))
            j=len(raw);raw.append(child[i]);raw_seed.append(seed[i]);raw_remove.append(ru[i]);raw_add.append(au[i])
            rows.append(dict(attempt_id=f'{namespace}-{method}-{rep}-{n}-{b}-{i}',method=method,replicate_id=rep,
                n_target=n,requested_bin=b,canonical_mask=key,structure_id=sid,scoring_structure_id=sid,raw_row=j,status='generated',
                diagnostic='primary',attempt=i,latent_dim=0,fit_seed=rep,error='',paired_id=f'{namespace}-{rep}-{n}-{b}-{i}',
                seed_source_row=int(indices[chosen[i]]),seed_source_sha256=sha256(base_data),
                seed_structure_id=str(bank['structure_ids'][chosen[i]]),swap_count=int(swaps[i]),
                removed_sites='|'.join(map(str,rem)),added_sites='|'.join(map(str,add)),
                paired_trial_id=f'{namespace}-{rep}-{n}-{b}-{i}'))
    if len(rows)!=expected: raise ValueError('Actual attempt budget differs')
    for filename in ('attempts.csv','attempts_all.csv'):
        with (out/filename).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    arrays=dict(occupancy=np.asarray(raw),seed_occupancy=np.asarray(raw_seed),
                remove_uniforms=np.asarray(raw_remove),add_uniforms=np.asarray(raw_add),
                uniforms=np.asarray(raw_remove),latents=np.zeros((len(raw),1),dtype=np.float32))
    np.savez_compressed(out/'raw_attempts.npz',**arrays)
    np.savez_compressed(out/'raw_primary.npz',**arrays)
    np.savez_compressed(out/'raw_diagnostics.npz',**{k:v[:0] for k,v in arrays.items()})
    masks=np.asarray([list(k) for k in sorted(all_keys)],dtype=np.uint8).reshape(-1,48)
    n=48-masks.sum(axis=1,dtype=np.int64)
    for name in ('candidates.npz','primary_candidates.npz'):
        np.savez_compressed(out/name,masks=masks,occupancy=1-masks,structure_ids=np.asarray([all_keys[k] for k in sorted(all_keys)]),
                            n_target=n,n_Li=n,n_atoms=host+n,parent_sha256=np.asarray(sha256(parent)))
    empty=np.empty((0,48),dtype=np.uint8)
    np.savez_compressed(out/'diagnostic_candidates.npz',masks=empty,occupancy=empty,structure_ids=np.asarray([],dtype='<U19'),
                        n_target=np.asarray([],dtype=int),n_Li=np.asarray([],dtype=int),n_atoms=np.asarray([],dtype=int),
                        parent_sha256=np.asarray(sha256(parent)))
    outputs={name:sha256(out/name) for name in ('attempts.csv','attempts_all.csv','raw_attempts.npz','raw_primary.npz','raw_diagnostics.npz','candidates.npz','primary_candidates.npz','diagnostic_candidates.npz')}
    receipt={'status':'FROZEN_BEFORE_SCORING','no_energy_filtering':True,'primary_attempts':len(rows),
             'protocol_sha256':sha256(out/'protocol.json'),'outputs':outputs,'checkpoints':[],
             'bank_receipt':binding(bank_dir/'receipt.json'),'source':binding(__file__),'source_sha256':sha256(__file__),
             'runtime_loader_source':binding(__file__),'total_attempts':len(rows),'new_MLIP_calls':0}
    write(out/'receipt.json',receipt);return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    b=s.add_parser('bank')
    for n in ('parent','base-data','schema','protected','output-dir'):b.add_argument('--'+n,type=Path,required=True)
    b.add_argument('--counts',type=int,nargs='+',required=True);b.add_argument('--bins',nargs='+',required=True)
    g=s.add_parser('generate')
    for n in ('manifest','parent','base-data','schema','history','protected','bank-dir','output-dir'):g.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();require_compute_allocation()
    if a.cmd=='bank':r=freeze_bank(a.parent,a.base_data,a.schema,a.protected,a.counts,a.bins,a.output_dir)
    else:r=generate(a.manifest,a.parent,a.base_data,a.schema,a.history,a.protected,a.bank_dir,a.output_dir)
    print(json.dumps({'status':r['status']},indent=2))
if __name__=='__main__':main()

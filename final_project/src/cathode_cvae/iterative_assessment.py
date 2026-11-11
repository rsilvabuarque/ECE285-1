"""Validate an iterative assessment and freeze an oracle-safe evaluator bundle.

The denominator and all raw random arrays are unchanged. Candidate identities
in the prior known or protected boundary are excluded before scoring; this
changes secondary all-attempt hit rates because those entries are not labeled.
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
from cathode_cvae.data import sha256
from cathode_cvae.iterative_acquisition import _base_lineage, _canonical_vacancy, _keys, _parent
from cathode_cvae.iterative_boundary import binding, write, verify_boundary
from cathode_cvae.runtime import require_compute_allocation


@lru_cache(maxsize=4)
def _orbit_set_cached(keys, shape, dtype, permutation_bytes):
    """Cache only exact key sets and exact parent-group bytes; no digest collisions."""
    permutations=np.frombuffer(permutation_bytes,dtype=np.dtype(dtype)).reshape(shape)
    a=np.asarray([[int(c) for c in key] for key in sorted(keys)],dtype=np.uint8).reshape(-1,48)
    return frozenset(_canonical_vacancy(1-a,permutations).tolist())


def orbit_set(keys, permutations):
    group=np.ascontiguousarray(permutations)
    # frozenset/cache key equality verifies all keys even if Python hashes collide.
    # Never expose the immutable cached value for callers to mutate.
    return set(_orbit_set_cached(frozenset(keys),group.shape,group.dtype.str,group.tobytes()))


def save_csv(path,rows):
    if not rows: raise ValueError('Empty attempt table')
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def verify_manifest(path, boundary_dir):
    m=json.loads(Path(path).read_text())
    if m.get('status')!='FROZEN_ITERATIVE_DEVELOPMENT_BEFORE_GENERATION' or m.get('purpose')!='assessment':
        raise ValueError('Expected frozen iterative DEVELOPMENT assessment manifest')
    boundary,known,protected=verify_boundary(boundary_dir)
    binding(m['boundary_receipt']['path'],m['boundary_receipt']['sha256'])
    if sha256(Path(boundary_dir)/'receipt.json')!=m['boundary_receipt']['sha256']:
        raise ValueError('Manifest names another boundary')
    for x in m['inputs'].values(): binding(x['path'],x['sha256'])
    for field in ('plan','source'): binding(m[field]['path'],m[field]['sha256'])
    for b in m.get('previous_manifests',[]): binding(b['path'],b['sha256'])
    merge=m.get('previous_merge_receipt')
    if merge: binding(merge['path'],merge['sha256'])
    if m['inputs']['parent']['sha256']!=boundary['parent']['sha256'] or m['inputs']['history']['sha256']!=boundary['history']['sha256']:
        raise ValueError('Manifest input does not match boundary')
    parent=m['inputs']['parent']['path'];data=m['inputs']['data']['path'];schema=m['inputs']['schema']['path']
    _,sch=_base_lineage(data,schema,parent,m['round'],None if merge is None else merge['path'])
    if sch['identity']!=m['expected_condition_schema_id']:
        raise ValueError('Schema identity differs')
    perms,host=_parent(parent)
    known=orbit_set(known,perms);protected=orbit_set(protected,perms)
    if not protected <= known: raise ValueError('Protected orbits not in known history')
    with np.load(data,allow_pickle=False) as z:
        if not set(_canonical_vacancy(z['occupancy'],perms).tolist()) <= known:
            raise ValueError('History omits training/validation/test dataset identities')
    for e in m['models']:
        if e.get('checkpoint'):
            binding(e['checkpoint'],e['expected_checkpoint_sha256'])
            binding(e['configuration']['path'],e['configuration']['sha256'])
            c=json.loads(Path(e['configuration']['path']).read_text())
            if (c.get('data_sha256')!=m['inputs']['data']['sha256'] or c.get('parent_sha256')!=m['inputs']['parent']['sha256']
                or c.get('condition_schema_id')!=sch['identity'] or c.get('seed')!=e.get('fit_seed')):
                raise ValueError('Acquired-data checkpoint metadata differs')
    return m,boundary,known,protected,perms,host


def filter_assessment(generated_dir,boundary_dir,generator_source,out_dir):
    src=Path(generated_dir).resolve();out=Path(out_dir)
    receipt=json.loads((src/'receipt.json').read_text())
    if receipt.get('status')!='FROZEN_BEFORE_SCORING' or not receipt.get('no_energy_filtering'):
        raise ValueError('Generator receipt does not freeze unfiltered proposals')
    binding(src/'protocol.json',receipt['protocol_sha256'])
    if sha256(generator_source)!=receipt['source_sha256']:
        raise ValueError('Generated output used a different source')
    binding(receipt['runtime_loader_source']['path'],receipt['runtime_loader_source']['sha256'])
    protocol=json.loads((src/'protocol.json').read_text())
    binding(protocol['manifest']['path'],protocol['manifest']['sha256'])
    manifest,boundary,known,protected,perms,host=verify_manifest(protocol['manifest']['path'],boundary_dir)
    if protocol.get('status')!='FROZEN_BEFORE_GENERATION_AND_SCORING' or not protocol.get('no_energy_filtering'):
        raise ValueError('Unfrozen generation protocol')
    for e in manifest['models']:
        if e.get('checkpoint'):
            matches=[x for x in receipt.get('checkpoints',[]) if x.get('method')==e['method'] and str(x.get('replicate_id',x.get('fit_seed','')))==str(e['replicate_id'])]
            if len(matches)!=1 or matches[0].get('actual_checkpoint_sha256')!=e['expected_checkpoint_sha256']:
                raise ValueError('Generation receipt does not bind the declared checkpoint')
    for name,b in manifest['inputs'].items():
        if protocol['inputs'][name]['sha256']!=b['sha256']:
            raise ValueError('Generation inputs differ from assessment manifest')
    if (protocol['namespace']!=manifest['namespace'] or protocol['condition_schema_id']!=manifest['expected_condition_schema_id']
        or protocol['expected_primary_attempts']!=manifest['expected_primary_attempts']
        or protocol['count_per_group']!=manifest['count_per_group']):
        raise ValueError('Generation protocol differs from frozen assessment')
    required=['attempts.csv','attempts_all.csv','raw_attempts.npz','raw_primary.npz','raw_diagnostics.npz',
              'candidates.npz','primary_candidates.npz','diagnostic_candidates.npz']
    for name in required: binding(src/name,receipt['outputs'][name])
    for name,digest in receipt['outputs'].items(): binding(src/name,digest)
    with (src/'attempts_all.csv').open(newline='') as f: rows=list(csv.DictReader(f))
    with (src/'attempts.csv').open(newline='') as f: input_primary=list(csv.DictReader(f))
    primary=[r for r in rows if r['diagnostic']=='primary']
    if (len(primary)!=manifest['expected_primary_attempts'] or (not manifest.get('diagnostics') and len(rows)!=len(primary)) or len(rows)!=receipt['total_attempts']
        or len(primary)!=receipt['primary_attempts'] or input_primary!=primary
        or len(set(r['attempt_id'] for r in rows))!=len(rows)):
        raise ValueError('Attempt denominator, ordering or identity changed')
    with np.load(src/'raw_attempts.npz',allow_pickle=False) as z: raw={k:z[k] for k in z.files}
    if any(k not in raw for k in ('occupancy','latents','uniforms')) or any(len(x)!=len(rows) for x in raw.values()):
        raise ValueError('Raw attempt/random arrays are not aligned')
    if raw['occupancy'].shape!=(len(rows),48) or raw['uniforms'].shape!=(len(rows),48):
        raise ValueError('Raw occupancy/uniform shape differs')
    valid=[i for i,r in enumerate(rows) if r['status']=='generated']
    keys=_canonical_vacancy(raw['occupancy'][valid],perms)
    for i,key in zip(valid,keys):
        r=rows[i];n=int(r['n_target']);sid=f'mask48_{int(key,2):012x}'
        if (r['canonical_mask']!=key or r['scoring_structure_id']!=sid or int(raw['occupancy'][i].sum())!=n):
            raise ValueError('Raw occupancy and generated orbit, ID or count differ')
    groups=Counter();original=Counter();excluded=Counter(); paired_u={};paired_z={}
    for i,r in enumerate(rows):
        if int(r['raw_row'])!=i: raise ValueError('Attempt/raw row ordering differs')
        original[r['status']]+=1;r['original_status']=r['status'];r['boundary_status']='original_non_generated'
        if r['status']=='generated':
            key=r['canonical_mask']
            status='protected' if key in protected else 'known' if key in known else 'eligible'
            r['boundary_status']=status
            if status!='eligible':
                r['status']=status+'_boundary';r['error']='Excluded before oracle registration';excluded[status]+=1
        if r['diagnostic']=='primary':
            groups[(r['method'],r['replicate_id'],int(r['n_target']),r['requested_bin'])]+=1
            pair=(r['replicate_id'],r['n_target'],r['requested_bin'],r['attempt'])
            u=raw['uniforms'][i]
            if pair in paired_u and not np.array_equal(u,paired_u[pair]): raise ValueError('Paired sampling uniforms differ')
            paired_u[pair]=u
            dim=int(r['latent_dim'])
            if dim:
                z=raw['latents'][i,:dim]
                if pair in paired_z and not np.array_equal(z[:min(dim,len(paired_z[pair]))],paired_z[pair][:min(dim,len(paired_z[pair]))]):
                    raise ValueError('Paired latent prefixes differ')
                if pair not in paired_z or dim>len(paired_z[pair]): paired_z[pair]=z.copy()
    expected={(e['method'],str(e['replicate_id']),n,b):manifest['count_per_group']
              for e in manifest['models'] for n in manifest['counts'] for b in manifest['requested_bins']}
    if dict(groups)!=expected: raise ValueError('Incomplete or unexpected assessment attempt cells')
    out.mkdir(parents=True,exist_ok=False)
    for name in ('protocol.json','raw_attempts.npz','raw_primary.npz','raw_diagnostics.npz'): shutil.copyfile(src/name,out/name)
    save_csv(out/'attempts_all.csv',rows);save_csv(out/'attempts.csv',[r for r in rows if r['diagnostic']=='primary'])
    def pool(name,selected):
        ids=sorted({r['canonical_mask'] for r in selected if r['status']=='generated'})
        masks=np.asarray([list(k) for k in ids],dtype=np.uint8).reshape(-1,48);n=48-masks.sum(1,dtype=np.int64)
        np.savez_compressed(out/name,masks=masks,occupancy=1-masks,
            structure_ids=np.asarray([f'mask48_{int(k,2):012x}' for k in ids]),n_target=n,n_Li=n,n_atoms=host+n,
            parent_sha256=np.asarray(manifest['inputs']['parent']['sha256']))
        return len(ids)
    unique=pool('candidates.npz',rows)
    pool('primary_candidates.npz',[r for r in rows if r['diagnostic']=='primary'])
    pool('diagnostic_candidates.npz',[r for r in rows if r['diagnostic']!='primary'])
    files=['protocol.json','raw_attempts.npz','raw_primary.npz','raw_diagnostics.npz','attempts.csv','attempts_all.csv',
           'candidates.npz','primary_candidates.npz','diagnostic_candidates.npz']
    result={'status':'FROZEN_BEFORE_SCORING','created_unix':time.time(),'protocol_sha256':sha256(out/'protocol.json'),
        'primary_attempts':len(primary),'total_attempts':len(rows),'unique_geometries':unique,
        'original_generation_receipt':binding(src/'receipt.json'),'generator_source':binding(generator_source),
        'adapter_source':binding(__file__),
        'runtime_loader_source':receipt['runtime_loader_source'],'manifest':binding(protocol['manifest']['path']),
        'boundary_receipt':binding(Path(boundary_dir)/'receipt.json'),'original_status_counts':dict(original),
        'boundary_exclusions':dict(excluded),'outputs':{name:sha256(out/name) for name in files},
        'all_attempt_denominators_preserved':True,'raw_arrays_unchanged':True,
        'scope':'Known and protected attempts excluded before scoring; novel yield retains all-attempt denominator. In-bin hit rates exclude their unknown labels.',
        'new_model_forwards':0,'new_MLIP_calls':0}
    write(out/'receipt.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('generated-dir','boundary-dir','generator-source','output-dir'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();require_compute_allocation()
    result=filter_assessment(a.generated_dir,a.boundary_dir,a.generator_source,a.output_dir)
    print(json.dumps({'status':result['status'],'primary_attempts':result['primary_attempts'],
                      'boundary_exclusions':result['boundary_exclusions']},indent=2))
if __name__=='__main__': main()

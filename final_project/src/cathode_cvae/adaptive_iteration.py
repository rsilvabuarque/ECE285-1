"""Prospective adaptive-label vacancy acquisition with fixed assessment metrics.

All geometry is a frozen 48-site parent. Energy conditions are learned labels,
not gradients or equilibrium probabilities. This module keeps the historical
fixed-schema pipeline unchanged and declares every new schema/round explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .conditions import BIN_NAMES, classify, fit_schema, load_schema, save_schema, schema_identity
from .data import sha256
from .iterative_acquisition import _canonical_vacancy, _keys, _parent
from .iterative_boundary import binding
from .runtime import require_compute_allocation


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot write an empty scientific table')
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def identity(sid):
    return f'mask48_{int(sid, 2):012x}'


def load_boundary(directory):
    p = Path(directory)
    receipt = json.loads((p/'receipt.json').read_text())
    if receipt.get('status') != 'FROZEN_ADAPTIVE_CANONICAL_BOUNDARY':
        raise ValueError('Require the separately verified canonical adaptive boundary')
    for name in ('parent', 'history', 'protected'):
        record = receipt[name]; binding(record['path'], record['sha256'])
    known = _keys(receipt['history']['path'], 48)
    protected = _keys(receipt['protected']['path'], 48)
    if not protected <= known:
        raise ValueError('Protected identities must be known')
    return receipt, known, protected


def bootstrap_boundary(boundary_dir, h0, assessment_dir, out):
    """Protect all eligible last-assessment orbits without reading their labels."""
    from .iterative_next_boundary import build, canonical_set
    assess = Path(assessment_dir)
    tasks = json.loads((assess/'generation_tasks.json').read_text())['tasks']
    if len({task['id'] for task in tasks}) != len(tasks):
        raise ValueError('Assessment task IDs repeat')
    pairs = [(assess/task['id']/'generated_raw', assess/task['id']/'generated_eligible')
             for task in tasks]
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    previous = build(boundary_dir, h0, pairs, out/'verified_previous')
    perms, _ = _parent(previous['parent']['path'])
    known = canonical_set(_keys(previous['history']['path'], 48), perms)
    protected = canonical_set(_keys(previous['protected']['path'], 48), perms)
    if not protected <= known:
        raise ValueError('Canonical protection is not included in history')
    write_json(out/'known_history.json', sorted(known))
    write_json(out/'protected.json', sorted(protected))
    receipt = dict(status='FROZEN_ADAPTIVE_CANONICAL_BOUNDARY', created_unix=time.time(),
                   parent=previous['parent'], history=binding(out/'known_history.json'),
                   protected=binding(out/'protected.json'), known_orbits=len(known),
                   protected_orbits=len(protected), no_energy_members_read=True,
                   verified_previous=binding(out/'verified_previous/receipt.json'),
                   inherited_h0_train_protected_orbits=previous['inherited_h0_train_protected_orbits'],
                   newly_protected_h0_train_orbits=previous['newly_protected_h0_train_orbits'],
                   source=binding(__file__))
    write_json(out/'receipt.json', receipt)
    return receipt


def fit_round_schema(data, parent, reference_schema, mode, out):
    """Use TRAIN energies only; retain empirical q20/q80 even for range edges."""
    ref = load_schema(reference_schema)
    expected_parent = ref.get('provenance', {}).get('parent_sha256')
    if expected_parent is not None and expected_parent != sha256(parent):
        raise ValueError('Reference schema belongs to another parent')
    if mode not in ('fixed', 'adaptive', 'range'):
        raise ValueError('Schema mode must be fixed, adaptive, or range')
    with np.load(data, allow_pickle=False) as z:
        roles = z['split'].astype(str)
        n = z['n_target'] if 'n_target' in z else z['n_Li']
        train = np.flatnonzero(roles == 'train')
        # NPZ decompresses the complete energy member, but only TRAIN elements
        # enter fitting, validation, extrema, support counts or provenance.
        energy = z['energy_eV_atom'][train]
        ids = z['structure_ids'][train].astype(str)
    if not len(train) or not np.isfinite(energy).all():
        raise ValueError('Need finite TRAIN reference energies')
    if ref['n_sites'] != 48:
        raise ValueError('This experiment uses a 48-site parent')
    if mode == 'fixed':
        schema = ref
    else:
        provenance = dict(ref.get('provenance', {}))
        provenance.update(data_sha256=sha256(data), parent_sha256=sha256(parent),
                          training_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                          adaptive_reference_schema_id=ref['identity'], adaptive_edge_mode=mode)
        schema = fit_schema(n[train], energy, np.full(len(train), 'train'), n_sites=48,
                            quantiles=(.2, .8), provenance=provenance)
        if mode == 'range':
            schema['edge_rule'] = 'TRAIN minimum + (0.2,0.8) * observed TRAIN range'
            schema['range_fractions'] = [.2, .8]
            schema['quantile_estimator'] = ('q20/q80 diagnostics use nearest rank; '
                'condition edges instead use declared observed-range fractions')
            for count, entry in schema['compositions'].items():
                values = energy[n[train] == int(count)]
                lo, hi = float(values.min()), float(values.max())
                edges = [lo + p*(hi-lo) for p in (.2, .8)]
                bins = np.searchsorted(edges, values, side='left')
                support = {name: int(np.sum(bins == i)) for i, name in enumerate(BIN_NAMES)}
                present = [name for name in BIN_NAMES if support[name]]
                entry.update(edges_eV_atom=edges, lower_condition_edge_eV_atom=edges[0],
                             upper_condition_edge_eV_atom=edges[1], observed_train_minimum_eV_atom=lo,
                             observed_train_maximum_eV_atom=hi, bin_counts=support,
                             supported_bins=present, empty_bins=[b for b in BIN_NAMES if b not in present],
                             support_status='full' if len(present)==3 else 'partial',
                             sparse=bool(len(values)<5 or len(present)<3), coincident_edges=edges[0]==edges[1])
            schema['identity'] = schema_identity(schema)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    save_schema(schema, out/'condition_schema.json')
    rows = []
    for count in sorted(set(n[train].tolist())):
        values = energy[n[train] == count]
        entry = schema['compositions'][str(int(count))]
        rows.append(dict(n_target=int(count), mode=mode, train_rows=len(values),
                         minimum_eV_atom=float(values.min()), maximum_eV_atom=float(values.max()),
                         low_edge_eV_atom=entry['edges_eV_atom'][0],
                         high_edge_eV_atom=entry['edges_eV_atom'][1],
                         q20_eV_atom=float(np.sort(values)[int(np.ceil(.2*len(values)))-1]),
                         q80_eV_atom=float(np.sort(values)[int(np.ceil(.8*len(values)))-1])))
    write_csv(out/'edges.csv', rows)
    receipt = dict(status='FROZEN_ADAPTIVE_ROUND_SCHEMA', mode=mode, created_unix=time.time(),
                   data=binding(data), parent=binding(parent), reference=binding(reference_schema),
                   schema=binding(out/'condition_schema.json'), schema_id=schema['identity'],
                   training_rows=len(train), training_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                   protected_energy_handling='Complete NPZ member materialized; only TRAIN subset used in fitting/statistics.',
                   source=binding(__file__))
    write_json(out/'receipt.json', receipt); return receipt


def freeze_seed_bank(data, parent, schema_path, boundary_dir, out):
    schema = load_schema(schema_path); boundary, _, protected = load_boundary(boundary_dir)
    if boundary['parent']['sha256'] != sha256(parent):
        raise ValueError('Seed-bank parent differs')
    perms, _ = _parent(parent)
    with np.load(data, allow_pickle=False) as z:
        roles=z['split'].astype(str); ii=np.flatnonzero(roles=='train')
        x=z['occupancy'][ii]; n=(z['n_target'] if 'n_target' in z else z['n_Li'])[ii]
        energy=z['energy_eV_atom'][ii]; ids=z['structure_ids'][ii].astype(str)
    keys=_canonical_vacancy(x,perms); keep=np.asarray([k not in protected for k in keys])
    labels=classify(schema,n,energy)
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'bank.npz',occupancy=x[keep],n_target=n[keep],energy_bin=labels[keep],
                        structure_ids=ids[keep],canonical_masks=keys[keep],parent_sha256=np.asarray(sha256(parent)))
    receipt=dict(status='FROZEN_ADAPTIVE_TRAIN_SEED_BANK',data=binding(data),parent=binding(parent),
                 schema=binding(schema_path),boundary=binding(Path(boundary_dir)/'receipt.json'),
                 bank=binding(out/'bank.npz'),seed_rows=int(keep.sum()),excluded_protected=int((~keep).sum()))
    write_json(out/'receipt.json',receipt); return receipt


def stream_seed(namespace, seed, count, requested):
    return int.from_bytes(hashlib.sha256(f'adaptive-v1|{namespace}|{seed}|{count}|{requested}'.encode()).digest()[:8],'little')


def propose(parent, schema_path, boundary_dir, out, *, namespace, seed, method,
            checkpoint=None, sampler=None, seed_bank=None, counts=(12,24,36),
            bins=BIN_NAMES, attempts_per_cell=512, queries_per_cell=256,
            device='cuda', threads=4, latent_width=32):
    if (checkpoint is None) == (sampler is None):
        raise ValueError('Choose exactly one neural checkpoint or sampler')
    if sampler not in (None,'uniform','mutation') or not namespace or not method:
        raise ValueError('Invalid sampler or namespace')
    if not 1 <= queries_per_cell <= attempts_per_cell:
        raise ValueError('Need 1 <= query cap <= attempted proposals')
    boundary, known, protected = load_boundary(boundary_dir)
    if boundary['parent']['sha256'] != sha256(parent):
        raise ValueError('Boundary parent differs')
    schema=load_schema(schema_path); perms,_=_parent(parent)
    if schema['n_sites']!=48 or set(bins)-set(BIN_NAMES):
        raise ValueError('Wrong schema or bins')
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    protocol=dict(status='FROZEN_ADAPTIVE_PROPOSALS_BEFORE_GENERATION',created_unix=time.time(),
                  parent=binding(parent),schema=binding(schema_path),schema_id=schema['identity'],
                  boundary=binding(Path(boundary_dir)/'receipt.json'),namespace=namespace,seed=int(seed),
                  method=method,sampler=sampler,checkpoint=None if checkpoint is None else binding(checkpoint),
                  seed_bank=None if seed_bank is None else binding(Path(seed_bank)/'receipt.json'),
                  counts=list(counts),bins=list(bins),attempts_per_cell=attempts_per_cell,
                  queries_per_cell=queries_per_cell,latent_stream_width=latent_width,
                  selection='First eligible unique canonical orbit per count/request cell; no energy filtering.',
                  teacher='UMA training labels; separate fixed schemas for assessment',source=binding(__file__))
    write_json(out/'protocol.json',protocol)
    model=cfg=None
    if checkpoint:
        import torch
        from .learning.generate_mask import load_model
        torch.set_num_threads(threads); torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model,cfg=load_model(checkpoint,device)
        if (cfg.get('parent_sha256')!=sha256(parent) or cfg.get('condition_schema_id')!=schema['identity']
            or cfg.get('condition_mode')!='multibin' or model.condition_dim!=4 or model.latent_dim>latent_width):
            raise ValueError('Checkpoint parent, schema, condition width, or latent width differs')
    bank=None
    if sampler=='mutation':
        if seed_bank is None: raise ValueError('Mutation requires a frozen common TRAIN seed bank')
        br=json.loads((Path(seed_bank)/'receipt.json').read_text())
        binding(br['bank']['path'],br['bank']['sha256'])
        if br['parent']['sha256']!=sha256(parent): raise ValueError('Seed-bank parent differs')
        with np.load(br['bank']['path'],allow_pickle=False) as z:bank={k:z[k] for k in z.files}
    rows=[];raw=[];latent_rows=[];uniform_rows=[];selected={};all_keys=set()
    for count in counts:
        if type(count) is not int or not 0<=count<=48: raise ValueError('Invalid Li count')
        for requested in bins:
            rng=np.random.default_rng(stream_seed(namespace,seed,count,requested))
            uniforms=rng.random((attempts_per_cell,48)).astype(np.float32)
            latents=rng.standard_normal((attempts_per_cell,latent_width)).astype(np.float32)
            swaps=np.zeros(attempts_per_cell,dtype=int);chosen=np.full(attempts_per_cell,-1,dtype=int)
            if checkpoint:
                from .conditions import encode_requests
                c=torch.as_tensor(encode_requests(schema,np.full(attempts_per_cell,count),
                                  np.full(attempts_per_cell,requested),mode='multibin'),device=device)
                with torch.no_grad():
                    x=model.sample(c,count,z=torch.as_tensor(latents[:,:model.latent_dim],device=device),
                                   uniforms=torch.as_tensor(uniforms,device=device)).cpu().numpy().astype(np.uint8)
            elif sampler=='uniform':
                x=np.zeros((attempts_per_cell,48),np.uint8)
                order=np.argsort(uniforms,axis=1,kind='stable');np.put_along_axis(x,order[:,:count],1,axis=1)
            else:
                pool=np.flatnonzero((bank['n_target']==count)&(bank['energy_bin'].astype(str)==requested))
                if not len(pool): raise ValueError('Frozen mutation bank lacks requested cell')
                chosen=rng.choice(pool,attempts_per_cell,replace=True); x=bank['occupancy'][chosen].copy()
                if count not in (0,48):
                    swaps=rng.integers(1,min(3,count,48-count)+1,size=attempts_per_cell)
                add_uniforms=rng.random((attempts_per_cell,48)).astype(np.float32)
                for i in range(attempts_per_cell):
                    occupied=np.flatnonzero(x[i]);vacant=np.flatnonzero(1-x[i]);k=swaps[i]
                    x[i,occupied[np.argsort(uniforms[i,occupied],kind='stable')[:k]]]=0
                    x[i,vacant[np.argsort(add_uniforms[i,vacant],kind='stable')[:k]]]=1
            if not np.all(x.sum(1)==count): raise ValueError('Generator violated count constraint')
            keys=_canonical_vacancy(x,perms); seen=set();accepted=0
            for i,key in enumerate(keys):
                if key in protected: status='protected_boundary'
                elif key in known: status='known_boundary'
                elif key in seen: status='duplicate_within_cell'
                elif accepted>=queries_per_cell: status='query_cap'
                else: status='selected';accepted+=1;selected[key]=identity(key)
                seen.add(key);all_keys.add(key);j=len(raw);raw.append(x[i]);latent_rows.append(latents[i]);uniform_rows.append(uniforms[i])
                rows.append(dict(attempt_id=f'{namespace}|{method}|{seed}|{count}|{requested}|{i}',
                                 method=method,seed=seed,n_target=count,requested_bin=requested,raw_row=j,
                                 canonical_mask=key,structure_id=identity(key),status=status,
                                 mutation_seed_row=int(chosen[i]),mutation_swaps=int(swaps[i])))
    ordered=sorted(selected);occ=np.asarray([[1-int(c) for c in key] for key in ordered],dtype=np.uint8).reshape(-1,48)
    np.savez_compressed(out/'candidates.npz',occupancy=occ,masks=1-occ,structure_ids=np.asarray([selected[k] for k in ordered]),
                        n_target=occ.sum(1),n_Li=occ.sum(1),parent_sha256=np.asarray(sha256(parent)))
    np.savez_compressed(out/'raw_attempts.npz',occupancy=np.asarray(raw,np.uint8),latents=np.asarray(latent_rows,np.float32),
                        uniforms=np.asarray(uniform_rows,np.float32),parent_sha256=np.asarray(sha256(parent)))
    write_csv(out/'attempts.csv',rows);write_json(out/'all_generated_keys.json',sorted(all_keys))
    outputs={name:sha256(out/name) for name in ('candidates.npz','raw_attempts.npz','attempts.csv','all_generated_keys.json')}
    receipt=dict(status='FROZEN_ADAPTIVE_STAGE_BEFORE_SCORING',created_unix=time.time(),protocol=binding(out/'protocol.json'),
                 outputs=outputs,attempts=len(rows),selected_unique=len(ordered),status_counts=dict(Counter(r['status'] for r in rows)),
                 parent=protocol['parent'],schema=protocol['schema'],boundary=protocol['boundary'],no_energy_filtering=True)
    write_json(out/'receipt.json',receipt);return receipt


def validate_stage(directory):
    p=Path(directory);r=json.loads((p/'receipt.json').read_text())
    if r.get('status')!='FROZEN_ADAPTIVE_STAGE_BEFORE_SCORING' or not r.get('no_energy_filtering'):
        raise ValueError('Stage was not frozen before scoring')
    binding(r['protocol']['path'],r['protocol']['sha256'])
    for name,digest in r['outputs'].items(): binding(p/name,digest)
    protocol=json.loads((p/'protocol.json').read_text())
    if protocol.get('status')!='FROZEN_ADAPTIVE_PROPOSALS_BEFORE_GENERATION':
        raise ValueError('Unexpected proposal protocol')
    for name in ('parent','schema','boundary','source'):
        b=protocol[name];binding(b['path'],b['sha256'])
    for name in ('checkpoint','seed_bank'):
        b=protocol.get(name)
        if b is not None:binding(b['path'],b['sha256'])
    if r['parent']!=protocol['parent'] or r['schema']!=protocol['schema'] or r['boundary']!=protocol['boundary']:
        raise ValueError('Stage inputs differ from frozen protocol')
    _,known,protected=load_boundary(Path(protocol['boundary']['path']).parent)
    perms,_=_parent(protocol['parent']['path'])
    rows=read_csv(p/'attempts.csv')
    expected=len(protocol['counts'])*len(protocol['bins'])*protocol['attempts_per_cell']
    if len(rows)!=expected or len(rows)!=r['attempts'] or len({x['attempt_id'] for x in rows})!=len(rows):
        raise ValueError('Attempt identities or denominator changed')
    with np.load(p/'raw_attempts.npz',allow_pickle=False) as z:
        x=z['occupancy'];latents=z['latents'];uniforms=z['uniforms']
    if (x.shape!=(len(rows),48) or uniforms.shape!=x.shape or
            latents.shape!=(len(rows),protocol['latent_stream_width'])):
        raise ValueError('Raw draw arrays have wrong shapes')
    actual=_canonical_vacancy(x,perms);seen=defaultdict(set);accepted=Counter();selected=set()
    for i,(row,key) in enumerate(zip(rows,actual)):
        cell=(int(row['n_target']),row['requested_bin'])
        if (int(row['raw_row'])!=i or row['canonical_mask']!=key or row['structure_id']!=identity(key)
                or int(x[i].sum())!=cell[0]):
            raise ValueError('Raw geometry/identity/count does not match attempt')
        if key in protected:status='protected_boundary'
        elif key in known:status='known_boundary'
        elif key in seen[cell]:status='duplicate_within_cell'
        elif accepted[cell]>=protocol['queries_per_cell']:status='query_cap'
        else:status='selected';accepted[cell]+=1;selected.add(key)
        seen[cell].add(key)
        if row['status']!=status:raise ValueError('Attempt eligibility or query-cap selection differs')
    if set(actual.tolist())!=set(json.loads((p/'all_generated_keys.json').read_text())):
        raise ValueError('All-generated identity history differs from raw attempts')
    with np.load(p/'candidates.npz',allow_pickle=False) as z:
        keys=_canonical_vacancy(z['occupancy'],perms);ids=z['structure_ids'].astype(str)
        if (set(keys.tolist())!=selected or len(set(keys.tolist()))!=len(keys)
                or not np.array_equal(z['masks'],1-z['occupancy'])
                or ids.tolist()!=[identity(key) for key in keys]):
            raise ValueError('Scoring pool differs from exact selected orbit union')
    for count in protocol['counts']:
        for requested in protocol['bins']:
            indices=[i for i,row in enumerate(rows) if (int(row['n_target']),row['requested_bin'])==(count,requested)]
            if len(indices)!=protocol['attempts_per_cell']:raise ValueError('Conditional-cell attempt budget changed')
            rng=np.random.default_rng(stream_seed(protocol['namespace'],protocol['seed'],count,requested))
            u=rng.random((len(indices),48)).astype(np.float32)
            z=rng.standard_normal((len(indices),protocol['latent_stream_width'])).astype(np.float32)
            if not np.array_equal(uniforms[indices],u) or not np.array_equal(latents[indices],z):
                raise ValueError('Saved paired random streams differ from declared namespace')
    return r


def verify_oracle(stages, oracle_root, collection_receipt, out, teacher='uma'):
    from .iterative_comparison import collect_oracle
    receipts=[validate_stage(p) for p in stages]
    parents={r['parent']['sha256'] for r in receipts}
    if len(parents)!=1: raise ValueError('Scored stages use different parents')
    expected=set()
    for p in stages:
        with np.load(Path(p)/'candidates.npz',allow_pickle=False) as z: expected.update(z['structure_ids'].astype(str).tolist())
    after=max(r['created_unix'] for r in receipts)
    evidence=collect_oracle(oracle_root,collection_receipt,expected,next(iter(parents)),teacher,after)
    receipt=dict(status='VERIFIED_FRESH_ADAPTIVE_ORACLE',created_unix=time.time(),teacher=teacher,
                 stages=[binding(Path(p)/'receipt.json') for p in stages],score=evidence['score'],
                 freeze_before_scoring_unix=after,oracle_evidence=evidence,unique_structures=len(expected),source=binding(__file__))
    write_json(out,receipt);return receipt


def scored_rows(stage, verifier):
    stage=Path(stage);s=validate_stage(stage);v=json.loads(Path(verifier).read_text())
    if v.get('status')!='VERIFIED_FRESH_ADAPTIVE_ORACLE': raise ValueError('Require full raw-score verification')
    if not any(b['sha256']==sha256(stage/'receipt.json') for b in v['stages']): raise ValueError('Verifier does not include stage')
    binding(v['score']['path'],v['score']['sha256'])
    with np.load(v['score']['path'],allow_pickle=False) as z:
        records={str(sid):(float(e),int(a),int(n)) for sid,e,a,n in zip(z['structure_ids'],z['energy_eV'],z['n_atoms'],z['n_Li'])}
    return s,records,v


def merge_training(base_data,parent,schema_path,stage,verifier,out,round_index=1,new_weight=1.):
    s,scores,v=scored_rows(stage,verifier)
    if v['teacher']!='uma' or s['parent']['sha256']!=sha256(parent) or s['schema']['sha256']!=sha256(schema_path):
        raise ValueError('Training requires the staged UMA parent/schema')
    if not np.isfinite(new_weight) or new_weight<=0: raise ValueError('New weight must be positive')
    schema=load_schema(schema_path);perms,_=_parent(parent)
    with np.load(base_data,allow_pickle=False) as z:base={k:z[k] for k in z.files}
    count_key='n_target' if 'n_target' in base else 'n_Li';rows=len(base[count_key])
    base_keys=set(_canonical_vacancy(base['occupancy'],perms).tolist())
    with np.load(Path(stage)/'candidates.npz',allow_pickle=False) as z:
        x=z['occupancy'];ids=z['structure_ids'].astype(str);n=z['n_target'];keys=_canonical_vacancy(x,perms)
    if set(keys.tolist()) & base_keys: raise ValueError('Acquired orbit already exists in base data')
    energy=np.asarray([scores[sid][0]/scores[sid][1] for sid in ids]);actual=classify(schema,n,energy)
    added={'occupancy':x,'n_Li':n,'n_target':n,'energy_eV_atom':energy,'split':np.full(len(x),'train'),
           'weight':np.full(len(x),new_weight),'structure_ids':ids,'canonical_masks':keys,
           'source':np.full(len(x),f'adaptive_round_{round_index}_uma')}
    merged={}
    for key,value in base.items():
        if value.ndim and len(value)==rows:
            if key not in added: raise ValueError('Unsupported row-aligned base field: '+key)
            new=added[key]
            if value.dtype.kind not in 'US':new=new.astype(value.dtype)
            merged[key]=np.concatenate([value,new],axis=0)
            same = (np.array_equal(merged[key][:rows],value) if value.dtype.kind in 'US'
                    else np.array_equal(merged[key][:rows],value,equal_nan=True))
            if not same:
                raise ValueError('Original base data changed')
        else:merged[key]=value
    merged['parent_sha256']=np.asarray(sha256(parent))
    out=Path(out);out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'training.npz',**merged)
    write_csv(out/'acquired_labels.csv',[dict(structure_id=sid,n_target=int(nn),energy_eV_atom=float(e),measured_bin=str(b),split='train')
              for sid,nn,e,b in zip(ids,n,energy,actual)])
    receipt=dict(status='MERGED_ADAPTIVE_TRAIN_ROUND',round=int(round_index),base_data=binding(base_data),parent=binding(parent),
                 schema=binding(schema_path),stage=binding(Path(stage)/'receipt.json'),verifier=binding(verifier),
                 training=binding(out/'training.npz'),base_rows=rows,added_train_rows=len(x),original_rows_value_identical=True,
                 labels_assigned_after_measured_energy=True,measured_bin_counts=dict(Counter(actual.tolist())),
                 new_weight=float(new_weight),source=binding(__file__))
    write_json(out/'receipt.json',receipt);return receipt


def summarize(stage,verifier,fixed_schema_path,out):
    stage=Path(stage);s,scores,v=scored_rows(stage,verifier);fixed=load_schema(fixed_schema_path)
    current=load_schema(s['schema']['path']);rows=read_csv(stage/'attempts.csv');groups=defaultdict(list)
    for row in rows:groups[(int(row['n_target']),row['requested_bin'])].append(row)
    result=[]
    for (count,requested),rr in sorted(groups.items()):
        selected={r['structure_id'] for r in rr if r['status']=='selected'}
        es=np.asarray([scores[sid][0]/scores[sid][1] for sid in sorted(selected)])
        measured=classify(fixed,np.full(len(es),count),es) if len(es) else np.asarray([],str)
        dynamic=classify(current,np.full(len(es),count),es) if len(es) else np.asarray([],str)
        low=fixed['compositions'][str(count)]['edges_eV_atom'][0];high=fixed['compositions'][str(count)]['edges_eV_atom'][1]
        result.append(dict(teacher=v['teacher'],method=rr[0]['method'],seed=rr[0]['seed'],n_target=count,requested_bin=requested,
                           attempts=len(rr),scored_unique=len(es),fixed_inbin_yield=float(np.sum(measured==requested)/len(rr)),
                           moving_inbin_yield=float(np.sum(dynamic==requested)/len(rr)),
                           fixed_low_yield=float(np.sum(es<=low)/len(rr)),fixed_high_yield=float(np.sum(es>high)/len(rr)),
                           minimum_eV_atom=float(es.min()) if len(es) else '',maximum_eV_atom=float(es.max()) if len(es) else '',
                           mean_eV_atom=float(es.mean()) if len(es) else '',known_excluded=sum(r['status']=='known_boundary' for r in rr),
                           protected_excluded=sum(r['status']=='protected_boundary' for r in rr)))
    out=Path(out);out.mkdir(parents=True,exist_ok=False);write_csv(out/'metrics.csv',result)
    receipt=dict(status='SUMMARIZED_ADAPTIVE_STAGE',stage=binding(stage/'receipt.json'),verifier=binding(verifier),
                 fixed_schema=binding(fixed_schema_path),outputs={'metrics.csv':sha256(out/'metrics.csv')},
                 definition='Unique selected novel orbits per all attempts; fixed metrics remain comparable across rebinning.',source=binding(__file__))
    write_json(out/'receipt.json',receipt);return receipt


def extend_boundary(boundary_dir,known_stages,protected_stages,out):
    boundary,known,protected=load_boundary(boundary_dir);sources=[]
    for scope,stages in [('known',known_stages),('protected',protected_stages)]:
        for p in stages:
            r=validate_stage(p)
            if r['parent']['sha256']!=boundary['parent']['sha256']: raise ValueError('Stage parent differs')
            known.update(json.loads((Path(p)/'all_generated_keys.json').read_text()))
            if scope=='protected':
                with np.load(Path(p)/'candidates.npz',allow_pickle=False) as z:keys=[''.join(map(str,m)) for m in z['masks']]
                protected.update(keys);known.update(keys)
            sources.append(dict(scope=scope,stage=binding(Path(p)/'receipt.json')))
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'known_history.json',sorted(known));write_json(out/'protected.json',sorted(protected))
    receipt=dict(status='FROZEN_ADAPTIVE_CANONICAL_BOUNDARY',created_unix=time.time(),parent=boundary['parent'],
                 history=binding(out/'known_history.json'),protected=binding(out/'protected.json'),known_orbits=len(known),
                 protected_orbits=len(protected),previous_boundary=binding(Path(boundary_dir)/'receipt.json'),
                 proposal_sources=sources,no_energy_members_read=True,source=binding(__file__))
    write_json(out/'receipt.json',receipt);return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('boundary');b.add_argument('--boundary-dir',type=Path,required=True);b.add_argument('--h0',type=Path,required=True);b.add_argument('--assessment-dir',type=Path,required=True);b.add_argument('--output-dir',type=Path,required=True)
    s=sub.add_parser('schema');s.add_argument('--data',type=Path,required=True);s.add_argument('--parent',type=Path,required=True);s.add_argument('--reference-schema',type=Path,required=True);s.add_argument('--mode',choices=['fixed','adaptive','range'],required=True);s.add_argument('--output-dir',type=Path,required=True)
    k=sub.add_parser('seed-bank');k.add_argument('--data',type=Path,required=True);k.add_argument('--parent',type=Path,required=True);k.add_argument('--schema',type=Path,required=True);k.add_argument('--boundary-dir',type=Path,required=True);k.add_argument('--output-dir',type=Path,required=True)
    g=sub.add_parser('propose');g.add_argument('--parent',type=Path,required=True);g.add_argument('--schema',type=Path,required=True);g.add_argument('--boundary-dir',type=Path,required=True);g.add_argument('--output-dir',type=Path,required=True);g.add_argument('--namespace',required=True);g.add_argument('--seed',type=int,required=True);g.add_argument('--method',required=True);g.add_argument('--checkpoint',type=Path);g.add_argument('--sampler',choices=['uniform','mutation']);g.add_argument('--seed-bank',type=Path);g.add_argument('--counts',nargs='+',type=int,default=[12,24,36]);g.add_argument('--bins',nargs='+',default=list(BIN_NAMES));g.add_argument('--attempts-per-cell',type=int,default=512);g.add_argument('--queries-per-cell',type=int,default=256);g.add_argument('--device',default='cuda');g.add_argument('--threads',type=int,default=4)
    v=sub.add_parser('verify-oracle');v.add_argument('--stage',type=Path,action='append',required=True);v.add_argument('--oracle-root',type=Path,required=True);v.add_argument('--collection-receipt',type=Path,required=True);v.add_argument('--output',type=Path,required=True);v.add_argument('--teacher',choices=['uma','mace'],default='uma')
    m=sub.add_parser('merge');m.add_argument('--base-data',type=Path,required=True);m.add_argument('--parent',type=Path,required=True);m.add_argument('--schema',type=Path,required=True);m.add_argument('--stage',type=Path,required=True);m.add_argument('--verifier',type=Path,required=True);m.add_argument('--output-dir',type=Path,required=True);m.add_argument('--round',type=int,default=1);m.add_argument('--new-weight',type=float,default=1.)
    e=sub.add_parser('summarize');e.add_argument('--stage',type=Path,required=True);e.add_argument('--verifier',type=Path,required=True);e.add_argument('--fixed-schema',type=Path,required=True);e.add_argument('--output-dir',type=Path,required=True)
    n=sub.add_parser('extend-boundary');n.add_argument('--boundary-dir',type=Path,required=True);n.add_argument('--known-stage',type=Path,action='append',default=[]);n.add_argument('--protected-stage',type=Path,action='append',default=[]);n.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv);require_compute_allocation()
    if a.command=='boundary':r=bootstrap_boundary(a.boundary_dir,a.h0,a.assessment_dir,a.output_dir)
    elif a.command=='schema':r=fit_round_schema(a.data,a.parent,a.reference_schema,a.mode,a.output_dir)
    elif a.command=='seed-bank':r=freeze_seed_bank(a.data,a.parent,a.schema,a.boundary_dir,a.output_dir)
    elif a.command=='propose':r=propose(a.parent,a.schema,a.boundary_dir,a.output_dir,namespace=a.namespace,seed=a.seed,method=a.method,checkpoint=a.checkpoint,sampler=a.sampler,seed_bank=a.seed_bank,counts=a.counts,bins=a.bins,attempts_per_cell=a.attempts_per_cell,queries_per_cell=a.queries_per_cell,device=a.device,threads=a.threads)
    elif a.command=='verify-oracle':r=verify_oracle(a.stage,a.oracle_root,a.collection_receipt,a.output,a.teacher)
    elif a.command=='merge':r=merge_training(a.base_data,a.parent,a.schema,a.stage,a.verifier,a.output_dir,a.round,a.new_weight)
    elif a.command=='summarize':r=summarize(a.stage,a.verifier,a.fixed_schema,a.output_dir)
    else:r=extend_boundary(a.boundary_dir,a.known_stage,a.protected_stage,a.output_dir)
    print(json.dumps({'status':r['status']}))


if __name__=='__main__':main()

"""Freeze key-only iterative boundaries and data-lineage-aware proposal manifests.

No energies are read while extending a boundary. Include *all* assessment
proposal bundles, including failed/diagnostic attempts, before selecting fits.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

from cathode_cvae.data import sha256
from cathode_cvae.iterative_acquisition import _base_lineage, _canonical_vacancy, _keys, _parent
from cathode_cvae.runtime import require_compute_allocation


def binding(path, expected=None):
    p = Path(path).resolve()
    digest = sha256(p)
    if expected is not None and digest != expected:
        raise ValueError(f"Input hash mismatch: {p}")
    return {"path": str(p), "sha256": digest}


def write(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def verify_boundary(directory):
    p = Path(directory)
    r = json.loads((p/'receipt.json').read_text())
    if r['status'] not in ('FROZEN_DEVELOPMENT_KEY_ONLY_BOUNDARY', 'FROZEN_ITERATIVE_KEY_ONLY_BOUNDARY'):
        raise ValueError('Wrong boundary receipt')
    for name in ('parent', 'history', 'protected'):
        binding(r[name]['path'], r[name]['sha256'])
    known = _keys(r['history']['path'], 48)
    protected = _keys(r['protected']['path'], 48)
    if not protected <= known:
        raise ValueError('Protected history is not a subset of known history')
    return r, known, protected


def generation_keys(directory, parent_hash, permutations):
    d = Path(directory).resolve()
    receipt = json.loads((d/'receipt.json').read_text())
    protocol = json.loads((d/'protocol.json').read_text())
    if receipt.get('status') != 'FROZEN_BEFORE_SCORING' or not receipt.get('no_energy_filtering'):
        raise ValueError('Generator receipt is not frozen and unfiltered')
    binding(d/'protocol.json', receipt['protocol_sha256'])
    if protocol['inputs']['parent']['sha256'] != parent_hash:
        raise ValueError('Proposal bundle belongs to another parent')
    binding(d/'candidates.npz', receipt['outputs']['candidates.npz'])
    # candidates.npz includes primary and diagnostic successful proposals.
    with np.load(d/'candidates.npz', allow_pickle=False) as z:
        if str(z['parent_sha256'].item()) != parent_hash:
            raise ValueError('Candidate parent differs')
        keys = set(_canonical_vacancy(1-z['masks'], permutations).tolist())
    return keys, {'receipt': binding(d/'receipt.json'), 'protocol': binding(d/'protocol.json'),
                  'candidates': binding(d/'candidates.npz'), 'orbits': len(keys)}


def freeze_history(boundary_dir, base_data, protect_generation, known_generation, output_dir):
    boundary, known, protected = verify_boundary(boundary_dir)
    parent_path = boundary['parent']['path']
    permutations, _ = _parent(parent_path)
    with np.load(base_data, allow_pickle=False) as z:
        keys = _canonical_vacancy(z['occupancy'], permutations)
        roles = z['split'].astype(str)
    if len(keys) != len(roles) or set(roles)-{'train','val','test'}:
        raise ValueError('Invalid input dataset roles')
    known.update(keys.tolist())
    protected.update(keys[np.isin(roles, ['val','test'])].tolist())
    records = []
    for scope, dirs in [('protected', protect_generation), ('known', known_generation)]:
        for d in dirs:
            ids, source = generation_keys(d, boundary['parent']['sha256'], permutations)
            known.update(ids)
            if scope == 'protected':
                protected.update(ids)
            records.append(dict(scope=scope, **source))
    if not records:
        raise ValueError('Declare at least one newly exposed proposal bundle')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    write(out/'known_history.json', sorted(known))
    write(out/'protected.json', sorted(protected))
    result = {'status':'FROZEN_ITERATIVE_KEY_ONLY_BOUNDARY', 'created_utc':datetime.now(timezone.utc).isoformat(),
              'parent': boundary['parent'], 'base_data':binding(base_data),
              'previous_boundary_receipt':binding(Path(boundary_dir)/'receipt.json'), 'proposal_sources':records,
              'history':binding(out/'known_history.json'), 'protected':binding(out/'protected.json'),
              'known_history_orbits':len(known), 'protected_orbits':len(protected),
              'inherited_train_protected_overlap':len(set(keys[roles=='train'].tolist()) & protected),
              'role_policy':'Existing roles preserved; new protected orbits excluded from additions.',
              'no_protected_energy_members_read':True}
    write(out/'receipt.json', result)
    return result


def freeze_manifest(plan_path, boundary_dir, parent, data, schema, round_index, previous_merge_receipt, output):
    plan = json.loads(Path(plan_path).read_text())
    if plan.get('status') != 'DECLARED_ITERATIVE_DEVELOPMENT' or plan.get('purpose') not in ('acquisition','assessment'):
        raise ValueError('Plan must declare iterative DEVELOPMENT acquisition or assessment')
    boundary, known, protected = verify_boundary(boundary_dir)
    parent_hash = sha256(parent)
    if parent_hash != boundary['parent']['sha256']:
        raise ValueError('Boundary parent differs')
    origin, request_schema = _base_lineage(data, schema, parent, round_index, previous_merge_receipt)
    permutations, _ = _parent(parent)
    with np.load(data, allow_pickle=False) as z:
        data_keys = set(_canonical_vacancy(z['occupancy'], permutations).tolist())
    if not data_keys <= known or not protected <= known:
        raise ValueError('History misses dataset or protected orbits')
    namespace = plan.get('namespace')
    if not isinstance(namespace, str) or not namespace:
        raise ValueError('A distinct, predeclared stream namespace is required')
    prior = []
    for path in plan.get('previous_manifests', []):
        prior_m = json.loads(Path(path).read_text())
        if prior_m.get('namespace') == namespace:
            raise ValueError('Fresh stream namespace duplicates a prior manifest')
        prior.append(binding(path))
    if plan['purpose'] == 'assessment' and not prior:
        raise ValueError('Assessment must declare prior namespaces to establish a fresh stream')
    counts, bins, count, width = plan['counts'], plan['requested_bins'], plan['count_per_group'], plan.get('latent_stream_width',32)
    if not counts or len(set(counts))!=len(counts) or any(type(n) is not int or not 0<=n<=48 for n in counts):
        raise ValueError('Invalid counts')
    if not bins or len(set(bins))!=len(bins) or set(bins)-{'low','middle','high'}:
        raise ValueError('Invalid energy bins')
    if type(count) is not int or count<1 or type(width) is not int or width<1:
        raise ValueError('Invalid draw/latent budget')
    models, seen = [], set()
    for raw in plan['models']:
        entry = dict(raw)
        identity = (entry.get('method'), str(entry.get('replicate_id','')))
        if not all(identity) or identity in seen:
            raise ValueError('Invalid or repeated method/replicate')
        seen.add(identity)
        if entry.get('checkpoint'):
            if entry.get('sampler'): raise ValueError('Mixed checkpoint and sampler')
            ck = binding(entry['checkpoint'],entry.get('expected_checkpoint_sha256'))
            cfg_path=Path(ck['path']).with_name('configuration.json')
            cfg=json.loads(cfg_path.read_text())
            if (cfg.get('seed')!=entry.get('fit_seed') or cfg.get('parent_sha256')!=parent_hash
                or cfg.get('data_sha256')!=sha256(data) or cfg.get('condition_schema_id')!=request_schema['identity']
                or cfg.get('condition_mode')!='multibin' or cfg.get('energy_target','deletion')!='deletion'):
                raise ValueError('Checkpoint lineage or conditioning differs')
            if type(cfg.get('latent_dim')) is not int or not 0<=cfg['latent_dim']<=width or cfg.get('model_format_version') not in (3,4,5):
                raise ValueError('Unsupported checkpoint format or latent width')
            entry.update(checkpoint=ck['path'], expected_checkpoint_sha256=ck['sha256'], configuration=binding(cfg_path))
        elif entry.get('sampler') not in ('mutation','uniform'):
            raise ValueError('Unknown sampler')
        models.append(entry)
    if not models: raise ValueError('No methods declared')
    inputs={'parent':binding(parent),'data':binding(data),'schema':binding(schema),'history':boundary['history']}
    m={'status':'FROZEN_ITERATIVE_DEVELOPMENT_BEFORE_GENERATION', 'created_utc':datetime.now(timezone.utc).isoformat(),
       'namespace':namespace,'purpose':plan['purpose'],'counts':counts,'requested_bins':bins,'count_per_group':count,
       'diagnostics':False,'latent_stream_width':width,'expected_primary_attempts':len(models)*len(counts)*len(bins)*count,
       'models':models,'inputs':inputs,'expected_condition_schema_id':request_schema['identity'],
       'origin_data_sha256':origin,'round':round_index,'plan':binding(plan_path),
       'boundary_receipt':binding(Path(boundary_dir)/'receipt.json'), 'previous_manifests':prior,
       'previous_merge_receipt':None if previous_merge_receipt is None else binding(previous_merge_receipt),
       'namespace_scope':'Freshness verified only against explicitly declared previous manifests.',
       'source':binding(__file__)}
    write(output,m)
    return m


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='cmd',required=True)
    h=sub.add_parser('history')
    for n in ('boundary-dir','base-data','output-dir'): h.add_argument('--'+n,type=Path,required=True)
    h.add_argument('--protect-generation',type=Path,action='append',default=[])
    h.add_argument('--known-generation',type=Path,action='append',default=[])
    m=sub.add_parser('manifest')
    for n in ('plan','boundary-dir','parent','data','schema','output'): m.add_argument('--'+n,type=Path,required=True)
    m.add_argument('--round',type=int,required=True);m.add_argument('--previous-merge-receipt',type=Path)
    a=p.parse_args(); require_compute_allocation()
    if a.cmd=='history': result=freeze_history(a.boundary_dir,a.base_data,a.protect_generation,a.known_generation,a.output_dir)
    else: result=freeze_manifest(a.plan,a.boundary_dir,a.parent,a.data,a.schema,a.round,a.previous_merge_receipt,a.output)
    print(json.dumps({'status':result['status']},indent=2))

if __name__=='__main__': main()

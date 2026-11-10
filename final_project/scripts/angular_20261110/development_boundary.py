"""Freeze and enforce the DEVELOPMENT-only mask boundary without reading old energies.

Three stages surround the existing generator/scorer/evaluator:
  freeze: union previously exposed canonical identities and explicitly protected
          H0/round1/paired/editor/FINAL identities, with source receipts;
  manifest: bind models, H0, geometry, request schema and new history before draws;
  filter: preserve every raw attempt but exclude protected identities before any
          oracle registration. The resulting bundle is consumed unchanged by
          scripts/overnight_20261108/evaluate_controls.py.

All mask keys use the original convention: 1 denotes a vacancy. No protected
energy member or old FINAL score is opened. No model or MLIP is evaluated here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path, expected=None):
    p = Path(path).absolute()
    actual = digest(p)
    if expected is not None and actual != expected:
        raise ValueError(f'Input bytes disagree with their prior receipt: {p}')
    return {'path': str(p), 'sha256': actual}


def save_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def save_csv(path, rows):
    if not rows:
        raise ValueError(f'Cannot infer CSV fields for empty {path}')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def canonical_keys(occupancy, permutations, batch_size=128):
    """Independent integer orbit canonicalization; occupancy 1=present.

    Site permutations are the same parent-preserving group used by generation;
    periodic image orbits and force vectors are not inferred from these masks.
    """
    x, p = np.asarray(occupancy), np.asarray(permutations)
    if x.ndim != 2 or not 1 <= x.shape[1] <= 64 or not np.isin(x, [0, 1]).all():
        raise ValueError('Occupancy must be binary and have 1..64 candidate sites')
    n = x.shape[1]
    if p.ndim != 2 or p.shape[1] != n or not np.array_equal(
            np.sort(p, axis=1), np.tile(np.arange(n), (len(p), 1))):
        raise ValueError('Invalid parent-site permutations')
    inverse = np.argsort(p, axis=1)
    result = []
    for start in range(0, len(x), batch_size):
        q = x[start:start + batch_size].astype(np.uint8, copy=False)
        best = np.full(len(q), np.iinfo(np.uint64).max, np.uint64)
        for perm in inverse:
            bits = np.packbits(1 - q[:, perm], axis=1, bitorder='big')
            padded = np.zeros((len(q), 8), np.uint8)
            padded[:, :bits.shape[1]] = bits
            np.minimum(best, padded.view('>u8').reshape(-1).astype(np.uint64), out=best)
        result.extend(format(int(code) >> (64 - n), f'0{n}b') for code in best)
    return np.asarray(result, dtype=f'<U{n}')


def _json_keys(path, sites):
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict):
        value = next(value[k] for k in ('keys', 'canonical_masks', 'known_keys', 'canonical_keys') if k in value)
    keys = set(map(str, value))
    if any(len(k) != sites or set(k) - {'0', '1'} for k in keys):
        raise ValueError(f'Unexpected canonical key encoding in {path}')
    return keys


def _check_embedded_parent(z, parent_hash):
    if 'parent_sha256' in z and str(z['parent_sha256'].item()) != parent_hash:
        raise ValueError('Identity source has a different parent')


def _npz_masks(path, parent, parent_hash, split=None, edit_pairs=False):
    """Return all identities and selected protected identities; never read energy."""
    with np.load(path, allow_pickle=False) as z:
        _check_embedded_parent(z, parent_hash)
        role = z['split'].astype(str) if split is not None else None
        splits = (split,) if isinstance(split, str) else tuple(split or ())
        if any(s not in set(role) for s in splits):
            raise ValueError(f'Missing required protected role {splits}: {path}')
        if edit_pairs:
            if 'parent_sha256' not in z:
                raise ValueError('Editor data must bind its parent')
            all_keys, protected = set(), {s: set() for s in splits}
            for data_key, key_field in [('seeds', 'canonical_seed_keys'), ('children', 'canonical_child_keys')]:
                actual = canonical_keys(z[data_key], parent['permutations'])
                if not np.array_equal(actual, z[key_field].astype(str)):
                    raise ValueError(f'Editor canonical identities differ for {data_key}')
                all_keys.update(actual.tolist())
                for s in splits:
                    protected[s].update(actual[role == s].tolist())
            return all_keys, (protected[split] if isinstance(split, str) else protected)
        if 'occupancy' in z:
            occ = z['occupancy']
        elif 'masks' in z:
            occ = 1 - z['masks']
        else:
            raise ValueError(f'No occupation or mask member: {path}')
        actual = canonical_keys(occ, parent['permutations'])
        if 'canonical_masks' in z and not np.array_equal(actual, z['canonical_masks'].astype(str)):
            raise ValueError(f'Saved canonical masks do not match parent group: {path}')
        if 'structure_ids' in z and split is None:
            ids = z['structure_ids'].astype(str)
            if all(s.startswith('mask48_') for s in ids) and len(actual) and len(actual[0]) == 48:
                expected = [f'mask48_{int(k, 2):012x}' for k in actual]
                if ids.tolist() != expected:
                    raise ValueError(f'Structure IDs do not match canonical masks: {path}')
        if 'parent_sha256' not in z and split is None:
            # Historical improvement FINAL stores the actual parent arrays, not a hash.
            for source, target in [('numbers', 'numbers'), ('positions_A', 'positions_A'),
                                   ('cell_A', 'cell_A'), ('li_indices', 'li_indices')]:
                if source not in z or target not in parent or not np.array_equal(z[source], parent[target]):
                    raise ValueError(f'Unbound parent geometry in {path}: {source}')
        all_keys = set(actual.tolist())
        if split is None:
            return all_keys, all_keys
        protected = {s: set(actual[role == s].tolist()) for s in splits}
        return all_keys, (protected[split] if isinstance(split, str) else protected)


def _registry_keys(path, parent_hash, sites):
    d = json.loads(Path(path).read_text())
    if d.get('parent_sha256') != parent_hash:
        raise ValueError(f'Oracle registry uses a different parent: {path}')
    width = (sites + 3) // 4
    keys = set()
    for row in d['physical_keys'].values():
        sid = str(row['structure_id'])
        prefix = f'mask{sites}_'
        if not sid.startswith(prefix) or len(sid[len(prefix):]) != width:
            raise ValueError(f'Unexpected physical registry identity: {sid}')
        key = format(int(sid[len(prefix):], 16), f'0{sites}b')
        if len(key) != sites:
            raise ValueError('Registry identity overflows parent sites')
        keys.add(key)
    return keys


def _receipt_output(path, key, receipt_name='receipt.json'):
    p = Path(path)
    receipt = p.parent / receipt_name
    obj = json.loads(receipt.read_text())
    value = obj['outputs'][key]
    expected = value['sha256'] if isinstance(value, dict) else value
    result = binding(p, expected)
    result['receipt'] = binding(receipt)
    return result


def freeze_boundary(overnight_root, prior_root, output_dir):
    old, prior, out = Path(overnight_root).absolute(), Path(prior_root).absolute(), Path(output_dir).absolute()
    static_manifest = old / 'runs/overnight_20261108/final/static_manifest.json'
    manifest = json.loads(static_manifest.read_text())
    source = {name: binding(value['path'], value['sha256']) for name, value in manifest['inputs'].items()}
    parent_path, data_path = Path(source['parent']['path']), Path(source['data']['path'])
    parent_hash = source['parent']['sha256']
    with np.load(parent_path, allow_pickle=False) as z:
        parent = {k: z[k] for k in ['numbers', 'positions_A', 'cell_A', 'li_indices', 'permutations']}
    sites = parent['permutations'].shape[1]
    if sites != 48:
        raise ValueError('This audited boundary is defined for the existing 48-site parent')
    history = _json_keys(source['history']['path'], sites)
    before = len(history)
    records = [{'name': 'frozen_historical_history', **source['history'], 'orbits': before}]
    protected, scopes = set(), {}

    def add_data(name, path, splits, expected, edit_pairs=False):
        nonlocal history, protected
        b = binding(path, expected)
        all_ids, roles = _npz_masks(path, parent, parent_hash, splits, edit_pairs)
        history.update(all_ids)
        for role, held in roles.items():
            protected.update(held)
            scopes[f'{name}_{role.upper()}'] = held
            records.append({'name': f'{name}_{role.upper()}', **b, 'all_orbits': len(all_ids),
                            'protected_orbits': len(held), 'role': role, 'energy_members_read': False})

    add_data('H0', data_path, ('val', 'test'), source['data']['sha256'])
    adaptation = json.loads((old/'artifacts/current/improvement_v1/data/adaptation_receipt.json').read_text())
    add_data('ROUND1', old/'artifacts/current/improvement_v1/training_round1.npz', ('val', 'test'), adaptation['training_sha256'])
    prep = old/'runs/overnight_20261108/paired/prepared'
    prep_receipt = json.loads((prep/'preparation.json').read_text())
    add_data('PAIRED', prep/'cohort.npz', ('val', 'test'), prep_receipt['outputs']['cohort.npz'])
    editor = prior/'runs/conditioning_20261107/editor/data_v2'
    editor_receipt = json.loads((editor/'preparation_receipt.json').read_text())
    add_data('EDITOR', editor/'edit_pairs.npz', ('val', 'heldout'), editor_receipt['outputs']['edit_pairs.npz'], True)
    with np.load(data_path, allow_pickle=False) as z:
        h0_train = set(z['canonical_masks'][z['split'].astype(str) == 'train'].astype(str).tolist())
    if len(h0_train) != 40157:
        raise ValueError('The inherited H0 TRAIN cohort is not the verified benchmark')

    first_final = old/'artifacts/current/improvement_v1/data/final.npz'
    first_receipt = json.loads((first_final.parent/'final_receipt.json').read_text())
    first_binding = binding(first_final, first_receipt['outputs']['all']['sha256'])
    all_ids, _ = _npz_masks(first_final, parent, parent_hash)
    history.update(all_ids); protected.update(all_ids); scopes['IMPROVEMENT_FINAL'] = all_ids
    records.append({'name': 'IMPROVEMENT_FINAL', **first_binding, 'receipt': binding(first_final.parent/'final_receipt.json'),
                    'protected_orbits': len(all_ids), 'energy_members_read': False})

    cond = prior/'runs/conditioning_20261107'
    folders = [('CONDITIONING_FINAL', cond/'final'),
               ('CONDITIONING_FINAL_MATCHED', cond/'final_matched_audit'),
               ('EDITOR_FINAL', cond/'editor/assessment_v1'),
               ('AGGREGATE_PRIOR_FINAL', cond/'aggregate_prior/assessment'),
               ('LATENT_DEVELOPMENT_AND_PROBES', cond/'development/latent'),
               ('OVERNIGHT_STATIC_FINAL', old/'runs/overnight_20261108/final/static_generated')]
    folders.extend((f'OVERNIGHT_PAIRED_{stage.upper()}', old/'runs/overnight_20261108/final/paired_generated/generated'/stage)
                   for stage in ['current_low', 'paired_deletion', 'paired_endpoint', 'shared_uniform'])
    for name, folder in folders:
        path = folder/'candidates.npz'
        b = _receipt_output(path, 'candidates.npz')
        all_ids, _ = _npz_masks(path, parent, parent_hash)
        history.update(all_ids); protected.update(all_ids); scopes[name] = all_ids
        records.append({'name': name, **b, 'protected_orbits': len(all_ids), 'energy_members_read': False})

    # Previously proposed development identities count as historical even if a
    # later failed or cached oracle call never entered a physical-key registry.
    development_paths = (sorted((cond/'development').glob('*/candidates.npz'))
        + sorted((old/'runs/overnight_20261108/development').glob('*/generated/candidates.npz')))
    for path in development_paths:
        receipt_name = 'receipt.json' if (path.parent/'receipt.json').exists() else 'generation_receipt.json'
        b = _receipt_output(path, 'candidates.npz', receipt_name)
        ids, _ = _npz_masks(path, parent, parent_hash)
        extra = len(ids - history)
        history.update(ids)
        records.append({'name': 'HISTORICAL_DEVELOPMENT_CANDIDATES', **b,
                        'known_orbits': len(ids), 'new_history_orbits': extra,
                        'scope': 'identities only; no oracle scores', 'energy_members_read': False})

    for name, path in [('PRIOR_ORACLE_REGISTRY', cond/'oracle/registry.json'),
                       ('OVERNIGHT_ORACLE_REGISTRY', old/'runs/overnight_20261108/oracle/registry.json')]:
        b = binding(path)
        ids = _registry_keys(path, parent_hash, sites)
        extra = len(ids - history)
        history.update(ids)
        records.append({'name': name, **b, 'known_orbits': len(ids), 'new_history_orbits': extra,
                        'scope': 'structure identities only; no oracle scores'})
    if not protected <= history or not h0_train <= history:
        raise ValueError('Known history does not contain all protected or H0 training orbits')
    if out.exists():
        raise FileExistsError(f'Preserve existing boundary directory: {out}')
    out.mkdir(parents=True)
    save_json(out/'known_history.json', sorted(history))
    save_json(out/'protected.json', sorted(protected))
    overlaps = {name: len(keys & h0_train) for name, keys in scopes.items()}
    receipt = {'status': 'FROZEN_DEVELOPMENT_KEY_ONLY_BOUNDARY', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': digest(__file__), 'original_static_manifest': binding(static_manifest),
        'parent': source['parent'], 'data': source['data'], 'request_schema': source['schema'],
        'original_history': source['history'], 'sources': records,
        'history': binding(out/'known_history.json'), 'protected': binding(out/'protected.json'),
        'historical_history_orbits': before, 'known_history_orbits': len(history), 'protected_orbits': len(protected),
        'protected_scope_counts': {k: len(v) for k, v in scopes.items()},
        'inherited_H0_TRAIN_overlap_by_scope': overlaps,
        'inherited_H0_TRAIN_overlap_union': len(protected & h0_train),
        'role_policy': 'H0 TRAIN is retained exactly; protected identities cannot enter new scoring or new acquisition/training additions. H0 VAL/TEST roles remain unchanged.',
        'no_protected_energy_members_read': True, 'new_model_forwards': 0, 'new_MLIP_calls': 0}
    save_json(out/'receipt.json', receipt)
    return receipt


def verify_boundary(path):
    root = Path(path)
    r = json.loads((root/'receipt.json').read_text())
    if r.get('status') != 'FROZEN_DEVELOPMENT_KEY_ONLY_BOUNDARY':
        raise ValueError('Expected a key-only frozen development boundary')
    for field in ('history', 'protected', 'parent', 'data', 'request_schema', 'original_history'):
        binding(r[field]['path'], r[field]['sha256'])
    if 'original_static_manifest' in r:
        binding(r['original_static_manifest']['path'], r['original_static_manifest']['sha256'])
    for source in r.get('sources', []):
        binding(source['path'], source['sha256'])
        if 'receipt' in source:
            binding(source['receipt']['path'], source['receipt']['sha256'])
    protected = _json_keys(r['protected']['path'], 48)
    history = _json_keys(r['history']['path'], 48)
    if len(protected) != r['protected_orbits'] or len(history) != r['known_history_orbits'] or not protected <= history:
        raise ValueError('Frozen history/protected coverage changed')
    return r, protected, history


def freeze_manifest(plan_path, boundary_dir, threshold_path, teacher_schema_items, output_path):
    from cathode_cvae.conditions import load_schema
    plan = json.loads(Path(plan_path).read_text())
    if plan.get('status') != 'DECLARED_DEVELOPMENT_COMPARISON':
        raise ValueError('Plan must explicitly declare DEVELOPMENT, not a held-out final')
    boundary, _, _ = verify_boundary(boundary_dir)
    source_root = Path(__file__).absolute().parents[2]
    inputs = {name: boundary[key] for name, key in [('parent', 'parent'), ('data', 'data'),
                                                       ('schema', 'request_schema'), ('history', 'history')]}
    request_schema = load_schema(inputs['schema']['path'])
    if request_schema['provenance']['data_sha256'] != inputs['data']['sha256'] or request_schema['provenance']['parent_sha256'] != inputs['parent']['sha256']:
        raise ValueError('Original request schema does not bind the protected H0 data/parent')
    teachers = {}
    for item in teacher_schema_items:
        name, path = item.split('=', 1)
        if name in teachers:
            raise ValueError('Repeated teacher schema')
        schema = load_schema(path)
        if schema['provenance'].get('parent_sha256') != inputs['parent']['sha256']:
            raise ValueError(f'Teacher schema uses a different parent: {name}')
        teachers[name] = {**binding(path), 'schema_id': schema['identity']}
    if 'uma' not in teachers or teachers['uma']['schema_id'] != request_schema['identity']:
        raise ValueError('UMA teacher and request schema differ')
    counts = plan.get('counts', [12, 24, 36])
    bins = plan.get('requested_bins', ['low', 'middle', 'high'])
    count = plan.get('count_per_group', 64)
    width = plan.get('latent_stream_width', 32)
    if not counts or len(set(counts)) != len(counts) or any(type(n) is not int or not 0 <= n <= 48 for n in counts):
        raise ValueError('Counts must be distinct valid integers')
    if not bins or len(set(bins)) != len(bins) or any(b not in ('low', 'middle', 'high') for b in bins):
        raise ValueError('Distinct declared energy bins required')
    if type(count) is not int or count < 1 or type(width) is not int or width < 1:
        raise ValueError('Positive attempt and common-latent stream budgets required')
    models, seen, method_reps = [], set(), {}
    for raw in plan['models']:
        entry = dict(raw)
        if not entry.get('method') or 'replicate_id' not in entry:
            raise ValueError('Every model/sampler needs method and replicate_id')
        identity = (entry['method'], str(entry['replicate_id']))
        if identity in seen:
            raise ValueError('Duplicate method/replicate identity')
        seen.add(identity); method_reps.setdefault(entry['method'], set()).add(str(entry['replicate_id']))
        if entry.get('checkpoint'):
            if entry.get('sampler'):
                raise ValueError('An entry is either a checkpoint or a sampler')
            ck = binding(entry['checkpoint'], entry.get('expected_checkpoint_sha256'))
            cfg_path = Path(ck['path']).with_name('configuration.json')
            cfg = json.loads(cfg_path.read_text())
            if (cfg.get('seed') != entry.get('fit_seed') or cfg.get('parent_sha256') != inputs['parent']['sha256']
                    or cfg.get('data_sha256') != inputs['data']['sha256'] or cfg.get('condition_schema_id') != request_schema['identity']
                    or cfg.get('condition_mode') != 'multibin' or cfg.get('energy_target', 'deletion') != 'deletion'):
                raise ValueError(f'Checkpoint metadata differs from declared H0 comparison: {ck["path"]}')
            if type(cfg.get('latent_dim')) is not int or not 0 <= cfg['latent_dim'] <= width:
                raise ValueError('Common random latent stream is too narrow')
            if cfg.get('model_format_version') not in (3, 4, 5):
                raise ValueError('Unrecognized checkpoint format; matching loader required')
            entry.update(checkpoint=ck['path'], expected_checkpoint_sha256=ck['sha256'],
                         configuration=binding(cfg_path), declared_latent_dim=cfg['latent_dim'],
                         declared_model_format_version=cfg['model_format_version'])
        elif entry.get('sampler') not in ('mutation', 'uniform'):
            raise ValueError('Unsupported declared baseline')
        models.append(entry)
    for pair in plan.get('contrasts', []):
        a, b = pair['method'], pair['reference']
        if a not in method_reps or b not in method_reps or method_reps[a] != method_reps[b]:
            raise ValueError('Every predeclared paired contrast needs the same replicate IDs')
    source_files = [source_root/'scripts/overnight_20261108'/name for name in
                    ['generate_controls.py', 'evaluate_controls.py', 'summarize_paired_sensitivity.py']]
    source_files += sorted((source_root/'src/cathode_cvae').rglob('*.py'))
    source_files.append(Path(__file__).absolute())
    source_bindings = [binding(path) for path in source_files]
    manifest = {'status': 'FROZEN_DEVELOPMENT_BEFORE_GENERATION', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'namespace': plan['namespace'], 'counts': counts, 'requested_bins': bins, 'count_per_group': count,
        'diagnostics': bool(plan.get('diagnostics', True)), 'latent_stream_width': width,
        'expected_primary_attempts': len(models) * len(counts) * len(bins) * count,
        'expected_diagnostic_attempts': (sum(bool(e.get('declared_latent_dim')) for e in models)
            * len(counts) * count if plan.get('diagnostics', True) and 'low' in bins else 0),
        'models': models, 'contrasts': plan.get('contrasts', []), 'inputs': inputs,
        'expected_condition_schema_id': request_schema['identity'], 'teacher_schemas': teachers,
        'fixed_thresholds': binding(threshold_path), 'boundary_receipt': binding(Path(boundary_dir)/'receipt.json'),
        'source_bindings': source_bindings, 'plan': binding(plan_path),
        'roles': 'DEVELOPMENT only. H0 VAL remains available for checkpoint selection under its original role. H0 TEST and previous FINAL energy labels are not used for fitting or selection.',
        'population_update': False, 'novel_angular_checkpoint_policy': 'Use matching source-bound versioned loader and strict state_dict; no padding or silent fallback.',
        'new_model_forwards': 0, 'new_MLIP_calls': 0}
    save_json(output_path, manifest)
    return manifest


def verify_manifest(path):
    m = json.loads(Path(path).read_text())
    if m.get('status') != 'FROZEN_DEVELOPMENT_BEFORE_GENERATION':
        raise ValueError('Not a frozen development manifest')
    bindings = [*m['inputs'].values(), *m['teacher_schemas'].values(), m['fixed_thresholds'], m['boundary_receipt'], m['plan'], *m['source_bindings']]
    for entry in m['models']:
        if entry.get('checkpoint'):
            bindings += [{'path': entry['checkpoint'], 'sha256': entry['expected_checkpoint_sha256']}, entry['configuration']]
    for b in bindings:
        binding(b['path'], b['sha256'])
    return m


def filter_bundle(generated_dir, boundary_dir, output_dir):
    generated, out = Path(generated_dir).absolute(), Path(output_dir).absolute()
    boundary, protected, _ = verify_boundary(boundary_dir)
    receipt = json.loads((generated/'receipt.json').read_text())
    protocol = json.loads((generated/'protocol.json').read_text())
    if receipt.get('status') != 'FROZEN_BEFORE_SCORING' or receipt['protocol_sha256'] != digest(generated/'protocol.json'):
        raise ValueError('Original proposal pool is not frozen before scoring')
    manifest_path = Path(protocol['manifest']['path'])
    if digest(manifest_path) != protocol['manifest']['sha256']:
        raise ValueError('Proposal manifest changed')
    manifest = verify_manifest(manifest_path)
    source_map = {Path(b['path']).name: b['sha256'] for b in manifest['source_bindings']}
    if (receipt['source_sha256'] != source_map.get('generate_controls.py')
            or receipt['runtime_loader_source']['sha256'] != source_map.get('generate_mask.py')):
        raise ValueError('Generation did not use the frozen generator and runtime loader sources')
    if manifest['boundary_receipt']['sha256'] != digest(Path(boundary_dir)/'receipt.json'):
        raise ValueError('Proposal manifest uses another protected boundary')
    for name, expected in receipt['outputs'].items():
        binding(generated/name, expected)
    for name, key in [('parent', 'parent'), ('data', 'data'), ('history', 'history')]:
        if protocol['inputs'][name]['sha256'] != boundary[key]['sha256']:
            raise ValueError(f'Proposal input differs from protected boundary: {name}')
    with np.load(boundary['parent']['path'], allow_pickle=False) as z:
        permutations = z['permutations']; sites = permutations.shape[1]; host = len(z['numbers']) - sites
    rows = read_csv(generated/'attempts_all.csv')
    primary = [r for r in rows if r['diagnostic'] == 'primary']
    if len(primary) != manifest['expected_primary_attempts'] or len(rows) != receipt['total_attempts']:
        raise ValueError('Frozen raw attempt denominator changed')
    if len({r['attempt_id'] for r in rows}) != len(rows):
        raise ValueError('Repeated raw attempt identity')
    with np.load(generated/'raw_attempts.npz', allow_pickle=False) as z:
        occ = z['occupancy']; uniforms = z['uniforms']; latents = z['latents']
    if occ.shape != (len(rows), sites) or uniforms.shape != occ.shape or len(latents) != len(rows):
        raise ValueError('Raw attempts and random streams are not row-aligned')
    generated_rows = np.asarray([r['status'] == 'generated' for r in rows])
    actual_keys = canonical_keys(occ[generated_rows], permutations)
    old_keys = [r['canonical_mask'] for r in rows if r['status'] == 'generated']
    if actual_keys.tolist() != old_keys:
        raise ValueError('Generated keys do not match independent symmetry canonicalization')
    by_group, common_u, common_z, original_statuses = Counter(), {}, {}, Counter()
    protected_count = 0
    for i, row in enumerate(rows):
        if int(row['raw_row']) != i:
            raise ValueError('Raw row order differs from attempt table')
        original_statuses[row['status']] += 1
        row['original_status'] = row['status']
        row['boundary_status'] = 'original_non_generated'
        if row['status'] == 'generated':
            key = row['canonical_mask']; n = int(row['n_target'])
            if int(np.sum(occ[i])) != n or row['scoring_structure_id'] != f'mask{sites}_{int(key, 2):0{(sites+3)//4}x}':
                raise ValueError('Attempt count or exact structure identity differs')
            row['boundary_status'] = 'protected' if key in protected else 'eligible'
            if key in protected:
                row['status'] = 'protected_boundary'
                row['error'] = 'Excluded before scoring to preserve existing protected roles'
                protected_count += 1
        if row['diagnostic'] == 'primary':
            by_group[(row['method'], row['replicate_id'], row['n_target'], row['requested_bin'])] += 1
            ukey = (row['replicate_id'], row['n_target'], row['attempt'])
            u = uniforms[i].tobytes()
            if ukey in common_u and common_u[ukey] != u:
                raise ValueError('Paired AR uniform streams differ')
            common_u[ukey] = u
            dim = int(row['latent_dim'])
            if dim:
                z = latents[i, :dim]
                if ukey in common_z:
                    old_z = common_z[ukey]
                    if not np.array_equal(z[:min(dim, len(old_z))], old_z[:min(dim, len(old_z))]):
                        raise ValueError('Paired latent stream prefixes differ')
                    if dim > len(old_z):
                        common_z[ukey] = z.copy()
                else:
                    common_z[ukey] = z.copy()
    expected = {(e['method'], str(e['replicate_id']), str(n), b): manifest['count_per_group']
                for e in manifest['models'] for n in manifest['counts'] for b in manifest['requested_bins']}
    if dict(by_group) != expected:
        raise ValueError('One or more declared method/fit/count/bin attempt registries are incomplete')
    if out.exists():
        raise FileExistsError(f'Preserve existing derived bundle: {out}')
    out.mkdir(parents=True)
    shutil.copyfile(generated/'protocol.json', out/'protocol.json')
    for name in ['raw_attempts.npz', 'raw_primary.npz', 'raw_diagnostics.npz']:
        shutil.copyfile(generated/name, out/name)
    save_csv(out/'attempts_all.csv', rows)
    save_csv(out/'attempts.csv', [r for r in rows if r['diagnostic'] == 'primary'])

    def write_pool(name, selected):
        codes = sorted({int(r['canonical_mask'], 2) for r in selected if r['status'] == 'generated'})
        masks = np.asarray([[(code >> j) & 1 for j in range(sites - 1, -1, -1)] for code in codes], np.uint8).reshape(-1, sites)
        np.savez_compressed(out/name, masks=masks, occupancy=1-masks,
            structure_ids=np.asarray([f'mask{sites}_{code:0{(sites+3)//4}x}' for code in codes]),
            n_target=sites-masks.sum(1), n_Li=sites-masks.sum(1), n_atoms=host+sites-masks.sum(1),
            parent_sha256=np.asarray(boundary['parent']['sha256']))
        return len(codes)
    eligible = write_pool('candidates.npz', rows)
    write_pool('primary_candidates.npz', [r for r in rows if r['diagnostic'] == 'primary'])
    write_pool('diagnostic_candidates.npz', [r for r in rows if r['diagnostic'] != 'primary'])
    registry = Counter((r['method'], r['replicate_id'], r['n_target'], r['requested_bin'], r['canonical_mask'], r['boundary_status'])
                       for r in rows if r['diagnostic'] == 'primary' and r['original_status'] == 'generated')
    registry_rows = [dict(method=k[0], replicate_id=k[1], n_target=k[2], requested_bin=k[3], canonical_mask=k[4],
                          scoring_structure_id=f'mask{sites}_{int(k[4], 2):0{(sites+3)//4}x}', boundary_status=k[5],
                          attempts=v) for k, v in sorted(registry.items())]
    if registry_rows:
        save_csv(out/'candidate_registry.csv', registry_rows)
    else:
        (out/'candidate_registry.csv').write_text('method,replicate_id,n_target,requested_bin,canonical_mask,scoring_structure_id,boundary_status,attempts\n')
    files = ['protocol.json','attempts.csv','attempts_all.csv','raw_attempts.npz','raw_primary.npz','raw_diagnostics.npz',
             'candidates.npz','primary_candidates.npz','diagnostic_candidates.npz','candidate_registry.csv']
    derived = {'status': 'FROZEN_BEFORE_SCORING', 'protocol_sha256': digest(out/'protocol.json'),
        'primary_attempts': len(primary), 'total_attempts': len(rows), 'unique_geometries': eligible,
        'source_sha256': digest(__file__), 'original_generation_receipt': binding(generated/'receipt.json'),
        'original_protocol': binding(generated/'protocol.json'), 'manifest': binding(manifest_path),
        'boundary_receipt': binding(Path(boundary_dir)/'receipt.json'),
        'protected_attempts_excluded_before_scoring': protected_count,
        'original_status_counts': dict(original_statuses), 'candidate_registry_scope': 'One row per primary method/replicate/count/bin/canonical orbit with attempt multiplicity; protected rows retained and tagged.',
        'all_attempt_denominators_preserved': True, 'raw_arrays_unchanged': True,
        'new_model_forwards': 0, 'new_MLIP_calls': 0, 'energy_members_read': False,
        'outputs': {name: digest(out/name) for name in files}}
    save_json(out/'receipt.json', derived)
    return derived


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    f = subs.add_parser('freeze'); f.add_argument('--overnight-root', type=Path, required=True); f.add_argument('--prior-root', type=Path, required=True); f.add_argument('--output-dir', type=Path, required=True)
    m = subs.add_parser('manifest'); m.add_argument('--plan', type=Path, required=True); m.add_argument('--boundary-dir', type=Path, required=True); m.add_argument('--thresholds', type=Path, required=True); m.add_argument('--teacher-schema', action='append', required=True); m.add_argument('--output', type=Path, required=True)
    v = subs.add_parser('verify'); v.add_argument('--manifest', type=Path, required=True)
    q = subs.add_parser('filter'); q.add_argument('--generated-dir', type=Path, required=True); q.add_argument('--boundary-dir', type=Path, required=True); q.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args(argv)
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    if a.command == 'freeze': result = freeze_boundary(a.overnight_root, a.prior_root, a.output_dir)
    elif a.command == 'manifest': result = freeze_manifest(a.plan, a.boundary_dir, a.thresholds, a.teacher_schema, a.output)
    elif a.command == 'verify': result = verify_manifest(a.manifest)
    else: result = filter_bundle(a.generated_dir, a.boundary_dir, a.output_dir)
    print(json.dumps({k: result[k] for k in ['status', 'expected_primary_attempts', 'protected_orbits', 'known_history_orbits', 'protected_attempts_excluded_before_scoring'] if k in result}, indent=2))
    return result


if __name__ == '__main__':
    main()

"""Verify selected development checkpoint latent sidecars against checkpoints.

Runs only on trusted local checkpoints explicitly bound by frozen development
manifests, and only in a compute allocation. No generated proposals, protected
outcomes, input energy members, model inference, or MLIP/DFT calls are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    def reject(value):
        raise ValueError(f'Nonfinite JSON constant {value}')
    return json.loads(Path(path).read_text(), parse_constant=reject)


def canonical(value):
    # JSON serialization also rejects NaN/Infinity in the embedded object.
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def bound(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'Frozen hash mismatch: {path}')
    return actual


def model_key(entry):
    return str(entry['method']), str(entry['replicate_id'])


def declarations(master_path, master_sha, manifest_paths):
    bound(master_path, master_sha)
    master = load_json(master_path)
    if master.get('status') != 'DECLARED_DEVELOPMENT_COMPARISON':
        raise ValueError('Unexpected master plan status')
    declared = {model_key(e): e for e in master['models']}
    if len(declared) != len(master['models']) or not declared:
        raise ValueError('Master plan has duplicate or no model declarations')
    manifest_sources, frozen = [], {}
    for path in manifest_paths:
        path = Path(path).resolve()
        m = load_json(path)
        if m.get('status') != 'FROZEN_DEVELOPMENT_BEFORE_GENERATION':
            raise ValueError('Expected frozen development manifest')
        for field in ('namespace', 'counts', 'requested_bins', 'count_per_group', 'diagnostics', 'latent_stream_width'):
            if m[field] != master[field]: raise ValueError(f'Manifest/master design differs: {field}')
        split_plan = m['plan']; bound(split_plan['path'], split_plan['sha256'])
        plan = load_json(split_plan['path'])
        if plan.get('status') != master['status']:
            raise ValueError('Per-replicate plan status differs from master')
        for field in ('namespace', 'counts', 'requested_bins', 'count_per_group', 'diagnostics', 'latent_stream_width'):
            if plan[field] != master[field]: raise ValueError(f'Split/master design differs: {field}')
        sub = {model_key(e): e for e in plan['models']}
        if len(sub) != len(plan['models']) or set(sub) != {model_key(e) for e in m['models']}:
            raise ValueError('Split plan/manifest arm sets differ')
        for e in m['models']:
            key = model_key(e)
            if key in frozen or key not in declared or sub[key] != declared[key]:
                raise ValueError('Frozen arm is repeated or does not match master declaration')
            for field in ('method', 'replicate_id', 'fit_seed', 'checkpoint', 'sampler'):
                if e.get(field) != declared[key].get(field):
                    raise ValueError(f'Frozen arm differs from master: {key}, {field}')
            frozen[key] = e
        manifest_sources.append({'path': str(path), 'sha256': sha256(path), 'split_plan': split_plan})
    if set(declared) != set(frozen):
        raise ValueError('Manifests do not cover every master-plan arm exactly once')
    return master, frozen, manifest_sources


def audit_one(entry, loader):
    checkpoint = Path(entry['checkpoint']).resolve()
    config_binding = entry['configuration']; config_path = Path(config_binding['path']).resolve()
    if config_path != checkpoint.with_name('configuration.json'):
        raise ValueError('Manifest configuration is not adjacent to checkpoint')
    config_hash = bound(config_path, config_binding['sha256'])
    cfg = load_json(config_path)
    dimension = entry['declared_latent_dim']
    if type(dimension) is not int or dimension < 0 or cfg.get('latent_dim') != dimension:
        raise ValueError('Manifest and configuration latent dimensions differ')
    if cfg.get('seed') != entry['fit_seed'] or cfg.get('model_format_version') != entry['declared_model_format_version']:
        raise ValueError('Manifest and configuration seed or format differ')
    if dimension == 0:
        if cfg.get('kind') != 'ar_baseline':
            raise ValueError('Zero-latent checkpoint is not declared as no-latent AR')
        return {'status': 'SKIPPED_NO_LATENT', 'latent_dim': 0, 'configuration_sha256': config_hash}
    checkpoint_hash = bound(checkpoint, entry['expected_checkpoint_sha256'])
    sidecar = Path(str(checkpoint) + '.latent.json')
    sidecar_hash = sha256(sidecar)
    sidecar_stats = load_json(sidecar)
    obj = loader(checkpoint, 'cpu')
    if not isinstance(obj, dict) or not isinstance(obj.get('configuration'), dict):
        raise ValueError('Trusted checkpoint lacks embedded configuration')
    embedded_cfg = obj['configuration']
    if canonical({k: v for k, v in embedded_cfg.items() if k != 'selected_validation_beta'}) != canonical(cfg):
        raise ValueError('Embedded and adjacent configuration differ')
    if embedded_cfg.get('seed') != entry['fit_seed'] or embedded_cfg.get('latent_dim') != dimension:
        raise ValueError('Checkpoint seed or latent dimension differs')
    beta = obj.get('selection_beta')
    if beta is None or embedded_cfg.get('selected_validation_beta') != beta:
        raise ValueError('Checkpoint selection rule is missing or inconsistent')
    embedded = obj.get('latent_statistics')
    if embedded is None or canonical(embedded) != canonical(sidecar_stats):
        raise ValueError('Latent sidecar differs from checkpoint embedded statistics')
    if embedded.get('epoch') != obj.get('epoch') or embedded.get('optimizer_updates') != obj.get('optimizer_updates'):
        raise ValueError('Latent statistics and checkpoint update/epoch differ')
    if embedded.get('pooled', {}).get('latent_dim') != dimension:
        raise ValueError('Pooled latent statistics dimension differs')
    if sha256(checkpoint) != checkpoint_hash or sha256(sidecar) != sidecar_hash or sha256(config_path) != config_hash:
        raise ValueError('Checkpoint, sidecar, or configuration changed during audit')
    return {'status': 'PASS', 'latent_dim': dimension, 'configuration_sha256': config_hash,
            'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_hash,
            'sidecar': str(sidecar), 'sidecar_sha256': sidecar_hash,
            'epoch': obj['epoch'], 'optimizer_updates': obj['optimizer_updates'], 'selection_beta': beta}


def run(master_path, master_sha, manifest_paths, output, loader):
    output = Path(output)
    if output.exists(): raise FileExistsError(f'Preserve existing receipt: {output}')
    _, frozen, manifests = declarations(master_path, master_sha, manifest_paths)
    results = []
    for key, entry in sorted(frozen.items()):
        base = {'method': key[0], 'replicate_id': key[1]}
        if not entry.get('checkpoint'):
            results.append({**base, 'status': 'SKIPPED_SAMPLER'}); continue
        try:
            result = audit_one(entry, loader)
        except Exception as exc:
            checkpoint = Path(entry['checkpoint']).resolve()
            sidecar = Path(str(checkpoint) + '.latent.json')
            result = {'status': 'FAIL', 'checkpoint': str(checkpoint),
                      'expected_checkpoint_sha256': entry.get('expected_checkpoint_sha256'),
                      'checkpoint_sha256': sha256(checkpoint) if checkpoint.is_file() else None,
                      'sidecar': str(sidecar), 'sidecar_sha256': sha256(sidecar) if sidecar.is_file() else None,
                      'error_type': type(exc).__name__, 'error': str(exc)}
        results.append({**base, **result})
    counts = {status: sum(r['status'] == status for r in results)
              for status in ('PASS', 'FAIL', 'SKIPPED_NO_LATENT', 'SKIPPED_SAMPLER')}
    receipt = {'status': 'FAIL' if counts['FAIL'] else 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(),
               'master_plan': {'path': str(Path(master_path).resolve()), 'sha256': master_sha},
               'source_sha256': sha256(__file__), 'manifests': manifests, 'counts': counts, 'arms': results,
               'scope': 'Trusted frozen checkpoint embedded latent statistics versus adjacent sidecar; exact parsed JSON equivalence; not latent interpretability or scientific validation',
               'new_model_forwards': 0, 'new_MLIP_calls': 0, 'protected_outcomes_read': False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f: json.dump(receipt, f, indent=2, allow_nan=False); f.write('\n')
    return receipt


def main():
    from cathode_cvae.learning.runtime import trusted_checkpoint
    from cathode_cvae.runtime import require_compute_allocation
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--master-plan', required=True, type=Path)
    p.add_argument('--master-sha256', required=True)
    p.add_argument('--manifest', action='append', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args(); require_compute_allocation()
    result = run(args.master_plan, args.master_sha256, args.manifest, args.output, trusted_checkpoint)
    print(json.dumps({'status': result['status'], 'counts': result['counts']}, indent=2))
    if result['status'] != 'PASS': raise SystemExit(1)


if __name__ == '__main__': main()

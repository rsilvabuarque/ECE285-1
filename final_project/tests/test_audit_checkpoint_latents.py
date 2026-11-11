"""Synthetic trusted-checkpoint and frozen-manifest tamper checks."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

path = Path(__file__).resolve().parents[1] / 'scripts/angular_20261110/audit_checkpoint_latents.py'
spec = importlib.util.spec_from_file_location('audit_checkpoint_latents', path)
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


def dump(path, obj): path.write_text(json.dumps(obj, indent=2) + '\n')
def binding(path): return {'path': str(path), 'sha256': audit.sha256(path)}
def loader(path, device): return torch.load(path, map_location=device, weights_only=False)


def fixture(root):
    cfg = {'seed': 123, 'latent_dim': 2, 'model_format_version': 5, 'kind': 'ar_cvae'}
    config = root / 'configuration.json'; dump(config, cfg)
    stats = {'epoch': 4, 'optimizer_updates': 27,
             'pooled': {'latent_dim': 2, 'KL_nats_by_dimension': [0.1, 0.2]}, 'by_composition': {}}
    ckpt = root / 'best_validation_beta_0p1.pt'
    torch.save({'configuration': {**cfg, 'selected_validation_beta': 0.1}, 'selection_beta': 0.1,
                'epoch': 4, 'optimizer_updates': 27, 'latent_statistics': stats}, ckpt)
    sidecar = Path(str(ckpt) + '.latent.json'); dump(sidecar, stats)
    model = {'method': 'example', 'replicate_id': 123, 'fit_seed': 123, 'checkpoint': str(ckpt)}
    uniform = {'method': 'uniform', 'replicate_id': 123, 'sampler': 'uniform'}
    plan = {'status': 'DECLARED_DEVELOPMENT_COMPARISON', 'namespace': 'synthetic',
            'counts': [12], 'requested_bins': ['low'], 'count_per_group': 2,
            'diagnostics': True, 'latent_stream_width': 4, 'models': [model, uniform]}
    master = root / 'master.json'; dump(master, plan)
    split = root / 'split.json'; dump(split, plan)
    entry = {**model, 'configuration': binding(config), 'expected_checkpoint_sha256': audit.sha256(ckpt),
             'declared_latent_dim': 2, 'declared_model_format_version': 5}
    manifest = root / 'manifest.json'
    dump(manifest, {**plan, 'status': 'FROZEN_DEVELOPMENT_BEFORE_GENERATION',
                    'plan': binding(split), 'models': [entry, uniform]})
    return master, manifest, ckpt, sidecar, config


def run(root, master, manifest):
    return audit.run(master, audit.sha256(master), [manifest], root / 'receipt.json', loader)


def test_exact_checkpoint_sidecar_and_receipt(tmp_path):
    master, manifest, ckpt, sidecar, config = fixture(tmp_path)
    r = run(tmp_path, master, manifest)
    assert r['status'] == 'PASS'
    assert r['counts'] == {'PASS': 1, 'FAIL': 0, 'SKIPPED_NO_LATENT': 0, 'SKIPPED_SAMPLER': 1}
    row = next(r for r in r['arms'] if r['method'] == 'example')
    assert row['checkpoint_sha256'] == audit.sha256(ckpt)
    assert row['sidecar_sha256'] == audit.sha256(sidecar)
    assert row['configuration_sha256'] == audit.sha256(config)
    with pytest.raises(FileExistsError): run(tmp_path, master, manifest)


def test_sidecar_tamper_fails_and_leaves_receipt(tmp_path):
    master, manifest, _, sidecar, _ = fixture(tmp_path)
    value = json.loads(sidecar.read_text()); value['pooled']['KL_nats_by_dimension'][0] = .11; dump(sidecar, value)
    r = run(tmp_path, master, manifest)
    assert r['status'] == 'FAIL'
    failed = next(r for r in r['arms'] if r['status'] == 'FAIL')
    assert 'sidecar differs' in failed['error']
    assert failed['sidecar_sha256'] == audit.sha256(sidecar)
    assert (tmp_path / 'receipt.json').exists()


def test_checkpoint_hash_change_is_rejected_before_deserialization(tmp_path):
    master, manifest, ckpt, _, _ = fixture(tmp_path)
    with ckpt.open('ab') as f: f.write(b'tamper')
    r = audit.run(master, audit.sha256(master), [manifest], tmp_path / 'receipt.json',
                  lambda *_: pytest.fail('unbound checkpoint must not be loaded'))
    assert r['status'] == 'FAIL'
    assert 'Frozen hash mismatch' in next(r for r in r['arms'] if r['status'] == 'FAIL')['error']


def test_seed_or_embedded_config_tamper_fails(tmp_path):
    master, manifest, ckpt, _, _ = fixture(tmp_path)
    obj = loader(ckpt, 'cpu'); obj['configuration']['seed'] = 456; torch.save(obj, ckpt)
    m = json.loads(manifest.read_text()); m['models'][0]['expected_checkpoint_sha256'] = audit.sha256(ckpt); dump(manifest, m)
    r = run(tmp_path, master, manifest)
    assert r['status'] == 'FAIL'
    assert 'configuration differ' in next(r for r in r['arms'] if r['status'] == 'FAIL')['error']


def test_missing_sidecar_reports_hashes_and_failure(tmp_path):
    master, manifest, ckpt, sidecar, _ = fixture(tmp_path)
    sidecar.unlink()
    r = run(tmp_path, master, manifest)
    failed = next(r for r in r['arms'] if r['status'] == 'FAIL')
    assert failed['checkpoint_sha256'] == audit.sha256(ckpt)
    assert failed['sidecar_sha256'] is None


def test_master_or_split_plan_mismatch_is_rejected(tmp_path):
    master, manifest, _, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='Frozen hash mismatch'):
        audit.run(master, '0'*64, [manifest], tmp_path / 'bad.json', loader)
    m = json.loads(manifest.read_text()); m['models'] = m['models'][:1]; dump(manifest, m)
    with pytest.raises(ValueError, match='arm sets differ'):
        run(tmp_path, master, manifest)

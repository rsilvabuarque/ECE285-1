"""Receipt and all-valid-pair checks using only synthetic four-site masks."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


path = Path(__file__).resolve().parents[1] / 'scripts/angular_20261110/all_valid_mask_diagnostics.py'
spec = importlib.util.spec_from_file_location('all_valid_mask_diagnostics', path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def fixture(root):
    gen = root / 'generated'; gen.mkdir()
    manifest = {'status': 'FROZEN_DEVELOPMENT_BEFORE_GENERATION', 'requested_bins': ['low', 'middle', 'high'],
                'counts': [2], 'count_per_group': 2, 'diagnostics': True,
                'models': [{'method': 'cvae', 'replicate_id': 1, 'declared_latent_dim': 2}]}
    manifest_path = root / 'manifest.json'; manifest_path.write_text(json.dumps(manifest))
    auxiliary = root / 'boundary.json'; auxiliary.write_text('{}')
    protocol = gen / 'protocol.json'; protocol.write_text('{}')
    masks = [[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0],
             [1, 0, 0, 1], [0, 0, 1, 1], [float('nan')]*4, [1, 0, 1, 0]]
    keys = ['0011', '0011', '0101', '0011', '0110', '0011', '', '0101']
    labels = ['low', 'middle', 'high', 'latent_B'] * 2
    rows = []
    for i, label in enumerate(labels):
        protected = i == 1
        original = 'invalid_mask' if i == 6 else 'generated'
        rows.append(dict(method='cvae', replicate_id='1', n_target='2', paired_id=f'pair-{i//4}',
                         attempt_id=f'attempt-{i}',
                         requested_bin='low' if label == 'latent_B' else label,
                         diagnostic='latent_B' if label == 'latent_B' else 'primary', fit_seed='11',
                         raw_row=i, original_status=original,
                         status='protected_boundary' if protected else original,
                         boundary_status='protected' if protected else 'eligible', canonical_mask=keys[i]))
    write_csv(gen / 'attempts_all.csv', rows)
    np.savez_compressed(gen / 'raw_attempts.npz', occupancy=np.asarray(masks),
                        uniforms=np.tile([.1, .2, .3, .4], (8, 1)),
                        latents=np.asarray([[1., 2.] if label != 'latent_B' else [3., 4.] for label in labels]))
    binding = lambda p: {'path': str(p), 'sha256': audit.digest(p)}
    receipt = {'status': 'FROZEN_BEFORE_SCORING', 'all_attempt_denominators_preserved': True,
               'raw_arrays_unchanged': True, 'total_attempts': 8, 'manifest': binding(manifest_path),
               'boundary_receipt': binding(auxiliary), 'original_generation_receipt': binding(auxiliary),
               'outputs': {name: audit.digest(gen / name) for name in ('attempts_all.csv', 'raw_attempts.npz', 'protocol.json')}}
    (gen / 'receipt.json').write_text(json.dumps(receipt))
    ev = root / 'evaluation'; ev.mkdir()
    ev_rows = []
    for index in range(2):
        for contrast in ('condition_middle_minus_low', 'condition_high_minus_low', 'alternate_latent_minus_low'):
            ev_rows.append(dict(teacher='uma', method='cvae', replicate_id='1', n_Li='2', paired_id=f'pair-{index}',
                                contrast=contrast, both_energies_known=str(not (index == 0 and contrast == 'condition_middle_minus_low')
                                                                           and not (index == 1 and contrast == 'condition_high_minus_low'))))
    write_csv(ev / 'paired_attempts.csv', ev_rows)
    outcomes = [dict(row, teacher='uma', energy_known=str(i not in (1, 6))) for i, row in enumerate(rows)]
    write_csv(ev / 'attempt_outcomes.csv', outcomes)
    ev_receipt = {'proposal_protocol_sha256': audit.digest(protocol), 'teachers': ['uma'],
                  'outputs': {name: audit.digest(ev / name) for name in ('paired_attempts.csv', 'attempt_outcomes.csv')}}
    (ev / 'receipt.json').write_text(json.dumps(ev_receipt))
    return gen, ev


def test_protected_masks_count_in_mask_statistics_and_not_energy_coverage(tmp_path):
    gen, ev = fixture(tmp_path)
    out = tmp_path / 'out'
    audit.run(gen, out, ev)
    summary = {r['contrast']: r for r in audit.rows_from(out / 'mask_summary.csv')}
    mid = summary['condition_middle_minus_low']
    assert mid['attempted_pairs'] == mid['valid_pairs'] == '2'
    assert mid['protected_valid_pairs'] == '1'
    assert float(mid['same_orbit_fraction']) == .5
    assert float(mid['mean_raw_Hamming']) == 2.
    assert summary['condition_high_minus_low']['valid_pairs'] == '1'
    coverage = {r['contrast']: r for r in audit.rows_from(out / 'energy_coverage.csv')}
    assert coverage['condition_middle_minus_low']['both_energies_known'] == '1'
    assert float(coverage['condition_middle_minus_low']['energy_pair_coverage']) == .5
    with pytest.raises(FileExistsError): audit.run(gen, out, ev)


def test_hash_mismatch_and_latent_or_uniform_leak_fail_closed(tmp_path):
    gen, _ = fixture(tmp_path)
    with (gen / 'attempts_all.csv').open('a') as f: f.write('tamper\n')
    with pytest.raises(ValueError, match='hash mismatch'): audit.run(gen, tmp_path / 'bad')


def test_distinct_latent_stream_is_required(tmp_path):
    gen, _ = fixture(tmp_path)
    raw_path = gen / 'raw_attempts.npz'
    with np.load(raw_path) as archive: arrays = {k: archive[k] for k in archive.files}
    arrays['latents'][3] = arrays['latents'][0]
    np.savez_compressed(raw_path, **arrays)
    receipt_path = gen / 'receipt.json'; receipt = json.loads(receipt_path.read_text())
    receipt['outputs']['raw_attempts.npz'] = audit.digest(raw_path); receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='latent inputs'): audit.run(gen, tmp_path / 'bad')


def test_evaluation_must_match_raw_attempts_even_with_updated_receipt(tmp_path):
    gen, ev = fixture(tmp_path)
    outcomes_path = ev / 'attempt_outcomes.csv'
    outcomes = audit.rows_from(outcomes_path)
    outcomes[0]['canonical_mask'] = '1111'
    write_csv(outcomes_path, outcomes)
    receipt_path = ev / 'receipt.json'; receipt = json.loads(receipt_path.read_text())
    receipt['outputs']['attempt_outcomes.csv'] = audit.digest(outcomes_path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='differs from frozen raw attempt'):
        audit.run(gen, tmp_path / 'bad', ev)

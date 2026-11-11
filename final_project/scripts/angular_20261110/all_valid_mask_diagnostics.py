"""Paired mask responses on a frozen development pool, including protected masks.

No energy values, prior assessments, model fits, or potential calls are read or
performed. The optional evaluation contributes only paired score-availability
flags. Hamming counts use the sampled site order; orbit equality uses the
independently checked canonical keys in the boundary-filtered receipt.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows_from(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def truth(value):
    if value not in ('True', 'False'):
        raise ValueError(f'Invalid boolean: {value!r}')
    return value == 'True'


def verify_file(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Receipt hash mismatch: {path}')


def run(generated, output, evaluation=None):
    generated, output = Path(generated), Path(output)
    receipt_path = generated / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('status') != 'FROZEN_BEFORE_SCORING'
            or not receipt.get('all_attempt_denominators_preserved')
            or not receipt.get('raw_arrays_unchanged')
            or 'boundary_receipt' not in receipt or 'original_generation_receipt' not in receipt):
        raise ValueError('A boundary-filtered frozen development receipt is required')
    for name in ('attempts_all.csv', 'raw_attempts.npz', 'protocol.json'):
        verify_file(generated / name, receipt['outputs'][name])
    for name in ('boundary_receipt', 'original_generation_receipt'):
        verify_file(receipt[name]['path'], receipt[name]['sha256'])
    manifest_binding = receipt['manifest']
    manifest_path = Path(manifest_binding['path'])
    verify_file(manifest_path, manifest_binding['sha256'])
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'FROZEN_DEVELOPMENT_BEFORE_GENERATION' or 'low' not in manifest['requested_bins']:
        raise ValueError('Manifest is not a declared development comparison')
    rows = rows_from(generated / 'attempts_all.csv')
    if len(rows) != receipt['total_attempts']:
        raise ValueError('Attempt count differs from receipt')
    with np.load(generated / 'raw_attempts.npz', allow_pickle=False) as archive:
        occupancy = archive['occupancy']; uniforms = archive['uniforms']; latents = archive['latents']
    if occupancy.ndim != 2 or uniforms.shape != occupancy.shape or len(latents) != len(rows) or len(occupancy) != len(rows):
        raise ValueError('Raw arrays do not align with attempts')

    groups = defaultdict(dict)
    for i, row in enumerate(rows):
        if int(row['raw_row']) != i:
            raise ValueError('Attempt rows are not aligned with raw arrays')
        label = row['requested_bin'] if row['diagnostic'] == 'primary' else row['diagnostic']
        key = (row['method'], row['replicate_id'], row['n_target'], row['paired_id'])
        if label in groups[key]:
            raise ValueError('Duplicate paired trial')
        groups[key][label] = (row, i)

    pairs = []
    pair_attempts = {}
    for key, group in sorted(groups.items()):
        if 'low' not in group:
            raise ValueError('Paired trial is missing LOW')
        a, i = group['low']
        for right, contrast in [('middle', 'condition_middle_minus_low'), ('high', 'condition_high_minus_low'),
                                ('latent_B', 'alternate_latent_minus_low')]:
            if right not in group:
                continue
            b, j = group[right]
            if not np.array_equal(uniforms[i], uniforms[j]):
                raise ValueError('Paired per-site uniforms differ')
            same_z = np.array_equal(latents[i], latents[j])
            if (right == 'latent_B' and same_z) or (right != 'latent_B' and not same_z):
                raise ValueError('Paired latent inputs do not match diagnostic design')
            valid = a['original_status'] == b['original_status'] == 'generated'
            if valid:
                x, y = occupancy[i], occupancy[j]
                if (not np.isin(x, [0, 1]).all() or not np.isin(y, [0, 1]).all()
                        or x.sum() != int(key[2]) or y.sum() != int(key[2])
                        or not a['canonical_mask'] or not b['canonical_mask']):
                    raise ValueError('Generated mask is invalid or lacks canonical identity')
            pairs.append(dict(method=key[0], replicate_id=key[1], fit_seed=a['fit_seed'], n_Li=int(key[2]),
                              paired_id=key[3], contrast=contrast, both_valid=valid,
                              raw_Hamming=int(np.count_nonzero(occupancy[i] != occupancy[j])) if valid else '',
                              same_orbit=(a['canonical_mask'] == b['canonical_mask']) if valid else '',
                              either_protected=(a['boundary_status'] == 'protected' or b['boundary_status'] == 'protected') if valid else ''))
            pair_attempts[(key[0], key[1], key[2], key[3], contrast)] = (a['attempt_id'], b['attempt_id'])

    cell_groups = defaultdict(list)
    for pair in pairs:
        cell_groups[pair['method'], pair['replicate_id'], pair['n_Li'], pair['contrast']].append(pair)
    expected = set()
    for model in manifest['models']:
        contrasts = []
        if 'middle' in manifest['requested_bins']: contrasts.append('condition_middle_minus_low')
        if 'high' in manifest['requested_bins']: contrasts.append('condition_high_minus_low')
        if manifest['diagnostics'] and model.get('declared_latent_dim', 0): contrasts.append('alternate_latent_minus_low')
        expected.update((model['method'], str(model['replicate_id']), n, contrast)
                        for n in manifest['counts'] for contrast in contrasts)
    if set(cell_groups) != expected:
        raise ValueError('Paired diagnostic cells differ from frozen manifest')
    cell_rows = []
    for key, subset in sorted(cell_groups.items()):
        if len(subset) != manifest['count_per_group']:
            raise ValueError('Paired attempt denominator differs from manifest')
        valid = [x for x in subset if x['both_valid']]
        cell_rows.append(dict(method=key[0], replicate_id=key[1], fit_seed=subset[0]['fit_seed'], n_Li=key[2], contrast=key[3],
                              attempted_pairs=len(subset), valid_pairs=len(valid), valid_coverage=len(valid)/len(subset),
                              protected_valid_pairs=sum(x['either_protected'] for x in valid),
                              mean_raw_Hamming=float(np.mean([x['raw_Hamming'] for x in valid])) if valid else '',
                              same_orbit_fraction=float(np.mean([x['same_orbit'] for x in valid])) if valid else ''))

    coverage_rows = []
    evaluation_receipt = None
    if evaluation is not None:
        evaluation = Path(evaluation)
        ev_path = evaluation / 'receipt.json'
        ev = json.loads(ev_path.read_text())
        if ev.get('proposal_protocol_sha256') != digest(generated / 'protocol.json'):
            raise ValueError('Evaluation did not use this frozen proposal protocol')
        verify_file(evaluation / 'paired_attempts.csv', ev['outputs']['paired_attempts.csv'])
        verify_file(evaluation / 'attempt_outcomes.csv', ev['outputs']['attempt_outcomes.csv'])
        raw_by_id = {r['attempt_id']: r for r in rows}
        if len(raw_by_id) != len(rows): raise ValueError('Repeated raw attempt ID')
        outcome_index = {}
        for row in rows_from(evaluation / 'attempt_outcomes.csv'):
            key = (row['teacher'], row['attempt_id'])
            if row['teacher'] not in ev['teachers'] or row['attempt_id'] not in raw_by_id or key in outcome_index:
                raise ValueError('Unexpected or repeated evaluation outcome')
            source = raw_by_id[row['attempt_id']]
            if any(row[field] != source[field] for field in ('raw_row', 'status', 'canonical_mask', 'paired_id')):
                raise ValueError('Evaluation outcome differs from frozen raw attempt')
            outcome_index[key] = truth(row['energy_known'])
        if len(outcome_index) != len(rows) * len(ev['teachers']):
            raise ValueError('Evaluation outcome coverage is incomplete')
        pair_index = {(p['method'], p['replicate_id'], str(p['n_Li']), p['paired_id'], p['contrast']): p for p in pairs}
        ev_groups = defaultdict(list)
        seen = set()
        for row in rows_from(evaluation / 'paired_attempts.csv'):
            key = (row['method'], row['replicate_id'], row['n_Li'], row['paired_id'], row['contrast'])
            teacher_key = (row['teacher'],) + key
            if row['teacher'] not in ev['teachers'] or key not in pair_index or teacher_key in seen:
                raise ValueError('Evaluation contains unexpected or repeated paired attempts')
            seen.add(teacher_key)
            known = truth(row['both_energies_known'])
            ids = pair_attempts[key]
            if known != all(outcome_index[(row['teacher'], attempt_id)] for attempt_id in ids):
                raise ValueError('Paired energy coverage differs from evaluated outcomes')
            if known and not pair_index[key]['both_valid']:
                raise ValueError('Invalid mask pair cannot have two known energies')
            ev_groups[(row['teacher'], key[0], key[1], int(key[2]), key[4])].append(known)
        for teacher in ev['teachers']:
            for key, subset in sorted(cell_groups.items()):
                flags = ev_groups.get((teacher,) + key, [])
                if len(flags) != len(subset):
                    raise ValueError('Evaluation lacks paired attempts for a teacher or cell')
                coverage_rows.append(dict(teacher=teacher, method=key[0], replicate_id=key[1], n_Li=key[2], contrast=key[3],
                                          attempted_pairs=len(flags), both_energies_known=sum(flags),
                                          energy_pair_coverage=sum(flags)/len(flags)))
        evaluation_receipt = {'path': str(ev_path.resolve()), 'sha256': digest(ev_path)}

    if output.exists():
        raise FileExistsError(f'Preserve existing output: {output}')
    output.mkdir(parents=True)
    files = [('paired_masks.csv', pairs), ('mask_summary.csv', cell_rows)]
    if evaluation is not None: files.append(('energy_coverage.csv', coverage_rows))
    for name, content in files:
        if not content: raise ValueError(f'No rows for {name}')
        with (output / name).open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(content[0])); writer.writeheader(); writer.writerows(content)
    result = {'status': 'PASS', 'scope': 'Development paired masks; raw-site Hamming and parent-symmetry orbit equality; all originally valid masks including protected identities',
              'source_sha256': digest(__file__), 'filtered_receipt': {'path': str(receipt_path.resolve()), 'sha256': digest(receipt_path)},
              'manifest': manifest_binding, 'evaluation_receipt': evaluation_receipt,
              'outputs': {name: digest(output / name) for name, _ in files}, 'new_MLIP_calls': 0, 'new_training_updates': 0}
    with (output / 'receipt.json').open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    return result


def main():
    from cathode_cvae.runtime import require_compute_allocation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated-dir', type=Path, required=True)
    parser.add_argument('--evaluation-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    require_compute_allocation()
    print(json.dumps(run(args.generated_dir, args.output_dir, args.evaluation_dir), indent=2))


if __name__ == '__main__': main()

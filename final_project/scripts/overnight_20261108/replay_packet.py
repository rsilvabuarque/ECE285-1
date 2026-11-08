"""Recompute cached all-attempt metrics from a verified, portable run packet.

No model is fitted or called. Checkpoint/identity hashes are exact; only final
floating-point table arithmetic has a small, explicitly numerical allowance.
All files named by the packet manifest are checked before analysis starts.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Packet path escapes its root')
    return path


def load_helper(name):
    path = Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare_csv(actual, expected):
    """Preserve identities/counts exactly; allow only cached-arithmetic roundoff.

    1e-12 for dimensionless summaries is far below one changed attempt in this
    packet. Energy-summary allowance is 1e-10 in the column's stated units.
    Neither allowance changes a class boundary, label, mask, seed or numerator.
    """
    with Path(actual).open(newline='') as f:
        a = list(csv.DictReader(f))
    with Path(expected).open(newline='') as f:
        b = list(csv.DictReader(f))
    if len(a) != len(b):
        raise ValueError('Replayed row count differs: ' + str(actual))
    adjusted = 0
    for i, (left, right) in enumerate(zip(a, b)):
        if left.keys() != right.keys():
            raise ValueError('Replayed column schema differs')
        for field, value in left.items():
            target = right[field]
            if value == target:
                continue
            identity = (field.endswith('_id') or field.endswith('_seed') or
                'sha' in field.lower() or 'mask' in field.lower() or
                field in {'method', 'teacher', 'fit_seed', 'replicate_id',
                          'status', 'requested_bin', 'measured_bin', 'diagnostic'})
            integers = bool(re.fullmatch(r'[+-]?\d+', value or '') and
                            re.fullmatch(r'[+-]?\d+', target or ''))
            summary = any(s in field.lower() for s in
                ['energy', 'yield', 'fraction', 'rate', 'mean', 'median', 'std',
                 'rms', 'spearman', 'kendall', 'correlation', 'descriptive_seed_sd'])
            if identity or integers or not summary:
                raise ValueError(f'Exact field differs at row{i}/{field}: {value!r} != {target!r}')
            try:
                x, y = float(value), float(target)
            except (ValueError, TypeError):
                raise ValueError(f'Non-numerical field differs: {field}') from None
            allowance = 1e-10 if 'energy' in field.lower() or 'eV' in field else 1e-12
            if not (math.isfinite(x) and math.isfinite(y) and abs(x - y) <= allowance):
                raise ValueError(f'Numerical field differs beyond roundoff: row{i}/{field}')
            adjusted += 1
    return dict(rows=len(a), roundoff_only_cells=adjusted)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--plots', action='store_true')
    a = p.parse_args(argv)
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    root = a.packet.resolve()
    manifest = json.loads((root / 'replay_manifest.json').read_text())
    if manifest.get('status') != 'FROZEN_COMPLETE_REPLAY_PACKET':
        raise ValueError('A completed, frozen packet is required')
    for relative, expected in manifest['files'].items():
        if digest(contained(root, relative)) != expected:
            raise ValueError('Packet bytes changed: ' + relative)
    a.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records = []
    for stage in manifest['stages']:
        output = a.output_dir / stage['name']
        resolve = lambda key: str(contained(root, stage[key]))
        if stage['kind'] == 'static':
            args = ['--generated', resolve('generated'), '--parent', resolve('parent'),
                    '--history', resolve('history'), '--training-data', resolve('training_data'),
                    '--output-dir', str(output)]
            for teacher, path in stage['scores'].items():
                args += ['--score', teacher + '=' + str(contained(root, path)),
                         '--schema', teacher + '=' + str(contained(root, stage['schemas'][teacher]))]
            if stage.get('fixed_thresholds'):
                args += ['--fixed-thresholds', resolve('fixed_thresholds')]
            if a.plots:
                args.append('--plots')
            load_helper('evaluate_controls').main(args)
        elif stage['kind'] == 'paired':
            args = ['--prepared', resolve('prepared'), '--registration', resolve('registration'),
                    '--attempts', resolve('attempts'), '--history', resolve('history'),
                    '--deletion-schema', resolve('deletion_schema'),
                    '--endpoint-schema', resolve('endpoint_schema'),
                    '--original-thresholds', resolve('fixed_thresholds'),
                    '--facts', resolve('facts'), '--facts-sha256', stage['facts_sha256'],
                    '--queue', str(root / 'NO_HPC_QUEUE_REQUIRED'),
                    '--output-dir', str(output), '--unseal-test']
            load_helper('paired_evaluate').main(args)
        else:
            raise ValueError('Unknown cached analysis kind')
        comparisons = {name: compare_csv(output / name, contained(root, expected))
                       for name, expected in stage['expected_tables'].items()}
        records.append(dict(name=stage['name'], kind=stage['kind'], comparisons=comparisons))
    result = dict(status='PASS', manifest_sha256=digest(root / 'replay_manifest.json'),
        source_sha256=digest(__file__), stages=records, elapsed_seconds=time.monotonic()-started,
        new_training_updates=0, new_MLIP_calls=0, new_proposals=0,
        scope='Cached metric replay, not independent scientific replication or physical validation')
    (a.output_dir / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()

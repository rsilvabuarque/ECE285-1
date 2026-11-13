"""Evaluate the frozen round-two proposals after both oracle collections pass.

The per-arm evaluator is the prospectively bound W3 source.  The aggregate is
the separately identified W4 exact-set cached verifier, which was introduced
after the assessment protocol was frozen.  This driver does not collect scores
or start oracle jobs; it refuses incomplete collections and existing outputs.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import multiprocessing
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BASE = Path('/global/cfs/cdirs/m5397/rsb/2026/classes')
RUN = BASE / 'ECE285_angular_goal_20261110/final_project/runs/angular_20261110'
ASSESS = RUN / 'iteration_round2/assessment'
W3 = BASE / 'ECE285_angular_round2_goal_20261111/final_project'
W4 = BASE / 'ECE285_angular_audit_goal_20261111/final_project'
OLD = BASE / 'ECE285_overnight_20261108/final_project'
PARENT = OLD / 'artifacts/current/core/data/parents/O3_N48_pilot.npz'
BOUNDARY = RUN / 'iteration_boundary_after_r2_acquisition'
THRESHOLD = RUN / 'iteration_main_train_thresholds.csv'
EVALUATOR = W3 / 'scripts/overnight_20261108/evaluate_controls.py'
TEACHERS = ('uma', 'mace')

# Evaluator workers import W3's actual, prospectively frozen implementation.
sys.path.insert(0, str(W3 / 'src'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def readiness():
    tasks = json.loads((ASSESS / 'generation_tasks.json').read_text())['tasks']
    registration = json.loads((ASSESS / 'registration_receipt.json').read_text())
    if len(tasks) != 87 or len({t['id'] for t in tasks}) != 87:
        raise ValueError('Expected 87 distinct frozen assessment arms')
    if registration['status'] != 'ALL_ASSESSMENT_BUNDLES_FROZEN_BEFORE_FRESH_REGISTRATION':
        raise ValueError('Assessment registration is not frozen')
    for teacher in TEACHERS:
        common = json.loads((ASSESS / f'common_{teacher}.json').read_text())
        if digest(EVALUATOR) != common['inputs']['evaluator_source']['sha256']:
            raise ValueError(f'{teacher}: evaluator differs from frozen source')
        receipt = json.loads((ASSESS / f'collected_{teacher}/receipt.json').read_text())
        if (receipt['status'] != 'PASS' or receipt['missing_tasks'] or
                receipt['failed_structures'] or
                receipt['model_counts'].get(teacher) != registration['unique_orbits'] or
                not (ASSESS / f'collected_{teacher}' / f'{teacher}.npz').is_file()):
            raise ValueError(f'{teacher}: collection incomplete or unexpected count')
        registry = ASSESS / f'oracle_{teacher}/registry.json'
        if receipt['registry_sha256'] != digest(registry):
            raise ValueError(f'{teacher}: collected registry hash mismatch')
    return tasks


def evaluate_one(item):
    task, teacher = item
    out = ASSESS / task['id']
    dest = out / f'evaluation_{teacher}'
    schema = OLD / 'artifacts/overnight_20261107/data' / f'condition_schema_{teacher}.json'
    score = ASSESS / f'collected_{teacher}' / f'{teacher}.npz'
    args = ['--generated', str(out / 'generated_eligible'), '--parent', str(PARENT),
            '--history', str(BOUNDARY / 'known_history.json'), '--training-data', task['data'],
            '--score', f'{teacher}={score}', '--schema', f'{teacher}={schema}',
            '--fixed-thresholds', str(THRESHOLD), '--fixed-threshold-source', 'main_train',
            '--output-dir', str(dest)]
    try:
        spec = importlib.util.spec_from_file_location('r2_frozen_evaluator', EVALUATOR)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with (out / f'evaluation_{teacher}.log').open('x') as log, \
                contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            mod.main(args)
        status = json.loads((dest / 'receipt.json').read_text())['status']
        return task['id'], teacher, status, None
    except BaseException as exc:
        return task['id'], teacher, 'FAILED', f'{type(exc).__name__}: {exc}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--check', action='store_true', help='Check inputs without writing outputs')
    opt = parser.parse_args()
    if opt.workers < 1:
        parser.error('--workers must be positive')
    tasks = readiness()
    if opt.check:
        print(json.dumps({'status': 'READY', 'arms': len(tasks), 'teachers': list(TEACHERS)}))
        return

    # Refuse to overwrite any prior attempt, including partial logs.
    for t in tasks:
        for teacher in TEACHERS:
            path = ASSESS / t['id']
            if (path / f'evaluation_{teacher}').exists() or (path / f'evaluation_{teacher}.log').exists():
                raise FileExistsError(f'Existing evaluation for {t["id"]}/{teacher}')
    for teacher in TEACHERS:
        for path in (ASSESS / f'comparison_{teacher}_cached_r2',
                     ASSESS / f'comparison_{teacher}_cached_r2.log'):
            if path.exists():
                raise FileExistsError(f'Existing aggregate output: {path}')
    jobs = [(task, teacher) for task in tasks for teacher in TEACHERS]
    results = []
    with ProcessPoolExecutor(max_workers=opt.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(evaluate_one, item) for item in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({'finished': len(results), 'total': len(jobs), 'task': result[0],
                              'teacher': result[1], 'status': result[2], 'error': result[3]}), flush=True)
    if any(result[2] != 'PASS' for result in results):
        raise RuntimeError('Some evaluations failed; preserve all logs and results')

    # W4 comparator records its post-freeze source status and verifies all raw
    # inputs, source bindings, assessment coverage, scores, and paired streams.
    env = os.environ.copy()
    env['PYTHONPATH'] = str(W4 / 'src')
    for teacher in TEACHERS:
        out = ASSESS / f'comparison_{teacher}_cached_r2'
        cmd = [sys.executable, '-m', 'cathode_cvae.iterative_comparison', 'aggregate',
               '--common', str(ASSESS / f'common_{teacher}.json'), '--boundary-dir', str(BOUNDARY),
               '--oracle-root', str(ASSESS / f'oracle_{teacher}'), '--collection-receipt',
               str(ASSESS / f'collected_{teacher}/receipt.json'), '--output-dir', str(out)]
        for task in tasks:
            root = ASSESS / task['id']
            cmd.extend(['--bundle', f'{task["policy"]}:{task["seed"]}='
                        f'{root / "generated_eligible"}={root / f"evaluation_{teacher}"}'])
        with (ASSESS / f'comparison_{teacher}_cached_r2.log').open('x') as log:
            print('COMMAND', repr(cmd), file=log, flush=True)
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, env=env)
        receipt = json.loads((out / 'receipt.json').read_text())
        if receipt['status'] != 'PASS':
            raise RuntimeError(f'{teacher} comparison failed verification')
    print(json.dumps({'status': 'BOTH_TEACHER_COMPARISONS_COMPLETE',
                      'evaluations': len(results), 'driver_sha256': digest(Path(__file__))}), flush=True)


if __name__ == '__main__':
    main()

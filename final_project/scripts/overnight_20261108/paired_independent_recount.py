"""Independent count/bin arithmetic from frozen endpoint facts and raw attempts.

This intentionally does not import the production evaluator or classifier.
Canonical-mask geometry was checked separately before scoring. Here energy
classes are reconstructed with the documented inclusive-lower-bin rule; every
raw attempt stays in the denominator. No teacher is called or model selected.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    r = a.run
    path = r/'final/compact_facts/facts.json'
    expected = '55571ac5ba4c609a780c6d3ae56f7fc638e76a18b84062fd8dd721aa2039359e'
    if sha(path) != expected:
        raise ValueError('Frozen facts identity differs')
    facts = json.loads(path.read_text())
    by_key = {v['canonical_vacancy_key']: v['record'] for v in facts['records'].values()}
    history = set(json.loads((r/'paired/final_history/history.json').read_text()))
    schemas = {t:json.loads((r/f'paired/full_collection/condition_schema_{t}.json').read_text()) for t in ['deletion','endpoint']}
    output, inputs, checked = [], {str(path):expected}, 0
    for stage in ['current_low','paired_deletion','paired_endpoint','shared_uniform']:
        rawpath = r/f'final/paired_generated/generated/{stage}/attempts_all.csv'
        raw = rows(rawpath); inputs[str(rawpath)] = sha(rawpath)
        groups = defaultdict(list)
        for row in raw:
            groups[row['method'],row['replicate_id'],int(row['n_target']),row['requested_bin']].append(row)
        recomputed = {}
        for key, attempts in groups.items():
            method, rep, n, request = key
            valid, novel = set(), set()
            useful = {t:set() for t in schemas}
            for row in attempts:
                mask = row['canonical_mask']
                if row['status'] != 'generated':
                    continue
                assert len(mask) == 48 and mask.count('1') == 48-n
                valid.add(mask)
                if mask not in history: novel.add(mask)
                record = by_key[mask]
                for target, schema in schemas.items():
                    if target == 'endpoint' and not record['endpoint_labels_usable']:
                        continue
                    total = record['initial_energy_eV' if target == 'deletion' else 'endpoint_energy_eV']
                    if total is None: continue
                    energy = total/record['n_atoms']
                    lo, hi = schema['compositions'][str(n)]['edges_eV_atom']
                    label = 'low' if energy <= lo else 'middle' if energy <= hi else 'high'
                    if mask not in history and label == request: useful[target].add(mask)
            recomputed[key] = dict(attempts=len(attempts), distinct_initial_orbits=len(valid),novel_initial_orbits=len(novel))
            for target in schemas:
                recomputed[key][target+'_useful_novel_orbits'] = len(useful[target])
                recomputed[key][target+'_useful_yield'] = len(useful[target])/len(attempts)
        cells_path = r/f'final/paired_evaluation/{stage}/cells.csv'
        inputs[str(cells_path)] = sha(cells_path)
        for cell in rows(cells_path):
            key = cell['method'],cell['replicate'],int(cell['n_target']),cell['requested_bin']
            for field, value in recomputed[key].items():
                actual = float(cell[field]) if field.endswith('yield') else int(cell[field])
                if isinstance(value, int): assert value == actual
                else: assert math.isclose(value,actual,rel_tol=0,abs_tol=1e-15)
                checked += 1
        fit_groups = defaultdict(list)
        for (method, rep, n, request), cell in recomputed.items():
            for target in schemas: fit_groups[method,target,rep].append(cell[target+'_useful_yield'])
        means = defaultdict(list)
        for (method,target,rep), values in fit_groups.items():
            means[method,target].append(statistics.mean(values))
        summary_path = r/f'final/paired_evaluation/{stage}/summary.csv'
        inputs[str(summary_path)] = sha(summary_path)
        for row in rows(summary_path):
            v = means[row['method'],row['target']]
            assert math.isclose(statistics.mean(v),float(row['mean_replicate_macro_yield']),rel_tol=0,abs_tol=1e-15)
            assert math.isclose(statistics.stdev(v),float(row['std_replicate_macro_yield']),rel_tol=0,abs_tol=1e-15)
            checked += 2
        output.append(dict(stage=stage,raw_attempts=len(raw),cells=len(groups),status='PASS'))
    a.output_dir.mkdir(parents=True,exist_ok=False)
    receipt = dict(status='PASS_INDEPENDENT_BIN_AND_DENOMINATOR_RECOUNT',stages=output,checked_fields=checked,
        facts_sha256=expected,inputs=inputs,new_MLIP_calls=0,new_proposals=0,new_model_fits=0,
        arithmetic_allowance='1e-15 dimensionless yield only; counts and mask identities exact',
        limit='Independent arithmetic and class reconstruction from authenticated scalar facts; not a second physical inference or replacement for raw geometry audit')
    (a.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()

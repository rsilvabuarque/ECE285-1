"""Freeze observed N48 occupation history without importing prior energy labels.

This is an allocated metadata preparation step. Canonical strings encode vacancy
1; the new learning/cohort arrays will explicitly encode present-ion occupancy 1.
The source union includes the latest frozen pre-assessment history and every
subsequently registered geometry, under the verified same parent identity.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prior-root', type=Path, required=True)
    a = p.parse_args()
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    root, prior = a.root.absolute(), a.prior_root.absolute()
    output = root/'runs/overnight_20261108/data'
    output.mkdir(parents=True, exist_ok=True)
    history_path = output/'intake_history.json'
    receipt_path = output/'intake_history_receipt.json'
    if history_path.exists() or receipt_path.exists():
        raise FileExistsError('Preserve the existing intake freeze')
    parent = root/'artifacts/current/core/data/parents/O3_N48_pilot.npz'
    parent_sha = sha(parent)
    expected = 'ae8f8bc8d7e767165b0172b396e3c723b15e09f4b79429ac01b7c78fd4741e95'
    if parent_sha != expected:
        raise ValueError('Primary parent changed from the verified starting state')
    run = prior/'runs/conditioning_20261107'
    names = ['data/history_before_development.json',
             'data/history_before_final.json',
             'editor/assessment_v1/history.json',
             'aggregate_prior/assessment/history.json']
    union, records = set(), []
    for name in names:
        path = run/name
        values = json.loads(path.read_text())
        if not isinstance(values, list) or any(not re.fullmatch('[01]{48}', s) for s in values):
            raise ValueError(f'Unexpected canonical-history encoding: {path}')
        before = len(union)
        union.update(values)
        records.append(dict(path=str(path), sha256=sha(path), rows=len(values),
                            added_orbits=len(union)-before))
    path = run/'oracle/registry.json'
    registry = json.loads(path.read_text())
    if registry['parent_sha256'] != parent_sha:
        raise ValueError('Registered geometries use a different parent')
    before = len(union)
    for item in registry['physical_keys'].values():
        match = re.fullmatch('mask48_([0-9a-f]{12})', item['structure_id'])
        if match is None:
            raise ValueError('Registry contains a different structure-ID convention')
        union.add(format(int(match.group(1), 16), '048b'))
    records.append(dict(path=str(path), sha256=sha(path),
        rows=len(registry['physical_keys']), added_orbits=len(union)-before,
        scope='Geometry identities only; no outcome energies are imported'))
    keys_path = run/'data/training_keys.json'
    train_keys = set(json.loads(keys_path.read_text()))
    if not train_keys <= union or len(train_keys) != 40157:
        raise ValueError('Protected H0 TRAIN identities are absent or changed')
    history_path.write_text(json.dumps(sorted(union))+'\n')
    receipt = dict(status='FROZEN_STARTING_HISTORY',
        frozen_UTC=datetime.now(timezone.utc).isoformat(), parent_sha256=parent_sha,
        history=dict(path=str(history_path), sha256=sha(history_path), orbits=len(union)),
        inputs=records, training_keys=dict(path=str(keys_path), sha256=sha(keys_path)),
        convention='48-character canonical VACANCY strings:1=vacant; not learning occupancy',
        role_policy='All old protected roles unchanged. New paired cohort rejects intake-known orbits before labels, records every raw draw, and assigns new roles before scoring.',
        historical_final_policy='Old finals motivate hypotheses only; no old FINAL labels enter new training or model selection.',
        new_labels_read=0, new_model_calls=0)
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'], orbits=len(union),
                         history=str(history_path)), indent=2))


if __name__ == '__main__':
    main()

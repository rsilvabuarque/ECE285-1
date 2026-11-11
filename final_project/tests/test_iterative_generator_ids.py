"""Generated scorer-ID schema is normalized without relaxing identity checks."""
import csv
import json
import pytest
from cathode_cvae.data import sha256
from test_iterative_acquisition import fixture_data, generation, run_stage, oracle, run_merge


def change_rows(generated, change):
    path=generated/'attempts.csv'
    with path.open(newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:change(row)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    receipt=generated/'receipt.json';data=json.loads(receipt.read_text())
    data['outputs']['attempts.csv']=sha256(path);receipt.write_text(json.dumps(data))


def test_scorer_id_only_stages_and_broadcasts_merge_labels(fixture_data):
    d=fixture_data;g=generation(d,d['base'])
    change_rows(g,lambda row:row.update(scoring_structure_id=row.pop('structure_id')))
    staged=d['root']/'stage';r=run_stage(d,g,staged)
    assert r['selected_unique_orbits']==2
    with (staged/'attempts.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    assert all(r['structure_id']==r['scoring_structure_id'] for r in rows)
    oracle_root,collection,_=oracle(d,staged);out=d['root']/'merged'
    run_merge(d,staged,oracle_root,collection,out)
    with (out/'attempts_scored.csv').open(newline='') as f:merged=list(csv.DictReader(f))
    assert sum(row['oracle_status']=='success' for row in merged)==3


def test_conflicting_identity_fields_rejected(fixture_data):
    d=fixture_data;g=generation(d,d['base'])
    def conflict(row):
        row['scoring_structure_id']=row['structure_id']
        if row['status']=='generated':row['scoring_structure_id']='mask48_ffffffffffff'
    change_rows(g,conflict)
    with pytest.raises(ValueError,match='structure_id and scoring_structure_id disagree'):
        run_stage(d,g,d['root']/'stage')

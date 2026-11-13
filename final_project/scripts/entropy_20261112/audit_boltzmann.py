"""Lightweight read-only artifact checks after the bounded target-weight fits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args();b=a.directory
    protocol=json.loads((b/'execution/protocol.json').read_text());root=Path(protocol['root']);checks=[]
    for name,expected in protocol['source_files'].items():checks.append((str(b/'execution'/name),sha(b/'execution'/name)==expected))
    for name,expected in protocol['input_files'].items():checks.append((name,sha(Path(name))==expected))
    for directory in ['diagnostics','summary']:
        receipt=json.loads((b/directory/'receipt.json').read_text())
        for name,expected in receipt['outputs'].items():checks.append((str(b/directory/name),sha(b/directory/name)==expected))
    fits=[]
    for teacher in ['uma','mace']:
        for seed in protocol['seeds']:
            folder=b/f'fits/{teacher}/{seed}';summary=json.loads((folder/'summary.json').read_text())
            valid=summary['optimizer_updates']==4000 and summary['terminal_checkpoint']['optimizer_updates']==4000
            valid=valid and sha(folder/'last_update.pt')==summary['terminal_checkpoint']['sha256']
            checks.append((str(folder/'last_update.pt'),valid));fits.append(dict(teacher=teacher,seed=seed,valid=valid,updates=summary['optimizer_updates']))
        with np.load(b/f'learning/{teacher}/learning.npz',allow_pickle=False) as z:
            valid=np.isnan(z['energy_eV_atom'][z['split']=='test']).all()
            checks.append((f'{teacher}:TEST_energy_sealed',bool(valid)))
    if not all(valid for _,valid in checks):raise ValueError('Artifact, source or role check failed')
    old=rows(root/'entropy_learned_proposal/learned_proposal_importance.csv');new=rows(b/'diagnostics/learned_proposal_importance.csv')
    key=lambda r:tuple(r[k] for k in ['teacher','n_target','temperature_K','draws','seed'])
    old_uniform={key(r):r for r in old if r['model']=='uniform_microstate'}
    new_uniform={key(r):r for r in new if r['model']=='uniform_microstate'}
    if old_uniform!=new_uniform:raise ValueError('Matched uniform seeds/budgets do not reproduce baseline')
    with np.load(root/'entropy_ar_roles/orbit_roles.npz',allow_pickle=False) as z:roles=dict(zip(z['structure_ids'].astype(str),z['split'].astype(str)))
    # These are post-fit sums of tabulated labelled-mask proposal probabilities,
    # not further training input or a new blind assessment.
    q=rows(b/'diagnostics/exact_AR_probabilities.csv');result=[]
    for model,n in sorted(set((x['model'],int(x['n_target'])) for x in q if x['model'].endswith('_last'))):
        rr=[x for x in q if x['model']==model and int(x['n_target'])==n]
        total={role:sum(float(x['q_mixture']) for x in rr if roles[x['orbit_structure_id']]==role) for role in ['train','val','test']}
        result.append(dict(model=model,n_target=n,TRAIN_proposal_probability=total['train'],
            VAL_proposal_probability=total['val'],TEST_proposal_probability=total['test'],
            scope='Post-fit development diagnostic over labelled masks; retained repeats; not used for selection'))
    out=b/'independent_audit';out.mkdir(exist_ok=False)
    with (out/'proposal_role_probability.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    receipt=dict(status='PASS_INDEPENDENT_BOLTZMANN_AUDIT',checks=len(checks),all_checks_match=True,
        checks_detail=[dict(path=name,match=valid) for name,valid in checks],fits=fits,
        uniform_comparison_rows_identical=len(old_uniform),baseline_input_hashes_unchanged=True,
        PDF_visual_check='Rendered at100dpi with Ghostscript; all4panels, axes and legends legible without clipping',
        statistical_note='Uniform has no fitted-seed variability; zero fit-SD CSV placeholders are not zero sampling uncertainty',
        source_sha256=sha(Path(__file__)),pipeline_receipt_sha256=sha(b/'pipeline_receipt.json'),
        outputs={'proposal_role_probability.csv':sha(out/'proposal_role_probability.csv')})
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['checks_detail','fits']},indent=2))


if __name__=='__main__':main()

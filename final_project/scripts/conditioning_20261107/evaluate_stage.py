"""Collect complete physical labels, evaluate, and select on development only."""
import argparse
import csv
import json
from pathlib import Path
from cathode_cvae.condition_evaluation import evaluate, plot_tables
from cathode_cvae.data import sha256
from cathode_cvae.pool_scoring import collect, atomic_json
from cathode_cvae.runtime import require_compute_allocation


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',required=True);p.add_argument('--generated',type=Path,required=True)
    p.add_argument('--history',type=Path);p.add_argument('--plots',action='store_true')
    a=p.parse_args();require_compute_allocation();root=a.root.resolve()
    run=root/'runs/conditioning_20261107';out=run/f'evaluation/{a.stage}'
    scores=run/f'collected/{a.stage}';r=collect(run/'oracle',scores)
    if r['status']!='PASS':raise ValueError('Do not select models using incomplete scored pools')
    history=a.history or run/'data/history_before_development.json'
    evaluate(a.generated/'attempts.csv',a.generated/'candidates.npz',{'uma':scores/'uma.npz'},
        {'uma':root/'artifacts/overnight_20261107/data/condition_schema_uma.json'},
        run/'data/training_keys.json',history,root/'artifacts/current/reports/generation_thresholds.csv',
        out,root/'artifacts/current/core/data/parents/O3_N48_pilot.npz')
    if a.plots:plot_tables(out)
    summary=read(out/'summary.csv');low={r['method']:r for r in read(out/'fixed_q20_low_request.csv')}
    ranked=[]
    for row in summary:
        method=row['method']
        if method in ['uniform','mutation']:continue
        ranked.append(dict(method=method,balanced_yield=float(row['mean_novel_inbin_yield']),
            low_q20_yield=float(low[method]['mean_novel_fixed_q20_yield'])))
    ranked.sort(key=lambda r:(-r['balanced_yield'],-r['low_q20_yield'],r['method']))
    print(json.dumps(dict(stage=a.stage,ranked=ranked),indent=2),flush=True)
    if a.stage=='policy':
        winner=ranked[0];name=winner['method']
        selected=dict(status='SELECTED_ON_DEVELOPMENT_ONLY',method=name,
            class_balance=bool(int(name.split('_')[0][-1])),
            selection_beta=.1 if 'sel0p1' in name else 1.,latent_dim=8,
            training_updates=4000,training_beta=.1,ranked_candidates=ranked,
            source_summary_sha256=sha256(out/'summary.csv'),
            decision_rule_sha256=sha256(root/'artifacts/conditioning_20261107/selection_rule.json'),
            data_scope='Protected round0 train; development proposals, no protected test labels',
            checkpoint_selection='Alternative validation beta checkpoints from the same six trajectories')
        atomic_json(root/'artifacts/conditioning_20261107/training_policy_selection.json',selected)


if __name__=='__main__':main()

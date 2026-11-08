"""Apply the previously frozen practical-margin rule to development results."""
import argparse
import csv
import json
from pathlib import Path
from cathode_cvae.data import sha256


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();art=root/'artifacts/conditioning_20261107';run=root/'runs/conditioning_20261107'
    policy=json.loads((art/'training_policy_selection.json').read_text())
    summary=read(run/'evaluation/latent/summary.csv')
    low={r['method']:r for r in read(run/'evaluation/latent/fixed_q20_low_request.csv')}
    candidates=[dict(latent_dim=int(r['method'][6:]),method=r['method'],
        balanced_yield=float(r['mean_novel_inbin_yield']),
        low_q20_yield=float(low[r['method']]['mean_novel_fixed_q20_yield']))
        for r in summary if r['method'].startswith('latent')]
    best_balanced=max(r['balanced_yield'] for r in candidates)
    best_low=max(r['low_q20_yield'] for r in candidates)
    eligible=[r for r in candidates if r['balanced_yield']>=best_balanced-.01 and r['low_q20_yield']>=best_low-.02]
    if not eligible:raise ValueError('No dimension satisfies both predeclared margins; inspect tradeoff rather than silently relax the rule')
    chosen=min(eligible,key=lambda r:r['latent_dim'])
    selected=dict(policy,method=f'selected_dense_d{chosen["latent_dim"]}',
        training_policy_method=policy['method'],latent_dim=chosen['latent_dim'],latent_candidates=candidates,
        latent_selection=chosen,latent_rule_sha256=sha256(art/'selection_rule.json'),
        latent_summary_sha256=sha256(run/'evaluation/latent/summary.csv'),
        scope='Training policy and dimension selected using development only; no final assessment generated yet')
    with (art/'selected_policy.json').open('x') as f:json.dump(selected,f,indent=2);f.write('\n')
    models=[]
    for seed in [20261106,20261107,20261108]:
        dim=chosen['latent_dim'];folder=f'balance_1_d8_{seed}' if dim==8 else f'latent_d{dim}_{seed}'
        ck=run/f'fits/{folder}/best_validation_beta_0p1.pt'
        models.append(dict(method='selected_dense',fit_seed=seed,checkpoint=str(ck),checkpoint_sha256=sha256(ck)))
    (run/'selected_models_run.json').write_text(json.dumps(dict(models=models),indent=2)+'\n')
    public=[dict(m,checkpoint=f'fits/{Path(m["checkpoint"]).parent.name}/{Path(m["checkpoint"]).name}') for m in models]
    (art/'selected_models.json').write_text(json.dumps(dict(root='artifacts/conditioning_20261107',models=public),indent=2)+'\n')
    print(json.dumps(dict(selected=chosen,candidates=candidates),indent=2))


if __name__=='__main__':main()

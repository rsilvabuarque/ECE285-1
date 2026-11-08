"""Prepare explicit fitting trajectories; never consumes assessment labels."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['policy','latent','symmetry','graph','graph_context'],required=True)
    p.add_argument('--selection',type=Path)
    a=p.parse_args();root=a.root.resolve();run=root/'runs/conditioning_20261107'
    policy=json.loads(a.selection.read_text()) if a.selection else {}
    tasks=[]
    if a.stage=='policy':
        variants=[(f'balance_{int(b)}_d8',dict(balance=b,dim=8)) for b in [False,True]]
    elif a.stage=='latent':
        variants=[(f'latent_d{d}',dict(dim=d)) for d in [4,16,32]]+[
            ('no_latent',dict(dim=0,kind='ar_baseline'))]
    elif a.stage=='symmetry':variants=[('symmetry_augment',dict(orientation='augment'))]
    elif a.stage=='graph':variants=[('graph_encoder',dict(encoder='graph'))]
    else:variants=[('graph_context',dict(encoder='graph',context=True))]
    for variant,options in variants:
        seeds=[20261106] if a.stage=='graph_context' else [20261106,20261107,20261108]
        for seed in seeds:
            dim=options.get('dim',policy.get('latent_dim',8))
            argv=['--data',str(root/'artifacts/current/improvement_v1/training_round0.npz'),
                '--parent',str(root/'artifacts/current/core/data/parents/O3_N48_pilot.npz'),
                '--condition-schema-in',str(root/'artifacts/overnight_20261107/data/condition_schema_uma.json'),
                '--condition-mode','multibin','--kind',options.get('kind','ar_cvae'),
                '--latent-dim',str(dim if dim else 8),'--encoder',options.get('encoder','dense'),
                '--orientation',options.get('orientation','canonical'),
                '--counts','12','24','36','--teacher','uma','--teacher-head','omat',
                '--geometry-protocol','pristine-derived deletion; unrelaxed; exact canonical parent coordinates',
                '--seed',str(seed),'--epochs','100','--minimum-epochs','100','--patience','100',
                '--max-updates','4000','--batch-size','512','--learning-rate','.001',
                '--beta-final','.1','--beta-warmup','20','--validation-beta','1',
                '--selection-betas','1','.1','--max-seconds','1200','--device','cuda','--threads','4',
                '--output-dir',str(run/f'fits/{variant}_{seed}')]
            if options.get('balance',policy.get('class_balance',False)):argv.append('--class-balance')
            if options.get('context'):argv.append('--graph-context')
            tasks.append(dict(id=f'{variant}_{seed}',variant=variant,fit_seed=seed,argv=argv))
    dest=run/f'{a.stage}_fit_tasks.json'
    with dest.open('x') as f:json.dump(dict(stage=a.stage,tasks=tasks,selection=policy),f,indent=2)
    print(json.dumps(dict(path=str(dest),full_trajectories=len(tasks)),indent=2))


if __name__=='__main__':main()

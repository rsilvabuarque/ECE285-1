"""Run the source-frozen N12 exact entropy and learned-proposal diagnostic."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from cathode_cvae.data import sha256
from cathode_cvae.runtime import require_compute_allocation


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--execution-dir',type=Path,required=True);opt=a.parse_args()
    require_compute_allocation();execution=opt.execution_dir.resolve();protocol=json.loads((execution/'protocol.json').read_text())
    for n,h in protocol['source_files'].items():
        if sha256(execution/n)!=h:raise ValueError('Frozen source changed: '+n)
    root=Path(protocol['root']);parent=root/'entropy_input/parent.npz';catalog=root/'entropy_input/orbit_representatives.npz'
    for path,key in [(parent,'parent_sha256'),(catalog,'catalog_sha256'),(root/'entropy_ar_roles/receipt.json','role_receipt_sha256'),(root/'entropy_ar_roles/orbit_roles.npz','role_sha256')]:
        if sha256(path)!=protocol[key]:raise ValueError('Frozen input changed: '+str(path))
    collection=root/'entropy_collected';receipt=json.loads((collection/'receipt.json').read_text())
    if receipt['status']!='PASS' or receipt['missing_tasks'] or receipt['failed_structures']:
        raise ValueError('Scoring collection incomplete')
    if any(receipt['model_counts'].get(t)!=123 for t in ('uma','mace')):raise ValueError('Expected87orbits+36orientations per teacher')
    env=os.environ.copy();env['PYTHONPATH']=str(execution/'src')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[key]='1'
    logs=root/'entropy_pipeline_logs';logs.mkdir(exist_ok=False);commands=[];start=time.time()
    def run(label,args):
        cmd=[sys.executable,*map(str,args)];record=dict(label=label,argv=cmd,start_unix=time.time());commands.append(record)
        with (logs/(label+'.log')).open('x') as f:
            subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        record['end_unix']=time.time();record['status']='COMPLETE';print(json.dumps(record),flush=True)
    script=execution/'scripts/entropy_20261112'
    run('export',['-u',script/'prepare_ar.py','export','--catalog',catalog,'--roles-dir',root/'entropy_ar_roles',
        '--score',collection/'uma.npz','--parent',parent,'--output-dir',root/'entropy_ar_learning'])
    def fit(seed):
        destination=root/'entropy_ar_fits'/str(seed)
        args=['-u','-m','cathode_cvae.learning.train_mask','--data',root/'entropy_ar_learning/learning.npz',
            '--parent',parent,'--output-dir',destination,'--kind','ar_baseline','--encoder','dense',
            '--latent-dim','0','--condition-mode','composition','--orientation','augment','--hidden-dim','32',
            '--graph-context','--context-kernel-mode','radial_shells','--context-shell-tolerance-A','1e-5',
            '--graph-cutoff-A','5.0','--max-updates','4000','--epochs','5000','--minimum-epochs','5000',
            '--patience','5000','--batch-size','64','--learning-rate','.001','--validation-samples','1',
            '--validation-beta','.1','--selection-betas','.1','--beta-schedule','reference_updates',
            '--beta-reference-updates-per-epoch','79','--beta-warmup','20','--beta-final','.1',
            '--teacher','uma','--label-inference-receipt',collection/'receipt.json',
            '--geometry-protocol','idealized O3 N12 pristine-derived deletion; fixed cell; no relaxation',
            '--seed',str(seed),'--device','cpu','--threads','1','--max-seconds','1800']
        run(f'fit_{seed}',args)
        final=json.loads((destination/'summary.json').read_text())
        # Trainer receipts differ historically; the checkpoint records the
        # authoritative update count and must meet the prospective budget.
        import torch
        ck=torch.load(destination/'last_update.pt',map_location='cpu',weights_only=False)
        if ck.get('optimizer_updates')!=4000:raise ValueError('Tiny AR did not reach4000updates')
        return final
    exact_args=['-u',script/'analyze.py','--catalog',catalog,'--score',f'uma={collection/"uma.npz"}',
        '--score',f'mace={collection/"mace.npz"}','--orientation-input',root/'entropy_input/symmetry_orientation_checks.npz',
        '--orientation-score',f'uma={collection/"uma.npz"}','--orientation-score',f'mace={collection/"mace.npz"}',
        '--output-dir',root/'entropy_analysis']
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(fit,s) for s in protocol['seeds']]
        futures.append(pool.submit(run,'exact_analysis',exact_args))
        for f in as_completed(futures):f.result()
    learned=['-u',script/'learned_proposal.py','--catalog',catalog,'--score',f'uma={collection/"uma.npz"}',
        '--score',f'mace={collection/"mace.npz"}','--output-dir',root/'entropy_learned_proposal','--device','cpu','--threads','1']
    for seed in protocol['seeds']:
        for tag,file in [('last','last_update.pt'),('best','best_validation_beta_0p1.pt')]:
            learned += ['--checkpoint',f'AR{seed}_{tag}={root/"entropy_ar_fits"/str(seed)/file}']
    run('learned_proposal',learned)
    result=dict(status='COMPLETE_ENTROPY_PIPELINE',protocol_sha256=sha256(execution/'protocol.json'),
        scorer_receipt_sha256=sha256(collection/'receipt.json'),commands=commands,start_unix=start,end_unix=time.time(),
        source_snapshot=str(execution),primary_last_update_fits=3,best_checkpoint_sensitivity_fits=3,
        exact_analysis_receipt_sha256=sha256(root/'entropy_analysis/receipt.json'),
        learned_proposal_receipt_sha256=sha256(root/'entropy_learned_proposal/receipt.json'))
    (root/'entropy_pipeline_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status']},indent=2))


if __name__=='__main__':main()

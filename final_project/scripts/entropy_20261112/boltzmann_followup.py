"""Bounded, TRAIN-only fixed-temperature target-measure follow-up to N12 AR."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from cathode_cvae.data import sha256
from cathode_cvae.entropy import boltzmann_training_weights,thermodynamics
from cathode_cvae.runtime import require_compute_allocation


def write_csv(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def read_csv(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def freeze(root,output):
    if output.exists():raise FileExistsError(output)
    baseline=json.loads((root/'entropy_pipeline_receipt.json').read_text())
    if baseline['status']!='COMPLETE_ENTROPY_PIPELINE':raise ValueError('Completed baseline required')
    old=root/'entropy_learning_execution';old_protocol=json.loads((old/'protocol.json').read_text())
    with np.load(root/'entropy_input/orbit_representatives.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    with np.load(root/'entropy_ar_roles/orbit_roles.npz',allow_pickle=False) as z:
        roles=z['split'].astype(str);ids=z['structure_ids'].astype(str)
    if not np.array_equal(ids,d['structure_ids'].astype(str)):raise ValueError('Frozen role order changed')
    with np.load(root/'entropy_input/parent.npz',allow_pickle=False) as z:
        host=len(z['numbers'])-len(z['target_indices'])
    execution=output/'execution';execution.mkdir(parents=True);inputs={};groups={}
    for name,expected in old_protocol['source_files'].items():
        if sha256(old/name)!=expected:raise ValueError('Baseline source changed')
        target=execution/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/name,target)
    project=Path(__file__).parents[2]
    for name in ['src/cathode_cvae/entropy.py','scripts/entropy_20261112/boltzmann_followup.py']:
        shutil.copy2(project/name,execution/name)
    for teacher in ['uma','mace']:
        path=root/f'entropy_collected/{teacher}.npz'
        with np.load(path,allow_pickle=False) as z:
            lookup={sid:i for i,sid in enumerate(z['structure_ids'].astype(str))};source=z['energy_eV'];status=z['status'].astype(str)
        # Only TRAIN/VAL values are indexed. TEST labels remain inert NaNs.
        energy=np.full(len(ids),np.nan)
        for i in np.flatnonzero(roles!='test'):
            j=lookup[ids[i]]
            if status[j]!='success' or not np.isfinite(source[j]):raise ValueError('TRAIN/VAL scoring incomplete')
            energy[i]=source[j]
        weight,group=boltzmann_training_weights(energy,d['n_target'],roles,d['multiplicity'],1000.)
        if not np.array_equal(weight[roles!='train'],d['multiplicity'][roles!='train']):raise ValueError('Heldout weights changed')
        # Verify no hidden consultation of VAL/TEST energies in the weighting.
        changed=energy.copy();changed[roles!='train']=1e99
        repeated,_=boltzmann_training_weights(changed,d['n_target'],roles,d['multiplicity'],1000.)
        if not np.array_equal(weight,repeated):raise ValueError('Weights depend on heldout energies')
        destination=output/'learning'/teacher;destination.mkdir(parents=True)
        np.savez_compressed(destination/'learning.npz',structure_ids=ids,occupancy=d['occupancy'],
            n_target=d['n_target'],energy_eV_atom=energy/(host+d['n_target']),split=roles,weight=weight,
            parent_sha256=np.array(sha256(root/'entropy_input/parent.npz')))
        receipt=dict(status='PASS_TRAIN_ONLY_WEIGHT_VALIDATION',teacher=teacher,temperature_K=1000.,
            weight_energy_units='TOTAL eV per cell; energy_eV_atom exists only for inherited training metadata',
            formula='g_o exp[-(E_o-Emin_TRAIN,n)/(kB T)], normalized to original TRAIN sum(g) separately per count',
            heldout_energy_perturbation_changes_weights=False,VAL_TEST_weights_unchanged=True,
            TEST_energy_values_exported=False,groups=group,source_score_sha256=sha256(path),
            learning_sha256=sha256(destination/'learning.npz'))
        (destination/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');groups[teacher]=receipt
    seeds=old_protocol['seeds'];commands=[]
    for teacher in ['uma','mace']:
        for seed in seeds:
            original=next(x['argv'] for x in baseline['commands'] if x['label']==f'fit_{seed}')
            argv=original.copy()
            replacements={'--data':output/f'learning/{teacher}/learning.npz',
                '--output-dir':output/f'fits/{teacher}/{seed}','--teacher':teacher,'--max-seconds':'1000'}
            for flag,value in replacements.items():argv[argv.index(flag)+1]=str(value)
            commands.append(dict(label=f'fit_{teacher}_{seed}',teacher=teacher,seed=seed,argv=argv))
    for path in [root/'entropy_input/parent.npz',root/'entropy_input/orbit_representatives.npz',
        root/'entropy_ar_roles/orbit_roles.npz',root/'entropy_ar_roles/receipt.json',
        root/'entropy_collected/receipt.json',root/'entropy_collected/uma.npz',root/'entropy_collected/mace.npz',
        root/'entropy_pipeline_receipt.json',root/'entropy_learned_proposal/learned_proposal_importance.csv',
        output/'learning/uma/learning.npz',output/'learning/mace/learning.npz']:
        inputs[str(path)]=sha256(path)
    sources={str(path.relative_to(execution)):sha256(path) for path in execution.rglob('*.py')}
    protocol=dict(status='FROZEN_BEFORE_BOLTZMANN_WEIGHTED_FITS',root=str(root),output=str(output),
        hypothesis='TRAIN-only fixed1000K Boltzmann weighting can improve own-teacher importance ESS/error over composition diversity training; cross-teacher mismatch may remain',
        exact_change='Only within-count TRAIN row weights and teacher-specific energy provenance change; architecture, roles, augmentation, seeds, update/draw budgets fixed',
        temperature_K=1000.,seeds=seeds,hidden_dim=32,batch_size=64,max_updates=4000,
        composition_only=True,class_balance=False,VAL_weighting='Original orbit multiplicity',
        primary='Terminal4000update;1000K own-teacher, equal-budget comparison; no fresh blind TEST claim',
        sensitivity='Validation-best and other temperatures; cross-teacher evaluation explicitly separate',
        model_support='10percent uniform labelled-mask mixture; unbiased Z requires support, not low variance',
        budget='Existing61215925 8CPU step;6fits concurrently;1000seconds maximum fit wall time;≤20min step; no new allocation/scoring/DFT',
        source_files=sources,input_files=inputs,commands=commands,weight_validation=groups)
    (execution/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'status':protocol['status'],'weight_validation':'PASS_BOTH_TEACHERS','fits':len(commands),'source_files':len(sources),'protocol_sha256':sha256(execution/'protocol.json')},indent=2))


def run(execution):
    require_compute_allocation();protocol=json.loads((execution/'protocol.json').read_text());root=Path(protocol['root']);out=Path(protocol['output'])
    for relative,expected in protocol['source_files'].items():
        if sha256(execution/relative)!=expected:raise ValueError('Frozen source changed')
    for name,expected in protocol['input_files'].items():
        if sha256(Path(name))!=expected:raise ValueError('Frozen input changed')
    logs=out/'logs';logs.mkdir(exist_ok=False);start=time.time();records=[]
    env=os.environ.copy();env['PYTHONPATH']=str(execution/'src')
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[name]='1'
    def command(record):
        record=record.copy();record['start_unix']=time.time();records.append(record)
        with (logs/(record['label']+'.log')).open('x') as f:subprocess.run(record['argv'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        record['end_unix']=time.time();record['status']='COMPLETE'
        if record['label'].startswith('fit_'):
            import torch
            destination=Path(record['argv'][record['argv'].index('--output-dir')+1])
            checkpoint=torch.load(destination/'last_update.pt',map_location='cpu',weights_only=False)
            if checkpoint.get('optimizer_updates')!=4000:raise ValueError('Fit did not reach prospective4000update target')
        print(json.dumps(record),flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending=[pool.submit(command,r) for r in protocol['commands']]
        for f in as_completed(pending):f.result()
    argv=[sys.executable,'-u',str(execution/'scripts/entropy_20261112/learned_proposal.py'),
        '--catalog',str(root/'entropy_input/orbit_representatives.npz'),'--score',f'uma={root/"entropy_collected/uma.npz"}',
        '--score',f'mace={root/"entropy_collected/mace.npz"}','--output-dir',str(out/'diagnostics'),'--device','cpu','--threads','1']
    for teacher in ['uma','mace']:
        for seed in protocol['seeds']:
            for tag,name in [('last','last_update.pt'),('best','best_validation_beta_0p1.pt')]:
                argv+=['--checkpoint',f'train{teacher.upper()}_AR{seed}_{tag}={out/"fits"/teacher/str(seed)/name}']
    command(dict(label='learned_proposal',argv=argv))
    summary(root,out)
    receipt=dict(status='COMPLETE_BOLTZMANN_WEIGHTED_FOLLOWUP',protocol_sha256=sha256(execution/'protocol.json'),
        start_unix=start,end_unix=time.time(),commands=records,new_MLIP_calls=0,new_DFT_calls=0,
        baseline_changed=False,primary_fits=6,primary_optimizer_updates=4000,
        diagnostic_receipt_sha256=sha256(out/'diagnostics/receipt.json'),summary_receipt_sha256=sha256(out/'summary/receipt.json'))
    (out/'pipeline_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':receipt['status'],'elapsed_seconds':receipt['end_unix']-start},indent=2))


def summary(root,out):
    old=read_csv(root/'entropy_learned_proposal/learned_proposal_importance.csv')
    new=read_csv(out/'diagnostics/learned_proposal_importance.csv');combined=[]
    for source,rows in [('baseline',old),('weighted',new)]:
        for row in rows:
            model=row['model']
            if model=='uniform_microstate':
                if source!='baseline':continue
                method='uniform_microstate';fit='uniform';training='none'
            elif not model.endswith('_last'):continue
            elif source=='baseline':method='composition_diversity_AR';fit=model.split('_')[0][2:];training='none'
            else:training=model.split('_')[0][5:].lower();method=f'Boltzmann_T1000_train_{training}';fit=model.split('_')[1][2:]
            combined.append(dict(row,method=method,fit=fit,training_teacher=training))
    groups=sorted(set((x['method'],x['teacher'],int(x['n_target']),float(x['temperature_K']),int(x['draws'])) for x in combined));table=[]
    for method,teacher,n,T,budget in groups:
        rr=[x for x in combined if x['method']==method and x['teacher']==teacher and int(x['n_target'])==n and float(x['temperature_K'])==T and int(x['draws'])==budget]
        fitmeans=[];naive=[];ess=[]
        for fit in sorted(set(x['fit'] for x in rr)):
            ff=[x for x in rr if x['fit']==fit]
            fitmeans.append(np.mean([abs(float(x['importance_entropy_kB'])-float(x['exact_entropy_kB'])) for x in ff]))
            naive.append(np.mean([abs(float(x['naive_microstate_histogram_entropy_kB'])-float(x['exact_entropy_kB'])) for x in ff]))
            ess.append(np.mean([float(x['importance_ESS'])/budget for x in ff]))
        table.append(dict(method=method,training_teacher=rr[0]['training_teacher'],evaluation_teacher=teacher,
            n_target=n,temperature_K=T,draws=budget,independent_fits=len(fitmeans),draw_seeds_per_fit=len(rr)//len(fitmeans),
            mean_absolute_entropy_error_kB=float(np.mean(fitmeans)),sd_fit_mean_absolute_error_kB=float(np.std(fitmeans,ddof=1)) if len(fitmeans)>1 else 0.,
            mean_absolute_naive_error_kB=float(np.mean(naive)),mean_weight_ESS_fraction=float(np.mean(ess))))
    dest=out/'summary';dest.mkdir(exist_ok=False);write_csv(dest/'matched_budget_comparison.csv',table)
    with np.load(root/'entropy_input/orbit_representatives.npz',allow_pickle=False) as z:
        ids=z['structure_ids'].astype(str);counts=z['n_target'];g=z['multiplicity']
    with np.load(root/'entropy_ar_roles/orbit_roles.npz',allow_pickle=False) as z:roles=z['split'].astype(str)
    support=[]
    for teacher in ['uma','mace']:
        with np.load(root/f'entropy_collected/{teacher}.npz',allow_pickle=False) as z:lookup=dict(zip(z['structure_ids'].astype(str),z['energy_eV']))
        E=np.array([lookup[s] for s in ids])
        for n in [3,6,9]:
            active=counts==n;prob=thermodynamics(E[active],g[active],1000.)['orbit_probability']
            support.append(dict(teacher=teacher,n_target=n,temperature_K=1000.,
                TRAIN_target_probability=float(prob[roles[active]=='train'].sum()),
                VAL_target_probability=float(prob[roles[active]=='val'].sum()),
                TEST_target_probability=float(prob[roles[active]=='test'].sum()),scope='Final closed diagnostic only; not consulted by weights or fit'))
    write_csv(dest/'physical_target_role_mass.csv',support)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(11,7.5));methods=['composition_diversity_AR','Boltzmann_T1000_train_uma','Boltzmann_T1000_train_mace','uniform_microstate']
    for j,teacher in enumerate(['uma','mace']):
        for method in methods:
            rr=sorted([x for x in table if x['method']==method and x['evaluation_teacher']==teacher and x['n_target']==6 and x['temperature_K']==1000.],key=lambda x:x['draws'])
            label=method.replace('Boltzmann_T1000_train_','Boltzmann: ').replace('composition_diversity_AR','Original AR').replace('uniform_microstate','Uniform')
            axs[0,j].plot([x['draws'] for x in rr],[x['mean_absolute_entropy_error_kB'] for x in rr],marker='o',label=label)
            axs[1,j].plot([x['draws'] for x in rr],[x['mean_weight_ESS_fraction'] for x in rr],marker='o',label=label)
        axs[0,j].set_title(f'{teacher.upper()} evaluation, Li6/12 at1000K')
        axs[0,j].set_ylabel('Mean absolute entropy error / kB');axs[1,j].set_ylabel('Weight ESS / draws')
        for ax in axs[:,j]:ax.set_xscale('log');ax.set_xlabel('Independent proposal draws');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'boltzmann_target_followup.pdf');plt.close(fig)
    lines=['# Fixed1000K Boltzmann-target follow-up','',
        'Six composition-only AR fits change only within-composition TRAIN weights to g exp[−(E−EminTRAIN)/(kB×1000K)]. '
        'Each composition preserves its original TRAIN sum(g); VAL/TEST weights remain g and TEST energies remain NaN. '
        'E is total eV per periodic cell. The same three fit seeds, architecture, augmentation, 4000updates, draw budgets '
        'and draw seeds are retained. Terminal checkpoints are primary; validation-best checkpoints remain sensitivity outputs.','',
        'Both teachers evaluate every proposal distribution. TRAIN-only Boltzmann weighting can miss probability carried '
        'by held-out orbits. The10%uniform mixture restores support for unbiased shifted Z estimates, but it does not '
        'guarantee efficient importance weights. The full reference was already analyzed before this follow-up; these '
        'are development comparisons, not a fresh blind TEST.','',
        'At Li6/12,1000K and8192draws (mean errors over three draw seeds within each fit, then three independent fits):','']
    for teacher in ['uma','mace']:
        for method in methods:
            x=next(x for x in table if x['method']==method and x['evaluation_teacher']==teacher and x['n_target']==6 and x['temperature_K']==1000. and x['draws']==8192)
            lines.append(f'- {teacher}, {method}: entropy MAE={x["mean_absolute_entropy_error_kB"]:.6g}kB/cell, fit SD={x["sd_fit_mean_absolute_error_kB"]:.6g}, ESS/draws={x["mean_weight_ESS_fraction"]:.6g}.')
    lines+=['','The target-role-mass table is a post-fit diagnostic. The comparison table includes n=3/6/9 and '
        '300/1000/5000K, with1000K own-teacher evaluation primary. All entropy estimators retain finite-budget bias; '
        'no finite-lattice entropy is promoted to physical LCO entropy.','']
    (dest/'RESULTS.md').write_text('\n'.join(lines))
    receipt=dict(status='COMPLETE_MATCHED_TARGET_MEASURE_COMPARISON',baseline_receipt_sha256=sha256(root/'entropy_pipeline_receipt.json'),
        weighted_diagnostic_sha256=sha256(out/'diagnostics/receipt.json'),primary='1000K own-teacher terminal checkpoints',
        outputs={p.name:sha256(p) for p in dest.iterdir() if p.is_file()})
    (dest/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('--root',type=Path,required=True);f.add_argument('--output-dir',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--execution-dir',type=Path,required=True)
    a=p.parse_args()
    if a.command=='freeze':freeze(a.root.resolve(),a.output_dir.resolve())
    else:run(a.execution_dir.resolve())


if __name__=='__main__':main()

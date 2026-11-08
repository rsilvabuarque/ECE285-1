"""Deduplicate primary relaxation attempts and report saved API-level counters.

Trajectory-position frames include the initial pose; labelled E/F/stress
snapshots are a separate stride-based subset. Neither is an independent-mask
count. These counters do not estimate undocumented internal NN forwards.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    from cathode_cvae.runtime import require_compute_allocation
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['prepared','queue','final-tasks','final-facts','preflight','variants','variant-queue','output-dir']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(argv);require_compute_allocation()
    initial_path=a.prepared/'tasks_all.json';tight_path=a.variants/'tight_0p01/tasks.json'
    initial=json.loads(initial_path.read_text());final=json.loads(a.final_tasks.read_text());tight=json.loads(tight_path.read_text())
    facts=json.loads(a.final_facts.read_text());preflight=json.loads(a.preflight.read_text())
    if facts['status']!='FROZEN_COMPLETE_ENDPOINT_FACTS' or facts['source_tasks_sha256']!=sha(a.final_tasks) or facts['assessment_preflight_sha256']!=sha(a.preflight):
        raise ValueError('Final facts are not bound to these frozen tasks/preflight')
    lanes={'acquisition':(initial['tasks'],a.queue),'final':(final['tasks'],a.queue),'tight_0p01':(tight['tasks'],a.variant_queue)}
    expected={'acquisition':6000,'final':7616,'tight_0p01':96}
    registry={};members={};rows=[]
    for lane,(tasks,queue) in lanes.items():
        if len(tasks)!=expected[lane]:raise ValueError('Prespecified task denominator changed: '+lane)
        keys=[]
        for task in tasks:
            key=(task['context_hash'],task['physical_key']);keys.append(key)
            members.setdefault(key,[]).append(lane)
            if key in registry:
                if registry[key]['task_id']!=task['task_id']:raise ValueError('Same physical task has incompatible task IDs')
                continue
            path=queue/'results'/task['task_id']/'result.json';raw=json.loads(path.read_text())
            if raw['context_hash']!=key[0] or raw['physical_key']!=key[1]:raise ValueError('Task/result identity changed')
            claim_path=queue/'claims'/(task['task_id']+'.json');claim=json.loads(claim_path.read_text())
            if claim['task']['physical_key']!=task['physical_key']:raise ValueError('Task/claim identity changed')
            if lane=='final':
                item=facts['records'][task['physical_key']]
                if sha(path)!=item['source']['result_sha256'] or sha(claim_path)!=item['source']['claim_sha256']:
                    raise ValueError('Final raw metadata changed since facts snapshot')
            row=dict(task_id=task['task_id'],context_hash=key[0],physical_key=key[1],status=raw['status'],
                n_target=raw['n_target'],EFS_attempts=raw['calculator_EFS_attempts'],EFS_successes=raw['calculator_EFS_successes'],EFS_failures=raw['calculator_EFS_failures'],
                optimizer_steps=raw['n_steps'],saved_trajectory_position_frames=raw['trajectory_frames'],saved_labelled_EFS_snapshots=raw['snapshots'],
                initial_energy_label_known=raw.get('initial_energy_eV') is not None,converged_endpoint_labels_usable=bool(raw['endpoint_labels_usable']),
                claimed_unix=claim['claimed_unix'],result_sha256=sha(path),claim_sha256=sha(claim_path),arrays_sha256=raw['arrays_sha256'])
            if row['EFS_attempts']!=row['EFS_successes']+row['EFS_failures']:raise ValueError('Saved API counter conservation fails')
            registry[key]=row
        if len(set(keys))!=len(keys):raise ValueError('Duplicate physical task within lane '+lane)
    def totals(keys):
        rr=[registry[k] for k in set(keys)]
        return dict(unique_optimization_attempts=len(rr),statuses=dict(Counter(r['status'] for r in rr)),
            EFS_attempts=sum(r['EFS_attempts'] for r in rr),EFS_successes=sum(r['EFS_successes'] for r in rr),EFS_failures=sum(r['EFS_failures'] for r in rr),
            saved_trajectory_position_frames=sum(r['saved_trajectory_position_frames'] for r in rr),saved_labelled_EFS_snapshots=sum(r['saved_labelled_EFS_snapshots'] for r in rr),
            initial_energy_labels_known=sum(r['initial_energy_label_known'] for r in rr),converged_usable_endpoints=sum(r['converged_endpoint_labels_usable'] for r in rr))
    initial_keys={(t['context_hash'],t['physical_key']) for t in initial['tasks']};final_keys={(t['context_hash'],t['physical_key']) for t in final['tasks']}
    main_keys=initial_keys|final_keys
    actual_main={p.parent.name for p in (a.queue/'results').glob('*/result.json')}
    declared_main={registry[k]['task_id'] for k in main_keys}
    fresh=sum(facts['records'][t['physical_key']]['claimed_unix']>=preflight['frozen_unix'] for t in final['tasks'])
    for key,row in registry.items():rows.append(dict(lanes='|'.join(members[key]),**row))
    a.output_dir.mkdir(parents=True,exist_ok=False)
    with (a.output_dir/'physical_tasks.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(status='PASS_DEDUPLICATED_PRIMARY_METADATA_COUNTERS',completed_UTC=datetime.now(timezone.utc).isoformat(),
        total=totals(registry),lanes={lane:totals((t['context_hash'],t['physical_key']) for t in tasks) for lane,(tasks,_) in lanes.items()},
        primary_queue_unique_results=totals(main_keys),primary_queue_result_directories=len(actual_main),
        queue_results_outside_declared_scopes=sorted(actual_main-declared_main),declared_main_results_missing=sorted(declared_main-actual_main),
        final_unique_tasks=len(final_keys),final_new_claims_after_joint_preflight=fresh,final_preexisting_compatible_claims=len(final_keys)-fresh,
        final_overlaps_initial_tasks=len(final_keys&initial_keys),final_claim_partition_reference_unix=preflight['frozen_unix'],
        source_bindings={str(path):sha(path) for path in [initial_path,a.final_tasks,a.final_facts,a.preflight,tight_path]},
        physical_task_table_sha256=sha(a.output_dir/'physical_tasks.csv'),source_sha256=sha(__file__),new_model_calls=0,
        definitions={'EFS_attempts':'Saved logical calculator energy/force/stress bundle API attempts, not undocumented internal backbone forwards',
            'trajectory_position_frames':'Saved accepted-trajectory coordinate frames including the initial pose, counted per header; not all are EFS-labelled or independent masks',
            'labelled_EFS_snapshots':'Saved stride-based E/F/stress snapshots counted per result header; not an independent-mask or TRAIN-row count',
            'primary_scope':'6000 acquisition + final unique tasks +96 tighter UMA controls, deduplicated by exact(context_hash,physical_key); MACE24 independent relaxations are a separate secondary lane'},
        scope='Metadata-only accounting; raw arrays were not reopened here. Numerical convergence is not DFT/ground-state validation. Allocation costs come from top-level Slurm receipts, not these API counters.')
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['status','total','primary_queue_unique_results','final_new_claims_after_joint_preflight','final_preexisting_compatible_claims']},indent=2))


if __name__=='__main__':main()

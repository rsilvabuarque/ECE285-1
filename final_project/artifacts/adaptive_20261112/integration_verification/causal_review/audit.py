"""Independent structured-record review; never opens training/score NPZ or PT."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def load(path):return json.loads(path.read_text())
def rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--causal-root',type=Path,required=True);opt=a.parse_args()
    b=opt.causal_root;out=Path(__file__).parent;checks=[];hashes={};skipped={}
    def check(name,value):
        checks.append(dict(check=name,passed=bool(value)))
    def bind(value):
        if isinstance(value,dict):
            if 'path'in value and 'sha256'in value:
                p=Path(value['path'])
                if p.suffix in ['.json','.csv','.py']:
                    actual=sha(p);hashes[str(p)]=actual;check('hash:'+str(p),actual==value['sha256'])
                else:skipped[str(p)]=value['sha256']
            for child in value.values():bind(child)
        elif isinstance(value,list):
            for child in value:bind(child)
    plan=load(b/'plan.json');protocol=load(b/'protocol.json');prep=load(b/'preparation_receipt.json');complete=load(b/'completion.json')
    models=load(b/'models.json');tasks=load(b/'taskfiles/fits.json');generation=load(b/'taskfiles/generation.json')
    for obj in [plan,protocol,prep,complete,tasks,generation,load(b/'H0_support_receipt.json')]:bind(obj)
    check('completed',complete['status']=='POSTHOC_CAUSAL_FOLLOWUP_COMPLETE')
    check('twelve_new_fits',len(tasks['tasks'])==protocol['new_fits']==prep['new_fits']==12)
    check('five_cells_per_family_seed',len(models)==30 and set(x['cell'] for x in models)=={'h0_fixed','h0_range_h0','h0_range_r1','aug_fixed','aug_range'})
    check('extra_H0_R1_edges_branch',prep['extra_H0_using_R1edges'] and protocol['extra_H0_using_R1_edges'])
    families=plan['families'];seeds=plan['fit_seeds'];configs={};modelmap={}
    initial={(x['family'],x['fit_seed']):x['initial'] for x in protocol['inputs']}
    for model in models:
        key=(model['family'],model['fit_seed'],model['cell']);modelmap[key]=model
        folder=Path(model['checkpoint']).parent;cfg=load(folder/'configuration.json');summary=load(folder/'summary.json');configs[key]=cfg
        for p in [folder/'configuration.json',folder/'summary.json']:
            hashes[str(p)]=sha(p)
        check('fit_2000:'+model['id'],summary['status']=='FIT_COMPLETE' and summary['optimizer_updates']==summary['terminal_checkpoint']['optimizer_updates']==cfg['maximum_updates']==2000)
        check('same_initial:'+model['id'],cfg['warm_start']==initial[(model['family'],model['fit_seed'])])
        check('seed_teacher_balance:'+model['id'],cfg['seed']==model['fit_seed'] and cfg['teacher']=='uma' and cfg['teacher_head']=='omat' and cfg['class_balance'])
        check('TEST_not_evaluated:'+model['id'],summary['heldout_test_evaluated'] is False)
        check('trainer_source:'+model['id'],cfg['trainer_sha256']==sha(Path(tasks['tasks'][0]['env']['PYTHONPATH'])/'cathode_cvae/learning/train_mask.py'))
        stage=Path(model['stage']);stage_protocol=load(stage/'protocol.json');stage_receipt=load(stage/'receipt.json')
        bind(stage_receipt)
        check('fresh_shared_stage:'+model['id'],stage_protocol['namespace']==plan['namespace'] and stage_protocol['seed']==model['draw_seed'] and stage_protocol['boundary']==protocol['boundary'] and stage_protocol['counts']==[12,24,36] and stage_protocol['attempts_per_cell']==stage_protocol['queries_per_cell']==128 and stage_receipt['attempts']==1152 and stage_receipt['no_energy_filtering'])
    for family in families:
        for i,seed in enumerate(seeds):
            group={cell:configs[(family,seed,cell)] for cell in ['h0_fixed','h0_range_h0','h0_range_r1','aug_fixed','aug_range']}
            check(f'draw_pairing:{family}:{seed}',{m['draw_seed'] for m in models if m['family']==family and m['fit_seed']==seed}=={plan['draw_seeds'][i]})
            check(f'H0_rows_identical:{family}:{seed}',{group[c]['data_sha256'] for c in ['h0_fixed','h0_range_h0','h0_range_r1']}=={plan['H0']['sha256']})
            check(f'augmented_rows_identical:{family}:{seed}',group['aug_fixed']['data_sha256']==group['aug_range']['data_sha256'])
            check(f'VAL_TEST_counts_preserved:{family}:{seed}',len({(x['val_rows'],x['test_rows']) for x in group.values()})==1)
            for n in [12,24,36]:
                edges=lambda cell:group[cell]['condition_schema']['compositions'][str(n)]['edges_eV_atom']
                check(f'fixed_edges:{family}:{seed}:{n}',edges('aug_fixed')==edges('h0_fixed'))
                check(f'matched_range_edges:{family}:{seed}:{n}',edges('aug_range')==edges('h0_range_r1'))
            # Merge receipt asserts value-identical original rows, not just roles.
            merge=Path(group['aug_fixed']['data']).parent/'receipt.json'
            md=load(merge);bind(md);check(f'merge_role_contract:{family}:{seed}',md['original_rows_value_identical'] and md['base_data']['sha256']==plan['H0']['sha256'])
    for collection in [tasks,generation]:
        name='fits' if collection is tasks else 'generation'
        for task in collection['tasks']:
            p=b/f'taskfiles/{name}.claims'/task['id']/'receipt.json';receipt=load(p)
            hashes[str(p)]=sha(p);check('worker:'+name+':'+task['id'],receipt['status']=='COMPLETE' and receipt['task']==task)
    edge_rows=rows(b/'edge_comparison.csv');support=rows(b/'H0_support_under_R1_edges.csv')
    changed=sum(x['exact_equal']=='False' for x in edge_rows)
    check('four_changed_edge_cells',len(edge_rows)==18 and changed==4)
    check('H0_support_nonempty',len(support)==18 and all(min(int(x[k]) for k in ['low_rows','middle_rows','high_rows'])>0 for x in support))
    for teacher in ['uma','mace']:
        verifier=load(b/f'evaluation/verified_{teacher}.json');collector=load(b/f'evaluation/collected_{teacher}/receipt.json')
        # Do not recurse into oracle_evidence: strict complete collectors and
        # their verifier stage contracts are sufficient for this bounded review.
        for stage in verifier['stages']:bind(stage)
        check('oracle_status:'+teacher,verifier['status']=='VERIFIED_FRESH_ADAPTIVE_ORACLE' and collector['status']=='PASS' and not collector['missing_tasks'] and not collector['failed_structures'] and collector['model_counts'][teacher]==verifier['unique_structures']==17343)
        check('oracle_same_stages:'+teacher,{x['path'] for x in verifier['stages']}=={str(Path(m['stage'])/'receipt.json') for m in models})
        for p in [b/f'evaluation/verified_{teacher}.json',b/f'evaluation/collected_{teacher}/receipt.json']:hashes[str(p)]=sha(p)
    aggregates=rows(b/'aggregate_metrics.csv');cells=rows(b/'all_cell_metrics.csv');given=rows(b/'paired_summary.csv');paired=rows(b/'paired_results.csv')
    lookup={(x['family'],int(x['fit_seed']),x['teacher'],x['cell']):x for x in aggregates}
    check('metric_dimensions',len(cells)==540 and len(aggregates)==60 and len(paired)==144 and len(given)==60)
    max_aggregate=0.
    for aggregate in aggregates:
        cc=[x for x in cells if x['model_id']==aggregate['model_id'] and x['teacher']==aggregate['teacher']]
        check('nine_cells:'+aggregate['model_id']+':'+aggregate['teacher'],len(cc)==9 and all(int(x['attempts'])==128 for x in cc))
        value=statistics.mean(float(x['fixed_inbin_yield']) for x in cc)
        max_aggregate=max(max_aggregate,abs(value-float(aggregate['balanced_fixed_yield'])))
    contrasts={'H0_relabeling':('h0_range_h0','h0_fixed'),'augmented_relabeling':('aug_range','aug_fixed'),
        'added_rows_fixed':('aug_fixed','h0_fixed'),'added_rows_identical_range_edges':('aug_range','h0_range_r1')}
    metrics={'balanced_fixed_yield':'balanced_fixed_yield','H0_fixed_low_yield':'low_tail_yield','H0_fixed_high_yield':'high_tail_yield'}
    independent=[];max_summary=0.;max_pair=0.
    for row in given:
        family,teacher,contrast,metric=row['family'],row['teacher'],row['contrast'],row['metric'];field=metrics[metric];values=[]
        for seed in seeds:
            get=lambda cell:float(lookup[(family,seed,teacher,cell)][field])
            if contrast=='rows_by_schema_interaction':value=100*((get('aug_range')-get('h0_range_r1'))-(get('aug_fixed')-get('h0_fixed')))
            else:
                policy,reference=contrasts[contrast];value=100*(get(policy)-get(reference))
                pr=next(x for x in paired if x['family']==family and x['teacher']==teacher and x['contrast']==contrast and x['metric']==metric and int(x['fit_seed'])==seed)
                max_pair=max(max_pair,abs(value-float(pr['gain_pp'])))
            values.append(value)
        mean,sd=statistics.mean(values),statistics.stdev(values)
        max_summary=max(max_summary,abs(mean-float(row['gain_pp_mean'])),abs(sd-float(row['gain_pp_sample_sd'])))
        independent.append(dict(family=family,teacher=teacher,contrast=contrast,metric=metric,fit_count=3,gain_pp_mean=mean,gain_pp_sample_sd=sd))
    check('CSV_arithmetic_identity',max(max_pair,max_summary,max_aggregate)<1e-12)
    for name in ['plan.json','protocol.json','completion.json','preparation_receipt.json','models.json','paired_results.csv','paired_summary.csv','aggregate_metrics.csv','all_cell_metrics.csv','edge_comparison.csv','H0_support_under_R1_edges.csv','prepare.py','support.py','run.py','analyze.py']:
        hashes[str(b/name)]=sha(b/name)
    with (out/'independent_paired_summary.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(independent[0]));w.writeheader();w.writerows(independent)
    warnings=[
        'Post-hoc mechanistic study prompted by previous range-policy gains; not blind confirmation.',
        'Relabeling recomputes class-balanced TRAIN weights, so it changes both conditions and effective row weights.',
        'Matched-edge added-row effects also include changed minibatches/class weights; they do not isolate information content alone.',
        'H0 with R1 edges uses acquired labels through the edge summary; only H0-own-range needs no acquired labels.',
        'Three paired fit/draw seeds give descriptive sample SD, not adjusted significance or proof of equivalence.',
        'Balanced gains are mainly high-bin success, not a demonstrated low-energy-search improvement.',
        'No NPZ or PT file was opened or rehashed in this audit. Dataset preservation, checkpoint warmstarts, and fresh scoring rely on structured writer/worker/verifier contracts.',
        'Schema source decompresses the complete NPZ energy member but indexes TRAIN only; training encodes TRAIN/VAL while TEST conditions remain inert.']
    passed=all(c['passed'] for c in checks)
    report=dict(status='PASS_STRUCTURED_CAUSAL_REVIEW_WITH_LIMITS' if passed else 'FAIL_STRUCTURED_CAUSAL_REVIEW',
        causal_root=str(b),source_sha256=sha(Path(__file__)),checks=len(checks),failed_checks=[x for x in checks if not x['passed']],
        check_details=checks,source_and_receipt_sha256=hashes,binary_bindings_not_opened=skipped,
        new_fits=12,reused_fits=18,models=30,assessment_attempts=34560,oracle_unique_per_teacher=17343,
        edge_cells_changed=changed,max_aggregate_difference=max_aggregate,max_paired_difference_pp=max_pair,max_summary_difference_pp=max_summary,
        identity_tolerance_pp=1e-12,tolerance_scope='Floating-point CSV arithmetic identity only, not physical or statistical tolerance',
        warnings=warnings,no_training_or_scoring_launched=True,no_TEST_energy_values_read=True,
        output_sha256={'independent_paired_summary.csv':sha(out/'independent_paired_summary.csv')})
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Structured causal-range review','',report['status'],'',
        'All paired signs use policy minus reference, in percentage points. The interaction is '
        '(augmented range − H0 with the same R1 range edges) − (augmented fixed − H0 fixed).','',
        '| Family | Teacher | Relabel H0 | Add rows, fixed | Add rows, identical range edges | Interaction |',
        '|---|---|---:|---:|---:|---:|']
    for family in families:
        for teacher in ['uma','mace']:
            values=[]
            for contrast in ['H0_relabeling','added_rows_fixed','added_rows_identical_range_edges','rows_by_schema_interaction']:
                x=next(x for x in independent if x['family']==family and x['teacher']==teacher and x['contrast']==contrast and x['metric']=='balanced_fixed_yield')
                values.append(f'{x["gain_pp_mean"]:+.5f} ± {x["gain_pp_sample_sd"]:.5f}')
            lines.append('| '+' | '.join([family,teacher,*values])+' |')
    lines+=['','± denotes sample SD across three paired fit/draw replicates.','']
    lines+=['- '+w for w in warnings]
    lines+=['',f'CSV arithmetic checks: max pair difference {max_pair:.3g} pp; max summary difference {max_summary:.3g} pp. '
        'The audit inspected structured sources, fit configurations, summaries, task/worker receipts, stage protocols and strict oracle verifiers. '
        'It did not repeat collectors, load checkpoints or inspect TEST energies.','']
    (out/'AUDIT.md').write_text('\n'.join(lines))
    print(json.dumps({k:report[k] for k in ['status','checks','failed_checks','new_fits','reused_fits','models','max_paired_difference_pp','max_summary_difference_pp']},indent=2))


if __name__=='__main__':main()

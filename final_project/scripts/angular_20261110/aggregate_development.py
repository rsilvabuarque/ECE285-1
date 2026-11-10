"""Aggregate receipt-verified DEVELOPMENT shards without filling missing pairs.

Input is the existing per-shard evaluate_controls output. Replicates, not
proposals, are the descriptive uncertainty unit. Sensitivity conditional on
both energies being known is reported together with its coverage, because a
protected-orbit exclusion can make the observed pairs nonrepresentative.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from development_boundary import binding, digest, read_csv, save_csv, save_json, verify_boundary, verify_manifest


def truth(x):
    if x in (True, 'True', 'true', '1'): return True
    if x in (False, 'False', 'false', '0'): return False
    raise ValueError(f'Unexpected boolean {x!r}')


def number(x, field):
    if x == '' or x is None:
        raise ValueError(f'Missing required {field}')
    v = float(x)
    if not np.isfinite(v): raise ValueError(f'Nonfinite {field}')
    return v


def summary(values):
    a = np.asarray(values, dtype=float)
    if not len(a) or not np.isfinite(a).all(): raise ValueError('No finite replicate values')
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else ''


def validate_shards(shards, collector_dir):
    """Check all consumed bytes and enforce an identical preregistered design."""
    collector_dir = Path(collector_dir).absolute()
    collect = json.loads((collector_dir/'receipt.json').read_text())
    if collect.get('status') != 'PASS': raise ValueError('Collector must have complete PASS coverage')
    common = None; seen_reps = set(); parsed = []; input_bindings = []
    for manifest_path, generated_dir, evaluation_dir in shards:
        manifest_path, generated_dir, evaluation_dir = map(lambda p: Path(p).absolute(),
                                                            (manifest_path, generated_dir, evaluation_dir))
        m = verify_manifest(manifest_path)
        verify_boundary(Path(m['boundary_receipt']['path']).parent)
        gen = json.loads((generated_dir/'receipt.json').read_text())
        proto = json.loads((generated_dir/'protocol.json').read_text())
        ev = json.loads((evaluation_dir/'receipt.json').read_text())
        if gen.get('status') != 'FROZEN_BEFORE_SCORING' or gen['protocol_sha256'] != digest(generated_dir/'protocol.json'):
            raise ValueError('Generated receipt/protocol invalid')
        if proto['manifest']['sha256'] != digest(manifest_path) or gen['manifest']['sha256'] != digest(manifest_path):
            raise ValueError('Generated pool does not bind its frozen manifest')
        if gen['boundary_receipt']['sha256'] != m['boundary_receipt']['sha256']:
            raise ValueError('Generated pool uses a different frozen boundary')
        binding(gen['original_generation_receipt']['path'], gen['original_generation_receipt']['sha256'])
        for name, expected in gen['outputs'].items(): binding(generated_dir/name, expected)
        if ev.get('status') != 'PASS' or any(ev.get('terminal_failed_candidate_labels', {}).values()):
            raise ValueError('Evaluation lacks complete successful coverage')
        if ev['proposal_protocol_sha256'] != digest(generated_dir/'protocol.json'):
            raise ValueError('Evaluation did not use this generated protocol')
        for name, expected in ev['outputs'].items(): binding(evaluation_dir/name, expected)
        source_hash = next((b['sha256'] for b in m['source_bindings'] if Path(b['path']).name=='evaluate_controls.py'),None)
        if ev['source_sha256'] != source_hash: raise ValueError('Evaluator differs from frozen source')
        if set(ev['score_sources']) != set(m['teacher_schemas']):
            raise ValueError('Evaluation teachers do not match the declared schemas')
        for teacher, source in ev['score_sources'].items():
            if source['schema_sha256'] != m['teacher_schemas'][teacher]['sha256']:
                raise ValueError('Measured teacher schema differs from frozen declaration')
            binding(collector_dir/f'{teacher}.npz', source['sha256'])
        fixed = ev.get('fixed_thresholds') or {}
        if fixed.get('sha256') != m['fixed_thresholds']['sha256'] or fixed.get('source') != 'main_train':
            raise ValueError('Evaluation threshold or reference source differs from frozen design')
        attempts = read_csv(generated_dir/'attempts_all.csv')
        attempt_lookup = {r['attempt_id']:r for r in attempts}
        if len(attempt_lookup) != len(attempts): raise ValueError('Repeated attempt identity')
        outcomes = read_csv(evaluation_dir/'attempt_outcomes.csv')
        seen = set()
        for r in outcomes:
            key=(r['teacher'],r['attempt_id'])
            if key in seen or r['teacher'] not in m['teacher_schemas'] or r['attempt_id'] not in attempt_lookup:
                raise ValueError('Repeated or unexpected scored attempt')
            seen.add(key); original=attempt_lookup[r['attempt_id']]
            for name in ('status','canonical_mask','scoring_structure_id','method','replicate_id','n_target','requested_bin','diagnostic','raw_row'):
                if r[name] != original[name]: raise ValueError(f'Evaluation changed filtered attempt {name}')
        if len(seen) != len(attempts)*len(m['teacher_schemas']): raise ValueError('Evaluation omitted attempts')
        reps = {str(e['replicate_id']) for e in m['models']}
        if len(reps) != 1: raise ValueError('Each input shard must contain exactly one replicate')
        rep = reps.pop()
        if rep in seen_reps: raise ValueError('A replicate is present in multiple shards')
        seen_reps.add(rep)
        methods = sorted(e['method'] for e in m['models'])
        if len(methods) != len(set(methods)): raise ValueError('Duplicate method in a shard')
        design = {'namespace':m['namespace'], 'counts':m['counts'], 'requested_bins':m['requested_bins'],
            'budget':m['count_per_group'], 'diagnostics':m['diagnostics'], 'width':m['latent_stream_width'],
            'methods':methods, 'contrasts':m['contrasts'], 'boundary':m['boundary_receipt']['sha256'],
            'latent_dims':{e['method']:e.get('declared_latent_dim',0) for e in m['models']},
            'inputs':{k:v['sha256'] for k,v in m['inputs'].items()},
            'teachers':{k:v['sha256'] for k,v in m['teacher_schemas'].items()},
            'threshold':m['fixed_thresholds']['sha256'],
            'sources':sorted((Path(b['path']).name, b['sha256']) for b in m['source_bindings'])}
        if common is None: common = design
        elif common != design: raise ValueError('Shards have differing frozen comparison designs')
        for c in m['contrasts']:
            if c['method'] not in methods or c['reference'] not in methods or c['method']==c['reference']:
                raise ValueError('Invalid declared control contrast')
        inputs = {'manifest':binding(manifest_path), 'generation_receipt':binding(generated_dir/'receipt.json'),
                  'evaluation_receipt':binding(evaluation_dir/'receipt.json'), 'replicate_id':rep}
        input_bindings.append(inputs)
        parsed.append((rep, m, generated_dir, evaluation_dir))
    if len(parsed) < 2: raise ValueError('At least two independent replicate shards required')
    if not common['contrasts']: raise ValueError('No control contrasts were declared before generation')
    return common, parsed, input_bindings, collect


def macro_yields(metrics, design, rep):
    """Require each declared teacher/method/count/bin cell and fixed denominator."""
    expected = {(t, method, rep, str(n), b) for t in design['teachers'] for method in design['methods']
                for n in design['counts'] for b in design['requested_bins']}
    actual = {}; by_arm = defaultdict(list)
    for r in metrics:
        key = (r['teacher'], r['method'], r['replicate_id'], str(int(r['n_Li'])), r['requested_bin'])
        if key in actual: raise ValueError(f'Duplicate metric cell {key}')
        actual[key] = r
        if int(r['attempts']) != design['budget']: raise ValueError('Proposal attempt denominator changed')
        distinct = int(r['distinct_novel_inbin_orbits']); scored = int(r['scored_attempts'])
        if not 0 <= distinct <= scored <= design['budget']: raise ValueError('Invalid coverage/yield counts')
        y = number(r['novel_inbin_yield_per_attempt'], 'yield')
        if not np.isclose(y, distinct/design['budget'], atol=1e-12, rtol=0):
            raise ValueError('Reported yield differs from recorded numerator/denominator')
        by_arm[(key[0], key[1])].append(r)
    if set(actual) != expected:
        raise ValueError(f'Metric cells incomplete or unexpected: missing={len(expected-set(actual))}, extra={len(set(actual)-expected)}')
    out=[]
    for (teacher,method), cells in sorted(by_arm.items()):
        fit_seeds={r.get('fit_seed','') for r in cells}
        if len(fit_seeds)!=1: raise ValueError('Fit seed changes across cells of the same replicate')
        macro = float(np.mean([float(r['novel_inbin_yield_per_attempt']) for r in cells]))
        qcells = [r for r in cells if r['requested_bin']=='low']
        qavailable = bool(qcells) and all(truth(r['fixed_q20_available']) for r in qcells)
        q = float(np.mean([number(r['novel_fixed_q20_yield_per_attempt'],'q20 yield') for r in qcells])) if qavailable else ''
        out.append(dict(teacher=teacher,method=method,replicate_id=rep,fit_seed=fit_seeds.pop(),macro_novel_inbin_yield=macro,
            low_request_fixed_q20_yield=q,fixed_q20_available=qavailable,
            attempted=sum(int(r['attempts']) for r in cells), scored=sum(int(r['scored_attempts']) for r in cells),
            score_coverage=sum(int(r['scored_attempts']) for r in cells)/sum(int(r['attempts']) for r in cells)))
    return out, actual


def contrasts_and_summary(per_seed, design):
    lookup={(r['teacher'],r['method'],r['replicate_id']):r for r in per_seed}
    if len(lookup)!=len(per_seed): raise ValueError('Repeated method/seed yield')
    reps=sorted({r['replicate_id'] for r in per_seed}); paired=[]
    for t in design['teachers']:
        for pair in design['contrasts']:
            a,b=pair['method'],pair['reference']
            for rep in reps:
                x=lookup[t,a,rep]['macro_novel_inbin_yield']; y=lookup[t,b,rep]['macro_novel_inbin_yield']
                paired.append(dict(teacher=t,method=a,reference=b,replicate_id=rep,
                    method_fit_seed=lookup[t,a,rep]['fit_seed'],reference_fit_seed=lookup[t,b,rep]['fit_seed'],
                    method_yield=x,reference_yield=y,difference_pp=100*(x-y)))
    groups=defaultdict(list)
    for r in paired: groups[r['teacher'],r['method'],r['reference']].append(r['difference_pp'])
    aggregate=[]
    for (teacher,method,reference), values in sorted(groups.items()):
        mean,sd=summary(values)
        aggregate.append(dict(teacher=teacher,method=method,reference=reference,n_replicates=len(values),
                              mean_difference_pp=mean,descriptive_SD_pp=sd))
    return paired,aggregate


def sensitivity(rows, rep, design):
    groups=defaultdict(list)
    for r in rows:
        if r['replicate_id'] != rep: raise ValueError('Paired diagnostic has wrong shard replicate')
        if r['contrast'] not in ('condition_middle_minus_low','condition_high_minus_low','alternate_latent_minus_low'):
            raise ValueError('Unknown paired response')
        if not truth(r['AR_uniforms_equal']): raise ValueError('Paired AR uniform inputs differ')
        same_z=truth(r['actual_z_equal'])
        if r['contrast']=='alternate_latent_minus_low' and same_z:
            raise ValueError('Alternate-latent probe used identical latent vectors')
        if r['contrast']!='alternate_latent_minus_low' and not same_z:
            raise ValueError('Condition probe did not keep the latent vector fixed')
        groups[r['teacher'],r['method'],rep,int(r['n_Li']),r['contrast']].append(r)
    out=[]
    contrasts=[]
    if 'low' in design['requested_bins']:
        if 'middle' in design['requested_bins']: contrasts.append('condition_middle_minus_low')
        if 'high' in design['requested_bins']: contrasts.append('condition_high_minus_low')
    expected = {(t, method, rep, n, contrast) for t in design['teachers'] for method in design['methods']
                for n in design['counts'] for contrast in contrasts}
    if design['diagnostics'] and 'low' in design['requested_bins']:
        expected |= {(t, method, rep, n, 'alternate_latent_minus_low') for t in design['teachers']
                     for method in design['methods'] if design['latent_dims'][method] > 0 for n in design['counts']}
    if set(groups) != expected:
        raise ValueError(f'Diagnostic cells incomplete or unexpected: missing={len(expected-set(groups))}, extra={len(set(groups)-expected)}')
    for (teacher,method,rep,n,contrast),rr in sorted(groups.items()):
        if len({r['paired_id'] for r in rr})!=len(rr): raise ValueError('Repeated paired attempt')
        if len(rr)!=design['budget']: raise ValueError('Paired diagnostic attempt denominator changed')
        known=[r for r in rr if truth(r['both_energies_known'])]
        d=np.array([number(r['energy_shift_meV_atom'],'paired energy shift') for r in known],float)
        out.append(dict(teacher=teacher,method=method,replicate_id=rep,n_Li=n,contrast=contrast,
            attempted_pairs=len(rr),known_pairs=len(known),missing_pairs=len(rr)-len(known),
            pair_coverage=len(known)/len(rr),
            mean_signed_meV_atom=float(d.mean()) if len(d) else '',
            mean_absolute_meV_atom=float(np.abs(d).mean()) if len(d) else '',
            rms_meV_atom=float(np.sqrt(np.square(d).mean())) if len(d) else '',
            mean_square_meV2_atom2=float(np.square(d).mean()) if len(d) else ''))
    return out


def sensitivity_summary(rows, design, replicates):
    """Equal-weight compositions within each seed; available pairs only."""
    groups=defaultdict(list)
    for r in rows: groups[r['teacher'],r['method'],r['replicate_id'],r['contrast']].append(r)
    fits=[]
    for (teacher,method,rep,contrast),rr in sorted(groups.items()):
        if {r['n_Li'] for r in rr} != set(design['counts']): raise ValueError('Missing diagnostic composition')
        if any(r['attempted_pairs'] != design['budget'] for r in rr): raise ValueError('Diagnostic paired denominator changed')
        any_empty = any(r['known_pairs']==0 for r in rr)
        fits.append(dict(teacher=teacher,method=method,replicate_id=rep,contrast=contrast,
            attempted_pairs=sum(r['attempted_pairs'] for r in rr),known_pairs=sum(r['known_pairs'] for r in rr),
            pair_coverage=sum(r['known_pairs'] for r in rr)/sum(r['attempted_pairs'] for r in rr),
            all_pairs_complete=all(r['known_pairs']==r['attempted_pairs'] for r in rr),
            complete_composition_support=not any_empty,
            macro_signed_meV_atom=float(np.mean([r['mean_signed_meV_atom'] for r in rr])) if not any_empty else '',
            macro_absolute_meV_atom=float(np.mean([r['mean_absolute_meV_atom'] for r in rr])) if not any_empty else '',
            macro_rms_meV_atom=float(np.sqrt(np.mean([r['mean_square_meV2_atom2'] for r in rr]))) if not any_empty else ''))
    aggregate=[]; grouped=defaultdict(list)
    for r in fits: grouped[r['teacher'],r['method'],r['contrast']].append(r)
    for (teacher,method,contrast),rr in sorted(grouped.items()):
        if {r['replicate_id'] for r in rr} != set(replicates): raise ValueError('Diagnostic missing for an entire replicate')
        available=[r for r in rr if r['complete_composition_support']]
        entry=dict(teacher=teacher,method=method,contrast=contrast,n_replicates=len(rr),
            supported_replicates=len(available),mean_pair_coverage=float(np.mean([r['pair_coverage'] for r in rr])),
            all_pairs_complete=all(r['all_pairs_complete'] for r in rr))
        for field in ('macro_signed_meV_atom','macro_absolute_meV_atom','macro_rms_meV_atom'):
            if available:
                mean,sd=summary([r[field] for r in available]);entry['mean_'+field]=mean;entry['SD_'+field]=sd
            else: entry['mean_'+field]=entry['SD_'+field]=''
        aggregate.append(entry)
    return fits,aggregate


def cell_contrasts(metrics, design, replicates):
    rows=[]
    for teacher in design['teachers']:
        for pair in design['contrasts']:
            for rep in replicates:
                for n in design['counts']:
                    for bin_name in design['requested_bins']:
                        a=metrics[teacher,pair['method'],rep,str(n),bin_name]
                        b=metrics[teacher,pair['reference'],rep,str(n),bin_name]
                        rows.append(dict(teacher=teacher,method=pair['method'],reference=pair['reference'],replicate_id=rep,
                            n_Li=n,requested_bin=bin_name,attempts_each=int(a['attempts']),
                            difference_pp=100*(float(a['novel_inbin_yield_per_attempt'])-float(b['novel_inbin_yield_per_attempt']))))
    return rows


def validate_registry(registry_path, collector_dir, collector_receipt, parsed, teachers):
    registry_path=Path(registry_path).absolute(); collector_dir=Path(collector_dir)
    if digest(registry_path) != collector_receipt['registry_sha256']:
        raise ValueError('Collector receipt does not bind the supplied registry')
    registry=json.loads(registry_path.read_text()); expected=set()
    if {v['model'] for v in registry['physical_keys'].values()} != set(teachers):
        raise ValueError('Registry teacher set differs from the frozen design')
    for _,m,generated,_ in parsed:
        if registry['parent_sha256'] != m['inputs']['parent']['sha256']: raise ValueError('Registry parent mismatch')
        with np.load(generated/'candidates.npz', allow_pickle=False) as z:
            expected.update(z['structure_ids'].astype(str).tolist())
    counts=[]
    for teacher in teachers:
        items=[v for v in registry['physical_keys'].values() if v['model']==teacher]
        ids={v['structure_id'] for v in items}
        if len(ids)!=len(items) or ids != expected: raise ValueError('Registry is not exactly the eligible shard union')
        with np.load(collector_dir/f'{teacher}.npz',allow_pickle=False) as z:
            score_ids=z['structure_ids'].astype(str).tolist()
            status=z['status'].astype(str); provenance=z['provenance'].astype(str)
        if len(score_ids)!=len(set(score_ids)) or set(score_ids)!=ids: raise ValueError('Collector identities differ from registry')
        if not all(s=='success' or (s=='cached' and p=='cached') for s,p in zip(status,provenance)):
            raise ValueError('Collector contains an uncertified score status')
        counts.append(dict(teacher=teacher,unique_physical_queries=len(items),
            new_queries=sum(i.get('provenance')=='new_scoring' for i in items),
            cached_queries=sum(i.get('provenance')=='exact_geometry_cache' for i in items),
            registered_tasks=sum(t['model']==teacher for t in registry['tasks'])))
        if counts[-1]['new_queries']+counts[-1]['cached_queries']!=len(items):
            raise ValueError('Unknown registry provenance')
    return counts


def plot_results(out, per_seed, paired_summary, sensitivity_aggregate, design):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=design['methods'];teachers=list(design['teachers'])
    fig,axes=plt.subplots(1,len(teachers),figsize=(6*len(teachers),max(4,0.27*len(methods)+1.5)),squeeze=False,sharey=True)
    for ax,teacher in zip(axes[0],teachers):
        for i,method in enumerate(methods):
            values=[100*r['macro_novel_inbin_yield'] for r in per_seed if r['teacher']==teacher and r['method']==method]
            mean,sd=summary(values)
            ax.errorbar(mean,i,xerr=sd if sd!='' else None,fmt='o',capsize=2,color='tab:blue')
        ax.set(title=teacher.upper(),xlabel='Novel, in-bin yield per attempt (%)',yticks=range(len(methods)),yticklabels=methods)
        ax.grid(axis='x',alpha=.2)
    axes[0,0].invert_yaxis();fig.tight_layout();fig.savefig(out/'macro_yield.pdf');plt.close(fig)
    labels=[(p['method'],p['reference']) for p in design['contrasts']]
    fig,axes=plt.subplots(1,len(teachers),figsize=(6*len(teachers),max(4,0.27*len(labels)+1.5)),squeeze=False,sharey=True)
    lookup={(r['teacher'],r['method'],r['reference']):r for r in paired_summary}
    for ax,teacher in zip(axes[0],teachers):
        for i,(method,reference) in enumerate(labels):
            r=lookup[teacher,method,reference];sd=r['descriptive_SD_pp']
            ax.errorbar(r['mean_difference_pp'],i,xerr=sd if sd!='' else None,fmt='o',capsize=2,color='tab:purple')
        ax.axvline(0,color='black',lw=.8);ax.grid(axis='x',alpha=.2)
        ax.set(title=teacher.upper(),xlabel='Paired yield difference (percentage points)',yticks=range(len(labels)),
               yticklabels=[f'{a} − {b}' for a,b in labels])
    axes[0,0].invert_yaxis();fig.tight_layout();fig.savefig(out/'paired_control_differences.pdf');plt.close(fig)
    # Available-pair responses: they are explicitly conditional on coverage.
    shown=[('condition_high_minus_low','mean_macro_signed_meV_atom','SD_macro_signed_meV_atom','High − low signed'),
           ('alternate_latent_minus_low','mean_macro_absolute_meV_atom','SD_macro_absolute_meV_atom','Latent resample absolute')]
    for contrast,avg,sdkey,title in shown:
        available=[r for r in sensitivity_aggregate if r['contrast']==contrast and r[avg]!='']
        if not available: continue
        selected=[m for m in methods if any(r['method']==m for r in available)]
        fig,axes=plt.subplots(1,len(teachers),figsize=(6*len(teachers),max(4,0.27*len(selected)+1.5)),squeeze=False,sharey=True)
        for ax,teacher in zip(axes[0],teachers):
            lookup={(r['teacher'],r['method']):r for r in available}
            for i,method in enumerate(selected):
                r=lookup.get((teacher,method))
                if r is not None:
                    sd=r[sdkey];ax.errorbar(r[avg],i,xerr=sd if sd!='' else None,fmt='o',capsize=2,color='tab:green')
            ax.axvline(0,color='black',lw=.8);ax.grid(axis='x',alpha=.2)
            ax.set(title=f'{teacher.upper()}: {title}',xlabel='Conditional energy response (meV/atom)',yticks=range(len(selected)),yticklabels=selected)
        axes[0,0].invert_yaxis();fig.suptitle('Available pairs only; see sensitivity tables for coverage',fontsize=9)
        fig.tight_layout();fig.savefig(out/f'sensitivity_{contrast}.pdf');plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard',nargs=3,action='append',required=True,metavar=('MANIFEST','FILTERED_GENERATION','EVALUATION'))
    p.add_argument('--collector-dir',type=Path,required=True)
    p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv)
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    design,parsed,input_bindings,collector=validate_shards(a.shard,a.collector_dir)
    query_cost=validate_registry(a.registry,a.collector_dir,collector,parsed,design['teachers'])
    per_seed=[];all_cells={};pair_rows=[];replicates=[]
    for rep,m,generated,evaluation in parsed:
        replicates.append(rep)
        seed,cells=macro_yields(read_csv(evaluation/'metrics.csv'),design,rep)
        original_seed={(r['teacher'],r['method'],r['replicate_id']):r for r in read_csv(evaluation/'seed_metrics.csv')}
        if len(original_seed)!=len(seed): raise ValueError('Existing seed-metric summary has unexpected coverage')
        for row in seed:
            original=original_seed[(row['teacher'],row['method'],rep)]
            if not np.isclose(row['macro_novel_inbin_yield'],number(original['macro_composition_bin_yield'],'original macro'),atol=1e-12,rtol=0):
                raise ValueError('Recomputed macro yield differs from the existing evaluator')
            if row['fixed_q20_available'] and not np.isclose(row['low_request_fixed_q20_yield'],
                    number(original['fixed_q20_LOW_request_yield'],'original low yield'),atol=1e-12,rtol=0):
                raise ValueError('Recomputed LOW q20 yield differs from the existing evaluator')
        per_seed.extend(seed);all_cells.update(cells)
        pair_rows.extend(sensitivity(read_csv(evaluation/'paired_attempts.csv'),rep,design))
    paired,paired_summary=contrasts_and_summary(per_seed,design)
    cells=cell_contrasts(all_cells,design,replicates)
    fits,sensitivity_aggregate=sensitivity_summary(pair_rows,design,replicates)
    yields=[];groups=defaultdict(list)
    for r in per_seed:groups[r['teacher'],r['method']].append(r)
    for (teacher,method),rr in sorted(groups.items()):
        mean,sd=summary([100*r['macro_novel_inbin_yield'] for r in rr])
        yields.append(dict(teacher=teacher,method=method,n_replicates=len(rr),mean_yield_percent=mean,
                           descriptive_SD_percentage_points=sd,mean_score_coverage=float(np.mean([r['score_coverage'] for r in rr]))))
    out=a.output_dir.absolute()
    if out.exists():raise FileExistsError(f'Preserve existing analysis: {out}')
    out.mkdir(parents=True)
    for name,rows in [('per_seed_yield.csv',per_seed),('yield_summary.csv',yields),('paired_control_differences.csv',paired),
                      ('paired_control_summary.csv',paired_summary),('paired_cell_differences.csv',cells),
                      ('sensitivity_by_composition.csv',pair_rows),('sensitivity_by_seed.csv',fits),
                      ('sensitivity_summary.csv',sensitivity_aggregate),('global_query_cost.csv',query_cost)]:
        if rows:save_csv(out/name,rows)
        else:(out/name).write_text('')
    plot_results(out,per_seed,paired_summary,sensitivity_aggregate,design)
    receipt={'status':'PASS_DEVELOPMENT_AGGREGATION','source_sha256':digest(__file__),
        'shards':input_bindings,'collector_receipt':binding(a.collector_dir/'receipt.json'),'registry':binding(a.registry),
        'replicates':replicates,'design':design,
        'estimands':{'yield':'Equal weight per requested bin and composition; distinct novel in-bin orbits per all attempts',
          'paired_difference':'Within replicate, method macro yield minus predeclared control; percentage points',
          'uncertainty':'Descriptive sample SD across independent fits/sampler replicates, not proposal-level standard error',
          'sensitivity':'Conditional on both teacher energies available; equal composition weight; report attempted/known pairs and potential selection bias',
          'cost':'Distinct globally registered physical teacher/orbit keys; not scheduler GPU node-hours'},
        'missing_values':'No imputation; empty responses indicate no available scores or fewer than two replicates for SD',
        'new_MLIP_calls':0,'new_training_updates':0,'outputs':{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()}}
    save_json(out/'receipt.json',receipt)
    print(json.dumps({'status':receipt['status'],'replicates':replicates,'queries':query_cost,'output_dir':str(out)},indent=2))


if __name__=='__main__':main()

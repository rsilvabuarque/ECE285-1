"""Freeze and aggregate paired multi-lineage DEVELOPMENT assessments.

A comparison protocol is frozen first. Each lineage's subsequent manifest plan
must bind that protocol before drawing. Score coverage and worker provenance are
checked from fresh task-level evidence, not inferred from collector output.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from cathode_cvae.data import sha256
from cathode_cvae.conditions import load_schema
from cathode_cvae.iterative_acquisition import _check_worker, _canonical_vacancy
from cathode_cvae.iterative_boundary import binding, verify_boundary, write
from cathode_cvae.iterative_assessment import verify_manifest
from cathode_cvae.pool_scoring import PROTOCOL,physical_key
from cathode_cvae.runtime import require_compute_allocation


def read_rows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))


def write_rows(p,rows,fields):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def freeze(plan_path,boundary_dir,out):
    p=json.loads(Path(plan_path).read_text());boundary,_,_=verify_boundary(boundary_dir)
    if p.get('status')!='DECLARED_MULTI_LINEAGE_DEVELOPMENT':raise ValueError('Expected declared development comparison')
    teacher=p.get('teacher')
    if teacher not in ('uma','mace'):raise ValueError('Unsupported teacher')
    if not p.get('lineages') or not p.get('contrasts'):raise ValueError('Declare all lineages and contrasts')
    lineages=[];seen=set()
    for e0 in p['lineages']:
        e=dict(e0);ident=(e.get('policy'),str(e.get('replicate_id','')))
        if not all(ident) or ident in seen or e.get('rng_kind') not in ('ar','fixed_mutation'):
            raise ValueError('Invalid policy/replicate/random-stream declaration')
        seen.add(ident)
        e['data']=binding(e['data'])
        e['checkpoint']=None if not e.get('checkpoint') else binding(e['checkpoint'])
        e['merge_receipt']=None if not e.get('merge_receipt') else binding(e['merge_receipt'])
        if e['rng_kind']=='ar' and e['checkpoint'] is None:raise ValueError('AR lineage needs a checkpoint')
        if e['rng_kind']=='fixed_mutation' and e['checkpoint'] is not None:raise ValueError('Fixed mutation is not a checkpoint model')
        lineages.append(e)
    for c in p['contrasts']:
        a={r for method,r in seen if method==c['policy']};b={r for method,r in seen if method==c['reference']}
        if not a or a!=b:raise ValueError('Every contrast needs matched replicate IDs')
    for field in ('counts','requested_bins'):
        if not p.get(field) or len(set(p[field]))!=len(p[field]):raise ValueError('Missing or duplicate assessment cells')
    if type(p.get('count_per_group')) is not int or p['count_per_group']<1:raise ValueError('Invalid attempt count')
    if not p.get('namespace'):raise ValueError('A shared comparison namespace is required')
    paths={name:binding(p[name]) for name in ('condition_schema','teacher_schema','thresholds','generator_source','evaluator_source')}
    condition=load_schema(paths['condition_schema']['path']); teacher_schema=load_schema(paths['teacher_schema']['path'])
    if condition.get('provenance',{}).get('parent_sha256')!=boundary['parent']['sha256'] or teacher_schema.get('provenance',{}).get('parent_sha256')!=boundary['parent']['sha256']:
        raise ValueError('Condition or teacher schema names a different parent')
    if teacher_schema.get('provenance',{}).get('teacher')!=teacher:
        raise ValueError('Teacher schema names a different teacher')
    threshold_rows=[x for x in read_rows(paths['thresholds']['path']) if x['teacher']==teacher and x.get('source','main_train')=='main_train']
    if len(threshold_rows)!=len(p['counts']) or {int(x['n_Li']) for x in threshold_rows}!=set(p['counts']) or any(not np.isfinite(float(x['q20_eV_atom'])) for x in threshold_rows):
        raise ValueError('Fixed thresholds must cover every count exactly once')
    if any(e['rng_kind']=='fixed_mutation' for e in lineages):
        paths['mutation_source']=binding(p['mutation_source'])
    if 'request_schema' in boundary and paths['condition_schema']['sha256']!=boundary['request_schema']['sha256']:
        raise ValueError('Comparison condition schema differs from boundary')
    result={'status':'FROZEN_MULTI_LINEAGE_BEFORE_GENERATION','created_unix':time.time(),
       'plan':binding(plan_path),'boundary_receipt':binding(Path(boundary_dir)/'receipt.json'),
       'parent':boundary['parent'],'history':boundary['history'],'protected':boundary['protected'],
       'teacher':teacher,'inputs':paths,'lineages':lineages,'contrasts':p['contrasts'],
       'counts':p['counts'],'requested_bins':p['requested_bins'],'count_per_group':p['count_per_group'],
       'namespace':p['namespace'],'source':binding(__file__)}
    write(out,result);return result


def _check_common(common,boundary_dir):
    c=json.loads(Path(common).read_text())
    if c.get('status')!='FROZEN_MULTI_LINEAGE_BEFORE_GENERATION':raise ValueError('Common protocol not frozen')
    boundary,_,_=verify_boundary(boundary_dir)
    for b in [c['plan'],c['source'],c['boundary_receipt'],c['parent'],c['history'],c['protected'],*c['inputs'].values()]:
        binding(b['path'],b['sha256'])
    if sha256(Path(boundary_dir)/'receipt.json')!=c['boundary_receipt']['sha256']:
        raise ValueError('Common boundary differs')
    for l in c['lineages']:
        for name in ('data','checkpoint','merge_receipt'):
            if l[name]:binding(l[name]['path'],l[name]['sha256'])
    return c


def collect_oracle(root,collection_path,expected,parent,teacher,after):
    root=Path(root);registry_path=root/'registry.json';registry=json.loads(registry_path.read_text())
    collection=json.loads(Path(collection_path).read_text())
    if collection.get('status')!='PASS' or collection.get('registry_sha256')!=sha256(registry_path):
        raise ValueError('Scoring requires a complete fresh collector bound to this registry')
    if registry['parent_sha256']!=parent:raise ValueError('Oracle parent differs')
    physical=registry['physical_keys'];wanted={physical_key(parent,format(int(s.split('_')[1],16),'048b'),teacher):s for s in expected}
    if set(physical)!=set(wanted):raise ValueError('Oracle physical IDs do not equal eligible union')
    for k,sid in wanted.items():
        v=physical[k]
        if v['model']!=teacher or v['structure_id']!=sid or v.get('provenance')!='new_scoring':
            raise ValueError('Old cached, wrong teacher or wrong geometry registered')
    records={}; evidence=[]; total_seconds=0.0
    for task in registry['tasks']:
        if task['model']!=teacher or task['geometry_protocol']!=PROTOCOL or task['parent_sha256']!=parent:
            raise ValueError('Scoring task identity differs')
        tid=task['task_id'];score=root/'scores'/teacher/f'{tid}.npz';meta_path=score.with_suffix('.json')
        owner_path=root/'claims'/tid/'owner.json';owner=json.loads(owner_path.read_text());meta=json.loads(meta_path.read_text())
        if (meta.get('status')!='complete' or meta.get('task')!=task or meta.get('output_sha256')!=sha256(score)
            or owner.get('worker')!=meta.get('worker') or float(owner.get('started_unix',0))<after):
            raise ValueError('Task did not start after assessment freeze, or raw receipt differs')
        binding(task['input_npz'],task['input_sha256']);binding(task['parent_npz'],task['parent_sha256'])
        worker=root/'workers'/f'{meta["worker"]}.json';_check_worker(worker,teacher)
        with np.load(task['input_npz'],allow_pickle=False) as z:
            src=z['structure_ids'][task['indices']].astype(str);masks=z['masks'][task['indices']]
        with np.load(task['parent_npz'],allow_pickle=False) as z: perms=z['permutations'];host=len(z['numbers'])-48
        keys=_canonical_vacancy(1-masks,perms)
        if [f'mask48_{int(k,2):012x}' for k in keys]!=src.tolist():
            raise ValueError('Scored input masks disagree with their physical IDs')
        with np.load(score,allow_pickle=False) as z:
            ids=z['structure_ids'].astype(str);status=z['status'].astype(str);energy=z['energy_eV'];atoms=z['n_atoms'];nl=z['n_Li'];seconds=z['seconds']
        if not np.array_equal(src,ids) or ids.tolist()!=task['expected_structure_ids'] or not np.all(status=='success'):
            raise ValueError('Task IDs/status disagree')
        for i,sid in enumerate(ids):
            row=meta['rows'][i]
            if (sid in records or sid not in expected or not np.isfinite(energy[i])
                or int(atoms[i])!=host+int(nl[i]) or int(nl[i])!=48-int(masks[i].sum())
                or row.get('structure_id')!=sid or row.get('status')!='success'
                or float(row.get('energy_eV',float('nan')))!=float(energy[i])
                or int(row.get('n_atoms',-1))!=int(atoms[i]) or int(row.get('n_Li',-1))!=int(nl[i])):
                raise ValueError('Duplicate, inconsistent or invalid raw score')
            if not np.isfinite(seconds[i]) or seconds[i]<0: raise ValueError('Invalid per-query scoring time')
            total_seconds+=float(seconds[i])
            records[sid]=(float(energy[i]),int(atoms[i]),int(nl[i]))
        evidence.append({'task':tid,'score':binding(score),'receipt':binding(meta_path),'owner':binding(owner_path),'worker':binding(worker)})
    if set(records)!=set(expected):raise ValueError('Raw scoring tasks do not cover exact eligible union')
    output=Path(collection_path).parent/f'{teacher}.npz'
    with np.load(output,allow_pickle=False) as z:
        # Compressed NPZ members decompress in full on each access. Load each
        # member once while preserving the same row-level provenance checks.
        collected={name:z[name] for name in ('structure_ids','status','provenance','energy_eV','n_atoms','n_Li')}
    ids=collected['structure_ids'].astype(str)
    if len(set(ids))!=len(ids) or set(ids)!=set(expected):raise ValueError('Collector score IDs differ from eligible union')
    for i,sid in enumerate(ids):
        if (collected['status'][i]!='success' or collected['provenance'][i]!='new'
            or (float(collected['energy_eV'][i]),int(collected['n_atoms'][i]),int(collected['n_Li'][i]))!=records[sid]):
            raise ValueError('Collector row differs from raw new score')
    return {'registry':binding(registry_path),'collection_receipt':binding(collection_path),'score':binding(output),
            'tasks':evidence,'unique_geometry_teacher_pairs':len(records),'scoring_tasks':len(evidence),
            'summed_inference_seconds':total_seconds}


def aggregate(common_path,boundary_dir,bundles,oracle_root,collection_receipt,output):
    c=_check_common(common_path,boundary_dir);common_sha=sha256(common_path)
    # bundle spec is POLICY:REPLICATE=assessment-safe-dir=evaluator-output-dir
    supplied={}
    for item in bundles:
        identity,safe,evaluation=item.split('=',2);policy,rep=identity.rsplit(':',1)
        if (policy,rep) in supplied:raise ValueError('Repeated lineage')
        supplied[(policy,rep)]=(Path(safe),Path(evaluation))
    declarations={(e['policy'],str(e['replicate_id'])):e for e in c['lineages']}
    if set(supplied)!=set(declarations):raise ValueError('Supplied lineages differ from protocol')
    union=set();safe_receipts=[];evaluation_info={};streams={};after=c['created_unix']
    for ident,(safe,evaluation) in supplied.items():
        decl=declarations[ident];r=json.loads((safe/'receipt.json').read_text());after=max(after,r['created_unix'])
        if r['status']!='FROZEN_BEFORE_SCORING':raise ValueError('Unfrozen assessment')
        for n,h in r['outputs'].items():binding(safe/n,h)
        binding(r['original_generation_receipt']['path'],r['original_generation_receipt']['sha256'])
        binding(r['generator_source']['path'],r['generator_source']['sha256'])
        binding(r['runtime_loader_source']['path'],r['runtime_loader_source']['sha256'])
        binding(r['adapter_source']['path'],r['adapter_source']['sha256'])
        manifest_path=Path(r['manifest']['path']);binding(manifest_path,r['manifest']['sha256'])
        m,_,_,_,_,_=verify_manifest(manifest_path,boundary_dir)
        protocol=json.loads((safe/'protocol.json').read_text())
        if (sha256(safe/'protocol.json')!=r['protocol_sha256'] or protocol['manifest']['sha256']!=sha256(manifest_path)
            or protocol['inputs']['data']['sha256']!=m['inputs']['data']['sha256']
            or protocol['inputs']['history']['sha256']!=m['inputs']['history']['sha256']
            or protocol['namespace']!=c['namespace']):
            raise ValueError('Safe proposal protocol does not bind this lineage and boundary')
        p=json.loads(Path(m['plan']['path']).read_text())
        declared_hashes=p.get('common_protocol_sha256s',[])
        if p.get('common_protocol_sha256'): declared_hashes=[*declared_hashes,p['common_protocol_sha256']]
        if common_sha not in declared_hashes:raise ValueError('Lineage plan did not bind common protocol before generation')
        if (m['namespace']!=c['namespace'] or m['counts']!=c['counts'] or m['requested_bins']!=c['requested_bins']
            or m['count_per_group']!=c['count_per_group'] or m['inputs']['data']['sha256']!=decl['data']['sha256']
            or m['inputs']['schema']['sha256']!=c['inputs']['condition_schema']['sha256']):
            raise ValueError('Lineage manifest differs from common protocol')
        models=[x for x in m['models'] if x['method']==ident[0] and str(x['replicate_id'])==ident[1]]
        if len(models)!=1 or len(m['models'])!=1:raise ValueError('Each lineage must contain its declared policy/replicate')
        if decl['rng_kind']=='fixed_mutation' and models[0].get('sampler')!='mutation':
            raise ValueError('Fixed mutation lineage does not name a mutation sampler')
        if decl['checkpoint'] and models[0].get('expected_checkpoint_sha256')!=decl['checkpoint']['sha256']:
            raise ValueError('Lineage checkpoint differs')
        if (m.get('previous_merge_receipt') or {}).get('sha256')!=(decl['merge_receipt'] or {}).get('sha256'):
            raise ValueError('Lineage merge differs')
        source_key='generator_source' if decl['rng_kind']=='ar' else 'mutation_source'
        if r['generator_source']['sha256']!=c['inputs'][source_key]['sha256']:
            raise ValueError('Generator source differs')
        with np.load(safe/'candidates.npz',allow_pickle=False) as z:union.update(z['structure_ids'].astype(str).tolist())
        rows=read_rows(safe/'attempts_all.csv')
        with np.load(safe/'raw_attempts.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        groups=Counter(); attempt_indices=defaultdict(set)
        for row in rows:
            if row['diagnostic']!='primary':continue
            n=int(row['n_target']);bin_=row['requested_bin'];groups[n,bin_]+=1;i=int(row['raw_row']);trial=(ident[1],n,bin_,row['attempt'])
            if row['method']!=ident[0] or row['replicate_id']!=ident[1]:raise ValueError('Attempt lineage differs')
            index=int(row['attempt'])
            if index in attempt_indices[n,bin_]:raise ValueError('Duplicate paired attempt index')
            attempt_indices[n,bin_].add(index)
            if decl['rng_kind']=='ar':
                dim=int(row['latent_dim'])
                if dim<0 or dim>raw['latents'].shape[1]:raise ValueError('Invalid actual latent dimension')
                value=(raw['uniforms'][i],raw['latents'][i,:dim])
                if trial in streams.get('ar',{}):
                    prev=streams['ar'][trial];dim=min(len(prev[1]),len(value[1]))
                    if not np.array_equal(prev[0],value[0]) or not np.array_equal(prev[1][:dim],value[1][:dim]):
                        raise ValueError('Cross-lineage AR streams differ')
                if trial not in streams.get('ar',{}) or len(value[1])>len(streams['ar'][trial][1]):
                    streams.setdefault('ar',{})[trial]=value
            else:
                value=(raw['seed_occupancy'][i],int(row['swap_count']),raw['remove_uniforms'][i],raw['add_uniforms'][i])
                if trial in streams.get('fixed_mutation',{}):
                    prev=streams['fixed_mutation'][trial]
                    if not all(np.array_equal(a,b) for a,b in zip(prev,value)):raise ValueError('Mutation seed/k/edit streams differ')
                streams.setdefault('fixed_mutation',{})[trial]=value
        if dict(groups)!={(n,b):c['count_per_group'] for n in c['counts'] for b in c['requested_bins']} or any(
            v!=set(range(c['count_per_group'])) for v in attempt_indices.values()):
            raise ValueError('Lineage attempt cells/denominators differ')
        er=json.loads((evaluation/'receipt.json').read_text())
        if (er.get('status')!='PASS' or er['proposal_protocol_sha256']!=sha256(safe/'protocol.json')
            or er['source_sha256']!=c['inputs']['evaluator_source']['sha256']):
            raise ValueError('Evaluation is incomplete or not bound to declared evaluator/proposals')
        for n,h in er['outputs'].items():binding(evaluation/n,h)
        if er['score_sources'][c['teacher']]['schema_sha256']!=c['inputs']['teacher_schema']['sha256']:
            raise ValueError('Evaluation teacher schema differs')
        if (er.get('fixed_thresholds') or {}).get('sha256')!=c['inputs']['thresholds']['sha256']:
            raise ValueError('Evaluation fixed thresholds differ')
        evaluation_info[ident]=(evaluation,er,rows)
        safe_receipts.append(binding(safe/'receipt.json'))
    oracle=collect_oracle(oracle_root,collection_receipt,union,c['parent']['sha256'],c['teacher'],after)
    for evaluation,er,_ in evaluation_info.values():
        if er['score_sources'][c['teacher']]['sha256']!=oracle['score']['sha256']:
            raise ValueError('Lineage evaluation used different score collection')
    seed=[];coverage=[]
    for ident,(evaluation,er,rows) in evaluation_info.items():
        metrics=read_rows(evaluation/'metrics.csv');selected=[r for r in metrics if r['teacher']==c['teacher'] and r['method']==ident[0] and r['replicate_id']==ident[1]]
        if {(int(r['n_Li']),r['requested_bin']) for r in selected}!={(n,b) for n in c['counts'] for b in c['requested_bins']}:
            raise ValueError('Evaluator metrics omit or add cells')
        low=[float(r['novel_fixed_q20_yield_per_attempt']) for r in selected if r['requested_bin']=='low' and r['fixed_q20_available']=='True']
        if len(low)!=len(c['counts']):raise ValueError('Fixed-q20 low thresholds missing')
        seed.append({'policy':ident[0],'replicate_id':ident[1],
            'balanced_novel_inbin':float(np.mean([float(r['novel_inbin_yield_per_attempt']) for r in selected])),
            'fixed_q20_low':float(np.mean(low))})
        statuses=Counter(r['status'] for r in rows if r['diagnostic']=='primary')
        coverage.append({'policy':ident[0],'replicate_id':ident[1],'attempts':sum(statuses.values()),
            'eligible_generated':statuses['generated'],'known_excluded':statuses['known_boundary'],
            'protected_excluded':statuses['protected_boundary'],
            'other_failed':sum(v for k,v in statuses.items() if k not in ('generated','known_boundary','protected_boundary')),
            'eligible_unique_ids':len({r['scoring_structure_id'] for r in rows if r['status']=='generated'}),
            'missing_scores':er['missing_or_nonterminal_candidate_labels'][c['teacher']],
            'terminal_failures':er['terminal_failed_candidate_labels'][c['teacher']]})
    pairs=[];lookup={(r['policy'],r['replicate_id']):r for r in seed}
    for contrast in c['contrasts']:
        a,b=contrast['policy'],contrast['reference']
        for rep in sorted(r for p,r in lookup if p==a):
            x,y=lookup[a,rep],lookup[b,rep]
            pairs.append({'policy':a,'reference':b,'replicate_id':rep,
              'balanced_difference':x['balanced_novel_inbin']-y['balanced_novel_inbin'],
              'low_difference':x['fixed_q20_low']-y['fixed_q20_low']})
    pair_summary=[]
    for contrast in c['contrasts']:
        selected=[x for x in pairs if x['policy']==contrast['policy'] and x['reference']==contrast['reference']]
        row={'policy':contrast['policy'],'reference':contrast['reference'],'paired_fits':len(selected)}
        for field in ('balanced_difference','low_difference'):
            v=[x[field] for x in selected];row[field+'_mean']=float(np.mean(v));row[field+'_seed_sd']=float(np.std(v,ddof=1)) if len(v)>1 else ''
        pair_summary.append(row)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    write_rows(out/'seed_metrics.csv',seed,list(seed[0]));write_rows(out/'coverage.csv',coverage,list(coverage[0]));write_rows(out/'paired_differences.csv',pairs,list(pairs[0]));write_rows(out/'paired_summary.csv',pair_summary,list(pair_summary[0]))
    result={'status':'PASS','common_protocol':binding(common_path),'assessment_receipts':safe_receipts,
       'evaluation_receipts':[binding(x[0]/'receipt.json') for x in evaluation_info.values()],
       'oracle_evidence':oracle,'global_oracle_calls':oracle['unique_geometry_teacher_pairs'],
       'standalone_eligible_queries':sum(r['eligible_unique_ids'] for r in coverage),
       'summed_inference_seconds':oracle['summed_inference_seconds'],
       'cost_scope':'Per-geometry worker inference seconds, summed across distinct physical tasks; allocation/node-hours require separate scheduler accounting.',
       'uncertainty_unit':'Independent model/fit replicates, never proposal rows.',
       'rng_scope':'AR streams compared only among AR lineages; literal seed/k/edit streams only among fixed mutation peers.',
       'outputs':{name:sha256(out/name) for name in ('seed_metrics.csv','coverage.csv','paired_differences.csv','paired_summary.csv')},'source':binding(__file__)}
    write(out/'receipt.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='cmd',required=True)
    f=s.add_parser('freeze');f.add_argument('--plan',type=Path,required=True);f.add_argument('--boundary-dir',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
    a=s.add_parser('aggregate')
    for name in ('common','boundary-dir','oracle-root','collection-receipt','output-dir'):a.add_argument('--'+name,type=Path,required=True)
    a.add_argument('--bundle',action='append',required=True)
    x=p.parse_args();require_compute_allocation()
    r=freeze(x.plan,x.boundary_dir,x.output) if x.cmd=='freeze' else aggregate(x.common,x.boundary_dir,x.bundle,x.oracle_root,x.collection_receipt,x.output_dir)
    print(json.dumps({'status':r['status']},indent=2))
if __name__=='__main__':main()

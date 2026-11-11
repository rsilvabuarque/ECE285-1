"""Descriptive agreement of UMA and MACE on a frozen DEVELOPMENT orbit union.

Fixed schema bins use each teacher's original TRAIN-only edges. Empirical LOW
sets use bottom-20%-with-ties of this selected generated-orbit union; they are
not a global density of states or a DFT assessment. Nothing here fits a model,
reads old assessment scores, or queries an MLIP.
"""
from __future__ import annotations

import argparse
import math
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, rankdata

from development_boundary import _json_keys, binding, digest, save_csv, save_json, verify_manifest
from aggregate_development import validate_registry
from cathode_cvae.conditions import classify, load_schema


def ratio(num, den):
    return num / den if den else ''


def overlap(left, right):
    left, right = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
    both=int(np.sum(left & right)); either=int(np.sum(left | right)); a=int(left.sum()); b=int(right.sum())
    return dict(uma_low=a,mace_low=b,overlap=both,union=either,jaccard=ratio(both,either),
                fraction_UMA_low_also_MACE=ratio(both,a),fraction_MACE_low_also_UMA=ratio(both,b))


def empirical_low(energy, fraction=0.2):
    """Nearest-rank cutoff, retaining all exact-energy ties at that cutoff."""
    e=np.asarray(energy,float)
    if e.ndim!=1 or not len(e) or not np.isfinite(e).all():raise ValueError('Nonempty finite energy vector required')
    k=math.ceil(fraction*len(e))
    cutoff=float(np.partition(e,k-1)[k-1])
    return e<=cutoff,cutoff


def analyze_count(count, ids, uma, mace, schema_uma, schema_mace):
    ids=np.asarray(ids).astype(str); u=np.asarray(uma,float); m=np.asarray(mace,float)
    if len(set(ids))!=len(ids) or not len(ids) or u.shape!=ids.shape or m.shape!=ids.shape:
        raise ValueError('Expected one paired energy per distinct orbit')
    if not np.isfinite(u).all() or not np.isfinite(m).all():raise ValueError('Nonfinite teacher energy')
    ru=rankdata(u,method='average');rm=rankdata(m,method='average')
    spearman=float(np.corrcoef(ru,rm)[0,1]) if len(ids)>1 and np.std(ru)>0 and np.std(rm)>0 else ''
    tau=float(kendalltau(u,m,variant='b').statistic) if len(ids)>1 and np.std(u)>0 and np.std(m)>0 else ''
    if spearman!='' and not np.isfinite(spearman):raise ValueError('Undefined Spearman correlation')
    if tau!='' and not np.isfinite(tau):raise ValueError('Undefined Kendall correlation')
    bu=classify(schema_uma,np.full(len(ids),count),u)
    bm=classify(schema_mace,np.full(len(ids),count),m)
    eu,cutu=empirical_low(u);em,cutm=empirical_low(m)
    schema_overlap=overlap(bu=='low',bm=='low');empirical_overlap=overlap(eu,em)
    summary=dict(n_Li=count,unique_shared_orbits=len(ids),spearman_average_rank=spearman,kendall_tau_b=tau,
        UMA_distinct_energies=len(np.unique(u)),MACE_distinct_energies=len(np.unique(m)),
        same_exact_average_rank_fraction=float(np.mean(ru==rm)),
        schema_bin_agreement_fraction=float(np.mean(bu==bm)),
        schema_low_UMA=schema_overlap['uma_low'],schema_low_MACE=schema_overlap['mace_low'],
        schema_low_overlap=schema_overlap['overlap'],schema_low_union=schema_overlap['union'],
        schema_low_jaccard=schema_overlap['jaccard'],
        schema_fraction_UMA_low_also_MACE=schema_overlap['fraction_UMA_low_also_MACE'],
        schema_fraction_MACE_low_also_UMA=schema_overlap['fraction_MACE_low_also_UMA'],
        empirical_low_UMA_cut_eV_atom=cutu,empirical_low_MACE_cut_eV_atom=cutm,
        empirical_low_UMA=empirical_overlap['uma_low'],empirical_low_MACE=empirical_overlap['mace_low'],
        empirical_low_overlap=empirical_overlap['overlap'],empirical_low_union=empirical_overlap['union'],
        empirical_low_jaccard=empirical_overlap['jaccard'],
        empirical_fraction_UMA_low_also_MACE=empirical_overlap['fraction_UMA_low_also_MACE'],
        empirical_fraction_MACE_low_also_UMA=empirical_overlap['fraction_MACE_low_also_UMA'])
    confused=Counter(zip(bu.tolist(),bm.tolist()));bins=('low','middle','high')
    confusion=[dict(n_Li=count,UMA_bin=x,MACE_bin=y,orbits=confused[x,y],
                    fraction_of_all_orbits=confused[x,y]/len(ids)) for x in bins for y in bins]
    rows=[dict(structure_id=sid,n_Li=count,UMA_energy_eV_atom=float(a),MACE_energy_eV_atom=float(b),
               UMA_average_rank=float(x),MACE_average_rank=float(y),UMA_schema_bin=c,MACE_schema_bin=d,
               UMA_empirical_low=bool(g),MACE_empirical_low=bool(h))
          for sid,a,b,x,y,c,d,g,h in zip(ids,u,m,ru,rm,bu,bm,eu,em)]
    return summary,confusion,rows


def validate_inputs(shards, collector_dir, registry_path, evaluation_receipt):
    """Use frozen manifests and identity-only candidate pools, never old scores."""
    collector_dir=Path(collector_dir).absolute();registry_path=Path(registry_path).absolute()
    collector=json.loads((collector_dir/'receipt.json').read_text())
    if collector.get('status')!='PASS':raise ValueError('Collector receipt is not PASS')
    ev=json.loads(Path(evaluation_receipt).read_text())
    if ev.get('status')!='PASS' or any(ev.get('terminal_failed_candidate_labels',{}).values()):
        raise ValueError('A complete fresh evaluator receipt is required to bind score bytes')
    parsed=[];common=None;seen=set();sources=[];candidate_counts={}
    protocol_hashes=set()
    for manifest_path,gen_path in shards:
        manifest_path,gen_path=Path(manifest_path).absolute(),Path(gen_path).absolute()
        m=verify_manifest(manifest_path)
        boundary=json.loads(Path(m['boundary_receipt']['path']).read_text())
        if boundary.get('status')!='FROZEN_DEVELOPMENT_KEY_ONLY_BOUNDARY':raise ValueError('Wrong boundary receipt')
        for field in ('history','protected'):binding(boundary[field]['path'],boundary[field]['sha256'])
        protected=_json_keys(boundary['protected']['path'],48);known=_json_keys(boundary['history']['path'],48)
        if (len(protected)!=boundary['protected_orbits'] or len(known)!=boundary['known_history_orbits']
                or not protected<=known or boundary['history']['sha256']!=m['inputs']['history']['sha256']):
            raise ValueError('Frozen protected/known identity boundary changed')
        if set(m['teacher_schemas'])!={'uma','mace'}:raise ValueError('This comparison requires UMA and MACE schemas')
        gen=json.loads((gen_path/'receipt.json').read_text());protocol=json.loads((gen_path/'protocol.json').read_text())
        if gen.get('status')!='FROZEN_BEFORE_SCORING' or gen['protocol_sha256']!=digest(gen_path/'protocol.json'):
            raise ValueError('Filtered generation is not frozen')
        protocol_hashes.add(gen['protocol_sha256'])
        if protocol['manifest']['sha256']!=digest(manifest_path) or gen['manifest']['sha256']!=digest(manifest_path):
            raise ValueError('Generation and manifest differ')
        if gen['boundary_receipt']['sha256']!=m['boundary_receipt']['sha256']:
            raise ValueError('Generation and protected boundary differ')
        binding(gen['original_generation_receipt']['path'],gen['original_generation_receipt']['sha256'])
        for name,h in gen['outputs'].items():binding(gen_path/name,h)
        reps={str(e['replicate_id']) for e in m['models']}
        if len(reps)!=1 or reps & seen:raise ValueError('Distinct one-replicate shards required')
        rep=reps.pop();seen.add(rep)
        state={'parent':m['inputs']['parent']['sha256'],'history':m['inputs']['history']['sha256'],
               'boundary':m['boundary_receipt']['sha256'],'counts':m['counts'],
               'schemas':{k:v['sha256'] for k,v in m['teacher_schemas'].items()}}
        if common is None:common=state
        elif common!=state:raise ValueError('Shards have incompatible parent, history, count or teacher schemas')
        with np.load(gen_path/'candidates.npz',allow_pickle=False) as z:
            ids=z['structure_ids'].astype(str); masks=z['masks']; n=z['n_Li']
            if masks.shape!=(len(ids),48) or len(n)!=len(ids):raise ValueError('Unexpected candidate shape')
            for sid,mask,count in zip(ids,masks,n):
                key=''.join(map(str,mask))
                if key in protected:raise ValueError('Protected orbit leaked into teacher agreement')
                if sid!=f'mask48_{int(key,2):012x}' or int(count)!=48-int(mask.sum()):
                    raise ValueError('Candidate identity/count mismatch')
                if sid in candidate_counts and candidate_counts[sid]!=int(count):raise ValueError('Inconsistent orbit count')
                candidate_counts[sid]=int(count)
        for field in ('conditions.py','pool_scoring.py'):
            expected=next((v['sha256'] for v in m['source_bindings'] if Path(v['path']).name==field),None)
            import cathode_cvae.conditions, cathode_cvae.pool_scoring
            actual=digest(cathode_cvae.conditions.__file__ if field=='conditions.py' else cathode_cvae.pool_scoring.__file__)
            if expected!=actual:raise ValueError(f'Runtime source differs from frozen manifest: {field}')
        sources.append(dict(replicate_id=rep,manifest=binding(manifest_path),generation_receipt=binding(gen_path/'receipt.json')))
        parsed.append((rep,m,gen_path,None))
    if not parsed:raise ValueError('At least one frozen shard required')
    if ev['proposal_protocol_sha256'] not in protocol_hashes:
        raise ValueError('Evaluator receipt does not belong to any supplied fresh shard')
    source_hash=next((v['sha256'] for v in parsed[0][1]['source_bindings']
                      if Path(v['path']).name=='evaluate_controls.py'),None)
    if ev['source_sha256']!=source_hash or set(ev['score_sources'])!={'uma','mace'}:
        raise ValueError('Evaluator source or teacher set differs from frozen manifest')
    for teacher in ('uma','mace'):
        score=ev['score_sources'][teacher]
        binding(collector_dir/f'{teacher}.npz',score['sha256'])
        if score['schema_sha256']!=common['schemas'][teacher]:
            raise ValueError('Evaluator used a different teacher schema')
    if set(candidate_counts.values())-set(common['counts']):raise ValueError('Candidate count outside frozen design')
    validate_registry(registry_path,collector_dir,collector,parsed,['uma','mace'])
    labels={}
    for teacher in ('uma','mace'):
        with np.load(collector_dir/f'{teacher}.npz',allow_pickle=False) as z:
            ids=z['structure_ids'].astype(str); energy=z['energy_eV'].astype(float)
            atoms=z['n_atoms'].astype(int); counts=z['n_Li'].astype(int)
        if not np.isfinite(energy).all() or np.any(atoms<=0):raise ValueError('Missing/nonfinite collected energy')
        labels[teacher]={sid:(e,at,n) for sid,e,at,n in zip(ids,energy,atoms,counts)}
    if set(labels['uma'])!=set(candidate_counts) or set(labels['mace'])!=set(candidate_counts):
        raise ValueError('Collected teacher identities do not equal eligible union')
    parent_path=Path(parsed[0][1]['inputs']['parent']['path'])
    with np.load(parent_path,allow_pickle=False) as z:host=len(z['numbers'])-len(z['li_indices'])
    for sid,n in candidate_counts.items():
        for teacher in ('uma','mace'):
            _,atoms,count=labels[teacher][sid]
            if count!=n or atoms!=host+n:raise ValueError('Teacher count/atom count disagrees with candidate')
    schema={t:load_schema(parsed[0][1]['teacher_schemas'][t]['path']) for t in ('uma','mace')}
    return common,sources,candidate_counts,labels,schema,parsed[0][1]['teacher_schemas']


def plot(out, orbit_rows, confusion, counts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(counts),figsize=(4.3*len(counts),4),squeeze=False)
    for ax,n in zip(axes[0],counts):
        rr=[r for r in orbit_rows if r['n_Li']==n];k=len(rr)
        if k:
            ax.scatter([r['UMA_average_rank']/k for r in rr],[r['MACE_average_rank']/k for r in rr],s=4,alpha=.22,rasterized=True)
        ax.plot([0,1],[0,1],color='black',lw=.8);ax.set(xlabel='UMA normalized rank',ylabel='MACE normalized rank',title=f'{n} Li; {k} orbits',xlim=(0,1),ylim=(0,1));ax.set_aspect('equal')
    fig.suptitle('Development union: rank 1 is lowest energy',fontsize=10);fig.tight_layout();fig.savefig(out/'teacher_ranks.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,len(counts),figsize=(4.2*len(counts),4),squeeze=False)
    bins=['low','middle','high']
    for ax,n in zip(axes[0],counts):
        vals={(r['UMA_bin'],r['MACE_bin']):r['orbits'] for r in confusion if r['n_Li']==n}
        a=np.array([[vals.get((x,y),0) for y in bins] for x in bins]);ax.imshow(a,cmap='Blues')
        for i in range(3):
            for j in range(3):ax.text(j,i,str(a[i,j]),ha='center',va='center',fontsize=8,color='white' if a[i,j] > a.max()/2 and a.max() else 'black')
        ax.set(xticks=range(3),xticklabels=bins,yticks=range(3),yticklabels=bins,xlabel='MACE schema bin',ylabel='UMA schema bin',title=f'{n} Li')
    fig.suptitle('Fixed TRAIN-schema energy bins; unique development orbits',fontsize=10);fig.tight_layout();fig.savefig(out/'teacher_schema_confusion.pdf');plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard',nargs=2,action='append',required=True,metavar=('MANIFEST','FILTERED_GENERATION'))
    p.add_argument('--collector-dir',type=Path,required=True);p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--evaluation-receipt',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv)
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    common,sources,candidate_counts,labels,schema,schema_bindings=validate_inputs(a.shard,a.collector_dir,a.registry,a.evaluation_receipt)
    summaries=[];confusions=[];orbits=[]
    for n in common['counts']:
        ids=sorted(sid for sid,count in candidate_counts.items() if count==n)
        if not ids:raise ValueError(f'No eligible shared orbits for {n} Li')
        u=[labels['uma'][sid][0]/labels['uma'][sid][1] for sid in ids]
        m=[labels['mace'][sid][0]/labels['mace'][sid][1] for sid in ids]
        summary,c,r=analyze_count(n,ids,u,m,schema['uma'],schema['mace'])
        summaries.append(summary);confusions.extend(c);orbits.extend(r)
    out=a.output_dir.absolute()
    if out.exists():raise FileExistsError(f'Preserve existing analysis: {out}')
    out.mkdir(parents=True)
    save_csv(out/'agreement_by_count.csv',summaries);save_csv(out/'schema_bin_confusion.csv',confusions);save_csv(out/'orbit_agreement.csv',orbits)
    plot(out,orbits,confusions,common['counts'])
    receipt={'status':'PASS_DEVELOPMENT_TEACHER_AGREEMENT','source_sha256':digest(__file__),
        'shared_orbits':len(orbits),'manifest_and_filtered_receipts':sources,
        'collector_receipt':binding(a.collector_dir/'receipt.json'),'registry':binding(a.registry),
        'fresh_evaluation_receipt':binding(a.evaluation_receipt),
        'teacher_scores':{t:binding(a.collector_dir/f'{t}.npz') for t in ('uma','mace')},
        'teacher_schemas':schema_bindings,'source_bindings':{
            'boundary_script':binding(Path(__file__).with_name('development_boundary.py')),
            'aggregation_script':binding(Path(__file__).with_name('aggregate_development.py'))},
        'definitions':{'rank':'Spearman Pearson correlation of average tie ranks, and Kendall tau-b; ascending energy, within composition',
            'schema_bins':'Original teacher-specific TRAIN-only fixed edges, not a shared absolute-energy cutoff',
            'schema_low_overlap':'Unique candidate orbits classified low by each teacher with its own frozen schema',
            'empirical_low_overlap':'Nearest-rank 20% cutoff in the selected generated orbit union, including all tied energies; not global density of states',
            'uncertainty':'Descriptive selected-union comparison, not an IID test or DFT validation'},
        'new_MLIP_calls':0,'new_training_updates':0,'outputs':{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()}}
    save_json(out/'receipt.json',receipt)
    print(json.dumps({'status':receipt['status'],'shared_orbits':len(orbits),'output_dir':str(out)},indent=2))


if __name__=='__main__':main()

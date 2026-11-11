"""Plot the fresh DEVELOPMENT angular-CVAE dimension sweeps.

Three families x five latent dimensions x three matched fits. The figures and
CSV tables distinguish descriptive replicate SD from statistical confidence.
Only current DEVELOPMENT summaries, all-valid masks and VAL sidecars are read.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from development_boundary import binding, digest, save_csv, save_json

FAMILIES=(('angle_hist_cvae','Angle histogram'),('angular_smooth_cvae','Smooth angles'),
          ('directional_l1_l2_cvae','Directional l1/l2'))
DIMENSIONS=(2,4,8,16,32)
COUNTS=(12,24,36)


def method_for(family,dimension):
    return family if dimension==4 else f'{family}_d{dimension}'


def rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def key_rows(values,fields):
    result={}
    for value in values:
        key=tuple(value[f] for f in fields)
        if key in result:raise ValueError(f'Duplicate input row: {key}')
        result[key]=value
    return result


def number(value,field):
    try:x=float(value)
    except (TypeError,ValueError):raise ValueError(f'Missing {field}') from None
    if not np.isfinite(x):raise ValueError(f'Nonfinite {field}')
    return x


def load_inputs(aggregate_path,mask_paths,fit_root):
    root=Path(aggregate_path).absolute().parent; agg_path=Path(aggregate_path).absolute();fit_root=Path(fit_root).resolve()
    agg=json.loads(agg_path.read_text())
    if agg.get('status')!='PASS_DEVELOPMENT_AGGREGATION':raise ValueError('Expected PASS development aggregation')
    seeds=tuple(sorted(map(str,agg['replicates'])))
    if len(seeds)!=3 or len(set(seeds))!=3:raise ValueError('Exactly three distinct matched seeds required')
    yield_path=root/'per_seed_yield.csv';binding(yield_path,agg['outputs']['per_seed_yield.csv'])
    inputs=[binding(agg_path),binding(yield_path)];manifests={}
    for s in agg['shards']:
        seed=str(s['replicate_id']);ref=s['manifest'];binding(ref['path'],ref['sha256'])
        if seed in manifests:raise ValueError('Duplicate seed manifest')
        m=json.loads(Path(ref['path']).read_text())
        if m.get('status')!='FROZEN_DEVELOPMENT_BEFORE_GENERATION':raise ValueError('Manifest not frozen development')
        manifests[seed]=(m,ref);inputs.append(ref)
    if set(manifests)!=set(seeds):raise ValueError('Aggregate and manifest seeds differ')
    masks={}
    for path in mask_paths:
        path=Path(path).absolute();r=json.loads(path.read_text())
        if r.get('status')!='PASS' or 'all originally valid masks' not in r.get('scope','').lower():
            raise ValueError('Expected all-valid development mask receipt')
        ref=r['manifest'];binding(ref['path'],ref['sha256'])
        match=[s for s,(_,mref) in manifests.items() if mref['sha256']==ref['sha256']]
        if len(match)!=1 or match[0] in masks:raise ValueError('Mask receipt manifest/seed is ambiguous or repeated')
        file=path.parent/'mask_summary.csv';binding(file,r['outputs']['mask_summary.csv'])
        masks[match[0]]=rows(file);inputs.extend([binding(path),binding(file)])
    if set(masks)!=set(seeds):raise ValueError('Mask receipt coverage differs from seed cohort')
    latent={};sidecars=[]
    for seed in seeds:
        m,_=manifests[seed]
        for family,_ in FAMILIES:
            for d in DIMENSIONS:
                method=method_for(family,d)
                matches=[x for x in m['models'] if x['method']==method and str(x['replicate_id'])==seed]
                if len(matches)!=1:raise ValueError(f'Missing angular model {method}/{seed}')
                e=matches[0]; ck=Path(e['checkpoint']).resolve()
                if not ck.is_relative_to(fit_root):raise ValueError(f'Angular checkpoint is outside fresh fit root: {ck}')
                checkpoint=binding(ck,e['expected_checkpoint_sha256']); cfg=e['configuration'];binding(cfg['path'],cfg['sha256'])
                config=json.loads(Path(cfg['path']).read_text())
                if int(config['latent_dim'])!=d or int(e['declared_latent_dim'])!=d or str(config['seed'])!=seed:
                    raise ValueError('Selected checkpoint dimension/seed mismatch')
                sidecar=Path(str(ck)+'.latent.json');val=json.loads(sidecar.read_text());p=val['pooled']
                if int(p['latent_dim'])!=d or int(p['rows'])!=int(config['val_rows']):raise ValueError('VAL sidecar/config mismatch')
                kl=np.asarray(p['KL_nats_by_dimension'],float);v=np.asarray(p['posterior_mean_variance_by_dimension'],float)
                if kl.shape!=(d,) or v.shape!=(d,) or not np.isfinite(kl).all() or not np.isfinite(v).all():
                    raise ValueError('Invalid VAL component arrays')
                if (np.any(kl < -1e-6) or np.any(v<0)
                    or not np.isclose(kl.mean(),p['mean_KL_nats_per_dimension'],atol=1e-8,rtol=1e-5)
                    or not np.isclose(kl.sum(),p['total_KL_nats'],atol=1e-8,rtol=1e-5)):
                    raise ValueError('Invalid KL components or sidecar totals')
                active={str(t):int(np.sum(v>t)) for t in (0.001,0.01,0.1)}
                if active!=p['active_components_by_cutoff']:raise ValueError('Activity counts disagree with VAL variances')
                latent[family,d,seed]=dict(mean_KL=float(kl.mean()),total_KL=float(kl.sum()),
                    active_001=active['0.001'],active_01=active['0.01'],active_1=active['0.1'],epoch=int(val['epoch']))
                sidecars.append(dict(family=family,dimension=d,seed=seed,checkpoint=checkpoint,configuration=cfg,
                    sidecar=binding(sidecar),sidecar_epoch=int(val['epoch'])))
    return seeds,rows(yield_path),masks,latent,inputs,sidecars


def extract(seeds,yields,masks,latent):
    yy=key_rows(yields,('teacher','method','replicate_id'))
    mm={s:key_rows(masks[s],('method','replicate_id','n_Li','contrast')) for s in seeds}
    result=[]
    for family,label in FAMILIES:
        for d in DIMENSIONS:
            method=method_for(family,d)
            for seed in seeds:
                row=dict(family=family,label=label,dimension=d,method=method,seed=seed)
                for teacher in ('uma','mace'):
                    y=yy[teacher,method,seed]
                    if y.get('fit_seed','')!=seed:raise ValueError('Yield row fit seed does not match replicate')
                    value=number(y['macro_novel_inbin_yield'],'yield');coverage=number(y['score_coverage'],'score coverage')
                    if not 0<=value<=1 or not 0<=coverage<=1:raise ValueError('Invalid yield/coverage')
                    row[f'{teacher}_yield_percent']=100*value;row[f'{teacher}_score_coverage']=coverage
                rr=[mm[seed][method,seed,str(n),'alternate_latent_minus_low'] for n in COUNTS]
                for r in rr:
                    valid=int(r['valid_pairs']);attempted=int(r['attempted_pairs'])
                    if valid<=0 or valid>attempted or not np.isclose(number(r['valid_coverage'],'mask coverage'),valid/attempted,atol=1e-12,rtol=0):
                        raise ValueError('Invalid all-valid pair coverage')
                fractions=[number(r['same_orbit_fraction'],'same-orbit fraction') for r in rr]
                if not all(0<=x<=1 for x in fractions):raise ValueError('Invalid same-orbit fraction')
                row['same_orbit_percent']=100*float(np.mean(fractions))
                row['valid_pair_coverage']=sum(int(r['valid_pairs']) for r in rr)/sum(int(r['attempted_pairs']) for r in rr)
                row.update(latent[family,d,seed]);result.append(row)
    return result


def summarize(points):
    result=[]
    for family,label in FAMILIES:
        for d in DIMENSIONS:
            p=[r for r in points if r['family']==family and r['dimension']==d]
            if len(p)!=3 or len({r['seed'] for r in p})!=3:raise ValueError('Expected three independent matched fit rows')
            row=dict(family=family,label=label,dimension=d,n_fits=3,
                     minimum_valid_pair_coverage=min(r['valid_pair_coverage'] for r in p),
                     minimum_uma_score_coverage=min(r['uma_score_coverage'] for r in p),
                     minimum_mace_score_coverage=min(r['mace_score_coverage'] for r in p))
            for name in ('uma_yield_percent','mace_yield_percent','mean_KL','total_KL','same_orbit_percent'):
                a=np.asarray([r[name] for r in p]);row[f'{name}_mean']=float(a.mean());row[f'{name}_SD']=float(a.std(ddof=1))
            result.append(row)
    return result


def plot(points,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig,axes=plt.subplots(3,3,figsize=(14,10),sharex=True,sharey='row',constrained_layout=True)
    specs=[('uma_yield_percent','UMA','#1261a0',-.13),('mace_yield_percent','MACE','#d26b23',.13)]
    seeds=sorted({r['seed'] for r in points});markers=['o','s','^']
    if len(seeds)!=3:raise ValueError('Exactly three distinct seed markers required')
    for col,(family,label) in enumerate(FAMILIES):
        for i,d in enumerate(DIMENSIONS):
            group=sorted((r for r in points if r['family']==family and r['dimension']==d),key=lambda r:r['seed'])
            if [r['seed'] for r in group]!=seeds:raise ValueError('Plot needs the same three fit seeds per condition')
            for metric,teacher,color,off in specs:
                vals=[r[metric] for r in group];mu=np.mean(vals);sd=np.std(vals,ddof=1);x=i+off
                axes[0,col].errorbar(x,mu,yerr=sd,fmt='D',ms=3,capsize=2,color=color)
                for j,value in enumerate(vals):axes[0,col].scatter(x+(j-1)*.035,value,s=17,marker=markers[j],alpha=.65,color=color)
            for ax,field,color in [(axes[1,col],'mean_KL','#366b43'),(axes[2,col],'same_orbit_percent','#724198')]:
                vals=[r[field] for r in group];ax.errorbar(i,np.mean(vals),yerr=np.std(vals,ddof=1),fmt='D',ms=3,capsize=2,color=color)
                for j,value in enumerate(vals):ax.scatter(i+(j-1)*.035,value,s=17,marker=markers[j],color=color,alpha=.65)
        axes[0,col].set_title(label);axes[2,col].set_ylim(-5,105)
        axes[2,col].set(xticks=range(5),xticklabels=DIMENSIONS,xlabel='Latent dimension')
        for ax in axes[:,col]:ax.grid(axis='y',alpha=.2)
    axes[0,0].set_ylabel('Balanced useful yield (%)');axes[1,0].set_ylabel('Weighted VAL KL\n(nats / dimension)');axes[2,0].set_ylabel('All-valid same orbit (%)')
    handles=[Line2D([],[],color='#1261a0',marker='o',linestyle='None',label='UMA'),
             Line2D([],[],color='#d26b23',marker='o',linestyle='None',label='MACE')]
    handles += [Line2D([],[],color='gray',marker=marker,linestyle='None',label=f'Seed {seed}') for seed,marker in zip(seeds,markers)]
    axes[0,0].legend(handles=handles,frameon=False,fontsize=7,ncol=2)
    fig.suptitle('Development dimension sweeps: points are matched seeds; diamonds ± descriptive SD',fontsize=12)
    fig.savefig(out/'angular_dimension_sweeps.pdf');fig.savefig(out/'angular_dimension_sweeps.png',dpi=160);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--aggregate-receipt',type=Path,required=True);p.add_argument('--mask-receipt',type=Path,action='append',required=True)
    p.add_argument('--fresh-fit-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args(argv)
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    seeds,yields,masks,latent,inputs,sidecars=load_inputs(a.aggregate_receipt,a.mask_receipt,a.fresh_fit_root)
    points=extract(seeds,yields,masks,latent);summary=summarize(points);out=a.output_dir.absolute()
    if out.exists():raise FileExistsError(f'Preserve existing output: {out}')
    out.mkdir(parents=True);save_csv(out/'angular_dimension_points.csv',points);save_csv(out/'angular_dimension_summary.csv',summary);plot(points,out)
    receipt=dict(status='PASS_DEVELOPMENT_ANGULAR_DIMENSION_FIGURE',source_sha256=digest(__file__),
        helper_source=binding(Path(__file__).with_name('development_boundary.py')),
        software=dict(python=sys.version,numpy=np.__version__,matplotlib=__import__('matplotlib').__version__),
        fresh_fit_root=str(a.fresh_fit_root.resolve()),replicates=seeds,inputs=inputs,selected_checkpoint_sidecars=sidecars,
        interpretation={'yield':'Teacher-specific balanced novel in-bin orbit yield per all attempts; equal count/request cells',
          'error_bars':'Descriptive sample SD across three matched fits; not confidence intervals or proof of convergence',
          'VAL_KL':'Weighted VAL posterior KL per latent coordinate, in nats/dimension; total KL also recorded',
          'activity':'Weighted VAL posterior-mean variance > 0.001/0.01/0.1; all thresholds are provisional descriptions',
          'latent_resampling':'Equal composition mean of same orbit among originally valid pairs, including protected identities; no energy use',
          'sidecar_link':'Selected checkpoint and adjacent VAL sidecar separately hashed; adjacency is not a cryptographic sidecar certificate'},
        new_MLIP_calls=0,new_training_updates=0,outputs={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    save_json(out/'receipt.json',receipt);print(json.dumps({'status':receipt['status'],'replicates':seeds,'output_dir':str(out)},indent=2))


if __name__=='__main__':main()

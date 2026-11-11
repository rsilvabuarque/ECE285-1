"""Plot a receipt-bound DEVELOPMENT comparison, never old assessment scores.

One PDF shows matched angular/control differences, radial dimension yields,
weighted VAL posterior KL/activity, and all-valid latent-resampling orbit overlap.
All dots are fit/replicate values; error bars are descriptive sample SD.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from development_boundary import binding, digest, save_csv, save_json

DIMENSIONS=(2,4,8,16,32)
COUNTS=(12,24,36)
TEACHERS=('uma','mace')
CONTRASTS=(('angle_hist_ar','redundant_6_ar','Angle histogram AR − 6-channel AR'),
 ('angle_hist_cvae','redundant_6_cvae','Angle histogram CVAE − 6-channel CVAE'),
 ('angular_smooth_ar','redundant_16_ar','Smooth angles AR − 16-channel AR'),
 ('angular_smooth_cvae','redundant_16_cvae','Smooth angles CVAE − 16-channel CVAE'),
 ('directional_l1_l2_ar','redundant_6_learned_ar','Directional AR − learned 6-channel AR'),
 ('directional_l1_l2_cvae','redundant_6_learned_cvae','Directional CVAE − learned 6-channel CVAE'))


def read_csv(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))


def finite(x, name):
    try:y=float(x)
    except (TypeError, ValueError):raise ValueError(f'Missing {name}') from None
    if not np.isfinite(y):raise ValueError(f'Nonfinite {name}')
    return y


def unique_map(rows, names):
    result={}
    for r in rows:
        key=tuple(r[n] for n in names)
        if key in result:raise ValueError(f'Duplicate input key {key}')
        result[key]=r
    return result


def validate_inputs(aggregate_receipt, mask_receipts):
    """Verify fresh receipts and derive selected-checkpoint/sidecar identities."""
    aggregate_receipt=Path(aggregate_receipt).absolute(); root=aggregate_receipt.parent
    a=json.loads(aggregate_receipt.read_text())
    if a.get('status')!='PASS_DEVELOPMENT_AGGREGATION':raise ValueError('Expected complete development aggregate')
    seeds=tuple(sorted(map(str,a['replicates'])))
    if len(seeds)!=3 or len(set(seeds))!=3:raise ValueError('Exactly three distinct matched replicates required')
    for name in ('paired_control_differences.csv','per_seed_yield.csv'):
        binding(root/name,a['outputs'][name])
    inputs=[binding(aggregate_receipt),binding(root/'paired_control_differences.csv'),binding(root/'per_seed_yield.csv')]
    manifests={}
    for shard in a['shards']:
        seed=str(shard['replicate_id']); ref=shard['manifest'];binding(ref['path'],ref['sha256'])
        if seed in manifests:raise ValueError('Duplicate replicate manifest')
        m=json.loads(Path(ref['path']).read_text())
        if m.get('status')!='FROZEN_DEVELOPMENT_BEFORE_GENERATION':raise ValueError('Manifest is not a development freeze')
        manifests[seed]=(m,ref);inputs.append(ref)
    if set(manifests)!=set(seeds):raise ValueError('Manifest seeds differ from aggregate')
    mask_data={}
    for path in mask_receipts:
        path=Path(path).absolute();r=json.loads(path.read_text())
        if r.get('status')!='PASS':raise ValueError('Mask diagnostic receipt is not PASS')
        manifest=r['manifest'];binding(manifest['path'],manifest['sha256'])
        matches=[s for s,(_,ref) in manifests.items() if ref['sha256']==manifest['sha256']]
        if len(matches)!=1:raise ValueError('Mask diagnostic does not bind exactly one aggregate replicate')
        seed=matches[0]
        if seed in mask_data:raise ValueError('Duplicate mask diagnostic replicate')
        if 'all originally valid masks' not in r.get('scope','').lower():raise ValueError('Mask diagnostic does not describe all-valid scope')
        summary=path.parent/'mask_summary.csv';binding(summary,r['outputs']['mask_summary.csv'])
        mask_data[seed]=read_csv(summary);inputs.extend([binding(path),binding(summary)])
    if set(mask_data)!=set(seeds):raise ValueError('Mask diagnostic replicate coverage differs')
    latent={};sidecar_bindings=[]
    for seed in seeds:
        manifest,_=manifests[seed]
        for d in DIMENSIONS:
            method=f'radial_cvae_d{d}'
            entries=[v for v in manifest['models'] if v['method']==method and str(v['replicate_id'])==seed]
            if len(entries)!=1:raise ValueError('Missing selected radial checkpoint')
            e=entries[0];checkpoint=Path(e['checkpoint'])
            ck=binding(checkpoint,e['expected_checkpoint_sha256']);cfg=e['configuration'];binding(cfg['path'],cfg['sha256'])
            c=json.loads(Path(cfg['path']).read_text())
            if str(c['seed'])!=seed or int(c['latent_dim'])!=d or int(e['declared_latent_dim'])!=d:
                raise ValueError('Checkpoint configuration seed/dimension mismatch')
            sidecar=Path(str(checkpoint)+'.latent.json');v=json.loads(sidecar.read_text());pooled=v['pooled']
            if int(pooled['latent_dim'])!=d or int(pooled['rows'])!=int(c['val_rows']):
                raise ValueError('Sidecar dimension or VAL row count differs from config')
            kl=np.asarray(pooled['KL_nats_by_dimension'],float);variance=np.asarray(pooled['posterior_mean_variance_by_dimension'],float)
            if kl.shape!=(d,) or variance.shape!=(d,) or not np.isfinite(kl).all() or not np.isfinite(variance).all():
                raise ValueError('Invalid weighted VAL latent statistics')
            if np.any(kl < -1e-6) or np.any(variance < 0):raise ValueError('Invalid KL or variance')
            if not np.isclose(kl.mean(),pooled['mean_KL_nats_per_dimension'],rtol=1e-5,atol=1e-8):
                raise ValueError('KL mean does not match component values')
            if not np.isclose(kl.sum(),pooled['total_KL_nats'],rtol=1e-5,atol=1e-8):
                raise ValueError('KL total does not match component values')
            active={str(t):int(np.sum(variance>t)) for t in (0.001,0.01,0.1)}
            if active!=pooled['active_components_by_cutoff']:raise ValueError('Latent activity does not match sidecar variances')
            latent[seed,d]=dict(mean_kl=float(kl.mean()),active_fraction=active['0.001']/d,
                active_001=active['0.001'],active_01=active['0.01'],active_1=active['0.1'],epoch=int(v['epoch']))
            sidecar_bindings.append(dict(seed=seed,dimension=d,checkpoint=ck,configuration=cfg,
                sidecar=binding(sidecar),sidecar_epoch=int(v['epoch']),scope='weighted VAL; checkpoint-adjacent sidecar'))
    return seeds,read_csv(root/'paired_control_differences.csv'),read_csv(root/'per_seed_yield.csv'),mask_data,latent,inputs,sidecar_bindings


def extract_data(seeds, paired_rows, yield_rows, masks, latent):
    paired=unique_map(paired_rows,('teacher','method','reference','replicate_id'))
    yields=unique_map(yield_rows,('teacher','method','replicate_id'))
    points=[]
    for method,reference,label in CONTRASTS:
        for teacher in TEACHERS:
            for seed in seeds:
                r=paired[(teacher,method,reference,seed)];p=finite(r['difference_pp'],'paired difference')
                x=finite(yields[teacher,method,seed]['macro_novel_inbin_yield'],'method yield')
                y=finite(yields[teacher,reference,seed]['macro_novel_inbin_yield'],'control yield')
                if not np.isclose(p,100*(x-y),atol=1e-9,rtol=0):raise ValueError('Paired difference disagrees with aggregate yield rows')
                for field,arm in [('method_fit_seed',method),('reference_fit_seed',reference)]:
                    if r.get(field,'') != yields[teacher,arm,seed].get('fit_seed',''):
                        raise ValueError('Paired fit seed disagrees with the individual arm')
                points.append(dict(panel='angular_comparison',teacher=teacher,method=method,control=reference,
                                   label=label,seed=seed,dimension='',metric='paired_difference_pp',value=p,coverage=''))
    for d in DIMENSIONS:
        method=f'radial_cvae_d{d}'
        for teacher in TEACHERS:
            for seed in seeds:
                r=yields[teacher,method,seed];coverage=finite(r['score_coverage'],'energy score coverage')
                if not 0<=coverage<=1:raise ValueError('Invalid score coverage')
                points.append(dict(panel='radial_yield',teacher=teacher,method=method,control='',label=f'D={d}',seed=seed,
                    dimension=d,metric='balanced_yield_percent',value=100*finite(r['macro_novel_inbin_yield'],'balanced yield'),coverage=coverage))
        for seed in seeds:
            val=latent[seed,d]
            for metric,value in [('weighted_VAL_KL_nats_per_dimension',val['mean_kl']),
                    ('VAL_active_fraction_cutoff_0p001',val['active_fraction']),
                    ('VAL_active_fraction_cutoff_0p01',val['active_01']/d),
                    ('VAL_active_fraction_cutoff_0p1',val['active_1']/d)]:
                points.append(dict(panel='latent_VAL',teacher='',method=method,control='',label=f'D={d}',seed=seed,
                    dimension=d,metric=metric,value=value,coverage=''))
            index=unique_map(masks[seed],('method','replicate_id','n_Li','contrast'))
            rr=[index[method,seed,str(n),'alternate_latent_minus_low'] for n in COUNTS]
            for r in rr:
                if int(r['valid_pairs'])<=0 or int(r['valid_pairs'])>int(r['attempted_pairs']):raise ValueError('Invalid all-valid pair coverage')
                if not np.isclose(finite(r['valid_coverage'],'pair coverage'),int(r['valid_pairs'])/int(r['attempted_pairs']),atol=1e-12,rtol=0):
                    raise ValueError('All-valid pair coverage is inconsistent')
            fractions=[finite(r['same_orbit_fraction'],'same-orbit fraction') for r in rr]
            if not all(0<=x<=1 for x in fractions):raise ValueError('Invalid same-orbit fraction')
            coverage=sum(int(r['valid_pairs']) for r in rr)/sum(int(r['attempted_pairs']) for r in rr)
            points.append(dict(panel='latent_resampling',teacher='',method=method,control='',label=f'D={d}',seed=seed,
                dimension=d,metric='same_orbit_percent',value=100*float(np.mean(fractions)),coverage=coverage))
    return points


def mean_sd(a):
    values=np.asarray(a,float)
    if len(values)!=3 or not np.isfinite(values).all():raise ValueError('Expected exactly three finite replicate points')
    return float(values.mean()),float(values.std(ddof=1))


def plot(points,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'uma':'#1261a0','mace':'#d26b23'}
    fig,axes=plt.subplots(2,2,figsize=(12,9),gridspec_kw={'height_ratios':[1.25,1]},constrained_layout=True)
    ax=axes[0,0]
    for i,(method,ref,label) in enumerate(CONTRASTS):
        for teacher,dy in [('uma',-.13),('mace',.13)]:
            vals=[r['value'] for r in points if r['panel']=='angular_comparison' and r['method']==method and r['teacher']==teacher]
            mean,sd=mean_sd(vals);y=i+dy
            ax.errorbar(mean,y,xerr=sd,fmt='D',ms=3,color=colors[teacher],capsize=2)
            ax.scatter(vals,[y-.025,y,y+.025],s=17,color=colors[teacher],alpha=.65,label=teacher.upper() if i==0 else None)
    ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(len(CONTRASTS)),[c[2] for c in CONTRASTS]);ax.invert_yaxis()
    ax.set(xlabel='Yield difference vs matched control (percentage points)',title='A. Angular features vs matched controls');ax.legend(frameon=False,fontsize=8)
    ax=axes[0,1]
    for i,d in enumerate(DIMENSIONS):
        for teacher,offset in [('uma',-.12),('mace',.12)]:
            vals=[r['value'] for r in points if r['panel']=='radial_yield' and r['dimension']==d and r['teacher']==teacher]
            mean,sd=mean_sd(vals);x=i+offset
            ax.errorbar(x,mean,yerr=sd,fmt='D',ms=3,color=colors[teacher],capsize=2)
            ax.scatter([x-.035,x,x+.035],vals,s=17,color=colors[teacher],alpha=.65,label=teacher.upper() if i==0 else None)
    ax.set(xticks=range(5),xticklabels=DIMENSIONS,xlabel='Radial CVAE latent dimension',ylabel='Balanced useful yield (%)',title='B. Radial dimension study');ax.legend(frameon=False,fontsize=8)
    ax=axes[1,0];other=ax.twinx()
    for i,d in enumerate(DIMENSIONS):
        kl=[r['value'] for r in points if r['panel']=='latent_VAL' and r['dimension']==d and r['metric']=='weighted_VAL_KL_nats_per_dimension']
        act=[100*r['value'] for r in points if r['panel']=='latent_VAL' and r['dimension']==d and r['metric']=='VAL_active_fraction_cutoff_0p001']
        mu,sd=mean_sd(kl);am,asd=mean_sd(act)
        ax.errorbar(i-.1,mu,yerr=sd,fmt='D',ms=3,color='#366b43',capsize=2);ax.scatter([i-.135,i-.1,i-.065],kl,s=17,color='#366b43',alpha=.65)
        other.errorbar(i+.1,am,yerr=asd,fmt='s',ms=3,color='#a54179',capsize=2);other.scatter([i+.065,i+.1,i+.135],act,s=17,color='#a54179',alpha=.65)
    ax.set(xticks=range(5),xticklabels=DIMENSIONS,xlabel='Radial CVAE latent dimension',ylabel='Weighted VAL KL (nats / dimension)',title='C. Posterior diagnostics on VAL')
    other.set_ylabel('Components active at 0.001 (%)',color='#a54179');other.set_ylim(-5,105)
    ax=axes[1,1]
    for i,d in enumerate(DIMENSIONS):
        vals=[r['value'] for r in points if r['panel']=='latent_resampling' and r['dimension']==d]
        mu,sd=mean_sd(vals);ax.errorbar(i,mu,yerr=sd,fmt='D',ms=3,color='#724198',capsize=2)
        ax.scatter([i-.035,i,i+.035],vals,s=17,color='#724198',alpha=.65)
    ax.set(xticks=range(5),xticklabels=DIMENSIONS,xlabel='Radial CVAE latent dimension',ylabel='Same symmetry orbit (%)',title='D. All-valid latent resampling');ax.set_ylim(-5,105)
    for ax in axes.flat:ax.grid(axis='x' if ax is axes[0,0] else 'y',alpha=.2)
    fig.suptitle('Development comparisons: points are matched seeds; diamonds ± descriptive SD',fontsize=12)
    fig.savefig(out/'development_summary.pdf');fig.savefig(out/'development_summary.png',dpi=160);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--aggregate-receipt',type=Path,required=True);p.add_argument('--mask-receipt',type=Path,action='append',required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv)
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    seeds,paired,yields,masks,latent,inputs,sidecars=validate_inputs(a.aggregate_receipt,a.mask_receipt)
    points=extract_data(seeds,paired,yields,masks,latent)
    out=a.output_dir.absolute()
    if out.exists():raise FileExistsError(f'Preserve existing report output: {out}')
    out.mkdir(parents=True);save_csv(out/'development_summary_points.csv',points);plot(points,out)
    receipt={'status':'PASS_DEVELOPMENT_SUMMARY_FIGURE','source_sha256':digest(__file__),
        'helper_source':binding(Path(__file__).with_name('development_boundary.py')),
        'software':{'python':sys.version,'numpy':np.__version__,
                    'matplotlib':__import__('matplotlib').__version__},
        'seeds':seeds,'inputs':inputs,'selected_checkpoint_sidecars':sidecars,
        'interpretation':{'yield':'Mean of equally weighted count/request cells; novel unique in-bin orbits divided by all attempts',
            'error_bars':'Sample SD of three matched replicate values; descriptive, not a confidence interval',
            'VAL_KL':'Weighted validation posterior KL summed over coordinates and divided by latent dimension, in nats per dimension',
            'active':'Weighted variance of VAL posterior mean > 0.001; provisional descriptive threshold, not an established standard',
            'latent_resampling':'Equal composition mean of same parent-symmetry orbit among all originally valid paired draws; energy labels not required',
            'sidecar_link':'Sidecars are adjacent to the manifest-selected checkpoint and match config seed/dimension/VAL rows; their bytes and the checkpoint bytes are separately bound. Adjacency is not an embedded cryptographic certificate.'},
        'new_MLIP_calls':0,'new_training_updates':0,'outputs':{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()}}
    save_json(out/'receipt.json',receipt)
    print(json.dumps({'status':receipt['status'],'seeds':seeds,'output_dir':str(out)},indent=2))


if __name__=='__main__':main()

"""Pristine-reference diagnostics and tie-aware, within-composition rankings.

All fits/statistics describe the inspected pools; no predictive calibration or
independent-MD confidence interval is claimed. Atomic identities are matched only
within a frame, never by subtracting unrelated pristine force vectors.
"""
from __future__ import annotations
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, kendalltau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from predict import HERE, ROOT, TINY, SLIDE

GPA=160.2176634
COLORS={'O3':'#0072B2','spinel':'#D55E00','O1':'#009E73','migration':'#CC79A7'}


def csv_out(frame, path):
    frame.to_csv(path,index=False)


def inversion_counts(a,b):
    """Strict discordant pairs in O(n log n), excluding exact ties in either score."""
    _, ranks=np.unique(b,return_inverse=True); tree=np.zeros(int(ranks.max())+2,dtype=np.int64)
    def count(j):
        out=0
        while j>0: out+=int(tree[j]);j-=j&-j
        return out
    def add(j):
        while j<len(tree): tree[j]+=1;j+=j&-j
    ix=np.argsort(a,kind='stable'); total=0;dis=0;start=0
    while start<len(ix):
        end=start+1
        while end<len(ix) and a[ix[end]]==a[ix[start]]: end+=1
        for k in ix[start:end]: dis+=total-count(int(ranks[k])+1)
        for k in ix[start:end]: add(int(ranks[k])+1)
        total+=end-start;start=end
    pairs=lambda counts:sum(int(n)*(int(n)-1)//2 for n in counts)
    nt=len(a)*(len(a)-1)//2
    td=pairs(np.unique(a,return_counts=True)[1]);tm=pairs(np.unique(b,return_counts=True)[1])
    tb=pairs(np.unique(np.stack([a,b],axis=1),axis=0,return_counts=True)[1])
    return dis,nt-td-tm+tb,td,tm


def group_metrics(g, mlcol='uma_energy_eV'):
    a=g.dft_energy_eV.to_numpy();b=g[mlcol].to_numpy(); n=len(g)
    ra=rankdata(a);rb=rankdata(b)
    nlow=max(1,int(math.ceil(.2*n)))
    sa=a<=np.sort(a)[nlow-1];sb=b<=np.sort(b)[nlow-1]
    intersection=int(np.sum(sa&sb));union=int(np.sum(sa|sb));best=b==b.min()
    dis,pairs,td,tm=inversion_counts(a,b)
    rmse=lambda v:float(np.sqrt(np.mean(v*v)))
    delta=b-a; nc=int(g.n_Co.iloc[0]);na=int(g.n_atoms.iloc[0])
    return dict(n=n,n_unique_DFT_energies=len(np.unique(a)),n_unique_MLIP_energies=len(np.unique(b)),
                exact_order_match=bool(np.array_equal(ra,rb)),informative_ordering=bool(pairs>0),
                spearman_rho=float(spearmanr(a,b).statistic) if n>1 and np.ptp(a)>0 and np.ptp(b)>0 else np.nan,
                kendall_tau_b=float(kendalltau(a,b).statistic) if n>1 and np.ptp(a)>0 and np.ptp(b)>0 else np.nan,
                discordant_pairs=dis,strict_pairs=pairs,inversion_fraction=dis/pairs if pairs else np.nan,
                DFT_tied_pairs=td,MLIP_tied_pairs=tm,
                DFT_bottom20_n=int(sa.sum()),MLIP_bottom20_n=int(sb.sum()),
                bottom20_precision=intersection/int(sb.sum()),bottom20_recall=intersection/int(sa.sum()),
                bottom20_Jaccard=intersection/union,
                selection_regret_meV_atom=float((a[best].mean()-a.min())/na*1000),
                selection_regret_meV_CoO2=float((a[best].mean()-a.min())/nc*1000),
                signed_bias_eV_cell=float(delta.mean()),error_sd_eV_cell=float(delta.std()),
                centered_rmse_meV_atom=rmse(delta-delta.mean())/na*1000,
                DFT_energy_span_meV_atom=float(np.ptp(a)/na*1000))


def rank_tables(df,out,name,mlcol='uma_energy_eV'):
    metrics=[];ranked=[]
    for extra in ([],['cohort'],['temperature_K'] if name.startswith('historical') else []):
        if not extra and metrics: continue
        keys=['phase','n_Co','n_Li']+extra
        for key,g in df.groupby(keys,dropna=False,sort=True):
            if not isinstance(key,tuple):key=(key,)
            info=dict(zip(keys,key));info['grouping']='phase_composition'+('+'+extra[0] if extra else '')
            metrics.append({**info,**group_metrics(g,mlcol)})
            if not extra:
                g=g.copy();g['DFT_rank']=rankdata(g.dft_energy_eV);g['MLIP_rank']=rankdata(g[mlcol])
                ranked.append(g.sort_values(['DFT_rank','structure_id']))
    result=pd.DataFrame(metrics)
    csv_out(result,out/f'{name}_rank_metrics.csv')
    csv_out(pd.concat(ranked),out/f'{name}_ranked_structures.csv')
    return result[result.grouping=='phase_composition']


def offset_fits(df,out,name):
    results=[]
    for key,g in df.groupby(['phase','cohort','n_Co']):
        d=(g.uma_energy_eV-g.dft_energy_eV).to_numpy();n=g.n_atoms.to_numpy();li=g.n_Li.to_numpy()
        models={'constant_total':np.ones((len(g),1)),'constant_per_atom':n[:,None]}
        if len(np.unique(li))>1:models['CoO2_host_plus_Li_reference']=np.stack([np.ones(len(g)),li],axis=1)
        for model,X in models.items():
            beta=np.linalg.lstsq(X,d,rcond=None)[0];r=d-X@beta
            results.append(dict(phase=key[0],cohort=key[1],n_Co=key[2],n=len(g),model=model,
                                coefficients_eV=json.dumps(beta.tolist()),
                                residual_RMSE_eV_cell=float(np.sqrt(np.mean(r*r))),
                                residual_RMSE_meV_atom=float(np.sqrt(np.mean((r/n)**2)))*1000,
                                note='All-data descriptive fit, not predictive correction or Hamiltonian alignment'))
    csv_out(pd.DataFrame(results),out/f'{name}_offset_models.csv')


def anchors(df, historical):
    """Historical: matched-T/P pristine ensemble; Hu: actual optimized O3 x=1 frame."""
    refs=[]; z=df.copy()
    for q in ('energy','force','pressure'):
        for m in ('dft','uma'):z[f'delta_{m}_{q}']=np.nan
    if historical:
        groups=list(z.groupby(['temperature_K','pressure_MPa']))
    else:
        groups=[(('O3_optimized_pristine',),z)]
    for key,g in groups:
        p=g[g.x==1] if historical else g[(g.phase=='O3')&(g.cohort=='optimized')&(g.x==1)]
        if p.empty:continue
        ref=dict(reference_group=str(key),n_pristine=len(p),
                 anchor='matched-T/P pristine ensemble mean' if historical else 'single archived optimized O3 Li12Co12O24 frame; common O3 endpoint, not phase-specific pristine')
        for m in ('dft','uma'):
            e=float((p[f'{m}_energy_eV']/p.n_Co).mean());f=float(p[f'{m}_force_rms'].mean());pr=float(p[f'{m}_pressure_eV_A3'].mean())
            ref.update({f'{m}_energy_eV_CoO2':e,f'{m}_force_RMS_eV_A':f,f'{m}_pressure_GPa':pr*GPA})
            z.loc[g.index,f'delta_{m}_energy']=g[f'{m}_energy_eV']/g.n_Co-e
            z.loc[g.index,f'delta_{m}_force']=g[f'{m}_force_rms']-f
            z.loc[g.index,f'delta_{m}_pressure']=g[f'{m}_pressure_eV_A3']*GPA-pr*GPA
        refs.append(ref)
    return z,pd.DataFrame(refs)


def parity(ax,x,y,c,label):
    ax.scatter(x,y,c=c,cmap='viridis',norm=Normalize(0,1),s=8,alpha=.45,rasterized=True)
    lo=min(float(np.min(x)),float(np.min(y)));hi=max(float(np.max(x)),float(np.max(y)))
    pad=max((hi-lo)*.04,1e-5);ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],'k--',lw=.8)
    ax.set_xlabel('DFT '+label);ax.set_ylabel('UMA '+label);ax.grid(alpha=.15)


def save_fig(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def reference_figure(df,out,name,historical):
    z,refs=anchors(df,historical)
    csv_out(refs,out/f'{name}_pristine_references.csv')
    csv_out(z,out/f'{name}_pristine_referenced_frames.csv')
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    cmap=plt.cm.viridis
    labels={'energy':'ΔE (eV/CoO₂)','force':'Δ force RMS (eV/Å)','pressure':'Δ pressure (GPa)'}
    good=z.dropna(subset=['delta_dft_energy'])
    for j,q in enumerate(('energy','force','pressure')):
        parity(axes[0,j],good[f'delta_dft_{q}'],good[f'delta_uma_{q}'],good.x,labels[q])
    for phase,g in z.groupby('phase'):
        color=COLORS[phase]
        for li,h in g.groupby('n_Li'):
            x=float(h.x.iloc[0]); dx=(h.delta_uma_energy-h.delta_dft_energy)*1000
            vals=[dx,h.force_rmse,h.stress_rmse*GPA]
            for j,v in enumerate(vals):
                axes[1,j].scatter(np.full(len(h),x),v,color=color,s=5,alpha=.10,rasterized=True)
                axes[1,j].errorbar(x,v.mean(),yerr=v.std(ddof=0),fmt='o',color=color,ms=4,capsize=3)
        for ax in axes[1]:ax.plot([],[],'o',color=color,label=phase)
    for ax,label in zip(axes[1],('Pristine-referenced energy error\n(meV/CoO₂)','Absolute force component RMSE\n(eV/Å)','Absolute six-component stress RMSE\n(GPa)')):
        ax.set_xlabel('Li fraction x in LiₓCoO₂');ax.set_ylabel(label);ax.grid(alpha=.15)
    axes[1,0].axhline(0,color='k',ls='--',lw=.8)
    axes[1,2].legend(frameon=False)
    title='Historical VASP snapshots: pinned UMA/omat replay' if historical else 'Hu archival DFT labels: new UMA/omat predictions (exact-geometry duplicates removed)'
    fig.suptitle(title,fontsize=15,y=.99)
    if historical:
        p=z[z.x==1]
        info=(f'Pristine pooled DFT/UMA: E={(p.dft_energy_eV/p.n_Co).mean():.5f}/{(p.uma_energy_eV/p.n_Co).mean():.5f} eV/CoO₂; '
              f'force RMS={p.dft_force_rms.mean():.4f}/{p.uma_force_rms.mean():.4f} eV/Å; '
              f'P={p.dft_pressure_eV_A3.mean()*GPA:.3f}/{p.uma_pressure_eV_A3.mean()*GPA:.3f} GPa.\n'
              'Subtraction uses separate pristine means at each recorded T/P; per-group values in CSV. Bars: frame SD, not CI.')
    else:
        r=refs.iloc[0]
        info=(f'O3 pristine: E DFT/UMA={r.dft_energy_eV_CoO2:.5f}/{r.uma_energy_eV_CoO2:.5f} eV/CoO₂; '
              f'force RMS={r.dft_force_RMS_eV_A:.4f}/{r.uma_force_RMS_eV_A:.4f} eV/Å; '
              f'P={r.dft_pressure_GPa:.3f}/{r.uma_pressure_GPa:.3f} GPa.\n'
              'Common O3 endpoint; spinel/O1/migration lack their own x=1. Bars: frame SD, not CI. Hu virial convention assumed from format.')
    fig.text(.5,.015,info+'\nForce RMS changes are scalar activity differences, not cross-frame force vectors. Bottom F/stress errors remain absolute.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.07,right=.89,top=.91,bottom=.15,hspace=.38,wspace=.36)
    cax=fig.add_axes([.925,.62,.012,.25])
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap=cmap),cax=cax,label='Li fraction x')
    save_fig(fig,out,name+'_pristine_referenced')
    return z,refs


def ranking_figure(df,out,name,metrics,mlcol='uma_energy_eV'):
    groups=list(df.groupby(['phase','n_Co','n_Li'],sort=True));ncols=4;nrows=math.ceil(len(groups)/ncols)
    fig,axes=plt.subplots(nrows,ncols,figsize=(17,3.2*nrows),squeeze=False)
    for ax,((phase,nc,nl),g) in zip(axes.flat,groups):
        ra=rankdata(g.dft_energy_eV);rb=rankdata(g[mlcol]);n=len(g)
        if name=='hu':
            for cohort,marker,color in [('DPGEN','o','#0072B2'),('optimized','^','#D55E00'),('migration','o','#CC79A7')]:
                ix_cohort=(g.cohort==cohort).to_numpy()
                ax.scatter(ra[ix_cohort],rb[ix_cohort],s=10 if cohort=='optimized' else 7,marker=marker,color=color,alpha=.6 if cohort=='optimized' else .35,rasterized=True)
        else:ax.scatter(ra,rb,s=8,color=COLORS[phase],alpha=.45,rasterized=True)
        ax.plot([1,max(n,2)],[1,max(n,2)],'k--',lw=.7)
        ix=metrics[(metrics.phase==phase)&(metrics.n_Co==nc)&(metrics.n_Li==nl)].iloc[0]
        title=f'{phase}: x={nl/nc:.4f}, n={n}\nρ={ix.spearman_rho:.3f}; τ={ix.kendall_tau_b:.3f}; top20 P={ix.bottom20_precision:.2f}'
        opt=g[g.cohort=='optimized']
        if name=='hu' and len(opt)>1:
            om=group_metrics(opt,mlcol)
            title+=f'\nOptimized only: n={len(opt)}, ρ={om["spearman_rho"]:.3f}, P={om["bottom20_precision"]:.2f}'
        ax.set_title(title,fontsize=8)
        # Full structure_N labels live in the ranked CSV; annotate selected candidates here.
        labels=set(np.argsort(g.dft_energy_eV.to_numpy())[:2])|set(np.argsort(g[mlcol].to_numpy())[:1])
        if n<=12:labels=set(range(n))
        for j in labels:
            ax.annotate(g.structure_id.iloc[j].replace('structure_','s'),(ra[j],rb[j]),fontsize=5,xytext=(2,3),textcoords='offset points')
        ax.set_xlabel('DFT rank (1 = lowest E)',fontsize=8);ax.set_ylabel('UMA rank',fontsize=8);ax.grid(alpha=.15);ax.tick_params(labelsize=7)
    for ax in axes.flat[len(groups):]:ax.set_visible(False)
    title='Original slide data: energy ordering by lithiation' if name=='historical' else 'Hu data: energy ordering within phase and composition'
    subtitle='Exact ties use average ranks; sN = structure_N; full identities and ordered lists are in the ranked CSV.'
    if name=='hu':subtitle+='\nBlue circles: DPGEN; orange triangles: optimized; purple: migration. Pooled frame ranks; optimized-only metrics are separate.'
    fig.suptitle(title+'\n'+subtitle,fontsize=12,y=1.0)
    fig.tight_layout(rect=(0,0,1,.965 if nrows<3 else .98))
    save_fig(fig,out,name+'_energy_rankings')


def cached_finetuned(out):
    u=TINY/'uma_finetune_workflow/live_checkpoint_analysis'
    frames=[]
    for label,folder in [('EF','analysis_full_completed_gpu_20260320_041048_job50282412_final'),('EFS','analysis_full_completed_gpu_efs_52418183_final')]:
        f=u/folder/'pre_post_metrics.csv'
        if not f.exists():continue
        z=pd.read_csv(f);z['n_Co']=24;z['n_Li']=np.rint(z.lithiation_pct/100*24).astype(int);z['phase']='O3'
        z['dft_energy_eV']=z.vasp_energy_eV
        for model,col in [('baseline','base_energy_eV'),('finetuned','ft_energy_eV')]:
            for nl,g in z.groupby('n_Li'):
                if 'natoms' not in g:g=g.assign(n_atoms=72+nl)
                else:g=g.assign(n_atoms=g.natoms)
                frames.append(dict(checkpoint=label,model=model,n_Li=nl,x=nl/24,**group_metrics(g,col)))
    if frames:csv_out(pd.DataFrame(frames),out/'cached_finetuned_test_rank_metrics.csv')


def main():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME','').startswith('nid'):
        raise RuntimeError('Run analysis in the compute allocation')
    out=HERE/'results';df=pd.read_csv(out/'frame_metrics.csv')
    unique=df.drop_duplicates(['dataset','geometry_key']).copy()
    summaries={}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for name in ('historical','hu'):
        g=unique[unique.dataset==name].copy()
        z,refs=reference_figure(g,out,name,name=='historical')
        offset_fits(g,out,name)
        metric=rank_tables(g,out,name)
        if name=='historical':
            old=g.copy();old['uma_energy_eV']=old.cached_uma_eV
            metric=rank_tables(old,out,'historical_cached')
            from independent_check import cached_reference
            independent=cached_reference()
            (out/'independent_cached_energy_check.json').write_text(json.dumps(independent,indent=2)+'\n')
            for check in independent:
                nl=round(check['lithiation_pct']*.24)
                actual=metric[metric.n_Li==nl].iloc[0]
                assert actual.discordant_pairs==check['inversions']
                assert actual.strict_pairs==check['strict_pairs']
                assert abs(actual.spearman_rho-check['rho'])<1e-12
                assert abs(actual.bottom20_precision-check['bottom20_precision'])<1e-12
            (out/'independent_validation.json').write_text(json.dumps(dict(status='PASS',
                checks='All seven cached-energy groups agree with independent standard-library average-rank correlations and quadratic pair inversion counts; 1e-12 is an arithmetic check, not a physical tolerance.'),indent=2)+'\n')
            ranking_figure(old,out,'historical',metric)
            replay=g[['structure_id','case_id','x','cached_uma_eV','uma_energy_eV']].copy()
            replay['fresh_minus_cached_eV']=replay.uma_energy_eV-replay.cached_uma_eV
            original=pd.read_csv(SLIDE).set_index('case_id')
            replay['cached_force_vector_RMS_error']=replay.case_id.map(original.rms_force_difference_eV_per_A)
            replay['fresh_force_vector_RMS_error']=g.force_rmse*np.sqrt(3)
            replay['fresh_minus_cached_force_error']=replay.fresh_force_vector_RMS_error-replay.cached_force_vector_RMS_error
            csv_out(replay,out/'historical_replay.csv')
        else:ranking_figure(g,out,name,metric)
        stats=[]
        for (phase,nc,nl),h in z.groupby(['phase','n_Co','n_Li']):
            stats.append(dict(phase=phase,n_Co=nc,n_Li=nl,x=nl/nc,n=len(h),
                              raw_bias_meV_atom=float(((h.uma_energy_eV-h.dft_energy_eV)/h.n_atoms*1000).mean()),
                              pristine_referenced_bias_meV_CoO2=float(((h.delta_uma_energy-h.delta_dft_energy)*1000).mean()),
                              force_component_RMSE_eV_A=float(np.sqrt(np.average(h.force_rmse**2,weights=h.n_atoms))),
                              stress_component_RMSE_GPa=float(np.sqrt(np.mean(h.stress_rmse**2))*GPA)))
        csv_out(pd.DataFrame(stats),out/f'{name}_composition_summary.csv')
        summaries[name]=dict(raw_n=int(np.sum(df.dataset==name)),unique_n=len(g),pristine_reference_count=len(refs),
                              rank_groups=len(metric),informative_rank_groups=int(metric.informative_ordering.sum()),
                              groups_exact_order=int((metric.exact_order_match & metric.informative_ordering).sum()),
                              median_rho=float(metric.loc[metric.informative_ordering,'spearman_rho'].median()),
                              median_bottom20_precision=float(metric.loc[metric.informative_ordering,'bottom20_precision'].median()))
    cached_finetuned(out)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':main()

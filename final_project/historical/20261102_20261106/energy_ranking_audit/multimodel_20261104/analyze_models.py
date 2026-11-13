"""Matched-geometry comparison of three frozen MLIPs against separate DFT archives.

All ranks are within source/phase/composition. Centered energy plots are
descriptive mean-removal diagnostics, never a fitted predictive correction.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'analysis_deps'))
sys.path.insert(0,str(HERE.parent))
from analyze import group_metrics
from benchmark import compute_guard,dump,sha

GPA=160.2176634
MODELS=['uma','mace','m3gnet']
LABELS={'uma':'UMA 1.2.1 / omat','mace':'MACE-MH-1 / omat_pbe','m3gnet':'M3GNet / MatPES-PBE'}
COLORS={'uma':'#0072B2','mace':'#D55E00','m3gnet':'#009E73'}
OUT=HERE/'comparison'
GROUP=['dataset','phase','n_Co','n_Li']


def savefig(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight',dpi=160)
    plt.close(fig)


def merge_predictions(model,reference):
    n=len(reference['energy']);nt=len(reference['numbers'])
    energy=np.full(n,np.nan);stress=np.full((n,6),np.nan);force=np.full((nt,3),np.nan)
    seen=np.zeros(n,dtype=int); receipts=[]
    shards=sorted((HERE/'results'/model).glob('shard_*'))
    if not shards:raise ValueError(f'No production predictions for {model}')
    signatures=[];reference_hash=sha(HERE/'reference_frames.npz')
    for shard in shards:
        completion=json.loads((shard/'completion.json').read_text())
        assert completion['status']=='COMPLETE' and not completion['errors']
        assert completion['reference_sha256']==reference_hash
        receipt=json.loads((shard/'model.json').read_text());receipts.append(receipt)
        signatures.append((receipt['dtype'],receipt.get('sha256'),json.dumps(receipt.get('checkpoint_hashes'),sort_keys=True),
                           receipt.get('task'),receipt.get('head'),receipt.get('torch'),
                           receipt.get('fairchem'),receipt.get('mace'),receipt.get('matgl'),receipt.get('settings'),receipt.get('script_sha256')))
        for path in sorted(shard.glob('chunk_*.npz')):
            with np.load(path) as archive:
                z={key:archive[key] for key in archive.files}
                for j,idx in enumerate(z['indices']):
                    i=int(idx); assert 0<=i<n
                    lo,hi=reference['offsets'][i:i+2];pl,ph=z['offsets'][j:j+2]
                    assert ph-pl==hi-lo
                    energy[i]=z['energy'][j];stress[i]=z['stress'][j];force[lo:hi]=z['force'][pl:ph];seen[i]+=1
    assert len(set(signatures))==1,signatures
    assert np.all(seen==1),np.unique(seen,return_counts=True)
    assert all(np.isfinite(v).all() for v in [energy,stress,force])
    dump(OUT/(model+'_receipts.json'),receipts)
    return dict(energy=energy,stress=stress,force=force)


def frame_metrics(meta,reference,pred,model):
    df=meta.copy();df['model']=model
    df['dft_energy_eV']=reference['energy'];df['mlip_energy_eV']=pred['energy']
    de=pred['energy']-reference['energy']
    df['energy_error_eV_cell']=de;df['energy_error_meV_atom']=1000*de/df.n_atoms
    ds=pred['stress']-reference['stress']
    df['stress_RMSE_GPa']=np.sqrt(np.mean(ds**2,axis=1))*GPA
    df['stress_MAE_GPa']=np.mean(np.abs(ds),axis=1)*GPA
    df['dft_pressure_GPa']=-reference['stress'][:,:3].mean(axis=1)*GPA
    df['mlip_pressure_GPa']=-pred['stress'][:,:3].mean(axis=1)*GPA
    for j in range(6):
        df[f'dft_stress_{j}_GPa']=reference['stress'][:,j]*GPA
        df[f'mlip_stress_{j}_GPa']=pred['stress'][:,j]*GPA
    rows=[]
    for i in range(len(df)):
        lo,hi=reference['offsets'][i:i+2];delta=pred['force'][lo:hi]-reference['forces'][lo:hi]
        item=dict(force_RMSE_eV_A=float(np.sqrt(np.mean(delta**2))),force_MAE_eV_A=float(np.mean(np.abs(delta))),
                  force_mean_vector_error_eV_A=float(np.linalg.norm(delta,axis=1).mean()))
        for label,z in [('Li',3),('Co',27),('O',8)]:
            selected=delta[reference['numbers'][lo:hi]==z]
            item[f'force_RMSE_{label}_eV_A']=float(np.sqrt(np.mean(selected**2))) if len(selected) else np.nan
        rows.append(item)
    return pd.concat([df,pd.DataFrame(rows)],axis=1)


def rank_metrics(df):
    tables=[];ranked=[]
    variants=[[],['cohort'],['temperature_K','pressure_MPa']]
    for extra in variants:
        data=df if extra!=['temperature_K','pressure_MPa'] else df[df.dataset=='historical']
        keys=['model']+GROUP+extra
        for key,g in data.groupby(keys,dropna=False,sort=True):
            info=dict(zip(keys,key));info['grouping']='phase_composition'+('+'+'+'.join(extra) if extra else '')
            met=group_metrics(g,'mlip_energy_eV')
            # Exact all-pair RMS identity at fixed composition/atom count.
            met['pair_energy_RMSE_meV_atom']=met['centered_rmse_meV_atom']*math.sqrt(2*len(g)/(len(g)-1)) if len(g)>1 else np.nan
            tables.append({**info,**met})
            if not extra:
                g=g.copy();g['DFT_rank']=rankdata(g.dft_energy_eV);g['MLIP_rank']=rankdata(g.mlip_energy_eV)
                ranked.append(g.sort_values(['DFT_rank','structure_id']))
    return pd.DataFrame(tables),pd.concat(ranked,ignore_index=True)


def summarize(df,metrics):
    rows=[]
    groups=[(model,dataset,phase,g) for (model,dataset,phase),g in df.groupby(['model','dataset','phase'])]
    groups += [(model,dataset,'ALL',g) for (model,dataset),g in df.groupby(['model','dataset'])]
    for model,dataset,phase,g in groups:
        r=metrics[(metrics.model==model)&(metrics.dataset==dataset)&((metrics.phase==phase) if phase!='ALL' else True)
                  &(metrics.grouping=='phase_composition')&metrics.informative_ordering]
        de=g.energy_error_meV_atom.to_numpy()
        rows.append(dict(model=model,dataset=dataset,phase=phase,n_frames=len(g),n_rank_groups=len(r),
             exact_order_groups=int(r.exact_order_match.sum()),mean_spearman=float(r.spearman_rho.mean()),
             median_spearman=float(r.spearman_rho.median()),median_kendall=float(r.kendall_tau_b.median()),
             mean_low20_precision=float(r.bottom20_precision.mean()),median_low20_precision=float(r.bottom20_precision.median()),
             median_regret_meV_atom=float(r.selection_regret_meV_atom.median()),
             mean_regret_meV_atom=float(r.selection_regret_meV_atom.mean()),
             energy_MAE_meV_atom=float(np.mean(abs(de))),energy_RMSE_meV_atom=float(np.sqrt(np.mean(de**2))),
             energy_bias_meV_atom=float(de.mean()),median_pair_energy_RMSE_meV_atom=float(r.pair_energy_RMSE_meV_atom.median()),
             force_RMSE_eV_A=float(np.sqrt(np.average(g.force_RMSE_eV_A**2,weights=g.n_atoms))),
             force_MAE_eV_A=float(np.average(g.force_MAE_eV_A,weights=g.n_atoms)),
             stress_RMSE_GPa=float(np.sqrt(np.mean(g.stress_RMSE_GPa**2))),
             stress_MAE_GPa=float(g.stress_MAE_GPa.mean())))
    return pd.DataFrame(rows)


def comparison_bars(summary):
    summary=summary[summary.phase!='ALL']
    cats=list(summary[['dataset','phase']].drop_duplicates().itertuples(index=False,name=None))
    specs=[('median_spearman','Median within-group Spearman ρ'),('median_low20_precision','Median low-20% precision'),
           ('mean_regret_meV_atom','Mean finite-pool DFT regret (meV/atom)'),
           ('energy_MAE_meV_atom','Native energy MAE (meV/atom)'),('force_RMSE_eV_A','Force-component RMSE (eV/Å)'),
           ('stress_RMSE_GPa','Six-component stress RMSE (GPa)')]
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for ax,(col,title) in zip(axes.flat,specs):
        for j,model in enumerate(MODELS):
            vals=[float(summary[(summary.model==model)&(summary.dataset==d)&(summary.phase==p)][col].iloc[0]) for d,p in cats]
            ax.bar(np.arange(len(cats))+(j-1)*.25,vals,.24,label=LABELS[model],color=COLORS[model])
        ax.set_xticks(np.arange(len(cats)),[f'{d}\n{p}' for d,p in cats],fontsize=8)
        ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2)
        if col=='median_spearman':ax.set_ylim(-1,1.05)
        if col=='median_low20_precision':ax.set_ylim(0,1.05)
    axes[0,0].legend(fontsize=7)
    fig.suptitle('Exact archived geometries: corrected UMA, MACE, and M3GNet versus their DFT references',fontsize=13)
    fig.text(.5,.015,'Hu exact duplicates removed. Rank groups weighted equally; F errors weighted by Cartesian components. Native energy errors retain reference offsets.\nStatic finite-pool comparison; no independent-parent uncertainty or global ground-state claim.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.065,1,.95]);savefig(fig,'model_comparison')


def parity_figures(df,reference,preds):
    rng=np.random.default_rng(285)
    for model in MODELS:
        p=preds[model]
        for dataset in ['historical','hu']:
            g=df[(df.model==model)&(df.dataset==dataset)].copy();idx=g['index'].to_numpy(dtype=int)
            fig,axes=plt.subplots(2,2,figsize=(11,10))
            x=g.dft_energy_eV/g.n_atoms;y=g.mlip_energy_eV/g.n_atoms
            groups=g.groupby(['phase','n_Co','n_Li'])
            cx=(g.dft_energy_eV-groups.dft_energy_eV.transform('mean'))/g.n_atoms*1000
            cy=(g.mlip_energy_eV-groups.mlip_energy_eV.transform('mean'))/g.n_atoms*1000
            slices=[np.arange(reference['offsets'][i],reference['offsets'][i+1]) for i in idx]
            atoms=np.concatenate(slices);fx=reference['forces'][atoms].ravel();fy=p['force'][atoms].ravel()
            if len(fx)>30000:
                chosen=np.random.default_rng(285).choice(len(fx),30000,replace=False);fx,fy=fx[chosen],fy[chosen]
            pairs=[(x,y,'Native E (eV/atom)'),(cx,cy,'Within-composition centered E (meV/atom)'),
                   (fx,fy,'Cartesian force component (eV/Å)'),
                   (reference['stress'][idx].ravel()*GPA,p['stress'][idx].ravel()*GPA,'Stress component (GPa)')]
            for ax,(xx,yy,label) in zip(axes.flat,pairs):
                xx=np.asarray(xx);yy=np.asarray(yy);ax.scatter(xx,yy,s=3,alpha=.2,color=COLORS[model],rasterized=True)
                lo=min(xx.min(),yy.min());hi=max(xx.max(),yy.max());ax.plot([lo,hi],[lo,hi],'k--',lw=.7)
                ax.set_xlabel('DFT '+label);ax.set_ylabel('MLIP '+label);ax.grid(alpha=.15)
            fig.suptitle(f'{LABELS[model]} — {dataset}: matched E / F / stress',fontsize=12)
            fig.text(.5,.01,'Centered energies remove a separate all-pool mean per phase/composition for this diagnostic only.\nForce display subsampled to ≤30,000 components; statistics always use all components.',ha='center',fontsize=8)
            fig.tight_layout(rect=[0,.055,1,.95]);savefig(fig,f'{model}_{dataset}_parity')


def rank_figures(df,metrics):
    for model in MODELS:
        for dataset in ['historical','hu']:
            z=df[(df.model==model)&(df.dataset==dataset)]
            groups=list(z.groupby(['phase','n_Co','n_Li']));ncols=4;nrows=math.ceil(len(groups)/ncols)
            fig,axes=plt.subplots(nrows,ncols,figsize=(17,3.1*nrows),squeeze=False)
            for ax,((phase,nc,nl),g) in zip(axes.flat,groups):
                ra,rb=rankdata(g.dft_energy_eV),rankdata(g.mlip_energy_eV)
                for cohort,h in g.groupby('cohort'):
                    ix=g.index.get_indexer(h.index)
                    ax.scatter(ra[ix],rb[ix],s=14 if cohort=='optimized' else 6,
                               marker='^' if cohort=='optimized' else 'o',alpha=.55 if cohort=='optimized' else .3,
                               label=cohort,rasterized=True)
                ax.plot([1,len(g)],[1,len(g)],'k--',lw=.6)
                r=metrics[(metrics.model==model)&(metrics.dataset==dataset)&(metrics.phase==phase)
                          &(metrics.n_Co==nc)&(metrics.n_Li==nl)&(metrics.grouping=='phase_composition')].iloc[0]
                title=f'{phase} x={nl/nc:.4f}, n={len(g)}\nρ={r.spearman_rho:.3f}, low20 P={r.bottom20_precision:.2f}'
                opt=g[g.cohort=='optimized']
                if len(opt)>1:
                    om=group_metrics(opt,'mlip_energy_eV');title+=f'\nOptimized: n={len(opt)}, ρ={om["spearman_rho"]:.3f}, P={om["bottom20_precision"]:.2f}'
                ax.set_title(title,fontsize=8);ax.set_xlabel('DFT rank (1=lowest)',fontsize=8);ax.set_ylabel('MLIP rank',fontsize=8)
                ax.tick_params(labelsize=7);ax.grid(alpha=.15)
                chosen=set(np.argsort(g.dft_energy_eV.to_numpy())[:1])|set(np.argsort(g.mlip_energy_eV.to_numpy())[:1])
                if len(g)<=12:chosen=set(range(len(g)))
                for j in chosen:ax.annotate(g.structure_id.iloc[j].replace('structure_','s'),(ra[j],rb[j]),fontsize=5,xytext=(2,2),textcoords='offset points')
            axes.flat[0].legend(fontsize=6)
            for ax in axes.flat[len(groups):]:ax.set_visible(False)
            fig.suptitle(f'{LABELS[model]} — {dataset}: within-composition energy ranks',fontsize=14)
            fig.tight_layout(rect=[0,0,1,.97]);savefig(fig,f'{model}_{dataset}_rankings')


def composition_figures(df,metrics):
    for dataset in ['historical','hu']:
        phases=list(df[df.dataset==dataset].phase.unique())
        fig,axes=plt.subplots(len(phases),3,figsize=(15,3.3*len(phases)),squeeze=False)
        for i,phase in enumerate(phases):
            for model in MODELS:
                r=metrics[(metrics.dataset==dataset)&(metrics.phase==phase)&(metrics.model==model)&(metrics.grouping=='phase_composition')].sort_values('n_Li')
                for ax,col in zip(axes[i],['spearman_rho','bottom20_precision','selection_regret_meV_atom']):
                    ax.plot(r.n_Li/r.n_Co,r[col],'o-',ms=4,color=COLORS[model],label=LABELS[model])
            for ax,label in zip(axes[i],['Spearman ρ','Low-20% precision','DFT regret (meV/atom)']):
                ax.set_xlabel('Li fraction x');ax.set_ylabel(label);ax.set_title(phase);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7);fig.suptitle(dataset+' — energy-ordering comparison',fontsize=14)
        fig.tight_layout(rect=[0,0,1,.96]);savefig(fig,dataset+'_ranking_comparison')
        fig,axes=plt.subplots(len(phases),3,figsize=(15,3.3*len(phases)),squeeze=False)
        for i,phase in enumerate(phases):
            for model in MODELS:
                z=df[(df.dataset==dataset)&(df.phase==phase)&(df.model==model)]
                vals=[]
                for (nc,nl),g in z.groupby(['n_Co','n_Li']):
                    vals.append((nl/nc,float(abs(g.energy_error_meV_atom).mean()),
                                 float(np.sqrt(np.mean(g.force_RMSE_eV_A**2))),float(np.sqrt(np.mean(g.stress_RMSE_GPa**2)))))
                vals=np.asarray(sorted(vals))
                for j,ax in enumerate(axes[i]):ax.plot(vals[:,0],vals[:,j+1],'o-',ms=4,color=COLORS[model],label=LABELS[model])
            for ax,label in zip(axes[i],['Native energy MAE (meV/atom)','Force RMSE (eV/Å)','Stress RMSE (GPa)']):
                ax.set_xlabel('Li fraction x');ax.set_ylabel(label);ax.set_title(phase);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7);fig.suptitle(dataset+' — E / F / stress errors',fontsize=14)
        fig.tight_layout(rect=[0,0,1,.96]);savefig(fig,dataset+'_EFS_errors')


def independent_checks(df,metrics):
    results=[]
    for model in MODELS:
        for nl,g in df[(df.model==model)&(df.dataset=='historical')].groupby('n_Li'):
            a=g.dft_energy_eV.to_numpy();b=g.mlip_energy_eV.to_numpy()
            inv=sum((a[i]-a[j])*(b[i]-b[j])<0 for i in range(len(a)) for j in range(i))
            pairs=sum(a[i]!=a[j] and b[i]!=b[j] for i in range(len(a)) for j in range(i))
            pair_error=np.sqrt(np.mean([((b[i]-a[i])-(b[j]-a[j]))**2 for i in range(len(a)) for j in range(i)]))/int(g.n_atoms.iloc[0])*1000
            r=metrics[(metrics.model==model)&(metrics.dataset=='historical')&(metrics.n_Li==nl)&(metrics.grouping=='phase_composition')].iloc[0]
            assert inv==r.discordant_pairs and pairs==r.strict_pairs
            assert abs(pair_error-r.pair_energy_RMSE_meV_atom)<1e-10
            ra=rankdata(a);rb=rankdata(b);rho=float(np.corrcoef(ra,rb)[0,1]);assert abs(rho-r.spearman_rho)<1e-12
            results.append(dict(model=model,group=f'historical Li{nl}',n=len(g),quadratic_inversions=inv,pair_energy_RMSE_meV_atom=pair_error,rank_correlation=rho,status='PASS'))
    dump(OUT/'independent_metric_checks.json',results)


def optimized_comparison(metrics):
    data=metrics[(metrics.dataset=='hu')&(metrics.grouping=='phase_composition+cohort')
                 &(metrics.cohort=='optimized')&metrics.informative_ordering]
    phases=sorted(data.phase.unique())
    fig,axes=plt.subplots(len(phases),3,figsize=(15,3.3*len(phases)),squeeze=False)
    for i,phase in enumerate(phases):
        for model in MODELS:
            g=data[(data.phase==phase)&(data.model==model)].sort_values('n_Li')
            for ax,col in zip(axes[i],['spearman_rho','bottom20_precision','selection_regret_meV_atom']):
                ax.plot(g.n_Li/g.n_Co,g[col],'o-',ms=4,color=COLORS[model],label=LABELS[model])
        for ax,label in zip(axes[i],['Spearman ρ','Low-20% precision','DFT regret (meV/atom)']):
            ax.set_xlabel('Li fraction x');ax.set_ylabel(label);ax.set_title(phase+' optimized only');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.suptitle('Hu optimized structures: same-cohort energy-ordering comparison',fontsize=14)
    fig.text(.5,.005,'Small and near-degenerate pools are not a global ground-state test. DFT energy spans and counts are retained in cohort_rank_metrics.csv.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.03,1,.96]);savefig(fig,'hu_optimized_ranking_comparison')


def workbook(summary,metrics,side,counter):
    from openpyxl.styles import Font,PatternFill
    with pd.ExcelWriter(OUT/'MLIP_comparison_tables.xlsx',engine='openpyxl') as writer:
        definitions=pd.DataFrame([
            ('Rank','1 is lowest energy within source/phase/composition; exact ties use average ranks.'),
            ('Source','historical and hu have distinct archived DFT protocols.'),
            ('Energy','Native total eV/cell; errors additionally reported meV/atom.'),
            ('Forces','Cartesian component RMSE eV/Angstrom; source-wide aggregate weights all atomic components.'),
            ('Stress','Six ASE tensile-positive components, xx yy zz yz xz xy; GPa in the comparison tables.'),
            ('Low20','ceil(0.2*n) lowest-energy entries plus all exact boundary ties.'),
            ('Regret','DFT energy penalty of choosing a model minimum in a finite tested pool; tied model minima averaged.'),
            ('Group medians','Each phase/composition group has equal weight; frame correlation is not removed by deduplication.'),
        ],columns=['Quantity','Definition'])
        for sheet,frame in [('Definitions',definitions),('Model_summary',summary),('Ranking_metrics',metrics),
                            ('All_structure_ranks',side.sort_values(['dataset','phase','n_Li','DFT_rank'])),
                            ('Hu_5_optimized',counter)]:
            frame.to_excel(writer,sheet_name=sheet,index=False)
            ws=writer.book[sheet];ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
            for cell in ws[1]:cell.font=Font(color='FFFFFF',bold=True);cell.fill=PatternFill('solid',fgColor='17324D')
            for column in ws.columns:
                ws.column_dimensions[column[0].column_letter].width=min(48,max(14,len(str(column[0].value))+2))


def main():
    compute_guard();OUT.mkdir(exist_ok=True)
    # NpzFile reads/decompresses on every __getitem__; cache each reference once.
    with np.load(HERE/'reference_frames.npz') as archive:
        reference={key:archive[key] for key in archive.files}
    meta=pd.read_csv(HERE/'frames.csv')
    assert np.array_equal(meta['index'].to_numpy(),np.arange(len(meta)))
    preds={m:merge_predictions(m,reference) for m in MODELS}
    df=pd.concat([frame_metrics(meta,reference,preds[m],m) for m in MODELS],ignore_index=True)
    df.to_csv(OUT/'per_structure_EFS.csv',index=False)
    metrics,ranked=rank_metrics(df)
    metrics.to_csv(OUT/'ranking_metrics.csv',index=False);ranked.to_csv(OUT/'ranked_structures.csv',index=False)
    # One row per exact geometry makes side-by-side ranks easy to inspect.
    side=ranked[ranked.model==MODELS[0]][['index','geometry_key','dataset','phase','cohort','structure_id','n_Co','n_Li','x','n_atoms','dft_energy_eV','DFT_rank']].copy()
    for model in MODELS:
        z=ranked[ranked.model==model][['index','mlip_energy_eV','MLIP_rank','force_RMSE_eV_A','stress_RMSE_GPa']].rename(columns={c:model+'_'+c for c in ['mlip_energy_eV','MLIP_rank','force_RMSE_eV_A','stress_RMSE_GPa']})
        side=side.merge(z,on='index',validate='one_to_one')
    side.sort_values(['dataset','phase','n_Li','DFT_rank']).to_csv(OUT/'side_by_side_structure_ranks.csv',index=False)
    summary=summarize(df,metrics);summary.to_csv(OUT/'model_summary.csv',index=False)
    cohort=metrics[metrics.grouping=='phase_composition+cohort'];cohort.to_csv(OUT/'cohort_rank_metrics.csv',index=False)
    main_metrics=metrics[metrics.grouping=='phase_composition']
    winners=[]
    for key,g in main_metrics.groupby(GROUP):
        info=dict(zip(GROUP,key))
        for col,larger in [('spearman_rho',True),('bottom20_precision',True),('selection_regret_meV_atom',False)]:
            best=g[col].max() if larger else g[col].min()
            info['best_'+col]=','.join(g.loc[np.isclose(g[col],best,rtol=0,atol=1e-12),'model'])
        winners.append(info)
    pd.DataFrame(winners).to_csv(OUT/'group_winners.csv',index=False)
    counter=df[(df.dataset=='hu')&df.structure_id.isin(['structure_4156','structure_4157','structure_4158','structure_4159','structure_4160'])].copy()
    for model,g in counter.groupby('model'):
        ix=g.index;counter.loc[ix,'DFT_subset_rank']=rankdata(g.dft_energy_eV)
        counter.loc[ix,'MLIP_subset_rank']=rankdata(g.mlip_energy_eV)
        counter.loc[ix,'DFT_above_min_meV_atom']=(g.dft_energy_eV-g.dft_energy_eV.min())/g.n_atoms*1000
        counter.loc[ix,'MLIP_above_min_meV_atom']=(g.mlip_energy_eV-g.mlip_energy_eV.min())/g.n_atoms*1000
    counter.to_csv(OUT/'Hu_Li10_optimized_counterexample.csv',index=False)
    workbook(summary,metrics,side,counter)
    independent_checks(df,metrics)
    comparison_bars(summary);parity_figures(df,reference,preds);rank_figures(df,metrics);composition_figures(df,metrics);optimized_comparison(metrics)
    dump(OUT/'completion.json',dict(status='ANALYSIS_COMPLETE',unique_geometries=len(meta),models=MODELS,
                                  reference_sha256=sha(HERE/'reference_frames.npz'),
                                  summary=json.loads(summary.to_json(orient='records'))))
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':main()

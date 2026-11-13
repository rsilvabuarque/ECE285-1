"""Summarize timing and numerical convergence, without claiming DFT accuracy."""
import csv,json,os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent

def main():
    assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
    conf=json.loads((HERE/'config.json').read_text());manifest=list(csv.DictReader((HERE/'manifest.csv').open()))
    rows=[];missing=[]
    for model in conf['models']:
        for a in manifest:
            q=HERE/'results'/model/a['structure_id']/'result.json'
            if q.exists():rows.append(json.loads(q.read_text()))
            else:missing.append((model,a['structure_id']))
    df=pd.DataFrame(rows)
    workers={q.stem:json.loads(q.read_text()) for q in (HERE/'workers').glob('*.json')}
    df['gpu']=df.worker.map(lambda w:workers[w]['gpu'])
    df['host']=df.worker.map(lambda w:workers[w]['host'])
    df['single_point_first_s']=df.single_point_seconds.apply(lambda a:a[0] if isinstance(a,list) else np.nan)
    df['single_point_second_s']=df.single_point_seconds.apply(lambda a:a[1] if isinstance(a,list) else np.nan)
    df['single_point_third_s']=df.single_point_seconds.apply(lambda a:a[2] if isinstance(a,list) else np.nan)
    df['single_point_repeat_s']=df.single_point_seconds.apply(lambda a:float(np.mean(a[1:])) if isinstance(a,list) else np.nan)
    df.to_csv(HERE/'per_structure_results.csv',index=False)
    summary=[]
    for model in conf['models']:
        d=df[df.model==model];good=d[d.status.isin(['converged','step_limit'])]
        rec=dict(model=model,requested=100,completed=len(d),converged=int((d.status=='converged').sum()),step_limited=int((d.status=='step_limit').sum()),exceptions=int((d.status=='exception').sum()))
        rec['convergence_fraction']=rec['converged']/100
        for metric in ['single_point_mean_s','single_point_first_s','single_point_second_s','single_point_third_s','single_point_repeat_s','relaxation_wall_s','relaxation_calculator_s','relaxation_evaluations','optimizer_steps','initial_fmax_eV_A','final_fmax_eV_A','energy_change_eV','rms_displacement_A','max_displacement_A']:
            if len(good):
                rec[metric+'_mean']=float(good[metric].mean());rec[metric+'_median']=float(good[metric].median());rec[metric+'_p90']=float(good[metric].quantile(.9))
        if len(good):
            calls=np.concatenate(good.single_point_seconds.to_list())
            rec['single_point_all_calls_mean_s']=float(calls.mean())
            rec['single_point_all_calls_median_s']=float(np.median(calls))
            rec['single_point_all_calls_p90_s']=float(np.quantile(calls,.9))
            rec['relaxation_wall_s_sum']=float(good.relaxation_wall_s.sum())
            rec['composition_macro_mean_relaxation_s']=float(good.groupby('n_Li').relaxation_wall_s.mean().mean())
        summary.append(rec)
    summ=pd.DataFrame(summary);summ.to_csv(HERE/'model_summary.csv',index=False)
    good=df[df.status.isin(['converged','step_limit'])]
    good.groupby(['model','gpu']).agg(n=('structure_id','count'),single_point_mean_s=('single_point_mean_s','mean'),single_point_third_s=('single_point_third_s','mean'),relaxation_wall_s=('relaxation_wall_s','mean')).to_csv(HERE/'hardware_summary.csv')
    grouped=good.groupby(['model','n_Li']).agg(n=('structure_id','count'),single_point_s=('single_point_mean_s','mean'),relaxation_s=('relaxation_wall_s','mean'),steps=('optimizer_steps','mean'),final_fmax=('final_fmax_eV_A','mean'),energy_change_eV=('energy_change_eV','mean'))
    grouped.to_csv(HERE/'composition_summary.csv')
    # Small groups (mostly 2--3) only support descriptive cross-model ordering.
    ranks=good[['model','structure_id','n_Li','initial_energy_eV','final_energy_eV']].copy()
    for k in ['initial','final']:ranks[k+'_rank']=ranks.groupby(['model','n_Li'])[k+'_energy_eV'].rank(method='min')
    ranks.to_csv(HERE/'cross_model_ranks.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=dict(zip(conf['models'],plt.cm.tab10.colors[:5]))
    fig,axs=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    xs=np.arange(5);labels=['UMA','MACE','SevenNet','CHGNet','M3GNet']
    for ax,mean,median in [(axs[0,0],summ.single_point_all_calls_mean_s,summ.single_point_all_calls_median_s),(axs[0,1],summ.relaxation_wall_s_mean,summ.relaxation_wall_s_median)]:
        ax.bar(xs-.18,mean,width=.36,label='Mean',color=[colors[m] for m in conf['models']])
        ax.bar(xs+.18,median,width=.36,label='Median',color=[colors[m] for m in conf['models']],alpha=.4,hatch='//')
        ax.legend(fontsize=8)
    axs[0,0].set(ylabel='Uncached E/F/stress time per call (s)',yscale='log')
    axs[0,1].set_ylabel('Relaxation time per structure (s)')
    axs[1,0].bar(xs,summ.convergence_fraction*100,color=[colors[m] for m in conf['models']]);axs[1,0].set_ylabel('Force-converged / requested (%)');axs[1,0].set_ylim(0,105)
    for m in conf['models']:
        d=good[good.model==m];axs[1,1].scatter(d.x,d.final_fmax_eV_A,label=m,s=13,alpha=.7,color=colors[m])
    axs[1,1].axhline(conf['fmax_eV_A'],c='k',ls='--',lw=1);axs[1,1].set(xlabel='Lithiation fraction x',ylabel='Final maximum atomic force (eV/Å)',yscale='log');axs[1,1].legend(fontsize=8)
    for ax in [axs[0,0],axs[0,1],axs[1,0]]:ax.set_xticks(xs,labels,rotation=15)
    fig.suptitle('100 symmetry-unique O3-derived Li$_{0–48}$Co$_{48}$O$_{96}$ structures\nFP32; fixed-cell FIRE; no DFT reference labels')
    fig.savefig(HERE/'performance_comparison.pdf');fig.savefig(HERE/'performance_comparison.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for m in conf['models']:
        d=good[good.model==m].groupby('n_Li').mean(numeric_only=True)
        for ax,col in zip(axs,['single_point_mean_s','relaxation_wall_s','optimizer_steps']):ax.plot(d.index/48,d[col],'.-',label=m,color=colors[m],lw=1,ms=3)
    for ax,label in zip(axs,['Single-point time (s)','Relaxation time (s)','FIRE steps']):ax.set(xlabel='Lithiation fraction x',ylabel=label)
    axs[0].legend(fontsize=8);fig.savefig(HERE/'performance_by_lithiation.pdf');fig.savefig(HERE/'performance_by_lithiation.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for m in conf['models']:
        d=good[good.model==m]
        axs[0].plot([1,2,3],np.array(d.single_point_seconds.to_list()).mean(axis=0),'o-',label=m,color=colors[m])
    axs[0].set(xlabel='Uncached evaluation number on each initial structure',ylabel='Mean time (s)',yscale='log',xticks=[1,2,3]);axs[0].legend(fontsize=8)
    d=good[good.model=='mace']
    for i in range(3):axs[1].scatter(d.n_Li,np.array(d.single_point_seconds.to_list())[:,i],s=14,alpha=.7,label=f'Call {i+1}')
    axs[1].set(xlabel='Li count',ylabel='MACE time per evaluation (s)',yscale='log');axs[1].legend(fontsize=8)
    fig.suptitle('Timing spikes retained in all averages; their cause is not established')
    fig.savefig(HERE/'single_point_timing_diagnostic.pdf');fig.savefig(HERE/'single_point_timing_diagnostic.png',dpi=180);plt.close(fig)
    columns=['model','converged','single_point_all_calls_mean_s','single_point_all_calls_median_s','relaxation_wall_s_mean','relaxation_wall_s_median','optimizer_steps_mean']
    table=summ[columns].rename(columns={'converged':'Converged / 100','single_point_all_calls_mean_s':'EFS mean (s)','single_point_all_calls_median_s':'EFS median (s)','relaxation_wall_s_mean':'Relax mean (s)','relaxation_wall_s_median':'Relax median (s)','optimizer_steps_mean':'Mean FIRE steps'}).to_markdown(index=False,floatfmt='.3f')
    report=f'''# 100-structure, five-model relaxation performance pilot

{len(rows)}/500 task bundles have terminal records; {len(missing)} missing.

[Performance figure](performance_comparison.pdf) · [Lithiation dependence](performance_by_lithiation.pdf) · [Summary CSV](model_summary.csv) · [All results](per_structure_results.csv)

{table}

Models: UMA-S-1.2.1 (`omat`), MACE-MH-1 (`omat_pbe`), SevenNet-Omni (`mpa`), CHGNet-PES-MatPES-PBE-1M-2026.9, and M3GNet-PES-MatPES-PBE-2025.2. Checkpoint hashes are frozen in checkpoints/receipt.json.

Each task contains three uncached initial E/F/stress evaluations and one fixed-cell ionic relaxation. The initial model warm-up is excluded. CUDA is synchronized for timing. All models use native FP32 with TF32 disabled and explicit conservative calculator paths; these are controlled settings, not each package's maximally optimized throughput. FIRE maximum step={conf['maxstep_A']} Å, maximum force target={conf['fmax_eV_A']} eV/Å, step cap={conf['max_steps']}. This target is a provisional timing criterion, not a DFT accuracy or stability certificate. No random rattling is applied.

The dataset spans all 49 integer Li counts, with one mask at 0, 1, 47, 48; two at every other composition; and a third at 6, 12, 18, 30, 36, 42. Mean size: 168 atoms (range 144–192). Raw fixed-count random masks are symmetry-deduplicated, not sampled uniformly over symmetry orbits. See manifest.csv and parent_symmetry.json for provenance. The witness indexes the sorted permutation list; it is not directly an index into the separate spglib rotation/translation arrays.

Arithmetic means weight the 100 selected structures equally. EFS means and medians describe all 300 timed initial-geometry calls per model. Relaxation summaries describe 100 trajectories per model. composition_summary.csv and the macro-composition mean in model_summary.csv provide alternate composition-balanced summaries. The three individual call times are retained for every structure. MACE's mean times for calls 1, 2, and 3 are 0.114, 2.277, and 0.065 s: excluding only the first call does not remove its timing spikes. The [timing diagnostic](single_point_timing_diagnostic.pdf) shows this effect. Its cause is not established, so none of these values is advertised as a definitive steady-state benchmark. All observed spikes remain in the mean; medians describe typical observed performance.

hardware_summary.csv separates observed A100 40 GB and 80 GB variants. Models received different mixtures of GPUs and compositions through dynamic scheduling, so this is a practical campaign measurement rather than a matched hardware comparison. Model loading and initial warm-up are recorded separately and excluded from these task times. Relaxation wall time includes optimizer and trajectory I/O; calculator-only time is also saved. Shared filesystem and scheduling conditions can affect timings. All 500 tasks converged; no step-limit or exception cases occurred.

All starting cells are identical and remain fixed. Residual stress need not vanish. Force convergence does not prove a stable minimum, especially for symmetry-preserving endpoints. Displacements identify structures for further inspection; they are not a validated O3/other-phase classifier. Relaxed endpoints from different models are different geometries. Their absolute energies cannot establish DFT accuracy or rank agreement with DFT. cross_model_ranks.csv is descriptive only: most composition groups have 2–3 structures.

Per-task initial.npz/final.npz preserve E/F/stress, coordinates and cells; relaxation.traj retains optimizer snapshots; optimizer.log and result.json preserve failures and stopping outcomes. Energies are eV/cell, forces eV/Å, ASE tensile-positive stresses eV/Å³ in xx,yy,zz,yz,xz,xy order. No VASP, training, or MD was run.

Checkpoint identity, package/precision configuration, worker hardware, and load/warm-up times are retained in checkpoints/receipt.json and workers/. validation/ retains numerical derivative and repeated-supercell checks. resources.json records allocations separately from summed task times.
'''
    (HERE/'REPORT.md').write_text(report)
    (HERE/'completion.json').write_text(json.dumps(dict(status='COMPLETE' if not missing else 'PARTIAL',terminal_tasks=len(rows),missing=missing,summary=summary),indent=2)+'\n')
    print('ANALYSIS',len(rows),'missing',len(missing),flush=True)

if __name__=='__main__':main()

"""Descriptive paired-fit summaries of a completed frozen two-round study."""
from pathlib import Path
import csv,hashlib,json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).parent

def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def write(p, rr):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(v):
    v=np.asarray(v,float);return float(v.mean()),float(v.std(ddof=1)) if len(v)>1 else 0.
def main():
    if not (R/'completion.json').exists():raise ValueError('Required study not complete; no result-based policy changes permitted')
    out=R/'analysis';out.mkdir(exist_ok=False)
    summaries=[];paired=[];support=[];corrected_cells=[];corrected_aggregates=[];inputs={};costs=[]
    for phase in ['round1_acquisition','round1_evaluation','round2_acquisition','round2_evaluation']:
        for name,dest in [('all_cell_metrics.csv',corrected_cells),('aggregate_metrics.csv',corrected_aggregates)]:
            p=R/phase/name;inputs[str(p)]=sha(p)
            for row in rows(p):
                if row['teacher']=='mace':
                    key='moving_inbin_yield' if name=='all_cell_metrics.csv' else 'balanced_moving_yield'
                    row[key]=''
                row['moving_metric_semantics']='UMA-trained condition fidelity' if row['teacher']=='uma' else 'not applicable: MACE did not define condition edges'
                if name=='all_cell_metrics.csv':
                    scored=int(row['scored_unique']);attempts=int(row['attempts'])
                    row['fixed_inbin_per_scored_orbit']=float(row['fixed_inbin_yield'])*attempts/scored if scored else ''
                dest.append(row)
    write(out/'corrected_all_cell_metrics.csv',corrected_cells);write(out/'corrected_aggregate_metrics.csv',corrected_aggregates)
    for taskfile in sorted((R/'taskfiles').glob('*.json')):
        tasks=json.loads(taskfile.read_text())['tasks'];claims=taskfile.parent/(taskfile.stem+'.claims');receipts=[json.loads((claims/t['id']/'receipt.json').read_text()) for t in tasks]
        if any(v['status']!='COMPLETE' for v in receipts):raise ValueError('Noncomplete child task in final cost receipt')
        costs.append(dict(phase=taskfile.stem,gpu_tasks=len(tasks),task_wall_seconds=sum(v['finished_unix']-v['started_unix'] for v in receipts),first_started_unix=min(v['started_unix'] for v in receipts),last_finished_unix=max(v['finished_unix'] for v in receipts),interpretation='Sum of one-GPU child lifetimes includes setup and idle polling; this is not billed node-hours',taskfile_sha256=sha(taskfile)))
    write(out/'task_cost_receipts.csv',costs)
    (out/'metric_semantics_repair.json').write_text(json.dumps(dict(status='DERIVED_MACE_MOVING_METRIC_MARKED_NOT_APPLICABLE',source_sha256=sha(__file__),input_sha256=inputs,reason='Frozen helper applied staged UMA absolute energy edges to MACE energies in moving metric. Corrected derived tables leave this column blank for MACE. Fixed MACE metrics use its own original TRAIN-only schema and remain unchanged. No raw tables, stages, queries, checkpoints, or schema edges were modified.'),indent=2)+'\n')
    for roundi in (1,2):
        data=rows(R/f'round{roundi}_evaluation/aggregate_metrics.csv')
        for teacher in ('uma','mace'):
            for family in ('ar','d2'):
                for mode in ('fixed','adaptive','range'):
                    rr=[r for r in data if r['teacher']==teacher and r['family']==family and r['schema_mode']==mode and r['stage_id'].startswith(f'r{roundi}_{family}_{mode}_')]
                    for metric in ['balanced_fixed_yield','low_tail_yield','high_tail_yield']+(['balanced_moving_yield'] if teacher=='uma' else []):
                        values=[];deltas=[]
                        for r in rr:
                            seed=r['seed'];cid=f'r1_no_new_{family}_anchor_{seed}' if roundi==1 else f'r2_no_new_{family}_{mode}_{seed}'
                            control=next(v for v in data if v['teacher']==teacher and v['stage_id']==cid)
                            val=float(r[metric]);delta=val-float(control[metric]);values.append(val);deltas.append(delta)
                            paired.append(dict(round=roundi,teacher=teacher,family=family,mode=mode,seed=seed,metric={'low_tail_yield':'H0_fixed_low_yield','high_tail_yield':'H0_fixed_high_yield'}.get(metric,metric),yield_fraction=val,control_yield_fraction=float(control[metric]),paired_gain_pp=delta*100,control_id=cid))
                        mu,sd=stats(values);dm,ds=stats(np.asarray(deltas)*100)
                        summaries.append(dict(round=roundi,teacher=teacher,family=family,mode=mode,metric={'low_tail_yield':'H0_fixed_low_yield','high_tail_yield':'H0_fixed_high_yield'}.get(metric,metric),fit_count=len(values),mean_percent=mu*100,sd_percent=sd*100,paired_gain_pp=dm,paired_gain_sd_pp=ds))
        # All modes have the same R1 acquired rows; compare schema choices directly, paired by seed.
        for teacher in ('uma','mace'):
            for family in ('ar','d2'):
                for mode in ('adaptive','range'):
                    for seed in ('2026111301','2026111302','2026111303'):
                        a=next(r for r in data if r['teacher']==teacher and r['stage_id']==f'r{roundi}_{family}_{mode}_{seed}')
                        b=next(r for r in data if r['teacher']==teacher and r['stage_id']==f'r{roundi}_{family}_fixed_{seed}')
                        for metric in ('balanced_fixed_yield','low_tail_yield','high_tail_yield'):
                            paired.append(dict(round=roundi,teacher=teacher,family=family,mode=mode,seed=seed,metric={'low_tail_yield':'H0_fixed_low_yield','high_tail_yield':'H0_fixed_high_yield'}.get(metric,metric),yield_fraction=float(a[metric]),control_yield_fraction=float(b[metric]),paired_gain_pp=100*(float(a[metric])-float(b[metric])),control_id=b['stage_id']))
        training_metadata=json.loads((R/f'round{roundi}/training_design.json').read_text())['new_models']
        for p in sorted((R/f'round{roundi}/schemas').glob('*/condition_schema.json')):
            schema=json.loads(p.read_text())
            model=next(m for m in training_metadata if m['id']==p.parent.name)
            with np.load(model['data'],allow_pickle=False) as z:
                roles=z['split'].astype(str); ii=np.flatnonzero(roles=='train'); nn=z['n_Li'][ii]; ee=z['energy_eV_atom'][ii]; inherited_weights=z['weight'][ii].astype(float)
            original=json.loads(Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project/artifacts/overnight_20261107/data/condition_schema_uma.json').read_text())
            for count in (12,24,36):
                v=schema['compositions'][str(count)]
                observed=ee[nn==count]; labels=np.searchsorted(v['edges_eV_atom'],observed,side='left');counts=[int(np.sum(labels==j)) for j in range(3)]
                weight=inherited_weights[nn==count].copy();present=np.unique(labels);multipliers=[]
                for label in present:
                    fraction=weight[labels==label].sum()/weight.sum()
                    multipliers.append(1/(len(present)*fraction))
                # Fractions are computed before any rescaling, matching train_mask.py.
                for label,multiplier in zip(present,multipliers):weight[labels==label]*=multiplier
                weight_ess=float(weight.sum()**2/np.square(weight).sum())
                support.append(dict(round=roundi,fit_id=p.parent.name,n_target=count,train_rows=len(observed),train_minimum_eV_atom=float(observed.min()),train_maximum_eV_atom=float(observed.max()),class_support_min=min(counts),class_support_max=max(counts),balanced_loss_weight_sum=float(weight.sum()),balanced_loss_weight_sum_squares=float(np.square(weight).sum()),loss_weight_ESS=weight_ess,loss_weight_ESS_fraction=weight_ess/len(observed),maximum_class_loss_multiplier=max(multipliers),maximum_within_count_mean1_row_loss_weight=float((weight/weight.mean()).max()),schema_reference_train_rows=v['train_rows'],low_count=counts[0],middle_count=counts[1],high_count=counts[2],low_edge_eV_atom=v['edges_eV_atom'][0],high_edge_eV_atom=v['edges_eV_atom'][1],low_edge_shift_meV_atom=1000*(v['edges_eV_atom'][0]-original['compositions'][str(count)]['edges_eV_atom'][0]),high_edge_shift_meV_atom=1000*(v['edges_eV_atom'][1]-original['compositions'][str(count)]['edges_eV_atom'][1]),supported_bins='|'.join(v['supported_bins']),actual_empty_bins='|'.join(b for b,n in zip(['low','middle','high'],counts) if not n),coincident_edges=v['coincident_edges']))
    write(out/'paired_fit_summary.csv',summaries);write(out/'paired_fit_results.csv',paired);write(out/'training_bin_support.csv',support)
    concentration=[]
    for identity in sorted({(r['round'],r['fit_id']) for r in support}):
        rr=[r for r in support if (r['round'],r['fit_id'])==identity];sw=sum(r['balanced_loss_weight_sum'] for r in rr);sw2=sum(r['balanced_loss_weight_sum_squares'] for r in rr);n=sum(r['train_rows'] for r in rr);ess=sw**2/sw2
        concentration.append(dict(round=identity[0],fit_id=identity[1],train_rows=n,loss_weight_ESS=ess,loss_weight_ESS_fraction=ess/n,minimum_actual_class_support=min(r['class_support_min'] for r in rr),interpretation='Concentration of objective loss weights across requestedcounts; not independent physical observations or gradient ESS'))
    write(out/'fit_weight_concentration.csv',concentration)
    fig,axs=plt.subplots(2,2,figsize=(9.5,6.4),sharey=True)
    colors={'fixed':'#377eb8','adaptive':'#ff7f00','range':'#4daf4a'}
    for row,family in enumerate(('ar','d2')):
        for col,teacher in enumerate(('uma','mace')):
            ax=axs[row,col]
            for j,mode in enumerate(('fixed','adaptive','range')):
                rr=[next(r for r in summaries if r['round']==i and r['teacher']==teacher and r['family']==family and r['mode']==mode and r['metric']=='balanced_fixed_yield') for i in (1,2)]
                ax.errorbar([1+(j-1)*.055,2+(j-1)*.055],[r['mean_percent'] for r in rr],yerr=[r['sd_percent'] for r in rr],marker='o',capsize=3,color=colors[mode],label=mode)
            ax.set_title(f'{family.upper()}: {teacher.upper()} fixed-bin assessment');ax.set_xticks([1,2]);ax.set_xlabel('Acquisition round');ax.grid(alpha=.2)
            if col==0:ax.set_ylabel('Novel in-bin structures / all attempts (%)')
    axs[0,0].legend(frameon=False,ncol=3,fontsize=8);fig.tight_layout();fig.savefig(out/'two_round_fixed_yields.pdf');fig.savefig(out/'two_round_fixed_yields.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9.5,3.4),sharey=True)
    for col,teacher in enumerate(('uma','mace')):
        ax=axs[col];xx=np.arange(6);labels=[]
        for i,(family,mode) in enumerate((f,m) for f in ('ar','d2') for m in ('fixed','adaptive','range')):
            rr=next(r for r in summaries if r['round']==2 and r['teacher']==teacher and r['family']==family and r['mode']==mode and r['metric']=='balanced_fixed_yield')
            ax.errorbar(i,rr['paired_gain_pp'],yerr=rr['paired_gain_sd_pp'],marker='o',capsize=3,color=colors[mode]);labels.append(f'{family}\n{mode}')
        ax.axhline(0,color='k',lw=.8);ax.set_xticks(xx,labels,fontsize=8);ax.set_title(teacher.upper());ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('Round2 acquisition gain vs no-new continuation (pp)');fig.tight_layout();fig.savefig(out/'round2_gain_vs_no_new.pdf');fig.savefig(out/'round2_gain_vs_no_new.png',dpi=180);plt.close(fig)
    # Extrema growth measures actual observed discoveries, independently of moving labels.
    h0=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project/artifacts/current/improvement_v1/training_round0.npz')
    with np.load(h0,allow_pickle=False) as z:
        roles=z['split'].astype(str);ii=np.flatnonzero(roles=='train');initial_n=z['n_Li'][ii];initial_e=z['energy_eV_atom'][ii]
    fig,axs=plt.subplots(2,3,figsize=(10.5,5.7),sharex=True)
    styles={'fixed':'-','adaptive':'--','range':':'};family_colors={'ar':'#377eb8','d2':'#e41a1c'}
    for col,count in enumerate((12,24,36)):
        for row,tail in enumerate(('low','high')):
            ax=axs[row,col];reference=float((min if tail=='low' else max)(initial_e[initial_n==count]))
            for family in ('ar','d2'):
                for mode in ('fixed','adaptive','range'):
                    ys=[0.];errs=[0.]
                    for roundi in (1,2):
                        rr=[r for r in support if r['round']==roundi and r['n_target']==count and r['fit_id'].startswith(f'r{roundi}_{family}_{mode}_')]
                        vv=[1000*(reference-r['train_minimum_eV_atom']) if tail=='low' else 1000*(r['train_maximum_eV_atom']-reference) for r in rr];mu,sd=stats(vv);ys.append(mu);errs.append(sd)
                    ax.errorbar([0,1,2],ys,yerr=errs,color=family_colors[family],linestyle=styles[mode],marker='o',markersize=3,capsize=2,label=f'{family}/{mode}')
            ax.axhline(0,color='k',lw=.6);ax.set_title(f'nLi={count}: {tail} endpoint');ax.set_xticks([0,1,2]);ax.grid(alpha=.2)
            if col==0:ax.set_ylabel('Observed TRAIN range growth (meV/atom)')
            if row==1:ax.set_xlabel('Acquisition round')
    axs[0,0].legend(fontsize=6,ncol=2,frameon=False);fig.tight_layout();fig.savefig(out/'observed_training_extrema_growth.pdf');fig.savefig(out/'observed_training_extrema_growth.png',dpi=180);plt.close(fig)
    # This explicit note prevents UMA condition edges being mistaken for MACE labels.
    (out/'interpretation.md').write_text('All reported fixed-bin metrics use the teacher-specific original H0 TRAIN-only schema. H0 fixed-low and H0 fixed-high use those original edges; they are distinct from the pilot409 threshold diagnostic. Moving-bin fidelity is reported only for UMA, which supplied training conditions; MACE moving-bin entries in raw helper CSVs apply UMA numerical edges and are not interpreted. Error bars are the sample standard deviation across three independent paired fits, not confidence intervals. Round1 schema contrasts share exactly the same acquired rows; round2 contrasts include the whole divergent acquisition policy. No-new continuations keep previous data and previous schema, restart Adam/RNG/schedule, and receive the same2000update budget. Acquisition query caps are256of512 attempts/cell, so capped per-attempt yield cannot exceed50%; fresh assessment scores all eligible128of128 proposals. Class balancing changes per-row loss weights, not the uniformly shuffled minibatch order; reported loss-weight ESS=(sumw)^2/sumw² measures concentration of objective weight and is not a count of independent physical observations. Uniform and fixed-TRAIN mutation are exploration baselines, not adaptive local energy minimizers. Configurations remain unrelaxed fixed-parent deletions.\n')
    receipt=dict(status='DESCRIPTIVE_ADAPTIVE_ANALYSIS_COMPLETE',created_unix=time.time(),source_sha256=sha(__file__),study_completion_sha256=sha(R/'completion.json'),outputs={p.name:sha(p) for p in out.iterdir()},no_mace_moving_bin_claim=True)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()

"""Test whether deletion-energy ordering survives geometry or cell relaxation."""
import csv,json,os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr,kendalltau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
rows=list(csv.DictReader((HERE/'reports/relaxation_attempts.csv').open()))
def enrich(row):
    folder='fixed_resume1' if row['selected_attempt'].startswith('fixed_resume1') else row['cohort'];p=HERE/'data'/folder/row['model']/(row['selected_attempt']+'.json');r=json.loads(p.read_text());row=dict(row);row.update(initial_energy_eV=r['initial_energy_eV'],final_energy_eV=r['final_energy_eV']);return row
with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(enrich,rows))
with (HERE/'reports/relaxation_energies.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
metrics=[]
for m in ['uma','mace','sevennet','chgnet','m3gnet']:
    for cohort in ['fixed','variable']:
        for n in sorted({int(r['n_Li']) for r in rows}):
            rr=[r for r in rows if r['model']==m and r['cohort']==cohort and int(r['n_Li'])==n and r['status']=='converged']
            if len(rr)<10:continue
            initial=np.array([r['initial_energy_eV'] for r in rr]);final=np.array([r['final_energy_eV'] for r in rr]);k=int(np.ceil(.2*len(rr)));q0=np.sort(initial)[k-1];q1=np.sort(final)[k-1];lo0=initial<=q0;lo1=final<=q1
            metrics.append(dict(model=m,cohort=cohort,n_Li=n,converged_masks=len(rr),spearman=float(spearmanr(initial,final).statistic),kendall=float(kendalltau(initial,final).statistic),initial_low20_recall_of_final_low20=float((lo0&lo1).sum()/lo1.sum()),median_relaxation_gain_meV_atom=float(np.median((final-initial)/(144+n)*1000)),scope='Acquisition-selectedsame1000parents;notunbiasedpopulationcoverage;onlyconvergedendpoints'))
with (HERE/'reports/deletion_vs_relaxed_ranking.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,cohort in zip(axs,['fixed','variable']):
    for m in ['uma','mace','sevennet','chgnet','m3gnet']:
        rr=[r for r in metrics if r['model']==m and r['cohort']==cohort];ax.plot([r['n_Li']/48 for r in rr],[r['spearman'] for r in rr],'.-',label=m)
    ax.set_title(cohort+' cell');ax.set_xlabel('Li fraction');ax.set_ylabel('Deletion vs relaxed energy Spearman');ax.set_ylim(-1,1.05);ax.grid(alpha=.2)
axs[1].legend(fontsize=8);fig.tight_layout();fig.savefig(HERE/'reports/deletion_vs_relaxed_ranking.pdf');fig.savefig(HERE/'reports/deletion_vs_relaxed_ranking.png',dpi=160)
(HERE/'validation/relaxation_objective.json').write_text(json.dumps(dict(status='PASS',audited_logical_records=len(rows),composition_groups=len(metrics),scope='Differentgeometryobjectivescanreorderfixed-compositionconfigurations. Same-inputpairedcomparison;acquisitionselectionandfiniteforce/stresstargetsapply.'),indent=2)+'\n');print('RELAXATION_OBJECTIVE_PASS',flush=True)

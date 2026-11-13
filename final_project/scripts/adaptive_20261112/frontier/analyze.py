"""Endpoint-search plots; energy changes are always within one Li count."""
from pathlib import Path
import csv,hashlib,json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
F=Path(__file__).parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert (F/'completion.json').exists();out=F/'analysis';out.mkdir(exist_ok=False)
 with (F/'all_metrics.csv').open() as f:data=list(csv.DictReader(f))
 colors={'greedy':'#e41a1c','random_walk':'#377eb8','uniform':'#4daf4a'}
 for teacher in ['uma','mace']:
  fig,axs=plt.subplots(2,3,figsize=(10.5,5.8))
  for i,label in enumerate(['low','high']):
   for j,count in enumerate([12,24,36]):
    ax=axs[i,j]
    for arm in ['greedy','random_walk','uniform']:
     xx=[0.];yy=[0.];ss=[0.]
     for step in [1,2,3,4]:
      rr=[r for r in data if r['teacher']==teacher and r['arm']==arm and int(r['n_target'])==count and r['requested_bin']==label and int(r['step'])==step];assert len(rr)==3
      gains=[float(r['desired_best_gain_meV_atom']) for r in rr]
      cumulative=[]
      for r in rr:cumulative.append(sum(int(z['selected_query_incidences']) for z in data if z['teacher']==teacher and z['arm']==arm and z['seed']==r['seed'] and int(z['n_target'])==count and z['requested_bin']==label and int(z['step'])<=step))
      xx.append(float(np.mean(cumulative)));yy.append(float(np.mean(gains)));ss.append(float(np.std(gains,ddof=1)))
     ax.errorbar(xx,yy,yerr=ss,color=colors[arm],marker='o',markersize=3,capsize=2,label=arm.replace('_',' '))
    ax.set_title(f'nLi={count}: {label} search');ax.set_xlabel('Cumulative newly scored cell proposals');ax.grid(alpha=.2)
    if j==0:ax.set_ylabel('Best observed gain vs common seeds\n(meV/atom, desired direction)')
  axs[0,0].legend(frameon=False,fontsize=8);fig.suptitle(teacher.upper()+' energy audit; all greedy decisions used UMA',fontsize=10);fig.tight_layout();fig.savefig(out/f'{teacher}_best_gain_per_query.pdf');fig.savefig(out/f'{teacher}_best_gain_per_query.png',dpi=180);plt.close(fig)
 (out/'interpretation.md').write_text('Greedy and energy-blind random-walk arms begin from exactly the same original unprotected H0 TRAIN masks and query exactly identical first-step neighbors. Each edit exchanges one present and one absent Li site anywhere in the cell; this is locality in occupancy/Hamming space, not a physical Li diffusion trajectory. Best-so-far includes every scored neighbor and is monotone by construction; the selected current path may move uphill. The uniform arm has equal attempt/query caps without identical edit semantics. Actual realized query counts differ when known/protected/duplicate masks are excluded, with no top-ups. All greedy choices use UMA; MACE audits the same acquired structures independently. Curves do not establish global extrema, DFT accuracy, equilibrium, or entropy. Error bars are sample SD across three paired seeds. Cell-query incidences are not additive physical cost across arms because the oracle de-duplicates shared geometry/model keys; use raw registry receipts for physical SCF cost.\n')
 (out/'receipt.json').write_text(json.dumps(dict(status='BOUNDED_FRONTIER_ANALYSIS_COMPLETE',created_unix=time.time(),source_sha256=sha(Path(__file__)),completion_sha256=sha(F/'completion.json'),outputs={p.name:sha(p) for p in out.iterdir()}),indent=2)+'\n')
if __name__=='__main__':main()

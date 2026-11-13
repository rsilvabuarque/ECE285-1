"""Light presentation redraws after all simulations; original figures remain unchanged."""
from pathlib import Path
import csv,hashlib,json,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
F=Path(__file__).parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert (F/'completion.json').exists();out=F/'presentation_v2';out.mkdir(exist_ok=False)
 with (F/'all_metrics.csv').open() as f:data=list(csv.DictReader(f))
 colors={'greedy':'#e41a1c','random_walk':'#377eb8','uniform':'#4daf4a'};plt.rcParams.update({'font.size':11,'axes.labelsize':11,'axes.titlesize':12})
 for teacher in ['uma','mace']:
  for kind in ['best_gain','current_vs_best']:
   fig,axs=plt.subplots(2,3,figsize=(12,6.7))
   for row,label in enumerate(['low','high']):
    for col,count in enumerate([12,24,36]):
     ax=axs[row,col]
     for arm in ['greedy','random_walk','uniform'] if kind=='best_gain' else ['greedy','random_walk']:
      for current in [False] if kind=='best_gain' else [False,True]:
       xx=[0.];yy=[0.];ss=[0.]
       for step in [1,2,3,4]:
        rr=[r for r in data if r['teacher']==teacher and r['arm']==arm and r['step']==str(step) and r['n_target']==str(count) and r['requested_bin']==label];assert len(rr)==3
        values=[(1 if label=='high' else -1)*(float(r['best_current_eV_atom'])-float(r['best_initial_eV_atom']))*1000 if current else float(r['desired_best_gain_meV_atom']) for r in rr]
        costs=[sum(int(z['selected_query_incidences']) for z in data if z['teacher']==teacher and z['arm']==arm and z['seed']==r['seed'] and z['n_target']==str(count) and z['requested_bin']==label and int(z['step'])<=step) for r in rr]
        xx.append(float(np.mean(costs)));yy.append(float(np.mean(values)));ss.append(float(np.std(values,ddof=1)))
       lab=arm.replace('_',' ')+(' current' if current else ' best-so-far' if kind!='best_gain' else '')
       ax.errorbar(xx,yy,yerr=ss if kind=='best_gain' else None,color=colors[arm],linestyle='--' if current else '-',marker='o',markersize=3,capsize=2,lw=1.7,label=lab)
     ax.set_title(f'nLi={count}: {label.upper()}');ax.set_xlabel('New query IDs per cell');ax.grid(alpha=.2)
     if col==0:ax.set_ylabel('Energy gain (meV/atom)')
   handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3 if kind=='best_gain' else 4,loc='upper center',bbox_to_anchor=(.52,.95),frameon=False,fontsize=10)
   fig.suptitle(teacher.upper()+': gains relative to each common starting best',fontsize=14,y=1.005)
   fig.text(.5,.015,'All greedy choices use UMA; LOW gain means lower energy, HIGH gain means higher energy. No global-extremum claim.',ha='center',fontsize=10)
   fig.subplots_adjust(left=.07,right=.99,bottom=.11,top=.83,hspace=.35,wspace=.22);fig.savefig(out/f'{teacher}_{kind}.pdf',bbox_inches='tight',pad_inches=.18);fig.savefig(out/f'{teacher}_{kind}.png',dpi=180,bbox_inches='tight',pad_inches=.18);plt.close(fig)
 summary=[]
 for teacher in ['uma','mace']:
  for arm in ['greedy','random_walk','uniform']:
   for count in [12,24,36]:
    for label in ['low','high']:
     rr=[r for r in data if r['teacher']==teacher and r['arm']==arm and r['n_target']==str(count) and r['requested_bin']==label and r['step']=='4'];values=[float(r['desired_best_gain_meV_atom']) for r in rr];costs=[sum(int(z['selected_query_incidences']) for z in data if z['teacher']==teacher and z['arm']==arm and z['seed']==r['seed'] and z['n_target']==str(count) and z['requested_bin']==label) for r in rr]
     summary.append(dict(teacher=teacher,arm=arm,n_target=count,requested_bin=label,best_gain_mean_meV_atom=float(np.mean(values)),best_gain_sample_sd_meV_atom=float(np.std(values,ddof=1)),mean_cumulative_query_IDs=float(np.mean(costs)),mean_symmetry_swap_distance=float(np.mean([float(r['mean_symmetry_swap_distance']) for r in rr])),diversity_sample=rr[0]['diversity_sample'],n_seeds=3))
 with (out/'final_summary.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 (out/'captions.md').write_text('Best-gain curves use the observed best among initial16 TRAIN starts and all queried neighbors, so best-so-far is monotone by construction. Current-versus-best curves show the best currently selected state separately: paths may leave an earlier best because every step chooses a new eligible neighbor, including uphill moves. These are not atomic diffusion trajectories; each edit swaps occupancy at any occupied/vacant site. Greedy and energy-blind walks share exact first-step proposals/queries and fixed one-swap/eight-neighbor budgets. Uniform has equal caps without an identical-edit claim. All comparisons are surrogate energies relative to the same starting pool best, not the entire H0 minimum or a global optimum. Sample SD is across three paired seeds. Geometry diversity uses minimum symmetry-reduced Hamming/2 distance between16current path masks or first16eligibleuniform proposals. Current mask uniqueness is constrained by shared novelty filtering and is not spontaneous diversity proof.\n')
 (out/'receipt.json').write_text(json.dumps(dict(status='FRONTIER_PRESENTATION_REDRAW_COMPLETE',created_unix=time.time(),source_sha256=sha(Path(__file__)),input_sha256=sha(F/'all_metrics.csv'),completion_sha256=sha(F/'completion.json'),outputs={p.name:sha(p) for p in out.iterdir()},no_new_queries_or_model_updates=True,original_outputs_unchanged=True),indent=2)+'\n')
if __name__=='__main__':main()

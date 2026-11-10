"""Render publication figures from the included saved-evidence CSVs only.

Run this file from any working directory; paths resolve beside this script.
No models, inference, training, cluster commands or external datasets are used.
"""
from pathlib import Path
import csv, hashlib, json, statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE=Path(__file__).absolute().parent; DATA=HERE/'data'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':7.2,'ytick.labelsize':7.2,'legend.fontsize':7,'axes.linewidth':.65,'lines.linewidth':1.1,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
BLUE='#2166ac'; ORANGE='#c75b12'; GRAY='#737373'
def read(name):
 with (DATA/name).open(newline='') as f:return list(csv.DictReader(f))
def sd(values):return statistics.stdev(values) if len(values)>1 else 0.
outputs={};text_bounds={}
def save(fig,name,title):
 fig.canvas.draw(); renderer=fig.canvas.get_renderer(); width,height=fig.canvas.get_width_height(); outside=[]
 for artist in fig.findobj(matplotlib.text.Text):
  if not artist.get_visible() or not artist.get_text():continue
  box=artist.get_window_extent(renderer)
  if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or box.x1 > width+1 or box.y1 > height+1):outside.append(dict(text=artist.get_text(),bounds=list(box.bounds)))
 text_bounds[name]=outside
 fig.savefig(HERE/(name+'.pdf'),metadata={'Title':title,'Creator':'Saved-evidence ECE285 figure renderer','CreationDate':None,'ModDate':None})
 fig.savefig(HERE/(name+'.png'))
 for ext in ['pdf','png']:
  p=HERE/(name+'.'+ext);outputs[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 plt.close(fig)
# Older, controlled dimension study: each point is an independent fit.
d=read('latent_dimension_fit_evidence.csv');fig,axs=plt.subplots(1,2,figsize=(5.5,2.25));fig.subplots_adjust(left=.105,right=.985,bottom=.25,top=.82,wspace=.43)
for ax,field,mult,label,title in [(axs[0],'useful_yield',100,'Prior useful yield (%)','a) Generation utility'),(axs[1],'val_reconstruction_nats',1,'VAL reconstruction (nats)','b) Reconstruction')]:
 means=[];errs=[]
 for k in [4,8,16,32]:
  values=[mult*float(x[field]) for x in d if int(x['D'])==k];means.append(statistics.mean(values));errs.append(sd(values))
  ax.scatter(np.asarray([k]*3)*np.asarray([.965,1,1.035]),values,color=GRAY,s=12,alpha=.7,zorder=4,linewidths=0)
 ax.errorbar([4,8,16,32],means,yerr=errs,color=BLUE,fmt='o-',ms=3.5,capsize=2.5,lw=1.15,zorder=3)
 ax.set_xscale('log',base=2);ax.set_xticks([4,8,16,32],labels=['4','8','16','32']);ax.minorticks_off();ax.set_xlabel('Latent dimension $D$');ax.set_ylabel(label);ax.set_title(title,loc='left',pad=5);ax.grid(axis='y',alpha=.18,lw=.5)
axs[0].axhline(100*.3304398148148148,color=GRAY,ls='--',lw=.8);axs[0].text(3.6,33.19,'Uniform reference',fontsize=7,color=GRAY);axs[0].set_ylim(26.5,35.2);axs[1].set_ylim(14.2,18)
fig.suptitle('Dense posterior, no spatial context: DEVELOPMENT',y=.975,fontsize=8.4)
fig.text(.5,.035,'Three fits per dimension; dots = fits, error bars = descriptive fit SD.',ha='center',fontsize=7)
save(fig,'latent_dimension','Latent dimension sensitivity in the dense/no-context development study')
# Current selected/context-matched encoders and distinct paired decoder probes.
components=read('latent_component_evidence.csv');inter=read('latent_intervention_fit_evidence.csv');fig,axs=plt.subplots(1,3,figsize=(5.5,2.6));fig.subplots_adjust(left=.091,right=.988,bottom=.24,top=.87,wspace=.62)
families=['A-shell','C-dense'];colors=[BLUE,ORANGE]
for panel,field,label in [(axs[0],'variance_of_posterior_mean',r'$\mathrm{Var}_{\mathrm{VAL}}(\mu_j)$'),(axs[1],'mean_posterior_variance',r'$E_{\mathrm{VAL}}[\sigma_j^2]$')]:
 for group,(family,color) in enumerate(zip(families,colors)):
  rr=[x for x in components if x['family']==family];seeds=sorted(set(x['seed'] for x in rr));positions=np.linspace(group-.2,group+.2,len(seeds))
  for x,seed in zip(positions,seeds):
   values=[float(v[field]) for v in rr if v['seed']==seed];panel.scatter(x+np.linspace(-.018,.018,4),values,s=8,color=color,alpha=.42,edgecolors='none',zorder=3);panel.scatter([x],[statistics.mean(values)],s=17,color=color,marker='D',edgecolor='white',linewidth=.35,zorder=4)
 panel.set_yscale('log');panel.set_xlim(-.45,1.45);panel.set_xticks([0,1],labels=['A-shell\n5 fits','C-dense\n3 fits']);panel.set_ylabel(label,labelpad=1.8);panel.grid(axis='y',which='major',alpha=.16,lw=.5)
axs[0].set_ylim(5e-6,1.7);axs[0].set_yticks([1e-5,1e-3,1e-1,1]);axs[0].axhline(.01,color='black',ls='--',lw=.7);axs[0].text(-.38,.014,'Activity cutoff',fontsize=7);axs[0].set_title('a) Encoder means',loc='left',pad=6)
axs[1].set_ylim(4e-5,2.2);axs[1].set_yticks([1e-4,1e-2,1]);axs[1].axhline(1,color='black',ls='--',lw=.7);axs[1].text(-.35,1.28,'Prior variance = 1',fontsize=7);axs[1].set_title('b) Posterior widths',loc='left',pad=6)
for k,(contrast,field,color,label) in enumerate([('condition_high_minus_low','mean_signed_delta_E_meV_atom',BLUE,r'Condition: signed $\Delta E$'),('alternate_latent_minus_low','mean_absolute_delta_E_meV_atom',ORANGE,r'Latent: $|\Delta E|$')]):
 for group,family in enumerate(families):
  values=[float(x[field]) for x in inter if x['family']==family and x['contrast']==contrast];x=group+(k-.5)*.31
  axs[2].bar(x,statistics.mean(values),.28,color=color,alpha=.85,label=label if group==0 else None,zorder=2);axs[2].errorbar(x,statistics.mean(values),yerr=sd(values),fmt='none',ecolor='black',capsize=2,lw=.7,zorder=3);axs[2].scatter(x+np.linspace(-.07,.07,len(values)),values,color='black',s=5,alpha=.7,zorder=4)
axs[2].set_xticks([0,1],labels=['A-shell\n5 fits','C-dense\n3 fits']);axs[2].set_ylabel('Energy response (meV/atom)',labelpad=1.8);axs[2].set_ylim(0,8);axs[2].set_title('c) Interventions',loc='left',pad=6);axs[2].grid(axis='y',alpha=.16,lw=.5);axs[2].legend(loc='upper right',frameon=False,handlelength=.9,handletextpad=.4,borderpad=.05,fontsize=7)
fig.text(.5,.035,'Dots: components (a–b), fit means (c). Diamonds: fit means. Error bars: fit SD.',ha='center',fontsize=7)
save(fig,'latent_mechanism','Current latent encoder statistics and separately controlled decoder interventions')
# Optional fixed-budget optimization illustration, not a convergence certificate.
curves=read('optimization_trace_evidence.csv');fits=read('latent_fit_evidence.csv');fig,axs=plt.subplots(1,2,figsize=(5.5,2.2));fig.subplots_adjust(left=.1,right=.985,bottom=.26,top=.85,wspace=.42)
for family,color in zip(families,colors):
 rr=[x for x in curves if x['family']==family];rr.sort(key=lambda r:int(r['updates']));x=[int(v['updates'])/1000 for v in rr]
 for ax,field in [(axs[0],'VAL_reconstruction_nats'),(axs[1],'VAL_KL_nats')]:
  vals=[float(v[field]) for v in rr];ax.plot(x,vals,color=color,label=family,lw=1)
  selected=next(f for f in fits if f['family']==family and f['seed']=='2026110801');point=next(v for v in rr if int(v['epoch'])==int(selected['selected_epoch']));ax.scatter(int(point['updates'])/1000,float(point[field]),color=color,s=27,marker='*',edgecolor='black',linewidth=.25,zorder=4)
for ax in axs:
 ax.axvline(1.502,color=GRAY,lw=.7,ls='--');ax.set_xlim(0,4.04);ax.set_xticks([0,1,2,3,4]);ax.set_xlabel('Optimizer updates (thousands)');ax.grid(axis='y',alpha=.16,lw=.5)
axs[0].set_ylabel('VAL reconstruction (nats)');axs[1].set_ylabel('VAL KL (nats)');axs[1].set_yscale('log');axs[0].legend(loc='lower left',frameon=False);axs[0].set_title('a) Reconstruction',loc='left');axs[1].set_title('b) Information cost',loc='left')
fig.suptitle('Example fixed-budget optimization: seed 2026110801',y=.975,fontsize=8.3)
fig.text(.5,.035,'Dashed: full KL weight at update 1502; stars: selected checkpoints. Fixed budget, not convergence.',ha='center',fontsize=7)
save(fig,'optimization_traces','Saved fixed4000update optimization traces for a prespecified fit seed')
receipt=dict(status='RENDERED_FROM_SAVED_EVIDENCE_ONLY',figure_width_inches=5.5,nominal_fonts_pt='7–8.4',outputs=outputs,renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),data_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.iterdir() if p.is_file()},outside_text_bounds=text_bounds,matplotlib_version=matplotlib.__version__,no_model_calls=True,no_new_training=True)
(HERE/'figure_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

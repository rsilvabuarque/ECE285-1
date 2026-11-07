"""Recompute matched-geometry DFT/MLIP selection diagnostics without new inference.

Every pool has one source, phase, composition, cell atom count, and geometry
cohort. No energies from different DFT sources or phases are mixed. Exact ties
use average ranks and tied boundaries expand selected sets.
"""
import argparse,csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def write(path,rows):
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def ranks(x):
 order=np.argsort(x,kind='stable');r=np.empty(len(x),float);i=0
 while i<len(x):
  j=i+1
  while j<len(x) and x[order[j]]==x[order[i]]:j+=1
  r[order[i:j]]=(i+j-1)/2+1;i=j
 return r

def metrics(d,m,natoms):
 d,m=np.asarray(d,float),np.asarray(m,float);n=len(d)
 rd,rm=ranks(d),ranks(m)
 rho=float(np.corrcoef(rd,rm)[0,1]) if n>1 and np.std(rd)>0 and np.std(rm)>0 else None
 error=(m-d)/natoms*1000
 pair=float(np.std(error,ddof=0)*math.sqrt(2*n/(n-1))) if n>1 else None
 low=m==m.min();regret=float(np.mean(d[low]-d.min())/natoms*1000)
 return dict(n=n,n_unique_DFT=len(np.unique(d)),n_unique_MLIP=len(np.unique(m)),
  spearman_rho=rho,top1_tie_mean_regret_meV_atom=regret,
  pair_gap_RMSE_meV_atom=pair,centered_error_RMSE_meV_atom=float(np.std(error,ddof=0)),
  mean_offset_meV_atom=float(np.mean(error)),DFT_span_meV_atom=float(np.ptp(d)/natoms*1000),
  exact_order_match=bool(np.array_equal(rd,rm)))

def topk(d,m,natoms,k):
 d,m=np.asarray(d,float),np.asarray(m,float);k=min(k,len(d))
 D=set(np.flatnonzero(d<=np.sort(d)[k-1]));M=set(np.flatnonzero(m<=np.sort(m)[k-1]))
 return dict(requested_k=k,selected_DFT=len(D),selected_MLIP=len(M),intersection=len(D&M),
   precision=len(D&M)/len(M),recall=len(D&M)/len(D),jaccard=len(D&M)/len(D|M),
   best_selection_regret_meV_atom=float((d[list(M)].min()-d.min())/natoms*1000),
   mean_selected_excess_meV_atom=float(np.mean(d[list(M)]-d.min())/natoms*1000))

def tests():
 d=np.array([0.,1.,2.]);m=np.array([10.,11.,12.]);r=metrics(d,m,1)
 assert r['spearman_rho']==1 and r['pair_gap_RMSE_meV_atom']==0 and r['top1_tie_mean_regret_meV_atom']==0
 m=np.array([0.,2.,4.]);assert metrics(d,m,1)['spearman_rho']==1 and metrics(d,m,1)['pair_gap_RMSE_meV_atom']>0
 assert topk(d,np.array([2.,1.,0.]),1,1)['best_selection_regret_meV_atom']==2000
 assert topk(d,np.array([0.,0.,3.]),1,1)['selected_MLIP']==2
 direct=np.mean([(m[i]-m[j]-d[i]+d[j])**2 for i in range(3) for j in range(i) ])**.5*1000
 assert np.isclose(direct,metrics(d,m,1)['pair_gap_RMSE_meV_atom'])

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();tests();a.output.mkdir(parents=True,exist_ok=True)
 rows=list(csv.DictReader(a.input.open()));required=['index','geometry_key','dataset','phase','cohort','structure_id','n_atoms','n_Co','n_Li','model','dft_energy_eV','mlip_energy_eV'];assert set(required)<=set(rows[0]);compact=[{k:r[k] for k in required} for r in rows]
 groups=defaultdict(list);seen=set();bygeom={}
 for r in compact:
  key=(r['model'],r['geometry_key']);assert key not in seen;seen.add(key)
  label=(float(r['dft_energy_eV']),int(r['n_atoms']),r['dataset'],r['phase'],r['cohort'],int(r['n_Co']),int(r['n_Li']))
  assert bygeom.get(r['geometry_key'],label)==label;bygeom[r['geometry_key']]=label
  for grouping in ['phase_composition','phase_composition_cohort']:
   cohort=r['cohort'] if grouping.endswith('_cohort') else 'all_cohorts'
   groups[(grouping,r['model'],r['dataset'],r['phase'],int(r['n_Co']),int(r['n_Li']),int(r['n_atoms']),cohort)].append(r)
 out=[];tops=[];counter=[]
 for key,rr in sorted(groups.items()):
  grouping,model,dataset,phase,nco,nli,natoms,cohort=key;D=np.array([float(r['dft_energy_eV']) for r in rr]);M=np.array([float(r['mlip_energy_eV']) for r in rr])
  base=dict(grouping=grouping,model=model,dataset=dataset,phase=phase,n_Co=nco,n_Li=nli,x=nli/nco,n_atoms=natoms,cohort=cohort)
  out.append(dict(base,**metrics(D,M,natoms)))
  for kn,kk in [('k1',1),('k5',5),('bottom20',math.ceil(.2*len(rr)))]:tops.append(dict(base,k_policy=kn,pool_n=len(rr),**topk(D,M,natoms,kk)))
  if dataset=='hu' and phase=='O3' and nco==12 and nli==10 and cohort=='optimized':
   rd,rm=ranks(D),ranks(M)
   for r,di,mi,ed,em in zip(rr,rd,rm,D,M):counter.append(dict(r,DFT_rank=di,MLIP_rank=mi,DFT_above_min_meV_atom=(ed-D.min())/natoms*1000,MLIP_above_min_meV_atom=(em-M.min())/natoms*1000))
 write(a.output/'pool_metrics.csv',out);write(a.output/'topk_metrics.csv',tops);write(a.output/'paired_energies.csv',compact);write(a.output/'Hu_O3_Li10_optimized_counterexample.csv',counter)
 # Source/cohort medians weight pools equally; do not pool unlike DFT energies.
 summary=[]
 for dataset,phase,cohort,model in sorted({(r['dataset'],r['phase'],r['cohort'],r['model']) for r in out if r['grouping']=='phase_composition_cohort'}):
  q=[r for r in out if (r['dataset'],r['phase'],r['cohort'],r['model'],r['grouping'])==(dataset,phase,cohort,model,'phase_composition_cohort') and r['n']>1 and r['spearman_rho'] is not None]
  if not q:continue
  t=[r for r in tops if (r['dataset'],r['phase'],r['cohort'],r['model'],r['grouping'],r['k_policy'])==(dataset,phase,cohort,model,'phase_composition_cohort','bottom20') and r['pool_n']>1]
  summary.append(dict(dataset=dataset,phase=phase,cohort=cohort,model=model,pools=len(q),frames=sum(r['n'] for r in q),median_rho=float(np.median([r['spearman_rho'] for r in q])),median_top20_precision=float(np.median([r['precision'] for r in t])),median_pair_gap_RMSE_meV_atom=float(np.median([r['pair_gap_RMSE_meV_atom'] for r in q])),mean_top1_regret_meV_atom=float(np.mean([r['top1_tie_mean_regret_meV_atom'] for r in q])),exact_order_pools=sum(r['exact_order_match'] for r in q)))
 write(a.output/'cohort_summary.csv',summary)
 # Independently check published pool-level values without relying on their summary.
 original=a.input.parent/'cohort_rank_metrics.csv';checks=[]
 if original.exists():
  existing=list(csv.DictReader(original.open()));old={(r['model'],r['dataset'],r['phase'],int(r['n_Co']),int(r['n_Li']),r['cohort']):r for r in existing if r['grouping']=='phase_composition+cohort'}
  for r in out:
   if r['grouping']!='phase_composition_cohort':continue
   k=(r['model'],r['dataset'],r['phase'],r['n_Co'],r['n_Li'],r['cohort'])
   if k not in old:continue
   prev=old[k]
   for new,oldcol in [('spearman_rho','spearman_rho'),('pair_gap_RMSE_meV_atom','pair_energy_RMSE_meV_atom'),('top1_tie_mean_regret_meV_atom','selection_regret_meV_atom')]:
    if r[new] is None:continue
    assert np.isclose(r[new],float(prev[oldcol]),atol=1e-9,rtol=1e-9),(k,new,r[new],prev[oldcol]);checks.append((k,new))
 receipt=dict(status='PASS',source_sha256=sha(a.input),source_rows=len(rows),unique_geometries=len(bygeom),models=sorted({r['model'] for r in compact}),pools=len(out),independent_field_checks=len(checks),script_sha256=sha(__file__),definitions=dict(pool='source/phase/nCo/nLi/nAtoms/cohort; alternate all-cohort summary is separately named',ties='Exact stored-energy ties have average ranks and expand top-k boundaries',regret='DFT penalty relative to minimum DFT member of complete evaluated pool',gaps='All unordered pair-gap RMSE via exact O(n) variance identity; per atom',summary='Unweighted across informative composition pools, not independent-frame inference'),limitations=['No new DFT or inference','Historical and Hu DFT protocols are separate','Correlated frames are not independent structures','FP64 audit differs from FP32 new deletion teacher campaign','No certification of archived SCF or magnetic convergence'])
 (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
 try:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,axs=plt.subplots(1,3,figsize=(11,3.3));colors={'uma':'#1f77b4','mace':'#d95f02','m3gnet':'#777777'}
  for model in ['uma','mace','m3gnet']:
   q=[r for r in out if r['grouping']=='phase_composition_cohort' and r['phase']=='O3' and r['dataset']=='historical' and r['cohort']=='rattled' and r['model']==model]
   q=sorted(q,key=lambda r:r['x']);axs[0].plot([r['x'] for r in q],[r['spearman_rho'] for r in q],'-o',label=model,color=colors[model],ms=3)
   q=[r for r in out if r['grouping']=='phase_composition_cohort' and r['phase']=='O3' and r['dataset']=='hu' and r['cohort']=='optimized' and r['model']==model and r['spearman_rho'] is not None]
   q=sorted(q,key=lambda r:r['x']);axs[1].plot([r['x'] for r in q],[r['spearman_rho'] for r in q],'-o',color=colors[model],ms=3)
   axs[2].plot([r['x'] for r in q],[r['pair_gap_RMSE_meV_atom'] for r in q],'-o',color=colors[model],ms=3)
  for ax,title in zip(axs,['Historical O3: rattled','Hu O3: optimized','Hu optimized: energy-gap errors']):ax.set_title(title);ax.set_xlabel('Li fraction x');ax.grid(alpha=.2)
  axs[0].set_ylabel('Spearman rank correlation');axs[1].set_ylabel('Spearman rank correlation');axs[2].set_ylabel('Pair-gap RMSE (meV/atom)');axs[0].legend(fontsize=8);axs[0].set_ylim(-1.05,1.05);axs[1].set_ylim(-1.05,1.05)
  fig.tight_layout();fig.savefig(a.output/'ranking_context.pdf');fig.savefig(a.output/'ranking_context.png',dpi=180);plt.close(fig)
 except ImportError:pass
if __name__=='__main__':main()

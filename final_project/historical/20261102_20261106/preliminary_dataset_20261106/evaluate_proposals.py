"""Audit frozen oracle shards and evaluate equal-attempt generative discovery."""
import csv,hashlib,json,math,os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
MODELS=['uma','mace','sevennet','chgnet','m3gnet']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writecsv(name,rows):
    with (HERE/'reports'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
    source=HERE/'data/generated_candidates.npz'
    with np.load(source) as z:p={k:z[k] for k in z.files}
    digest=sha(source);N=len(p['structure_ids']);lookup={s:i for i,s in enumerate(p['structure_ids'])};assert len(lookup)==N
    expected=np.ones((N,len(p['numbers'])),bool);expected[:,p['li_indices']]=~p['masks'].astype(bool)
    counts=expected.sum(1);nli=48-p['masks'].sum(1);tasks=list(map(json.loads,(HERE/'data/generated_tasks.jsonl').read_text().splitlines()))
    all_results={};workers=set()
    def readtask(t):
        assert t['input_sha256']==digest
        base=HERE/'data/generated'/t['model']/t['task_id'];r=json.loads(base.with_suffix('.json').read_text())
        assert r['status']=='complete'
        for k in ['task_id','model','input_npz','input_sha256','start','stop']:assert r[k]==t[k]
        with np.load(base.with_suffix('.npz')) as z:
            sl=slice(t['start'],t['stop']);ids=np.arange(t['start'],t['stop']);B=len(ids)
            assert np.array_equal(z['structure_ids'],p['structure_ids'][sl]) and np.array_equal(z['atom_presence'],expected[sl])
            E=z['energy_eV'];F=z['forces_parent_eV_A'];S=z['stress_eV_A3'];seconds=z['seconds']
            assert E.shape==(B,) and F.shape==(B,len(p['numbers']),3) and S.shape==(B,6) and seconds.shape==(B,)
            assert all(np.isfinite(a).all() for a in [E,F,S,seconds]) and (seconds>0).all()
            assert (F[~expected[sl]]==0).all()
            rms=np.sqrt((F*F).sum((1,2))/(3*counts[sl]));fm=np.linalg.norm(F,axis=2).max(1)
            return ids,E.copy(),S.copy(),seconds.copy(),rms,fm,r['worker']
    for m in MODELS:
        seen=np.zeros(N,int);E=np.empty(N);S=np.empty((N,6));seconds=np.empty(N);rms=np.empty(N);fm=np.empty(N)
        with ThreadPoolExecutor(max_workers=6) as pool:
            for ids,ee,ss,tt,rr,ff,worker in pool.map(readtask,[t for t in tasks if t['model']==m]):
                assert not seen[ids].any();seen[ids]+=1;E[ids]=ee;S[ids]=ss;seconds[ids]=tt;rms[ids]=rr;fm[ids]=ff;workers.add((m,worker))
        assert (seen==1).all()
        path=HERE/'data'/f'generated_scalar_{m}.npz';np.savez_compressed(path,structure_ids=p['structure_ids'],energy_eV=E,n_atoms=counts,n_Li=nli,stress_eV_A3=S,seconds=seconds,force_component_rms_eV_A=rms,force_max_eV_A=fm)
        all_results[m]=dict(E=E/counts,seconds=seconds)
        print('GENERATED_AUDIT_MODEL',m,N,flush=True)
    cfg=json.loads((HERE/'config.json').read_text())
    for m,w in sorted(workers):
        r=json.loads((HERE/'workers'/f'{w}.json').read_text());prior=json.loads((Path(cfg['fixed_pilot'])/'validation'/f'{m}.json').read_text())['receipt']
        for k in ['checkpoint_hashes','dtype','tf32','package_version','torch']:assert r[k]==prior[k],(m,k,w)
        for filename,d in r['source_hashes'].items():assert sha(HERE/filename)==d,(filename,w)
    splits={r['structure_id']:r for r in csv.DictReader((HERE/'splits/lineage_splits.csv').open())}
    thresholds={};thresholdrows=[]
    for m in MODELS:
        with np.load(HERE/'data'/f'learning_O3_N48_pilot_{m}.npz') as z:d={k:z[k] for k in z.files}
        train=np.array([splits[s]['split']=='train' for s in d['structure_ids']]);ee=d['energy_eV']/d['n_atoms']
        for n in [9,12,15,21,24,27,33,36,39]:
            primary=n in [12,24,36]
            ids=(d['n_Li']==n)&train if primary else (d['n_Li']==n)
            vals=ee[ids];q=float(np.sort(vals)[math.ceil(.2*len(vals))-1]);mn=float(vals.min())
            thresholds[m,n]=(q,mn)
            thresholdrows.append(dict(teacher=m,n_Li=n,source='main_train' if primary else 'withheld_catalog_assessment_only',rows=len(vals),q20_eV_atom=q,minimum_eV_atom=mn,empirical_low_fraction=float((vals<=q).mean())))
    writecsv('generation_thresholds.csv',thresholdrows)
    groups=defaultdict(list)
    for r in csv.DictReader((HERE/'cvae/proposal_attempts.csv').open()):groups[r['method'],int(r['n_Li']),int(r['seed'])].append(r)
    rows=[];paired=[]
    for (method,n,seed),rr in sorted(groups.items()):
        assert len(rr)==512
        ids=np.array([lookup[r['scoring_structure_id']] for r in rr]);unique=np.unique(ids);novel=np.array([r['novel_to_training']=='True' for r in rr]);assert novel.all()
        assert all(r['canonical_mask'].count('0')==n for r in rr)
        assert (nli[ids]==n).all() and sorted(int(r['attempt']) for r in rr)==list(range(512))
        assert all(r['scoring_structure_id']==f"gen48_{int(r['canonical_mask'],2):012x}" for r in rr)
        for m in MODELS:
            q,mn=thresholds[m,n];E=all_results[m]['E'];delta=1000*(E[ids]-q);du=1000*(E[unique]-q)
            row=dict(teacher=m,method=method,n_Li=n,seed=seed,attempts=len(ids),unique_orbits=len(unique),duplicate_fraction=1-len(unique)/len(ids),novel_attempt_fraction=float(novel.mean()),projected_count_valid_fraction=1.,distinct_novel_low_orbits=int((E[unique]<=q).sum()),novel_low_yield_per_attempt=float((E[unique]<=q).sum()/len(ids)),unique_low_fraction=float((E[unique]<=q).mean()),attempt_median_delta_meV_atom=float(np.median(delta)),attempt_p10_delta_meV_atom=float(np.quantile(delta,.1)),minimum_delta_meV_atom=float(delta.min()),unique_median_delta_meV_atom=float(np.median(du)),unique_p10_delta_meV_atom=float(np.quantile(du,.1)),minimum_minus_reference_min_meV_atom=float(1000*(E[ids].min()-mn)),standalone_unique_oracle_seconds=float(all_results[m]['seconds'][unique].sum()),threshold_scope='train' if n in [12,24,36] else 'withheld_assessment_only')
            for key,label in [('raw_expected_Li_count','expected'),('raw_threshold_Li_count','threshold_mode'),('raw_Bernoulli_Li_count','Bernoulli_draw')]:
                values=np.array([float(r[key]) if r[key] else np.nan for r in rr]);ok=np.isfinite(values)
                row[f'raw_{label}_mean_abs_count_error']=float(np.abs(values[ok]-n).mean()) if ok.any() else ''
                row[f'raw_{label}_exact_count_fraction']=float((values[ok]==n).mean()) if ok.any() and label!='expected' else ''
            rows.append(row)
    assert len(rows)==555
    writecsv('generation_metrics.csv',rows)
    for (method,n,seed),rr in sorted(groups.items()):
        if method!='uma_cvae':continue
        other=groups.get(('uma_cvae_high',n,seed))
        if other is None:continue
        other={int(r['attempt']):r for r in other};low=np.array([lookup[r['scoring_structure_id']] for r in rr]);high=np.array([lookup[other[int(r['attempt'])]['scoring_structure_id']] for r in rr])
        for m in MODELS:
            diff=1000*(all_results[m]['E'][low]-all_results[m]['E'][high]);paired.append(dict(teacher=m,n_Li=n,seed=seed,paired_attempts=len(diff),mean_low_minus_notlow_meV_atom=float(diff.mean()),median_low_minus_notlow_meV_atom=float(np.median(diff)),fraction_low_below_notlow=float((diff<0).mean())))
    if paired:writecsv('conditioning_paired_response.csv',paired)
    ranks=[]
    for n in [12,24,36]:
        ii=nli==n
        for i,m in enumerate(MODELS):
            for other in MODELS[i+1:]:ranks.append(dict(n_Li=n,teacher_a=m,teacher_b=other,shared_orbits=int(ii.sum()),spearman=float(spearmanr(all_results[m]['E'][ii],all_results[other]['E'][ii]).statistic)))
    writecsv('generated_teacher_rank_agreement.csv',ranks)
    methods=sorted({r['method'] for r in rows if r['threshold_scope']=='train'});fig,axs=plt.subplots(1,5,figsize=(19,6),sharey=True)
    for ax,m in zip(axs,MODELS):
        for j,method in enumerate(methods):
            v=[r['novel_low_yield_per_attempt'] for r in rows if r['teacher']==m and r['method']==method and r['threshold_scope']=='train']
            ax.barh(j,np.mean(v),color='steelblue',alpha=.6);ax.scatter(v,[j]*len(v),s=10,color='black',alpha=.65)
        ax.set_title(m);ax.set_xlabel('Distinct novel low-tail orbits / 512 attempts');ax.set_xlim(0,1);ax.grid(axis='x',alpha=.2)
    axs[0].set_yticks(range(len(methods)),[s.replace('_high','_not-low') for s in methods]);fig.suptitle('Fresh fixed-cell deletion geometries: 3 compositions × 3 training seeds per method\nLow-tail threshold fitted within each teacher/composition training set; dots are composition–seed values',fontsize=11);fig.tight_layout();fig.savefig(HERE/'reports/generation_low_tail_yield.pdf');fig.savefig(HERE/'reports/generation_low_tail_yield.png',dpi=160);plt.close(fig)
    receipt=dict(status='PASS',tasks=len(tasks),unique_orbits=N,model_structure_pairs=N*5,summary_rows=len(rows),attempt_groups=len(groups),workers=len(workers),input_sha256=digest,shared_union_oracle_seconds={m:float(all_results[m]['seconds'].sum()) for m in MODELS},inference_latency='Not timed; no end-to-end speed claim.',interpretation='Fresh MLIP-oracle consistency, not DFT accuracy. Raw-mask uniform baseline is orbit-size weighted. Projection ensures composition. _high means outside bottom20%, not highest20%. Held compositions are within-range interpolation.',vector_join='Only invariant scalar results may be joined to other symmetry orientations without a transformation witness.')
    (HERE/'validation/generated.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()

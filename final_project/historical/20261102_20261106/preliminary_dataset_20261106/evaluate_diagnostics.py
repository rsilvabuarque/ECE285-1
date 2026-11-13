"""Check prior-decoded geometries and a bounded actual-archive phase diagnostic."""
import csv,hashlib,json,os,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;MODELS=['uma','mace','sevennet','chgnet','m3gnet']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writecsv(name,rows):
    with (HERE/'reports'/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def collect(stage):
    tasks=list(map(json.loads,(HERE/'data'/f'{stage}_tasks.jsonl').read_text().splitlines()));path=Path(tasks[0]['input_npz']);digest=sha(path)
    with np.load(path) as z:p={k:z[k] for k in z.files}
    B=len(p['structure_ids']);results={};workers=set()
    for m in MODELS:
        E=np.full(B,np.nan);F=np.full((*p['atom_presence'].shape,3),np.nan);S=np.full((B,6),np.nan);seen=np.zeros(B,int)
        for t in [t for t in tasks if t['model']==m]:
            assert t['input_sha256']==digest;base=HERE/'data'/stage/m/t['task_id'];r=json.loads(base.with_suffix('.json').read_text());assert r['status']=='complete'
            for key in ['task_id','model','input_npz','input_sha256','start','stop']:assert r[key]==t[key]
            workers.add((m,r['worker']))
            with np.load(base.with_suffix('.npz')) as z:
                idx=z['indices'];assert np.array_equal(idx,np.arange(t['start'],t['stop'])) and not seen[idx].any();seen[idx]+=1
                assert np.array_equal(z['structure_ids'],p['structure_ids'][idx])
                valid=p.get('valid',np.ones(B,bool))[idx]
                assert ((z['statuses']=='complete')==valid).all()
                assert np.isfinite(z['energy_eV'][valid]).all() and np.isfinite(z['forces_parent_eV_A'][valid]).all() and np.isfinite(z['stress_eV_A3'][valid]).all()
                assert (z['forces_parent_eV_A'][valid][~p['atom_presence'][idx][valid]]==0).all()
                E[idx]=z['energy_eV'];F[idx]=z['forces_parent_eV_A'];S[idx]=z['stress_eV_A3']
        assert (seen==1).all();results[m]=(E,F,S)
    cfg=json.loads((HERE/'config.json').read_text())
    for m,w in workers:
        r=json.loads((HERE/'workers'/f'{w}.json').read_text());prior=json.loads((Path(cfg['fixed_pilot'])/'validation'/f'{m}.json').read_text())['receipt']
        for key in ['checkpoint_hashes','dtype','tf32','package_version','torch']:assert r[key]==prior[key]
        for fn,h in r['source_hashes'].items():assert sha(HERE/fn)==h
    return p,results,dict(tasks=len(tasks),frames=B,model_structure_pairs=B*5,input_sha256=digest,workers=len(workers))
def main():
    assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
    stages=['decoded_score','phase_context','decoded_relax']
    while True:
        missing=[]
        for stage in stages:
            for t in map(json.loads,(HERE/'data'/f'{stage}_tasks.jsonl').read_text().splitlines()):
                if not (HERE/'data'/stage/t['model']/(t['task_id']+'.json')).exists():missing.append(t['task_id'])
        if not missing:break
        print('DIAGNOSTICS_WAIT',len(missing),flush=True);time.sleep(30)
    p,results,receipt=collect('decoded_score');N=p['atom_presence'].sum(1);rows=[]
    for m,(E,F,S) in results.items():
        rms=np.sqrt((F*F).sum((1,2))/(3*N));fm=np.linalg.norm(F,axis=2).max(1)
        for i,sid in enumerate(p['source_mask_ids']):
            base=int(np.flatnonzero((p['source_mask_ids']==sid)&(p['methods']=='deletion'))[0])
            rows.append(dict(teacher=m,source_mask_id=str(sid),structure_id=str(p['structure_ids'][i]),method=str(p['methods'][i]),n_Li=int(N[i]-144),valid=bool(p['valid'][i]),energy_eV=float(E[i]),energy_minus_deletion_meV_atom=float((E[i]-E[base])/N[i]*1000),force_component_rms_eV_A=float(rms[i]),force_max_eV_A=float(fm[i]),force_rms_over_deletion=float(rms[i]/rms[base]),stress_max_abs_component_GPa=float(abs(S[i]).max()*160.21766208)))
    writecsv('decoded_geometry_scores.csv',rows);summary=[]
    for m in MODELS:
        for method in np.unique(p['methods']):
            rr=[r for r in rows if r['teacher']==m and r['method']==method]
            summary.append(dict(teacher=m,method=method,masks=len(rr),median_energy_minus_deletion_meV_atom=float(np.median([r['energy_minus_deletion_meV_atom'] for r in rr])),fraction_lower_energy=float(np.mean([r['energy_minus_deletion_meV_atom']<0 for r in rr])),median_force_rms_ratio=float(np.median([r['force_rms_over_deletion'] for r in rr])),fraction_fmax_below_005=float(np.mean([r['force_max_eV_A']<.05 for r in rr]))))
    writecsv('decoded_geometry_summary.csv',summary)
    tasks=list(map(json.loads,(HERE/'data/decoded_relax_tasks.jsonl').read_text().splitlines()));rel=[]
    for t in tasks:
        r=json.loads((HERE/'data/decoded_relax/uma'/(t['task_id']+'.json')).read_text());assert r['status'] in ['converged','step_limit']
        assert r['input_sha256']==t['input_sha256'] and r['task_id']==t['task_id']
        rel.append(dict(source_mask_id=t['source_mask_id'],method=t['method'],n_Li=t['n_Li'],converged=r['converged'],steps=r['steps'],calculator_calls=r['calculator_calls'],wall_seconds=r['wall_seconds'],initial_energy_eV=r['initial_energy_eV'],final_energy_eV=r['final_energy_eV'],final_fmax_eV_A=r['final_fmax_eV_A']))
    ref={r['source_mask_id']:r for r in rel if r['method']=='deletion'}
    for r in rel:
        b=ref[r['source_mask_id']];r['steps_saved_vs_deletion']=b['steps']-r['steps'];r['endpoint_energy_difference_meV_atom']=1000*(r['final_energy_eV']-b['final_energy_eV'])/(144+r['n_Li'])
    writecsv('decoded_matched_relaxations.csv',rel)
    pp,phase,phreceipt=collect('phase_context');pr=[]
    for m,(E,F,S) in phase.items():
        for phase_name in np.unique(pp['phase']):
            ids=np.flatnonzero(pp['phase']==phase_name);present=pp['atom_presence'][ids];nn=present.sum(1);de=(E[ids]-pp['dft_energy_eV'][ids])/nn*1000;df=(F[ids]-pp['dft_forces_parent_eV_A'][ids])[present];ds=(S[ids]-pp['dft_stress_eV_A3'][ids])*160.21766208
            pr.append(dict(teacher=m,phase=str(phase_name),frames=len(ids),raw_energy_MAE_meV_atom=float(np.abs(de).mean()),raw_energy_bias_meV_atom=float(de.mean()),force_component_RMSE_eV_A=float(np.sqrt((df*df).mean())),force_component_MAE_eV_A=float(abs(df).mean()),stress_component_RMSE_GPa=float(np.sqrt((ds*ds).mean())),stress_component_MAE_GPa=float(abs(ds).mean())))
    writecsv('phase_context_DFT_metrics.csv',pr)
    fig,axs=plt.subplots(1,2,figsize=(11,4));methods=['regressor','cvae_draw0','cvae_draw1','cvae_draw2']
    for j,m in enumerate(MODELS):
        rr=[r for r in summary if r['teacher']==m];a={r['method']:r for r in rr}
        for ax,key in zip(axs,['median_energy_minus_deletion_meV_atom','median_force_rms_ratio']):ax.plot(range(4),[a[k][key] for k in methods],'o-',label=m)
    axs[0].set_ylabel('Median energy − deletion (meV/atom)');axs[1].set_ylabel('Median force RMS / deletion force RMS')
    for ax,baseline in zip(axs,[0,1]):ax.axhline(baseline,color='gray',ls='--');ax.set_xticks(range(4),methods,rotation=20);ax.grid(alpha=.2)
    axs[1].legend(fontsize=8);fig.suptitle('Fresh-mask geometry decoding at supplied fixed cell; 128 matched masks\nThree prior draws from one CVAE fit, not independent training seeds',fontsize=11);fig.tight_layout();fig.savefig(HERE/'reports/decoded_geometry.pdf');fig.savefig(HERE/'reports/decoded_geometry.png',dpi=160);plt.close(fig)
    receipt.update(status='PASS',matched_relaxations=len(rel),matched_independent_masks=len(ref),phase_context=phreceipt,scope='Onefactorizedfixedcellgeometryprototype; priorEFSandmatchedrelaxations, notproofglobalminimum orphasegenerality. ArchiveDFTparameter/energy-reference differences remain; bounded16perphase diagnostic notbenchmark-ranking.')
    (HERE/'validation/diagnostics.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()

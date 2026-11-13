"""Audit exact small-cell AR likelihood and its entropy importance estimates.

Only a no-latent AR is accepted: a CVAE decoder at one z is not its marginal
proposal density. Exact enumeration supplies the reference and audits q's
normalization; finite-budget importance uses a nonzero uniform support mixture.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
import torch
from cathode_cvae.conditions import encode_requests
from cathode_cvae.data import sha256
from cathode_cvae.entropy import KB_EV_K,thermodynamics
from cathode_cvae.learning.generate_mask import load_model
from cathode_cvae.runtime import require_compute_allocation


def named(items):
    out={}
    for item in items:
        name,path=item.split('=',1)
        if name in out:raise ValueError('Repeated named input')
        out[name]=Path(path)
    return out


def csv_file(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def estimate(energy,q,temperature,draws,seed):
    rng=np.random.default_rng(seed);chosen=rng.choice(len(q),draws,p=q)
    reference=float(energy.min());delta=energy-reference;beta=1/(KB_EV_K*temperature)
    logw=-beta*delta[chosen]-np.log(q[chosen]);logsum=float(logsumexp(logw))
    w=np.exp(logw-logsum);mean=float(w@delta[chosen]);logZ=logsum-np.log(draws)
    f=np.bincount(chosen,minlength=len(q))/draws;nz=f>0
    return dict(draws=draws,seed=seed,temperature_K=temperature,
        importance_entropy_kB=float(logZ+beta*mean),
        naive_microstate_histogram_entropy_kB=float(-np.sum(f[nz]*np.log(f[nz]))),
        importance_ESS=float(1/np.sum(w*w)),unique_microstates=int(np.count_nonzero(f)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--checkpoint',action='append',required=True,help='MODEL=trusted no-latent AR checkpoint')
    p.add_argument('--score',action='append',required=True,help='TEACHER=exhaustive scalar NPZ')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=1)
    p.add_argument('--requested-bin',default='low')
    p.add_argument('--exploration',type=float,default=.1)
    p.add_argument('--seeds',type=int,nargs='+',default=[2026111201,2026111202,2026111203])
    p.add_argument('--draws',type=int,nargs='+',default=[128,512,2048,8192])
    p.add_argument('--sampler-audit-draws',type=int,default=8192)
    p.add_argument('--normalization-tolerance',type=float,default=5e-5,
        help='Provisional dimensionless FP32 sum(q) error check, not physical convergence')
    a=p.parse_args();require_compute_allocation()
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    if not 0<a.exploration<=1 or a.sampler_audit_draws<1:raise ValueError('Invalid mixture/budget')
    torch.set_num_threads(a.threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with np.load(a.catalog,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    raw=d['raw_occupancy'];rawcounts=raw.sum(1);ids=d['structure_ids'].astype(str)
    n_sites=raw.shape[1];powers=(1<<np.arange(n_sites-1,-1,-1)).astype(np.int64)
    if not np.array_equal(raw@powers,np.arange(len(raw))):raise ValueError('Unexpected raw ordering')
    checkpoints=named(a.checkpoint);scorepaths=named(a.score);energies={}
    for teacher,path in scorepaths.items():
        with np.load(path,allow_pickle=False) as z:
            lookup={sid:i for i,sid in enumerate(z['structure_ids'].astype(str))}
            energy=z['energy_eV'];status=z['status'] if 'status' in z else np.full(len(energy),'success')
        index=np.array([lookup[sid] for sid in ids]);e=energy[index]
        if not np.isfinite(e).all() or not np.all(status[index].astype(str)=='success'):raise ValueError('Exhaustive labels failed')
        energies[teacher]=e[d['raw_to_orbit']]
    a.output_dir.mkdir(parents=True);normalization=[];quality=[];sampling=[];estimates=[];probability_rows=[]
    for name,path in checkpoints.items():
        model,cfg=load_model(path,a.device)
        if model.latent_dim!=0 or cfg['kind']!='ar_baseline':raise ValueError('Exact marginal diagnostic requires no-latent AR')
        if model.n_sites!=n_sites:raise ValueError('Checkpoint parent size differs')
        mode=cfg['condition_mode'];schema=cfg.get('condition_schema')
        if schema is None:raise ValueError('Explicit condition schema required')
        for n in (3,6,9):
            rawix=np.flatnonzero(rawcounts==n);x=raw[rawix]
            request='unconditioned' if mode=='composition' else a.requested_bin
            cc=encode_requests(schema,np.full(len(x),n),np.full(len(x),request),mode=mode)
            logs=[]
            with torch.no_grad():
                for start in range(0,len(x),256):
                    xx=torch.as_tensor(x[start:start+256],dtype=torch.float32,device=a.device)
                    c=torch.as_tensor(cc[start:start+256],dtype=torch.float32,device=a.device)
                    logs.extend((-model.reconstruction(xx,c,None,exact_count=True)).double().cpu().numpy())
            logs=np.asarray(logs);mass=float(np.exp(logsumexp(logs)));error=abs(mass-1)
            normalization.append(dict(model=name,n_target=n,probability_sum=mass,absolute_error=error,
                tolerance=a.normalization_tolerance,scope='Exact enumeration of AR product probabilities; FP32 logits'))
            if error>a.normalization_tolerance:raise ValueError('AR likelihood fails exact normalization audit')
            q_ar=np.exp(logs-logsumexp(logs));q=(1-a.exploration)*q_ar+a.exploration/len(x)
            for i,raw_index in enumerate(rawix):
                probability_rows.append(dict(model=name,n_target=n,raw_occupancy_code=int(raw_index),
                    orbit_structure_id=ids[d['raw_to_orbit'][raw_index]],q_AR=float(q_ar[i]),q_mixture=float(q[i])))
            # Independently sample the actual decoder and compare with its full
            # likelihood, rather than assuming a correct ancestral sampler.
            generator=torch.Generator(device=a.device);generator.manual_seed(2026111207)
            generated=[]
            with torch.no_grad():
                for start in range(0,a.sampler_audit_draws,512):
                    batch=min(512,a.sampler_audit_draws-start)
                    c=torch.as_tensor(np.repeat(cc[:1],batch,axis=0),dtype=torch.float32,device=a.device)
                    generated.extend(model.sample(c,n,generator=generator).cpu().numpy())
            generated=np.asarray(generated);codes=generated@powers
            if not np.all(generated.sum(1)==n):raise AssertionError('AR count constraint failed')
            frequency=np.bincount(codes.astype(int),minlength=len(raw))[rawix]/len(generated)
            sampling.append(dict(model=name,n_target=n,draws=len(generated),
                exact_AR_vs_empirical_total_variation=float(.5*np.abs(q_ar-frequency).sum()),
                scope='Finite empirical-frequency error over many states; no threshold establishes physical validity'))
            for teacher,rawenergy in energies.items():
                e=rawenergy[rawix]
                for T in (300.,1000.,5000.):
                    exact=thermodynamics(e,np.ones(len(e)),T);physical=exact['orbit_probability']
                    nz=physical>0
                    quality.append(dict(model=name,teacher=teacher,n_target=n,temperature_K=T,
                        physical_entropy_kB=exact['entropy_kB'],
                        proposal_mixture_entropy_kB=float(-np.sum(q*np.log(q))),
                        proposal_vs_physical_total_variation=float(.5*np.abs(q-physical).sum()),
                        physical_to_proposal_KL_nats=float(np.sum(physical[nz]*(np.log(physical[nz])-np.log(q[nz]))))))
                    for draws in a.draws:
                        for seed in a.seeds:
                            result=estimate(e,q,T,draws,seed)
                            estimates.append(dict(model=name,teacher=teacher,n_target=n,
                                exact_entropy_kB=exact['entropy_kB'],**result))
    # Same draw budgets and seeds under uniform labelled-mask proposals provide
    # a direct baseline for any claimed efficiency gain from the learned q.
    for teacher,rawenergy in energies.items():
        for n in (3,6,9):
            e=rawenergy[rawcounts==n];uniform=np.full(len(e),1/len(e))
            for T in (300.,1000.,5000.):
                exact=thermodynamics(e,np.ones(len(e)),T)
                for draws in a.draws:
                    for seed in a.seeds:
                        estimates.append(dict(model='uniform_microstate',teacher=teacher,n_target=n,
                            exact_entropy_kB=exact['entropy_kB'],**estimate(e,uniform,T,draws,seed)))
    for filename,rows in [('AR_normalization.csv',normalization),('AR_sampler_audit.csv',sampling),
        ('learned_proposal_quality.csv',quality),('learned_proposal_importance.csv',estimates),
        ('exact_AR_probabilities.csv',probability_rows)]:csv_file(a.output_dir/filename,rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    for name in [next(iter(checkpoints)),'uniform_microstate']:
        for teacher in energies:
            rr=[r for r in estimates if r['model']==name and r['teacher']==teacher and r['n_target']==6 and r['temperature_K']==1000.]
            for key,style in [('naive_microstate_histogram_entropy_kB',':'),('importance_entropy_kB','-')]:
                budgets=sorted(set(r['draws'] for r in rr));means=[np.mean([abs(r[key]-r['exact_entropy_kB']) for r in rr if r['draws']==b]) for b in budgets]
                axs[0].plot(budgets,means,style,marker='o',label=f'{name}/{teacher}: '+('naive' if key.startswith('naive') else 'importance'))
            budgets=sorted(set(r['draws'] for r in rr));means=[np.mean([r['importance_ESS']/r['draws'] for r in rr if r['draws']==b]) for b in budgets]
            axs[1].plot(budgets,means,marker='o',label=f'{name}/{teacher}')
    axs[0].set_ylabel('Mean absolute entropy error / kB');axs[1].set_ylabel('Weight ESS / proposal draws')
    for ax in axs:ax.set_xscale('log');ax.set_xlabel('Learned-mixture draws');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('N12 first named AR versus uniform; exact reference, Li6/12 at1000K')
    fig.tight_layout();fig.savefig(a.output_dir/'learned_AR_entropy_diagnostic.pdf');plt.close(fig)
    receipt=dict(status='COMPLETE_LEARNED_AR_IMPORTANCE_DIAGNOSTIC',catalog_sha256=sha256(a.catalog),
        checkpoint_sha256={k:sha256(v) for k,v in checkpoints.items()},score_sha256={k:sha256(v) for k,v in scorepaths.items()},
        normalization_max_absolute_error=max(r['absolute_error'] for r in normalization),
        normalization_tolerance=a.normalization_tolerance,exploration=a.exploration,
        scope='Exact no-latent AR probabilities over labelled microstates, mixed with uniform support; total energy eV',
        limitations=['No CVAE marginal-likelihood claim','Finite-budget importance ratios and entropy have estimator bias',
        'Small fixed lattice and MLIP energies only; no entropy transfer to N48/physical LCO claim'],
        source_sha256=sha256(Path(__file__)))
    receipt['outputs']={p.name:sha256(p) for p in a.output_dir.iterdir() if p.is_file()}
    (a.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

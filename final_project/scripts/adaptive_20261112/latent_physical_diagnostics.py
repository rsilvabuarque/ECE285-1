"""Probe physical information and generate matched z interventions.

The probe fits TRAIN only and evaluates VAL. Its target is a nearest-shell
vacancy-pair fraction, residualized within TRAIN composition/energy-bin cells.
Probe accuracy on posterior means is not evidence of useful prior generation.
Raw generated masks here have NOT been checked against protected histories;
the integrating owner must filter identities before any new oracle scoring.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cathode_cvae.runtime import require_compute_allocation
from cathode_cvae.graph import PeriodicGraph
from cathode_cvae.learning.generate_mask import load_model, resolve_request
from cathode_cvae.learning.runtime import device_name, trusted_checkpoint
from cathode_cvae.learning.train_mask import canonicalize, encode_train_val_conditions
from cathode_cvae.learning.physical_auxiliary import (prepare_targets,
    vacancy_pair_values, make_head, head_prediction)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_ridge(features, target, alpha=1.0):
    """Unweighted TRAIN ridge regression; intercept is not penalized."""
    f=np.asarray(features,dtype=np.float64); y=np.asarray(target,dtype=np.float64)
    x=np.column_stack((np.ones(len(f)),f))
    penalty=np.eye(x.shape[1])*alpha;penalty[0,0]=0
    coef=np.linalg.solve(x.T@x+penalty,x.T@y)
    return coef


def predict_ridge(coef, features):
    return np.column_stack((np.ones(len(features)),features))@coef


def regression_summary(target,prediction):
    y=np.asarray(target,float);p=np.asarray(prediction,float)
    mse=float(np.mean((y-p)**2)); denom=float(np.mean((y-y.mean())**2))
    from scipy.stats import spearmanr
    correlation=None if np.ptp(y)==0 or np.ptp(p)==0 else float(spearmanr(y,p).statistic)
    return dict(rows=len(y),MSE=mse,R2_vs_VAL_mean=1-mse/denom if denom>0 else None,
                Spearman=correlation)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','data','parent','output-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--draws',type=int,default=128)
    p.add_argument('--probe-only',action='store_true')
    a=p.parse_args(argv);require_compute_allocation();a.device=device_name(a.device)
    if a.draws<1 or a.batch_size<1:raise ValueError('draws and batch size must be positive')
    a.output_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,cfg=load_model(a.checkpoint,a.device)
    if cfg.get('data_sha256')!=digest(a.data) or cfg.get('parent_sha256')!=digest(a.parent):
        raise ValueError('Diagnostic data/parent differ from fitted checkpoint')
    if cfg.get('descriptor_schema'):raise ValueError('This diagnostic requires energy-only conditions')
    graph=PeriodicGraph.from_dict(cfg['graph'])
    with np.load(a.data,allow_pickle=False) as z:
        roles=z['split'].astype(str);active=np.flatnonzero(np.isin(roles,['train','val']))
        # The compressed energy member is materialized, but TEST values are not indexed.
        energy=z['energy_eV_atom'];n=z['n_target'] if 'n_target' in z else z['n_Li']
        cond,bins,_=encode_train_val_conditions(cfg['condition_schema'],n,energy,roles,cfg['condition_mode'])
        x=z['occupancy'][active];ids=z['structure_ids'][active];n=n[active]
        cond,bins,roles=cond[active],bins[active],roles[active]
    with np.load(a.parent,allow_pickle=False) as z:perms=z['permutations']
    inverse=torch.as_tensor(np.argsort(perms,axis=1),device=a.device)
    xt=torch.as_tensor(x,dtype=torch.float32,device=a.device);xt,_=canonicalize(xt,inverse)
    ct=torch.as_tensor(cond,dtype=torch.float32,device=a.device)
    mu=[];logvar=[]
    with torch.no_grad():
        for start in range(0,len(x),a.batch_size):
            m,l=model.encode(xt[start:start+a.batch_size],ct[start:start+a.batch_size])
            mu.append(m.cpu().numpy());logvar.append(l.cpu().numpy())
    mu=np.concatenate(mu);logvar=np.concatenate(logvar)
    raw,genuine,_,target_metadata=prepare_targets(x,n,bins,roles,graph,mode='latent',seed=a.seed,
        shell_tolerance_A=cfg.get('context_shell_tolerance_A',cfg.get('decoder_context_spec',{}).get('shell_tolerance_A',1e-5)))
    train=np.flatnonzero(roles=='train');val=np.flatnonzero(roles=='val')
    center=mu[train].mean(0);scale=mu[train].std(0);scale=np.where(scale>0,scale,1)
    normalized=(mu-center)/scale
    condition_coef=fit_ridge(cond[train],genuine[train]);latent_features=np.column_stack((cond,normalized))
    latent_coef=fit_ridge(latent_features[train],genuine[train])
    rng=np.random.default_rng(a.seed+99);shuffled=genuine.copy()
    for count,bin_ in sorted({(int(n[i]),str(bins[i])) for i in train}):
        members=train[(n[train]==count)&(bins[train]==bin_)]
        shuffled[members]=genuine[rng.permutation(members)]
    shuffled_coef=fit_ridge(latent_features[train],shuffled[train])
    predictions={'condition_only':predict_ridge(condition_coef,cond[val]),
                 'posterior_mu_plus_condition':predict_ridge(latent_coef,latent_features[val]),
                 'shuffled_TRAIN_target_mu_plus_condition':predict_ridge(shuffled_coef,latent_features[val])}
    eps=rng.standard_normal(mu[val].shape).astype(np.float32)
    posterior=mu[val]+np.exp(.5*logvar[val])*eps
    predictions['mu_fitted_probe_on_posterior_sample']=predict_ridge(latent_coef,np.column_stack((cond[val],(posterior-center)/scale)))
    predictions['mu_fitted_probe_on_independent_prior']=predict_ridge(latent_coef,np.column_stack((cond[val],(eps-center)/scale)))
    head_cfg=cfg.get('physical_auxiliary')
    if head_cfg:
        ck=trusted_checkpoint(a.checkpoint,a.device)
        head=make_head(model.latent_dim,model.condition_dim,head_cfg['mode'],123,a.device)
        head.load_state_dict(ck['physical_auxiliary_state_dict']);head.eval()
        with torch.no_grad():
            for name,z in [('auxiliary_head_posterior_mu',mu[val]),('auxiliary_head_posterior_sample',posterior),('auxiliary_head_independent_prior',eps)]:
                predictions[name]=head_prediction(head,torch.as_tensor(z,device=a.device),ct[val],head_cfg['mode']).cpu().numpy()
    summary={name:regression_summary(genuine[val],value) for name,value in predictions.items()}
    np.savez_compressed(a.output_dir/'VAL_probe_predictions.npz',structure_ids=ids[val],n_target=n[val],
        measured_energy_bin=bins[val],raw_vacancy_pair_fraction=raw[val],genuine_standardized_target=genuine[val],**predictions)
    result=dict(status='PASS',checkpoint_sha256=digest(a.checkpoint),data_sha256=digest(a.data),
        parent_sha256=digest(a.parent),source_sha256=digest(__file__),latent_dim=model.latent_dim,
        fit_seed=cfg['seed'],diagnostic_seed=a.seed,TRAIN_rows=len(train),VAL_rows=len(val),
        TEST_processed=False,target=target_metadata,probe_alpha=1.0,
        probe_scope='Unweighted TRAIN-only ridge coefficients and latent scaling; VAL targets never fit or tune coefficients',
        prior_prediction_scope='Independent prior z is compared with the same VAL mask target only as a mismatch diagnostic; this is not a physical generation score',
        metrics=summary)
    if not a.probe_only:
        outcomes=[];all_occ=[];all_ids=[];all_n=[];all_labels=[];all_interventions=[]
        for count in cfg['counts']:
            for bin_index,bin_ in enumerate(['low','middle','high']):
                cv,requested,effective,identity,_=resolve_request(cfg,count,requested_bin=bin_)
                stream=np.random.default_rng(a.seed+1009*int(count)+97*bin_index)
                width=max(4,model.latent_dim);z0=stream.standard_normal((a.draws,width)).astype(np.float32)[:,:model.latent_dim]
                z1=stream.standard_normal((a.draws,width)).astype(np.float32)[:,:model.latent_dim]
                uniforms=stream.random((a.draws,model.n_sites),dtype=np.float32)
                requests=[('primary',z0),('latent_resample',z1),('latent_zero',np.zeros_like(z0))]
                for v in [-2.,-1.,0.,1.,2.]:
                    zz=z0.copy()
                    if model.latent_dim:zz[:,0]=v
                    requests.append((f'coordinate0_{v:g}',zz))
                c=torch.as_tensor(np.tile(cv,(a.draws,1)),device=a.device)
                u=torch.as_tensor(uniforms,device=a.device);base_codes=None;base_raw=None
                for name,z in requests:
                    with torch.no_grad():occ=model.sample(c,int(count),z=torch.as_tensor(z,device=a.device),uniforms=u)
                    canonical,codes=canonicalize(occ,inverse);occupancy=canonical.cpu().numpy().astype(np.uint8)
                    values,_=vacancy_pair_values(occupancy,graph)
                    sid=np.array([f'mask{model.n_sites}_{int(code):0{(model.n_sites+3)//4}x}' for code in codes])
                    if name=='primary':base_codes=codes;base_raw=values
                    for i in range(a.draws):outcomes.append(dict(n_target=int(count),requested_bin=bin_,intervention=name,attempt=i,
                        structure_id=str(sid[i]),vacancy_pair_fraction=float(values[i]),same_orbit_as_primary=bool(codes[i]==base_codes[i]),
                        pair_fraction_change_from_primary=float(values[i]-base_raw[i])))
                    all_occ.append(occupancy);all_ids.append(sid);all_n.extend([count]*a.draws)
                    all_labels.extend([bin_]*a.draws);all_interventions.extend([name]*a.draws)
        with (a.output_dir/'generated_physical_outcomes.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(outcomes[0]));writer.writeheader();writer.writerows(outcomes)
        occupancy=np.concatenate(all_occ)
        np.savez_compressed(a.output_dir/'raw_intervention_proposals.npz',occupancy=occupancy,masks=1-occupancy,
            structure_ids=np.concatenate(all_ids),n_target=np.asarray(all_n),requested_bin=np.asarray(all_labels),
            intervention=np.asarray(all_interventions),parent_sha256=np.asarray(digest(a.parent)))
        result['generation']=dict(attempts=len(outcomes),draws_per_cell_intervention=a.draws,
            unique_raw_orbits=len(set(np.concatenate(all_ids).tolist())),
            protection_status='UNFILTERED_RAW; filter all prior TRAIN/VAL/TEST/assessment histories before oracle use',
            coordinate_scope='Untrained latent-axis intervention at -2,-1,0,1,2; not a semantic physical coordinate',
            paired_stream='Same per-cell prior prefix (width at least4) and uniform matrices across models with same diagnostic seed; baseline AR has zero latent coordinates')
    (a.output_dir/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'latent_dim':model.latent_dim,'metrics':summary}),flush=True)


if __name__=='__main__':main()

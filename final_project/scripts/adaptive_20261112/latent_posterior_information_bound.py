"""Exact empirical Gaussian-pair upper bound on conditional sampled-z information.

For qbar=(1/N)sum_j q_j, convexity of KL in its second argument gives
I(X;Z|fixed_c)=mean_i KL(q_i||qbar) <= mean_ij KL(q_i||q_j).
This finite-VAL empirical bound is not a population or thermodynamic statement.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from cathode_cvae.runtime import require_compute_allocation
require_compute_allocation();import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args();p=args.run_root.resolve();sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest();rows=[]
for j,name in enumerate(['D2','Sampled auxiliary lambda10','Mean auxiliary lambda10']):
 q=p/f'fixed_condition_embedding/embedding_{j}.npz'
 with np.load(q,allow_pickle=False) as z:mu=z['posterior_mu'].astype(np.float64);lv=z['posterior_logvar'].astype(np.float64)
 delta=lv[:,None,:]-lv[None,:,:];dm=mu[:,None,:]-mu[None,:,:]
 # expm1(delta)-delta preserves the tiny variance contribution; the direct
 # expression exp(delta)-1-delta suffers cancellation near equal Gaussians.
 pair=.5*np.sum(np.expm1(delta)-delta+dm**2*np.exp(-lv[None,:,:]),axis=-1)
 if not np.isfinite(pair).all() or pair.min() < -1e-25:raise FloatingPointError('Invalid pairwise Gaussian divergence')
 prior=.5*np.sum(mu**2+np.expm1(lv)-lv,axis=-1)
 rows.append(dict(model=name,rows=len(mu),latent_dim=mu.shape[1],empirical_information_upper_bound_nats=float(pair.mean()),upper_bound_bits=float(pair.mean()/np.log(2)),mean_KL_to_standard_Normal_nats=float(prior.mean()),SD_mu=np.std(mu,axis=0,dtype=np.float64).tolist(),RMS_posterior_sigma=np.sqrt(np.exp(lv).mean(0)).tolist(),embedding_sha256=sha(q)))
result=dict(status='PASS',source_sha256=sha(__file__),condition=dict(n_target=24,measured_energy_bin='middle',fit_seed=2026111301),rows=rows,proof='I=mean_i KL(q_i||mean_j q_j) <= mean_ij KL(q_i||q_j) by convexity in second KL argument; q_i diagonalGaussian encoder, empirical uniformVALrows atfixedc',units='Natural logarithms give nats; divide bylog2 forbits',scope='Illustrative firstseed fixedcondition empirical distribution only. Deterministicmu canencode C_s while its tinyvariation is overwhelmed by posteriornoise. No newmodelinference, targetnormalization, probes, trainingororaclecalls.')
with (p/'posterior_information_bound.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result,indent=2),flush=True)

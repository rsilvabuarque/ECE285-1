"""Terminal TRAIN-only gradient-path audit; no optimization or TEST targets."""
from pathlib import Path
import argparse,json,hashlib,csv
import numpy as np
import torch
from cathode_cvae.runtime import require_compute_allocation
from cathode_cvae.graph import PeriodicGraph
from cathode_cvae.learning.generate_mask import load_model
from cathode_cvae.learning.runtime import trusted_checkpoint
from cathode_cvae.learning.train_mask import canonicalize,encode_train_val_conditions
from cathode_cvae.learning.physical_auxiliary import prepare_targets,make_head,head_prediction,auxiliary_latent
from cathode_cvae.learning.models import gaussian_kl
require_compute_allocation();torch.set_num_threads(4)
parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--run-root',type=Path,required=True);a=parser.parse_args();p=a.run_root.resolve();b=p.parent;r=b/'latent_diagnostics';a.output_dir.mkdir(exist_ok=False)
sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest();rows=[];bindings=[]
data=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project/artifacts/current/improvement_v1/training_round0.npz');parent=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project/artifacts/current/core/data/parents/O3_N48_pilot.npz')
with np.load(data,allow_pickle=False) as z:roles=z['split'].astype(str);occupancy=z['occupancy'];counts=z['n_target'] if 'n_target' in z else z['n_Li'];energy=z['energy_eV_atom']
with np.load(parent,allow_pickle=False) as z:perms=z['permutations']
train=np.flatnonzero(roles=='train');chosen=np.random.default_rng(2026111501).choice(train,size=512,replace=False)
paths=[b/'latent_phase2/fits'/f'lambda10_{mode}_d2_{seed}'/'last_update.pt' for mode in ['latent','condition','shuffled'] for seed in range(2026111301,2026111306)]
paths += [p/'fits'/f'mean_lambda10_{mode}_d2_{seed}'/'last_update.pt' for mode in ['latent','condition','shuffled'] for seed in range(2026111301,2026111306)]
for ckpath in paths:
 model,cfg=load_model(ckpath,'cpu');headcfg=cfg['physical_auxiliary'];graph=PeriodicGraph.from_dict(cfg['graph']);mode=headcfg['mode'];weight=headcfg['weight']
 cond,bins,_=encode_train_val_conditions(cfg['condition_schema'],counts,energy,roles,cfg['condition_mode'])
 raw,genuine,target,meta=prepare_targets(occupancy,counts,bins,roles,graph,mode=mode,seed=cfg['seed']+65537)
 for k,cell in meta['cells'].items():
  if cell!=headcfg['cells'][k]:raise ValueError('Target normalization differs from trained metadata')
 if mode=='shuffled' and meta['shuffle_map_sha256']!=headcfg['shuffle_map_sha256']:raise ValueError('Shuffled target permutation changed')
 # Independent nearest-edge product for actual sampled TRAIN masks.
 source,destination,images,distance=graph.mobile_edges();take=distance<=distance.min()+1e-5;vv=1-occupancy[chosen].astype(float);manual=(vv[:,source[take]]*vv[:,destination[take]]).sum(1)/int(take.sum())
 if not np.array_equal(raw[chosen],manual):raise ValueError('Independent pair-target check differs')
 xt,_=canonicalize(torch.tensor(occupancy[chosen],dtype=torch.float32),torch.tensor(np.argsort(perms,axis=1)));ct=torch.tensor(cond[chosen],dtype=torch.float32)
 head=make_head(model.latent_dim,model.condition_dim,mode,123,'cpu');ck=trusted_checkpoint(ckpath,'cpu');head.load_state_dict(ck['physical_auxiliary_state_dict']);torch.manual_seed(2026111503)
 mu,lv=model.encode(xt,ct);z=mu+torch.exp(.5*lv)*torch.randn_like(mu);recon=model.reconstruction(xt,ct,z,exact_count=True).mean();kl=gaussian_kl(mu,lv).mean();aux=(head_prediction(head,auxiliary_latent(mu,z,headcfg.get('posterior_input','sampled')),ct,mode)-torch.tensor(target[chosen])).square().mean();params=list(model.named_parameters())
 def norm(loss,prefix=None):
  pp=[v for name,v in params if prefix is None or name.startswith(prefix)];grads=torch.autograd.grad(loss,pp,retain_graph=True,allow_unused=True);return float(sum((g.square().sum() for g in grads if g is not None),torch.tensor(0.)).sqrt())
 lv_grad=torch.autograd.grad(weight*aux,lv,retain_graph=True,allow_unused=True)[0];lv_norm=0.0 if lv_grad is None else float(lv_grad.norm());e_aux=norm(weight*aux,'encoder.');d_aux=norm(weight*aux) # All-model = encoder path only.
 row=dict(family=ckpath.parent.name.rsplit('_',1)[0],seed=cfg['seed'],weight=weight,mode=mode,posterior_input=headcfg.get('posterior_input','sampled'),auxiliary_logvar_tensor_gradient_norm=lv_norm,rows=512,reconstruction=float(recon.detach()),KL=float(kl.detach()),auxiliary_MSE=float(aux.detach()),encoder_reconstruction_gradient_norm=norm(recon,'encoder.'),encoder_weighted_auxiliary_gradient_norm=e_aux,model_weighted_auxiliary_gradient_norm=d_aux,model_reconstruction_gradient_norm=norm(recon),model_combined_gradient_norm_before_clipping=norm(recon+.1*kl+weight*aux),zero_encoder_aux_expected=mode=='condition',target_manual_exact=True)
 if mode=='condition' and e_aux!=0:raise ValueError('Condition-only head incorrectly backpropagates into encoder')
 rows.append(row);bindings.append(dict(path=str(ckpath),sha256=sha(ckpath)))
with (a.output_dir/'terminal_TRAIN_gradient_audit.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
receipt=dict(status='PASS',source_sha256=sha(__file__),TRAIN_batch_seed=2026111501,posterior_noise_seed=2026111503,unweighted_rows=512,scope='Read-only terminal TRAIN minibatch gradient norms, diagnostic unweighted mean; this is not an optimizer step or convergence test. Genuine training used inherited/class-balanced weights. Condition-only generator path exactly zero required.',manual_target='Independent actual nearest-shell directed edge products exactly match prepared target for all512 TRAIN rows in every model',checkpoint_bindings=bindings,TEST_targets_computed=False)
(a.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('GRADIENT_AUDIT_PASS',len(rows),flush=True)

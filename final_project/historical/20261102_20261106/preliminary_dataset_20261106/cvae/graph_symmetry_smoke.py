"""Untrained architectural diagnostic: global versus site stochastic variables."""
import json,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ase import Atoms
from ase.neighborlist import neighbor_list
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sampling import conditional_count,top_count
ROOT=Path(__file__).resolve().parent.parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
torch.manual_seed(19);torch.set_num_threads(4);rng=np.random.default_rng(19)
with np.load(ROOT/'data/parents/O3_N48_pilot.npz') as z:p={k:z[k] for k in z.files}
li=Atoms(numbers=np.full(48,3),positions=p['positions_A'][p['li_indices']],cell=p['cell_A'],pbc=True)
i,j=neighbor_list('ij',li,3.1,self_interaction=False);A=np.zeros((48,48));np.add.at(A,(i,j),1);degree=A.sum(1);assert np.all(degree==6)
adj=torch.tensor(A/degree[:,None],dtype=torch.float64)
class Decoder(nn.Module):
    def __init__(self):
        super().__init__();self.self1=nn.Linear(12,32);self.neigh1=nn.Linear(12,32);self.self2=nn.Linear(32,32);self.neigh2=nn.Linear(32,32);self.out=nn.Linear(32,1)
    def forward(self,x,a):
        h=torch.nn.functional.silu(self.self1(x)+self.neigh1(a@x));h=torch.nn.functional.silu(self.self2(h)+self.neigh2(a@h));return self.out(h).squeeze(-1)
model=Decoder().double();B=256;global_z=torch.tensor(rng.normal(size=(B,8)),dtype=torch.float64)
global_features=torch.cat([torch.ones((B,48,1)),torch.full((B,48,1),.5),global_z[:,None,:].expand(-1,48,-1),torch.zeros((B,48,2))],dim=-1)
site_features=global_features.clone();site_features[:,:,-2:]=torch.tensor(rng.normal(size=(B,48,2)))
with torch.no_grad():g=model(global_features,adj).numpy();s=model(site_features,adj).numpy()
variance_global=float(g.var(axis=1).max());variance_site=float(s.var(axis=1).mean());assert variance_global<1e-20 and variance_site>1e-8
perm=p['permutations'][17];assert np.array_equal(A[perm][:,perm],A)
with torch.no_grad():equiv_error=float((model(site_features[:,perm],adj)-torch.tensor(s[:,perm])).abs().max())
assert equiv_error<1e-10
# Replace only roundoff-sized site differences by their exact symmetry limit.
constant=np.repeat(g.mean(axis=1,keepdims=True),48,axis=1)
det=np.zeros((B,48),np.uint8);det[:,:24]=1
uniform=conditional_count(constant,24,rng);site=top_count(s,24,rng)
inv=np.argsort(p['permutations'],axis=1);powers=1<<np.arange(47,-1,-1,dtype=np.int64)
def unique_orbits(o):return len({int(np.min((1-v)[inv].astype(np.int64)@powers)) for v in o})
counts=[unique_orbits(det),unique_orbits(uniform),unique_orbits(site)]
result={'status':'PASS','untrained_synthetic_architecture_test':True,'sites':48,'periodic_neighbors_per_site':6,'draws':B,'global_only_max_site_logit_variance':variance_global,'site_noise_mean_logit_variance':variance_site,'parent_symmetry_permutation_equivariance_max_error':equiv_error,'unique_orbits':dict(zip(['global_deterministic_ties','global_exact_count_sampling','site_noise_top_count'],counts)),'interpretation':'Identicalsitefeatures+globalz giveconstantmarginals. Exactcountsamplingcanstillproduceuniformmasks; energydependentorderingrequireslearnedjointcorrelations. Sitenoisebreakssamplesymmetrywhilelawremainsequivariant; thisdoesnotdemonstratelearnedenergyutility.'}
(ROOT/'validation/graph_symmetry_smoke.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True);axes[0].plot(g[0],label='Global latent only');axes[0].plot(s[0],label='Global + site noise');axes[0].set(xlabel='Parent Li-site index',ylabel='Untrained decoder logit');axes[0].legend(fontsize=8)
axes[1].bar(range(3),counts);axes[1].set_xticks(range(3),['Global: fixed ties','Global: count sampling','Site noise'],rotation=15);axes[1].set_ylabel('Distinct mask orbits / 256 draws');fig.suptitle('Untrained architecture diagnostic; no energy optimization tested')
fig.savefig(ROOT/'reports/graph_symmetry_smoke.pdf');fig.savefig(ROOT/'reports/graph_symmetry_smoke.png',dpi=160);print(result,flush=True)

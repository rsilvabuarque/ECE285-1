"""Variational local edits with exact numbers of removals and additions.

One means a present mobile ion. The seed is a genuine inference input. A child
is visible only to the posterior encoder and to likelihood evaluation. Removal
and addition sets refer to the original seed's occupied and vacant sites,
respectively, so a removal cannot be immediately undone by reinsertion.
"""
import numpy as np
import torch
from torch import nn

from .models import exact_count_nll, sample_exact_count


EDIT_FORMAT_VERSION=1
BIN_ORDER=("low","middle","high")


def encode_condition(n_target,energy_bin,swap_count,n_sites=48):
    """Dimensionless count fraction, measured/requested child bin, and k/3."""
    n,b,k=np.broadcast_arrays(np.asarray(n_target),np.asarray(energy_bin).astype(str),np.asarray(swap_count))
    if not np.equal(n,np.floor(n)).all() or not np.equal(k,np.floor(k)).all():
        raise ValueError("Counts and swap budgets must be integers")
    if np.any((n<0)|(n>n_sites)) or not np.isin(k,[1,2,3]).all() or not np.isin(b,BIN_ORDER).all():
        raise ValueError("Unsupported count, child bin or one/two/three swap budget")
    if np.any(np.minimum(n,n_sites-n)<k):raise ValueError("The seed cannot support the requested distinct edits")
    c=np.zeros(n.shape+(5,),np.float32);c[...,0]=n/n_sites
    for j,label in enumerate(BIN_ORDER):c[...,j+1]=b==label
    c[...,4]=k/3
    return c


def _counts(k,batch,device):
    result=torch.as_tensor(k,device=device,dtype=torch.long)
    if result.ndim==0:result=result.expand(batch)
    if result.shape!=(batch,) or bool(((result<1)|(result>3)).any()):
        raise ValueError("One swap budget in {1,2,3} is needed per seed")
    return result


def _validate_seed(seed,k):
    if seed.ndim!=2 or not bool(((seed==0)|(seed==1)).all()):
        raise ValueError("Seeds must be binary(batch,sites) occupations")
    counts=_counts(k,len(seed),seed.device)
    available=seed.sum(1)
    if bool(((available<counts)|(seed.shape[1]-available<counts)).any()):
        raise ValueError("The fixed seed lacks enough occupied or vacant sites")
    return counts


def _subset_nll(logits,target,eligible,k):
    """Exact-k likelihood on only the genuinely eligible seed sites."""
    if logits.shape!=target.shape or eligible.shape!=logits.shape:
        raise ValueError("Subset logits, targets and eligibility must agree")
    if bool(((target!=0)&(target!=1)).any()) or bool(((target>0)&~eligible).any()):
        raise ValueError("An edit selected an ineligible seed site")
    if not torch.equal(target.sum(1).long(),k):raise ValueError("Observed edit count differs from its declared budget")
    result=logits.new_zeros(len(logits));sizes=eligible.sum(1)
    for size in torch.unique(sizes).tolist():
        rows=torch.nonzero(sizes==size,as_tuple=False).flatten()
        positions=torch.nonzero(eligible[rows],as_tuple=False)[:,1].reshape(len(rows),int(size))
        values=logits[rows].gather(1,positions);observed=target[rows].gather(1,positions)
        result[rows]=exact_count_nll(values,observed,k[rows])
    return result


@torch.no_grad()
def _subset_sample(logits,eligible,k,*,uniforms=None,generator=None):
    if eligible.shape!=logits.shape:raise ValueError("One eligibility flag is needed per site")
    if uniforms is not None and (uniforms.shape!=logits.shape or not bool(((uniforms>=0)&(uniforms<1)).all())):
        raise ValueError("Subset uniforms must have(seed,site)shape and lie in[0,1)")
    result=torch.zeros_like(logits);sizes=eligible.sum(1)
    for size in torch.unique(sizes).tolist():
        rows=torch.nonzero(sizes==size,as_tuple=False).flatten()
        positions=torch.nonzero(eligible[rows],as_tuple=False)[:,1].reshape(len(rows),int(size))
        local_uniforms=None if uniforms is None else uniforms[rows].gather(1,positions)
        chosen=sample_exact_count(logits[rows].gather(1,positions),k[rows],generator=generator,uniforms=local_uniforms)
        result[rows]=result[rows].scatter(1,positions,chosen)
    return result


def edit_targets(seed,child,k):
    counts=_validate_seed(seed,k)
    if child.shape!=seed.shape or not bool(((child==0)|(child==1)).all()) or not torch.equal(child.sum(1),seed.sum(1)):
        raise ValueError("Children must be binary, same-parent and composition preserving")
    removed=((seed==1)&(child==0)).to(seed.dtype)
    added=((seed==0)&(child==1)).to(seed.dtype)
    if not torch.equal(removed.sum(1).long(),counts) or not torch.equal(added.sum(1).long(),counts):
        raise ValueError("A copied child or incorrect exact edit budget is not a supported target")
    return removed,added,counts


class SeedEditCVAE(nn.Module):
    """q(z|seed,child,c), then p(remove|seed,z,c)p(add|seed,remove,z,c)."""
    def __init__(self,n_sites=48,condition_dim=5,latent_dim=4,hidden_dim=128):
        super().__init__()
        if min(n_sites,condition_dim,latent_dim,hidden_dim)<1:raise ValueError("Editor dimensions must be positive")
        self.n_sites=int(n_sites);self.condition_dim=int(condition_dim)
        self.latent_dim=int(latent_dim);self.hidden_dim=int(hidden_dim)
        self.encoder=nn.Sequential(nn.Linear(2*n_sites+condition_dim,hidden_dim),nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim),nn.SiLU())
        self.mu=nn.Linear(hidden_dim,latent_dim);self.logvar=nn.Linear(hidden_dim,latent_dim)
        self.remove_decoder=nn.Sequential(nn.Linear(n_sites+condition_dim+latent_dim,hidden_dim),nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,n_sites))
        self.add_decoder=nn.Sequential(nn.Linear(2*n_sites+condition_dim+latent_dim,hidden_dim),nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,n_sites))

    def encode(self,seed,child,c):
        if seed.shape!=child.shape or seed.shape[1]!=self.n_sites or c.shape!=(len(seed),self.condition_dim):
            raise ValueError("Posterior seed, child and condition shapes disagree")
        h=self.encoder(torch.cat((seed,child,c),dim=-1))
        return self.mu(h),self.logvar(h).clamp(-12,8)

    def removal_logits(self,seed,c,z):
        return self.remove_decoder(torch.cat((seed,c,z),dim=-1))

    def addition_logits(self,seed,removed,c,z):
        return self.add_decoder(torch.cat((seed,removed,c,z),dim=-1))

    def reconstruction(self,seed,child,c,z,k):
        removed,added,counts=edit_targets(seed,child,k)
        remove_loss=_subset_nll(self.removal_logits(seed,c,z),removed,seed.bool(),counts)
        add_loss=_subset_nll(self.addition_logits(seed,removed,c,z),added,~seed.bool(),counts)
        return remove_loss+add_loss

    @torch.no_grad()
    def sample(self,seed,c,k,*,z=None,generator=None,uniforms_remove=None,uniforms_add=None):
        """Generate from seed,z,c only; no target child or posterior is consulted."""
        counts=_validate_seed(seed,k)
        if seed.shape[1]!=self.n_sites or c.shape!=(len(seed),self.condition_dim):
            raise ValueError("Inference seed/condition differs from this fixed-parent model")
        if z is None:z=torch.randn((len(seed),self.latent_dim),device=seed.device,dtype=seed.dtype,generator=generator)
        if z.shape!=(len(seed),self.latent_dim):raise ValueError("Explicit latent shape disagrees with checkpoint")
        removed=_subset_sample(self.removal_logits(seed,c,z),seed.bool(),counts,uniforms=uniforms_remove,generator=generator)
        added=_subset_sample(self.addition_logits(seed,removed,c,z),~seed.bool(),counts,uniforms=uniforms_add,generator=generator)
        child=seed-removed+added
        if not torch.equal(child.sum(1),seed.sum(1)) or not torch.equal((child!=seed).sum(1),2*counts):
            raise AssertionError("Exact seed-edit sampling lost its count or raw2k constraint")
        return dict(child=child,removed=removed,added=added)


@torch.no_grad()
def uniform_edits(seed,k,*,generator=None,uniforms_remove=None,uniforms_add=None):
    """Uniform exact-k subsets, with the identical seed/edit information."""
    counts=_validate_seed(seed,k);zero=torch.zeros_like(seed)
    removed=_subset_sample(zero,seed.bool(),counts,uniforms=uniforms_remove,generator=generator)
    added=_subset_sample(zero,~seed.bool(),counts,uniforms=uniforms_add,generator=generator)
    return dict(child=seed-removed+added,removed=removed,added=added)

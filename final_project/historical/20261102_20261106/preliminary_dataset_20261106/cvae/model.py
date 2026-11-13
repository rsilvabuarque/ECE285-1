"""Fixed-parent vacancy CVAE. Occupancy1 means present Li; no MLIP imports."""
import torch
from torch import nn
class MaskCVAE(nn.Module):
    def __init__(self,n_sites=48,condition_dim=2,latent_dim=8):
        super().__init__();self.n_sites=n_sites;self.condition_dim=condition_dim;self.latent_dim=latent_dim
        self.encoder=nn.Sequential(nn.Linear(n_sites+condition_dim,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU())
        self.mu=nn.Linear(64,latent_dim);self.logvar=nn.Linear(64,latent_dim)
        self.decoder=nn.Sequential(nn.Linear(latent_dim+condition_dim,64),nn.SiLU(),nn.Linear(64,128),nn.SiLU(),nn.Linear(128,n_sites))
    def encode(self,x,c):
        h=self.encoder(torch.cat([x,c],dim=-1));return self.mu(h),self.logvar(h).clamp(-12,8)
    def decode(self,z,c):return self.decoder(torch.cat([z,c],dim=-1))
    def forward(self,x,c):
        mu,logvar=self.encode(x,c);z=mu+torch.exp(.5*logvar)*torch.randn_like(mu)
        return self.decode(z,c),mu,logvar
def loss_terms(logits,x,mu,logvar):
    reconstruction=nn.functional.binary_cross_entropy_with_logits(logits,x,reduction='none').sum(dim=-1)
    kl=.5*(mu.square()+logvar.exp()-1-logvar).sum(dim=-1)
    return reconstruction,kl

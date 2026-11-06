"""Fixed parent atom-index displacement models; occupancy is supplied context."""
import torch
from torch import nn
class DisplacementModel(nn.Module):
    def __init__(self,context_dim=65,kind='cvae',latent_dim=16,coordinates=576):
        super().__init__();self.kind=kind;self.latent_dim=latent_dim;self.coordinates=coordinates
        if kind=='cvae':
            self.encoder=nn.Sequential(nn.Linear(context_dim+coordinates,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU())
            self.mu=nn.Linear(128,latent_dim);self.logvar=nn.Linear(128,latent_dim)
        self.decoder=nn.Sequential(nn.Linear(context_dim+(latent_dim if kind=='cvae' else 0),128),nn.SiLU(),nn.Linear(128,256),nn.SiLU(),nn.Linear(256,coordinates))
    def encode(self,u,c):
        h=self.encoder(torch.cat([u,c],-1));return self.mu(h),self.logvar(h).clamp(-12,8)
    def decode(self,z,c):return self.decoder(torch.cat([z,c],-1)) if self.kind=='cvae' else self.decoder(c)
    def forward(self,u,c):
        if self.kind=='regressor':return self.decoder(c),None,None
        mu,logvar=self.encode(u,c);return self.decode(mu+torch.exp(.5*logvar)*torch.randn_like(mu),c),mu,logvar

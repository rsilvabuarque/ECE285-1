"""Fixed-parent binary occupancy models for the bounded improvement experiment.

Occupancy one means a present mobile ion. ``condition[:, 0]`` is its requested
site fraction; other components encode versioned energy/descriptor classes.
Exact-count losses and draws preserve composition without post-hoc bit changes.
Optional physical graphs describe the fixed parent, not relaxed coordinates.
No MLIP, coordinate relaxation, or energy gradient is part of these models.
"""

import torch
from torch import nn
from torch.nn import functional as F


def gaussian_kl(mu, logvar):
    """Per-row KL to N(0,I); empty baseline latent dimensions yield zero."""
    return 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(dim=-1)


def _counts(n, batch, n_sites, device):
    counts = torch.as_tensor(n, device=device, dtype=torch.long)
    if counts.ndim == 0:
        counts = counts.expand(batch)
    if counts.shape != (batch,):
        raise ValueError("n must be a scalar or one count per batch row")
    return counts


def _log_partitions(logits):
    """Log partition for every cardinality, using a differentiable recurrence.

State k sums weights of k occupied sites. A large finite negative sentinel
avoids undefined logaddexp derivatives for two impossible (-inf) branches.
It is far below any weight reachable by this finite-logit FP32 experiment.
"""
    batch, n_sites = logits.shape
    negative = torch.finfo(logits.dtype).min / 100
    dp = logits.new_full((batch, n_sites + 1), negative)
    dp[:, 0] = 0
    unavailable = logits.new_full((batch, 1), negative)
    for i in range(n_sites):
        occupied = torch.cat((unavailable, dp[:, :-1] + logits[:, i:i + 1]), dim=1)
        dp = torch.logaddexp(dp, occupied)
    return dp


def exact_count_nll(logits, x, n=None):
    """Per-row -log p(x|sum(x)=n), for binary x and factorized logits.

The exact conditional likelihood is exp(sum_i logits_i*x_i)/Z_n.
This trains a count-constrained decoder directly, rather than projecting a
decoder trained with independent Bernoulli likelihood at inference time.
"""
    if logits.shape != x.shape:
        raise ValueError("logits and x must have the same (batch, sites) shape")
    if n is None:
        n = x.sum(dim=1).round().long()
    counts = _counts(n, len(x), x.shape[1], logits.device)
    log_z = _log_partitions(logits).gather(1, counts[:, None]).squeeze(1)
    return log_z - (logits * x).sum(dim=1)


@torch.no_grad()
def sample_exact_count(logits, n, generator=None, uniforms=None):
    """Exact conditional Bernoulli draws with row-specific target counts."""
    batch, n_sites = logits.shape
    remaining = _counts(n, batch, n_sites, logits.device).clone()
    if bool(((remaining < 0) | (remaining > n_sites)).any()):
        raise ValueError("Requested count must be between zero and n_sites")
    if uniforms is not None:
        if uniforms.shape != logits.shape or not bool(((uniforms >= 0) & (uniforms < 1)).all()):
            raise ValueError("Uniform draws must have shape (batch, sites) and lie in [0,1)")
    negative = torch.finfo(logits.dtype).min / 100
    suffix = [None] * (n_sites + 1)
    dp = logits.new_full((batch, n_sites + 1), negative)
    dp[:, 0] = 0
    suffix[n_sites] = dp
    unavailable = logits.new_full((batch, 1), negative)
    for i in range(n_sites - 1, -1, -1):
        occupied = torch.cat((unavailable, dp[:, :-1] + logits[:, i:i + 1]), dim=1)
        dp = torch.logaddexp(dp, occupied)
        suffix[i] = dp
    result = torch.zeros_like(logits)
    for i in range(n_sites):
        denominator = suffix[i].gather(1, remaining[:, None]).squeeze(1)
        numerator = logits[:, i] + suffix[i + 1].gather(
            1, (remaining - 1).clamp_min(0)[:, None]).squeeze(1)
        probability = (numerator - denominator).clamp_max(0).exp()
        random_value = (torch.rand(batch, device=logits.device, generator=generator)
                        if uniforms is None else uniforms[:, i])
        draw = random_value < probability
        take = ((remaining == n_sites - i) | draw) & (remaining > 0)
        result[:, i] = take.to(logits.dtype)
        remaining -= take.long()
    return result


class DenseCVAE(nn.Module):
    """Exact legacy architecture and state_dict names; count loss is optional."""
    def __init__(self, n_sites=48, condition_dim=2, latent_dim=8):
        super().__init__()
        self.n_sites = n_sites
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.has_latent = True
        self.encoder = nn.Sequential(nn.Linear(n_sites + condition_dim, 128), nn.SiLU(),
                                     nn.Linear(128, 64), nn.SiLU())
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim + condition_dim, 64), nn.SiLU(),
                                     nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, n_sites))

    def encode(self, x, c):
        h = self.encoder(torch.cat((x, c), dim=-1))
        return self.mu(h), self.logvar(h).clamp(-12, 8)

    def decode_logits(self, z, c, x=None):
        return self.decoder(torch.cat((z, c), dim=-1))

    def decode(self, z, c):
        return self.decode_logits(z, c)

    def reconstruction(self, x, c, z, exact_count=True):
        logits = self.decode_logits(z, c)
        if exact_count:
            return exact_count_nll(logits, x)
        return F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(-1)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode_logits(z, c), mu, logvar

    @torch.no_grad()
    def sample(self, c, n, z=None, generator=None, uniforms=None):
        if z is None:
            z = torch.randn((len(c), self.latent_dim), device=c.device,
                            dtype=c.dtype, generator=generator)
        return sample_exact_count(self.decode_logits(z, c), n, generator, uniforms)


class ARCVAE(nn.Module):
    """Dense encoder and teacher-forced GRU occupancy decoder.

Training predicts each site's occupancy from previous true occupancies.
Generation feeds back sampled occupancies. Forced zero/one choices needed to
reach the requested count have conditional probability one and zero loss.
Site order is the original fixed parent order, not an equivariance claim.
"""
    def __init__(self, n_sites=48, condition_dim=2, latent_dim=8, hidden_dim=64,
                 graph_encoder=None, graph_context=None, remaining_count_features=False):
        super().__init__()
        self.n_sites = n_sites
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.has_latent = latent_dim > 0
        self.graph_context = graph_context
        self.encoder_kind = "graph" if graph_encoder is not None else "dense"
        self.remaining_count_features=bool(remaining_count_features)
        if self.has_latent:
            if graph_encoder is None:
                self.encoder = nn.Sequential(nn.Linear(n_sites + condition_dim, 128), nn.SiLU(),
                                             nn.Linear(128, 64), nn.SiLU())
                self.mu = nn.Linear(64, latent_dim)
                self.logvar = nn.Linear(64, latent_dim)
            else:
                self.encoder = graph_encoder
        self.initial_hidden = nn.Linear(condition_dim + latent_dim, hidden_dim)
        context_dim = 0 if graph_context is None else graph_context.feature_dim
        self.gru = nn.GRU(2 + condition_dim + latent_dim + context_dim + 2*int(self.remaining_count_features), hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)

    def encode(self, x, c):
        if not self.has_latent:
            empty = x.new_zeros((len(x), 0))
            return empty, empty
        if self.encoder_kind == "graph":
            return self.encoder(x, c)
        h = self.encoder(torch.cat((x, c), dim=-1))
        return self.mu(h), self.logvar(h).clamp(-12, 8)

    def _context(self, z, c):
        return torch.cat((c, z), dim=-1) if self.has_latent else c

    def decode_logits(self, z, c, x=None):
        if x is None:
            raise ValueError("AR teacher-forced logits require x; use sample for new masks")
        ctx = self._context(z, c)
        previous = F.pad(x[:, :-1], (1, 0))[:, :, None]
        index = torch.linspace(0, 1, self.n_sites, device=x.device, dtype=x.dtype)
        site = index[None, :, None].expand(len(x), -1, -1)
        inputs = torch.cat((previous, ctx[:, None, :].expand(-1, self.n_sites, -1), site), dim=-1)
        if self.graph_context is not None:
            inputs = torch.cat((inputs, self.graph_context.teacher_features(x)), dim=-1)
        if self.remaining_count_features:
            # n is a supplied condition, never x.sum(): future target labels
            # must not enter a site's decoder through the count feature.
            target=torch.round(c[:,0]*self.n_sites)[:,None]
            used=F.pad(x[:,:-1],(1,0)).cumsum(1)
            left=torch.arange(self.n_sites,0,-1,device=x.device,dtype=x.dtype)[None].expand_as(x)
            inputs=torch.cat((inputs,torch.stack(((target-used)/self.n_sites,left/self.n_sites),dim=-1)),dim=-1)
        h0 = torch.tanh(self.initial_hidden(ctx))[None]
        values, _ = self.gru(inputs, h0)
        return self.output(values).squeeze(-1)

    def reconstruction(self, x, c, z, exact_count=True):
        logits = self.decode_logits(z, c, x)
        losses = F.binary_cross_entropy_with_logits(logits, x, reduction="none")
        if exact_count:
            target = x.sum(dim=1, keepdim=True)
            used_before = x.cumsum(dim=1) - x
            remaining = target - used_before
            sites_left = torch.arange(self.n_sites, 0, -1, device=x.device)[None]
            free = (remaining > 0) & (remaining < sites_left)
            losses = losses * free
        return losses.sum(dim=-1)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode_logits(z, c, x), mu, logvar

    @torch.no_grad()
    def sample(self, c, n, z=None, generator=None, uniforms=None):
        batch = len(c)
        if z is None:
            z = torch.randn((batch, self.latent_dim), device=c.device,
                            dtype=c.dtype, generator=generator)
        remaining = _counts(n, batch, self.n_sites, c.device).clone()
        if self.remaining_count_features and not torch.equal(torch.round(c[:,0]*self.n_sites).long(),remaining):
            raise ValueError("Condition count disagrees with the hard sampler count")
        if bool(((remaining < 0) | (remaining > self.n_sites)).any()):
            raise ValueError("Requested count must be between zero and n_sites")
        if uniforms is not None:
            if uniforms.shape != (batch, self.n_sites) or not bool(((uniforms >= 0) & (uniforms < 1)).all()):
                raise ValueError("Uniform draws must have shape (batch, sites) and lie in [0,1)")
        ctx = self._context(z, c)
        hidden = torch.tanh(self.initial_hidden(ctx))[None]
        previous = c.new_zeros((batch, 1))
        result = c.new_zeros((batch, self.n_sites))
        for i in range(self.n_sites):
            site = c.new_full((batch, 1), i / max(1, self.n_sites - 1))
            inputs = torch.cat((previous, ctx, site), dim=-1)
            if self.graph_context is not None:
                inputs = torch.cat((inputs, self.graph_context.step_features(result, i)), dim=-1)
            if self.remaining_count_features:
                left=c.new_full((batch,),self.n_sites-i)
                inputs=torch.cat((inputs,torch.stack((remaining.to(c.dtype)/self.n_sites,left/self.n_sites),dim=-1)),dim=-1)
            inputs = inputs[:, None, :]
            values, hidden = self.gru(inputs, hidden)
            probability = self.output(values[:, 0]).squeeze(-1).sigmoid()
            random_value = (torch.rand(batch, device=c.device, generator=generator)
                            if uniforms is None else uniforms[:, i])
            draw = random_value < probability
            take = ((remaining == self.n_sites - i) | draw) & (remaining > 0)
            previous = take.to(c.dtype)[:, None]
            result[:, i] = previous[:, 0]
            remaining -= take.long()
        return result


class ARBaseline(ARCVAE):
    """Same autoregressive decoder without latent variables or an encoder."""
    def __init__(self, n_sites=48, condition_dim=2, hidden_dim=64, graph_context=None,remaining_count_features=False):
        super().__init__(n_sites, condition_dim, latent_dim=0, hidden_dim=hidden_dim,
                         graph_context=graph_context,remaining_count_features=remaining_count_features)


def make_model(kind, n_sites=48, condition_dim=2, latent_dim=8, hidden_dim=64, cond_dim=None,
               encoder_kind="dense", graph_data=None, graph_context=False, context_spec=None,remaining_count_features=False):
    if cond_dim is not None:
        condition_dim = cond_dim
    if encoder_kind not in {"dense", "graph"}:
        raise ValueError("encoder_kind must be dense or graph")
    if latent_dim < 0:
        raise ValueError("latent_dim must be nonnegative")
    if not isinstance(hidden_dim, int) or hidden_dim < 1:
        raise ValueError("hidden_dim must be a positive integer")
    if context_spec is not None and not graph_context:
        raise ValueError("A decoder context specification requires graph_context")
    if remaining_count_features and kind not in {"ar","ar_cvae","ar_baseline"}:
        raise ValueError("Remaining-count features are implemented for AR decoders only")
    encoder = context = None
    if encoder_kind == "graph" or graph_context:
        if graph_data is None:
            raise ValueError("Graph variants require an explicit serialized physical graph")
        if kind not in {"ar", "ar_cvae", "ar_baseline"}:
            raise ValueError("Graph variants are implemented for autoregressive models only")
        from ..graph import PeriodicGraph, CausalGraphContext
        from .graph_model import GraphOccupationEncoder
        graph = PeriodicGraph.from_dict(graph_data)
        if graph.n_sites != n_sites:
            raise ValueError("Graph and model mobile-site dimensions differ")
        if encoder_kind == "graph":
            if kind == "ar_baseline":
                raise ValueError("The no-latent AR baseline has no encoder")
            encoder = GraphOccupationEncoder(graph, condition_dim, latent_dim)
        if graph_context:
            spec = {} if context_spec is None else context_spec
            if spec.get("version", "causal-context-v1") not in {"causal-context-v1","causal-context-v2"}:
                raise ValueError("Unsupported decoder context specification")
            mode=spec.get("kernel_mode","legacy")
            if mode!="legacy" and spec.get("version")!="causal-context-v2":raise ValueError("Radial-shell context needs its explicit version2 specification")
            context = CausalGraphContext(graph, spec.get("association", "physical"), spec.get("site_permutation"),mode,spec.get("shell_tolerance_A",1e-5))
            observed = context.to_spec()
            for key in ("feature_dim", "kernels_sha256"):
                if key in spec and spec[key] != observed[key]:
                    raise ValueError("Decoder context specification disagrees with reconstructed " + key)
    if kind in {"dense", "dense_bce", "dense_count"}:
        return DenseCVAE(n_sites, condition_dim, latent_dim)
    if kind in {"ar", "ar_cvae"}:
        return ARCVAE(n_sites, condition_dim, latent_dim, hidden_dim, encoder, context,remaining_count_features)
    if kind == "ar_baseline":
        return ARBaseline(n_sites, condition_dim, hidden_dim, context,remaining_count_features)
    raise ValueError(f"Unknown model kind {kind!r}")

"""Causal ACSF-inspired angular pair descriptors on a fixed parent graph.

The parent graph's directed periodic image edges are kept distinct until after
neighbor pairs are constructed. Inputs contain only already assigned mobile
occupations. Output is a sampler descriptor, not an energy or force model.
"""

import hashlib
import itertools
import math

import numpy as np
import torch
from torch import nn

from .graph import CausalGraphContext


class SmoothAngularContext(nn.Module):
    """Radial shell fractions plus smooth occupation-weighted pair features.

    For every center and unordered pair of distinct periodic image edges e,f,
    the fixed geometric weight is

      2**(1-zeta) * (1 + lambda*cos(theta))**zeta
          * exp[-alpha*((r_e/r_c)**2 + (r_f/r_c)**2)].

    r_c is the fixed *feature-graph* cutoff, not a force-field cutoff. All
    ratios and alpha are dimensionless. The default alpha=0 and alpha=2 provide
    broad and more short-distance-weighted profiles; zeta=1,2 and lambda=+/-1
    provide low-order angular responses. These are provisional model choices,
    not optimized tolerances or universal physical parameters.

    We sum over all mobile-mobile neighbor image pairs from the given graph,
    including cross-shell pairs. For each fixed basis the geometric weights are
    normalized by their sum at the center, then multiplied by two causally known
    occupied or two causally known vacant indicators. Future sites contribute to
    neither pair channel and remain explicitly unknown in the radial context.

    The angular and radial functions are smooth within the fixed graph. The
    existing graph uses a hard radius cutoff, so this context must not be used
    to claim a globally smooth coordinate-dependent potential or force field.
    """

    def __init__(self, graph, alphas=(0., 2.), zetas=(1, 2), lambdas=(1, -1),
                 shell_tolerance_A=1e-5):
        super().__init__()
        self.n_sites = graph.n_sites
        self.cutoff_A = float(graph.cutoff_A)
        self.shell_tolerance_A = float(shell_tolerance_A)
        self.alphas = tuple(float(x) for x in alphas)
        self.zetas = tuple(int(x) for x in zetas)
        self.lambdas = tuple(int(x) for x in lambdas)
        if (not self.alphas or not all(math.isfinite(x) and x >= 0 for x in self.alphas)
                or not self.zetas or any(int(x) != x or x < 1 for x in zetas)
                or not self.lambdas or any(int(x) != x or int(x) not in (-1, 1) for x in lambdas)):
            raise ValueError("angular basis needs nonnegative finite alphas, positive integer zetas, and lambda +/-1")
        self.radial = CausalGraphContext(
            graph, kernel_mode="radial_shells", shell_tolerance_A=shell_tolerance_A
        )
        self.graph_hash = graph.to_dict()["graph_hash"]
        self.basis = list(itertools.product(self.alphas, self.zetas, self.lambdas))

        source, destination, images, distances = graph.mobile_edges()
        mobile_pos = graph.parent.positions_A[graph.parent.mobile_indices]
        displacement = mobile_pos[destination] - mobile_pos[source] + images @ graph.parent.cell_A
        unit = displacement / distances[:, None]
        pairs = []
        for center in range(self.n_sites):
            edges = np.flatnonzero(source == center)
            pairs.append(list(itertools.combinations(edges.tolist(), 2)))
        max_pairs = max(map(len, pairs), default=0)
        left = np.zeros((self.n_sites, max_pairs), dtype=np.int64)
        right = np.zeros_like(left)
        valid = np.zeros((self.n_sites, max_pairs), dtype=bool)
        weights = np.zeros((len(self.basis), self.n_sites, max_pairs), dtype=np.float64)
        for center, center_pairs in enumerate(pairs):
            if not center_pairs:
                continue
            edges = np.asarray(center_pairs, dtype=np.int64)
            e, f = edges[:, 0], edges[:, 1]
            size = len(e)
            left[center, :size] = destination[e]
            right[center, :size] = destination[f]
            valid[center, :size] = True
            cosine = np.clip(np.sum(unit[e] * unit[f], axis=-1), -1., 1.)
            distance_sq = (distances[e] / self.cutoff_A) ** 2 + (distances[f] / self.cutoff_A) ** 2
            for index, (alpha, zeta, lam) in enumerate(self.basis):
                angular = 2. ** (1 - zeta) * (1. + lam * cosine) ** zeta
                unnormalized = angular * np.exp(-alpha * distance_sq)
                total = unnormalized.sum()
                if total > 0:
                    weights[index, center, :size] = unnormalized / total

        self.register_buffer("pair_left", torch.as_tensor(left, dtype=torch.long))
        self.register_buffer("pair_right", torch.as_tensor(right, dtype=torch.long))
        self.register_buffer("pair_valid", torch.as_tensor(valid, dtype=torch.bool))
        self.register_buffer("pair_weights", torch.as_tensor(weights, dtype=torch.float32))
        self.feature_dim = self.radial.feature_dim + 2 * len(self.basis)

    @staticmethod
    def _products(left, right, known):
        # Mask each factor before the product. In particular, a future NaN is
        # neither an assigned vacancy nor an input to arithmetic or gradients.
        zero = torch.zeros_like(left)
        left_occupied = torch.where(known, left, zero)
        right_occupied = torch.where(known, right, zero)
        left_vacant = torch.where(known, 1 - left, zero)
        right_vacant = torch.where(known, 1 - right, zero)
        return left_occupied * right_occupied, left_vacant * right_vacant

    def features(self, occupancy, assigned):
        """Features for all centers with one explicit assignment mask per row."""
        if (occupancy.ndim != 2 or occupancy.shape[1] != self.n_sites
                or assigned.shape != occupancy.shape):
            raise ValueError("occupancy and assigned must be matching (batch, sites) arrays")
        assigned = assigned.bool()
        known = (self.pair_valid[None] & assigned[:, self.pair_left]
                 & assigned[:, self.pair_right])
        oo, vv = self._products(occupancy[:, self.pair_left], occupancy[:, self.pair_right], known)
        weights = self.pair_weights.to(dtype=occupancy.dtype)
        angular_oo = torch.einsum("bim,kim->bik", oo, weights)
        angular_vv = torch.einsum("bim,kim->bik", vv, weights)
        return torch.cat((self.radial.features(occupancy, assigned), angular_oo, angular_vv), dim=-1)

    def teacher_features(self, occupancy):
        """At center i, use only ground-truth occupations with index j<i."""
        if occupancy.ndim != 2 or occupancy.shape[1] != self.n_sites:
            raise ValueError("occupancy must have shape (batch, mobile sites)")
        center = torch.arange(self.n_sites, device=occupancy.device)[:, None]
        known = (self.pair_valid & (self.pair_left < center) & (self.pair_right < center))[None]
        oo, vv = self._products(occupancy[:, self.pair_left], occupancy[:, self.pair_right], known)
        weights = self.pair_weights.to(dtype=occupancy.dtype)
        angular_oo = torch.einsum("bim,kim->bik", oo, weights)
        angular_vv = torch.einsum("bim,kim->bik", vv, weights)
        return torch.cat((self.radial.teacher_features(occupancy), angular_oo, angular_vv), dim=-1)

    def step_features(self, occupancy, step):
        """Features for one autoregressive site using its sampled prefix."""
        if occupancy.ndim != 2 or occupancy.shape[1] != self.n_sites:
            raise ValueError("occupancy must have shape (batch, mobile sites)")
        if not 0 <= step < self.n_sites:
            raise ValueError("step outside mobile-site range")
        left, right = self.pair_left[step], self.pair_right[step]
        known = (self.pair_valid[step] & (left < step) & (right < step))[None]
        oo, vv = self._products(occupancy[:, left], occupancy[:, right], known)
        weights = self.pair_weights[:, step].to(dtype=occupancy.dtype)
        angular_oo = torch.einsum("bm,km->bk", oo, weights)
        angular_vv = torch.einsum("bm,km->bk", vv, weights)
        return torch.cat((self.radial.step_features(occupancy, step), angular_oo, angular_vv), dim=-1)

    def to_spec(self):
        """Describe the architecture and bind its immutable pair geometry."""
        digest = hashlib.sha256()
        for value, dtype in ((self.pair_left, "<i8"), (self.pair_right, "<i8"),
                             (self.pair_valid, "u1"), (self.pair_weights, "<f4")):
            digest.update(value.detach().cpu().numpy().astype(dtype).tobytes())
        return {
            "version": "causal-angular-smooth-v1",
            "association": "physical",
            "kernel_mode": "angular_smooth",
            "graph_hash": self.graph_hash,
            "shell_tolerance_A": self.shell_tolerance_A,
            "feature_graph_cutoff_A": self.cutoff_A,
            "dimensionless_radial_alphas": list(self.alphas),
            "angular_zetas": list(self.zetas),
            "angular_lambdas": list(self.lambdas),
            "basis_order": [list(x) for x in self.basis],
            "pair_states": ["occupied_occupied", "vacant_vacant"],
            "feature_dim": self.feature_dim,
            "radial_spec": self.radial.to_spec(),
            "angular_buffers_sha256": digest.hexdigest(),
            "scope": "Fixed-parent sampler features; hard feature-graph cutoff is not an MLIP force cutoff",
        }

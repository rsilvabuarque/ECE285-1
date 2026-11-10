"""Causal angular pair histograms for an autoregressive fixed-parent sampler.

Angles describe candidate mobile-site images, including vacancies. The decoder
must never see an unassigned occupation; angles and cell geometry are known.
The output is an invariant local descriptor, not a claim that the fixed-order
generator is itself permutation equivariant or a differentiable force field.
"""

import hashlib
from itertools import combinations

import numpy as np
import torch
from torch import nn

from .graph import CausalGraphContext


class AngularHistogramContext(nn.Module):
    """Radial fractions plus same-state nearest-shell pair-angle fractions.

    Three fixed angular bins, with boundaries 90 and 150 degrees, correspond
    to the 60/120/180 degree pair classes of the O3 Li nearest-neighbor ring.
    The bins are a provisional representation choice, not a physical cutoff
    or an optimality claim. Other parent geometries can use the same coarse
    bins, whose effect must be validated for that geometry.
    """

    def __init__(self, graph, shell_tolerance_A=1e-5):
        super().__init__()
        self.n_sites = graph.n_sites
        self.radial = CausalGraphContext(graph, kernel_mode="radial_shells",
                                         shell_tolerance_A=shell_tolerance_A)
        self.shell_tolerance_A = float(shell_tolerance_A)
        source, destination, images, distance = graph.mobile_edges()
        positions = graph.parent.positions_A[graph.parent.mobile_indices]
        vectors = positions[destination] - positions[source] + images @ graph.parent.cell_A
        nearest = float(np.min(distance))
        # Each edge is one periodic image. Pair images before any aggregation
        # of edges with a common site index, preserving their distinct angles.
        edge_groups = [np.flatnonzero((source == i) &
                        (distance <= nearest + shell_tolerance_A)) for i in range(self.n_sites)]
        pair_groups = [list(combinations(group.tolist(), 2)) for group in edge_groups]
        max_pairs = max(1, max(map(len, pair_groups), default=0))
        first = np.zeros((self.n_sites, max_pairs), dtype=np.int64)
        second = np.zeros_like(first)
        valid = np.zeros_like(first, dtype=bool)
        weights = np.zeros((self.n_sites, 3, max_pairs), dtype=np.float64)
        for i, pairs in enumerate(pair_groups):
            for p, (e, f) in enumerate(pairs):
                first[i, p], second[i, p] = destination[e], destination[f]
                valid[i, p] = True
                cosine = np.dot(vectors[e], vectors[f]) / (distance[e] * distance[f])
                # Nearest angular centers 60, 120, 180 degrees have boundaries
                # 90 and 150 degrees. Pair identity remains image-aware.
                angle = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
                angle_bin = 0 if angle < 90. else (1 if angle < 150. else 2)
                weights[i, angle_bin, p] = 1.
        counts = weights.sum(axis=-1, keepdims=True)
        weights = np.divide(weights, counts, out=np.zeros_like(weights), where=counts > 0)
        self.register_buffer("pair_first", torch.as_tensor(first, dtype=torch.long))
        self.register_buffer("pair_second", torch.as_tensor(second, dtype=torch.long))
        self.register_buffer("pair_valid", torch.as_tensor(valid, dtype=torch.bool))
        self.register_buffer("pair_weights", torch.as_tensor(weights, dtype=torch.float32))
        self.feature_dim = self.radial.feature_dim + 6
        self.graph_hash = graph.to_dict()["graph_hash"]

    def _angular(self, occupancy, assigned, first, second, valid, weights):
        # Where masking occurs BEFORE arithmetic; NaN future occupation labels
        # therefore cannot leak through a zero multiplier or its gradient.
        left_known = assigned[..., first] & valid
        right_known = assigned[..., second] & valid
        left = torch.where(left_known, occupancy[..., first], torch.zeros_like(occupancy[..., first]))
        right = torch.where(right_known, occupancy[..., second], torch.zeros_like(occupancy[..., second]))
        vacant_left = torch.where(left_known, 1 - occupancy[..., first], torch.zeros_like(left))
        vacant_right = torch.where(right_known, 1 - occupancy[..., second], torch.zeros_like(right))
        occupied_pairs = left * right
        vacant_pairs = vacant_left * vacant_right
        if first.ndim == 2:
            present = torch.einsum("bip,ikp->bik", occupied_pairs, weights)
            vacant = torch.einsum("bip,ikp->bik", vacant_pairs, weights)
        else:
            present = torch.einsum("bp,kp->bk", occupied_pairs, weights)
            vacant = torch.einsum("bp,kp->bk", vacant_pairs, weights)
        return torch.cat((present, vacant), dim=-1)

    def features(self, occupancy, assigned):
        if occupancy.ndim != 2 or occupancy.shape[1] != self.n_sites or assigned.shape != occupancy.shape:
            raise ValueError("occupancy/assigned must be matching (batch, mobile sites)")
        angular = self._angular(occupancy, assigned.bool(), self.pair_first, self.pair_second,
                                self.pair_valid, self.pair_weights.to(dtype=occupancy.dtype))
        return torch.cat((self.radial.features(occupancy, assigned), angular), dim=-1)

    def teacher_features(self, occupancy):
        if occupancy.ndim != 2 or occupancy.shape[1] != self.n_sites:
            raise ValueError("occupancy must have shape (batch, mobile sites)")
        position = torch.arange(self.n_sites, device=occupancy.device)[:, None]
        first_known = (self.pair_first < position) & self.pair_valid
        second_known = (self.pair_second < position) & self.pair_valid
        a, b = occupancy[:, self.pair_first], occupancy[:, self.pair_second]
        present_a = torch.where(first_known[None], a, torch.zeros_like(a))
        present_b = torch.where(second_known[None], b, torch.zeros_like(b))
        vacant_a = torch.where(first_known[None], 1-a, torch.zeros_like(a))
        vacant_b = torch.where(second_known[None], 1-b, torch.zeros_like(b))
        weights = self.pair_weights.to(dtype=occupancy.dtype)
        present = torch.einsum("bip,ikp->bik", present_a * present_b, weights)
        vacant = torch.einsum("bip,ikp->bik", vacant_a * vacant_b, weights)
        return torch.cat((self.radial.teacher_features(occupancy), present, vacant), dim=-1)

    def step_features(self, occupancy, step):
        if not 0 <= step < self.n_sites:
            raise ValueError("step outside mobile-site range")
        assigned = torch.arange(self.n_sites, device=occupancy.device)[None] < step
        angular = self._angular(occupancy, assigned.expand_as(occupancy),
                                self.pair_first[step], self.pair_second[step],
                                self.pair_valid[step], self.pair_weights[step].to(dtype=occupancy.dtype))
        return torch.cat((self.radial.step_features(occupancy, step), angular), dim=-1)

    def to_spec(self):
        digest = hashlib.sha256()
        for name in ("pair_first", "pair_second", "pair_valid", "pair_weights"):
            value = getattr(self, name).detach().cpu().numpy()
            digest.update(name.encode())
            digest.update(value.tobytes())
        return dict(version="causal-angular-hist-v1", kernel_mode="angle_hist",
                    association="physical", shell_tolerance_A=self.shell_tolerance_A,
                    angle_bin_boundaries_degrees=[90., 150.],
                    feature_dim=self.feature_dim, radial_spec=self.radial.to_spec(),
                    angular_buffers_sha256=digest.hexdigest(), graph_hash=self.graph_hash,
                    scope="Nearest mobile-site image pairs; occupied-occupied and vacancy-vacancy; causal prefix")


class RedundantRadialContext(nn.Module):
    """Control with extra GRU inputs containing no new angular information.

    Additional columns repeat existing radial fractions. This matches raw
    decoder input dimension/parameter count, but collinearity means it does
    not equalize every notion of effective model capacity or optimization.
    """

    def __init__(self, graph, extra_features=6, shell_tolerance_A=1e-5, learned_mix=False):
        super().__init__()
        if not isinstance(extra_features, int) or extra_features < 1:
            raise ValueError("extra_features must be a positive integer")
        self.radial = CausalGraphContext(graph, kernel_mode="radial_shells",
                                         shell_tolerance_A=shell_tolerance_A)
        self.extra_features = extra_features
        self.learned_mix = bool(learned_mix)
        if self.learned_mix:
            # Six-by-six linear mixing adds 36 weights for the directional
            # context's raw-parameter control but no new geometry information.
            self.mix = nn.Linear(extra_features, extra_features, bias=False)
            nn.init.eye_(self.mix.weight)
        self.shell_tolerance_A = float(shell_tolerance_A)
        self.feature_dim = self.radial.feature_dim + extra_features

    def _append(self, radial):
        idx = torch.arange(self.extra_features, device=radial.device) % self.radial.feature_dim
        repeated = radial.index_select(-1, idx)
        return torch.cat((radial, self.mix(repeated) if self.learned_mix else repeated), dim=-1)

    def features(self, occupancy, assigned):
        return self._append(self.radial.features(occupancy, assigned))

    def teacher_features(self, occupancy):
        return self._append(self.radial.teacher_features(occupancy))

    def step_features(self, occupancy, step):
        return self._append(self.radial.step_features(occupancy, step))

    def to_spec(self):
        return dict(version="causal-redundant-radial-v1", kernel_mode="redundant_radial",
                    association="physical", shell_tolerance_A=self.shell_tolerance_A,
                    extra_features=self.extra_features, learned_mix=self.learned_mix, feature_dim=self.feature_dim,
                    radial_spec=self.radial.to_spec(),
                    scope="Repeated radial inputs; raw parameter-count control, not new geometry")

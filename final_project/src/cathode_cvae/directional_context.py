"""Learned directional graph messages for causal fixed-parent decoding.

The node states are occupied, vacant, and undecided candidate mobile sites.
Periodic image edges remain distinct while unit vectors and rank-two tensors
are assembled. Scalar outputs are invariant under global O(3) transformations;
the *complete* fixed-order autoregressive generator is not thereby equivariant.
No future occupation, energy label, or relaxation enters these descriptors.
"""

import hashlib
import math

import numpy as np
import torch
from torch import nn

from .graph import CausalGraphContext


class DirectionalGraphContext(nn.Module):
    """Radial shells plus a learnable, equivariant l=1/l=2 message layer.

    Each directed mobile-site edge carries the unit displacement u from its
    center to a particular periodic image. For each shell and occupation state,
    the edge messages u and u u^T - I/3 are summed and divided by the number of
    geometric shell neighbors. Two learned scalar weight banks mix those state
    and shell channels into ``channels`` vector and symmetric-traceless tensor
    channels. Their Gram-matrix entries are rotational/reflection invariants.

    The output appends both upper-triangular Gram matrices to the existing
    occupied/vacant/undecided radial fractions. Learned scalar weights are part
    of the checkpoint; only geometric buffers should be checked against the
    reconstruction from ``graph``. The architecture supports arbitrary parent
    species and fixed cells, but has not established cross-parent transfer.
    """

    def __init__(self, graph, channels=2, shell_tolerance_A=1e-5):
        super().__init__()
        if not isinstance(channels, int) or channels < 1:
            raise ValueError("directional channels must be a positive integer")
        self.n_sites = graph.n_sites
        self.channels = channels
        self.radial = CausalGraphContext(
            graph, kernel_mode="radial_shells", shell_tolerance_A=shell_tolerance_A
        )
        self.shell_tolerance_A = float(shell_tolerance_A)
        self.shell_centers_A = list(self.radial.shell_centers_A)
        self.n_shells = len(self.shell_centers_A)
        self.graph_hash = graph.to_dict()["graph_hash"]

        source, destination, images, distance = graph.mobile_edges()
        mobile_pos = graph.parent.positions_A[graph.parent.mobile_indices]
        displacement = mobile_pos[destination] - mobile_pos[source] + images @ graph.parent.cell_A
        unit = displacement / distance[:, None]
        centers = np.asarray(self.shell_centers_A)
        shell = np.argmin(np.abs(distance[:, None] - centers[None]), axis=1)
        if np.any(np.abs(distance - centers[shell]) > self.shell_tolerance_A):
            raise ValueError("directional edge does not match its radial shell")

        shape = (self.n_shells, self.n_sites, self.n_sites)
        count = np.zeros(shape, dtype=np.float64)
        vector = np.zeros(shape + (3,), dtype=np.float64)
        quadrupole = np.zeros(shape + (3, 3), dtype=np.float64)
        np.add.at(count, (shell, source, destination), 1.0)
        np.add.at(vector, (shell, source, destination), unit)
        q_edge = unit[:, :, None] * unit[:, None, :] - np.eye(3)[None] / 3.0
        np.add.at(quadrupole, (shell, source, destination), q_edge)

        # Normalization counts geometric image edges, not present atoms. The
        # denominator is independent of occupations and remains valid when all
        # neighbors are undecided. An empty shell row contributes zero.
        degree = count.sum(axis=-1, keepdims=True)
        vector = np.divide(vector, degree[..., None], out=np.zeros_like(vector),
                           where=degree[..., None] > 0)
        quadrupole = np.divide(quadrupole, degree[..., None, None],
                               out=np.zeros_like(quadrupole),
                               where=degree[..., None, None] > 0)
        self.register_buffer("vector_geometry", torch.as_tensor(vector, dtype=torch.float32))
        self.register_buffer("quadrupole_geometry", torch.as_tensor(quadrupole, dtype=torch.float32))
        gram_i, gram_j = torch.triu_indices(channels, channels)
        self.register_buffer("gram_i", gram_i)
        self.register_buffer("gram_j", gram_j)

        # A state/shell-dependent scalar weight produces an equivariant vector
        # or tensor message. Contractions mix directions without a fixed frame.
        self.vector_weights = nn.Parameter(torch.empty(channels, 3 * self.n_shells))
        self.quadrupole_weights = nn.Parameter(torch.empty(channels, 3 * self.n_shells))
        nn.init.xavier_uniform_(self.vector_weights)
        nn.init.xavier_uniform_(self.quadrupole_weights)
        self.feature_dim = self.radial.feature_dim + channels * (channels + 1)

    @staticmethod
    def _states(occupancy, assigned):
        if occupancy.ndim != 2 or assigned.shape != occupancy.shape:
            raise ValueError("occupancy and assigned must be matching (batch, sites) arrays")
        assigned = assigned.to(dtype=torch.bool)
        # Mask before arithmetic so NaN sentinels in unassigned sites cannot
        # contaminate features or gradients of an earlier decision.
        present = torch.where(assigned, occupancy, torch.zeros_like(occupancy))
        vacant = torch.where(assigned, 1 - occupancy, torch.zeros_like(occupancy))
        unknown = (~assigned).to(dtype=occupancy.dtype)
        return torch.stack((present, vacant, unknown), dim=-1)

    def _readout(self, vectors, tensors):
        # Last two axes of vectors/tensors are (shell*state, 3/9); leading axes
        # are either (batch, site) for teacher forcing or just (batch) at a step.
        vector = torch.einsum("...md,cm->...cd", vectors, self.vector_weights)
        tensor = torch.einsum("...md,cm->...cd", tensors, self.quadrupole_weights)
        vector_gram = torch.einsum("...ad,...bd->...ab", vector, vector)
        tensor_gram = torch.einsum("...ad,...bd->...ab", tensor, tensor)
        return torch.cat((vector_gram[..., self.gram_i, self.gram_j],
                          tensor_gram[..., self.gram_i, self.gram_j]), dim=-1)

    def features(self, occupancy, assigned):
        """Features for all centers using one explicit assigned mask per row."""
        if occupancy.ndim != 2 or occupancy.shape[1] != self.n_sites:
            raise ValueError("occupancy must have shape (batch, mobile sites)")
        states = self._states(occupancy, assigned)
        vector = self.vector_geometry.to(dtype=occupancy.dtype)
        tensor = self.quadrupole_geometry.to(dtype=occupancy.dtype).flatten(-2)
        v = torch.einsum("bjc,sijd->biscd", states, vector).flatten(-3, -2)
        q = torch.einsum("bjc,sijd->biscd", states, tensor).flatten(-3, -2)
        return torch.cat((self.radial.features(occupancy, assigned), self._readout(v, q)), dim=-1)

    def teacher_features(self, occupancy):
        """Feature at i uses only true occupations at positions j<i."""
        if occupancy.ndim != 2 or occupancy.shape[1] != self.n_sites:
            raise ValueError("occupancy must have shape (batch, mobile sites)")
        past = (torch.arange(self.n_sites, device=occupancy.device)[None, :]
                < torch.arange(self.n_sites, device=occupancy.device)[:, None])
        present = torch.where(past[None], occupancy[:, None, :],
                              torch.zeros_like(occupancy[:, None, :]))
        vacant = torch.where(past[None], 1 - occupancy[:, None, :],
                             torch.zeros_like(occupancy[:, None, :]))
        vector = self.vector_geometry.to(dtype=occupancy.dtype)
        tensor = self.quadrupole_geometry.to(dtype=occupancy.dtype).flatten(-2)

        def moments(geometry):
            known_present = torch.einsum("bij,sijd->bisd", present, geometry)
            known_vacant = torch.einsum("bij,sijd->bisd", vacant, geometry)
            total = geometry.sum(dim=2).permute(1, 0, 2)[None]
            unknown = total - known_present - known_vacant
            return torch.stack((known_present, known_vacant, unknown), dim=-2).flatten(-3, -2)

        v, q = moments(vector), moments(tensor)
        return torch.cat((self.radial.teacher_features(occupancy), self._readout(v, q)), dim=-1)

    def step_features(self, occupancy, step):
        """Features only for the current center and its sampled prefix."""
        if occupancy.ndim != 2 or occupancy.shape[1] != self.n_sites:
            raise ValueError("occupancy must have shape (batch, mobile sites)")
        if not 0 <= step < self.n_sites:
            raise ValueError("step outside mobile-site range")
        assigned = torch.arange(self.n_sites, device=occupancy.device)[None] < step
        states = self._states(occupancy, assigned.expand_as(occupancy))
        vector = self.vector_geometry[:, step].to(dtype=occupancy.dtype)
        tensor = self.quadrupole_geometry[:, step].to(dtype=occupancy.dtype).flatten(-2)
        v = torch.einsum("bjc,sjd->bscd", states, vector).flatten(-3, -2)
        q = torch.einsum("bjc,sjd->bscd", states, tensor).flatten(-3, -2)
        return torch.cat((self.radial.step_features(occupancy, step), self._readout(v, q)), dim=-1)

    def to_spec(self):
        """Architecture and immutable geometry needed for safe reconstruction."""
        digest = hashlib.sha256()
        for value in (self.vector_geometry, self.quadrupole_geometry):
            digest.update(value.detach().cpu().numpy().astype("<f4").tobytes())
        fixed_digest = digest.copy()
        for value in (self.gram_i, self.gram_j):
            fixed_digest.update(value.detach().cpu().numpy().astype("<i8").tobytes())
        return {
            "version": "causal-directional-v1",
            "association": "physical",
            "kernel_mode": "directional_l1_l2",
            "graph_hash": self.graph_hash,
            "shell_tolerance_A": self.shell_tolerance_A,
            "shell_centers_A": self.shell_centers_A,
            "channels": self.channels,
            "feature_dim": self.feature_dim,
            "radial_spec": self.radial.to_spec(),
            "directional_geometry_sha256": digest.hexdigest(),
            "directional_buffers_sha256": fixed_digest.hexdigest(),
            "scope": "One learned equivariant message layer; invariant readout; fixed-order decoder not equivariant",
        }

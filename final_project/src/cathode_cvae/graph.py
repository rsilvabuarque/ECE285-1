"""Periodic parent graphs and causal occupation descriptors.

Edges are geometric neighbors, not established ion migration pathways.  Every
directed edge retains its integer cell image; distinct periodic images are not
silently merged before distance features are built. No occupation labels enter
the graph construction.
"""

from dataclasses import dataclass
import json

import numpy as np
import torch
from torch import nn

from .parent_spec import ParentSpec, _json_hash


@dataclass
class PeriodicGraph:
    parent: ParentSpec
    source: np.ndarray
    destination: np.ndarray
    image_shifts: np.ndarray
    distances_A: np.ndarray
    cutoff_A: float
    rule: str = "ase-neighbor-list-directed-cutoff-v1"

    def __post_init__(self):
        self.source = np.asarray(self.source, dtype=np.int64)
        self.destination = np.asarray(self.destination, dtype=np.int64)
        self.image_shifts = np.asarray(self.image_shifts, dtype=np.int64)
        self.distances_A = np.asarray(self.distances_A, dtype=float)
        if (self.source.ndim != 1 or self.destination.shape != self.source.shape
                or self.distances_A.shape != self.source.shape or self.image_shifts.shape != (len(self.source), 3)):
            raise ValueError("invalid periodic edge arrays")
        if np.any(self.source < 0) or np.any(self.destination < 0) or np.any(self.source >= len(self.parent.numbers)) or np.any(self.destination >= len(self.parent.numbers)):
            raise ValueError("edge index outside full parent")
        if not np.isfinite(self.distances_A).all() or np.any(self.distances_A <= 0):
            raise ValueError("edges must have positive finite physical distances")
        if not np.isfinite(self.cutoff_A) or self.cutoff_A <= 0 or np.any(self.distances_A >= self.cutoff_A + 1e-10):
            raise ValueError("edge distances exceed positive cutoff")
        if np.any(self.image_shifts[:, ~self.parent.pbc]):
            raise ValueError("periodic edge shift in a nonperiodic direction")
        delta = (self.parent.positions_A[self.destination] - self.parent.positions_A[self.source]
                 + self.image_shifts @ self.parent.cell_A)
        if not np.allclose(np.linalg.norm(delta, axis=1), self.distances_A, atol=1e-9, rtol=1e-10):
            raise ValueError("edge distances do not agree with coordinates/cell images")

    @property
    def n_sites(self):
        return self.parent.n_sites

    @staticmethod
    def _identity(value):
        # A geometric graph's identity must not depend on an NPZ container hash,
        # audit software version, or numerical residual in provenance. These are
        # retained in the serialized parent but are not different model inputs.
        payload = {k: value[k] for k in ("schema", "source", "destination", "image_shifts", "distances_A", "cutoff_A", "rule")}
        payload["parent_hash"] = value["parent"]["parent_hash"]
        return _json_hash(payload)

    def to_dict(self):
        d = dict(schema="periodic-parent-graph-v1", parent=self.parent.to_dict(),
                 source=self.source.tolist(), destination=self.destination.tolist(),
                 image_shifts=self.image_shifts.tolist(), distances_A=self.distances_A.tolist(),
                 cutoff_A=float(self.cutoff_A), rule=self.rule,
                 scope="Full host chemistry plus all candidate mobile sites; geometric edges are not barrier labels")
        return dict(d, graph_hash=self._identity(d))

    @classmethod
    def from_dict(cls, value):
        d = dict(value)
        expected = d.pop("graph_hash", None)
        if d.get("schema") != "periodic-parent-graph-v1" or expected != cls._identity(d):
            raise ValueError("unrecognized or modified periodic graph")
        return cls(ParentSpec.from_dict(d["parent"]), d["source"], d["destination"],
                   d["image_shifts"], d["distances_A"], d["cutoff_A"], d["rule"])

    def mobile_edges(self):
        mapping = np.full(len(self.parent.numbers), -1, dtype=int)
        mapping[self.parent.mobile_indices] = np.arange(self.n_sites)
        keep = (mapping[self.source] >= 0) & (mapping[self.destination] >= 0)
        return mapping[self.source[keep]], mapping[self.destination[keep]], self.image_shifts[keep], self.distances_A[keep]

    def shell_audit(self, shell_tolerance_A=1e-5):
        """Report connected components, shells and mobile layer communication.

        The shell merge tolerance is an explicitly reported numerical grouping
        device, not a physical interaction radius or diffusion criterion.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        def components(src, dst, n):
            mat = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
            number, labels = connected_components(mat, directed=False)
            return int(number), np.bincount(labels).tolist()
        def shells(distances):
            result = []
            for d in np.sort(distances):
                if not result or d - result[-1][0] > shell_tolerance_A:
                    result.append([float(d), 1])
                else:
                    result[-1][1] += 1
            return [dict(distance_A=x, directed_image_edges=n) for x, n in result]
        a, b, shifts, distances = self.mobile_edges()
        nf, sf = components(self.source, self.destination, len(self.parent.numbers))
        nm, sm = components(a, b, self.n_sites)
        reverse = {(int(j), int(i), tuple(-s)) for i, j, s in zip(self.source, self.destination, self.image_shifts)}
        forward = {(int(i), int(j), tuple(s)) for i, j, s in zip(self.source, self.destination, self.image_shifts)}
        result = dict(parent_hash=self.parent.parent_hash, cutoff_A=self.cutoff_A, graph_rule=self.rule,
                      full_parent_nodes=len(self.parent.numbers), mobile_nodes=self.n_sites,
                      host_species=sorted(set(map(int, self.parent.numbers[self.parent.host_indices]))),
                      directed_full_image_edges=len(self.source), directed_mobile_image_edges=len(a),
                      nonzero_image_self_edges=int(np.sum((self.source == self.destination) & np.any(self.image_shifts != 0, axis=1))),
                      reverse_edge_complete=reverse == forward, full_connected_components=nf, full_component_sizes=sf,
                      mobile_connected_components=nm, mobile_component_sizes=sm, mobile_shells=shells(distances),
                      shell_merge_tolerance_A=shell_tolerance_A,
                      shell_tolerance_provenance="Provisional numerical grouping of nearly equal parent distances; no barrier interpretation")
        if self.parent.site_groups is not None:
            g = self.parent.site_groups
            result["cross_group_mobile_image_edges"] = int(np.sum(g[a] != g[b]))
            result["site_groups"] = sorted(set(map(int, g)))
        return result


def build_periodic_graph(parent, cutoff_A=5.0):
    """All full-parent neighbors at distance < cutoff, including image identity.

    The default 5 Å is a *provisional model feature radius*, not a proven physical
    convergence threshold. Inspect ``shell_audit`` and test another radius when
    interpreting a model benefit or transferring to another parent.
    """
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    if not isinstance(parent, ParentSpec):
        raise TypeError("build_periodic_graph requires an explicit ParentSpec")
    if not np.isfinite(cutoff_A) or cutoff_A <= 0:
        raise ValueError("graph cutoff must be positive Å")
    atoms = Atoms(numbers=parent.numbers, positions=parent.positions_A, cell=parent.cell_A, pbc=parent.pbc)
    src, dst, shift, distance = neighbor_list("ijSd", atoms, cutoff_A, self_interaction=False)
    # Deterministic edge order; do not collapse repeated endpoint pairs/images.
    order = np.lexsort((shift[:, 2], shift[:, 1], shift[:, 0], dst, src))
    return PeriodicGraph(parent, src[order], dst[order], shift[order], distance[order], float(cutoff_A))


def mobile_kernel_matrices(graph, shell_tolerance_A=1e-5):
    """Two neighbor kernels: nearest shell and exp(-r/r_nearest) to cutoff.

    Matrix row is queried mobile site; column is neighboring mobile site.
    Repeated cell images contribute separately. Self-image edges refer to the
    same site's occupation, which remains unknown until that site is assigned.
    """
    a, b, _, distance = graph.mobile_edges()
    if len(distance) == 0:
        raise ValueError("graph cutoff contains no mobile-mobile edges")
    nearest = float(distance.min())
    kernels = np.zeros((2, graph.n_sites, graph.n_sites), float)
    np.add.at(kernels[0], (a, b), (distance <= nearest + shell_tolerance_A).astype(float))
    np.add.at(kernels[1], (a, b), np.exp(-distance / nearest))
    return kernels, nearest


class CausalGraphContext(nn.Module):
    """Six fixed graph features, never future occupation labels.

    For each of two kernels return occupied/vacant/unassigned neighbor weight
    fractions. Teacher forcing at site i can use only occupations j<i. Values at
    all other sites are explicitly masked before arithmetic, so changing future
    labels cannot alter an earlier prediction. These are engineered features,
    not learned message passing or an equivariance claim.
    """
    feature_dim = 6

    def __init__(self, graph):
        super().__init__()
        self.n_sites = graph.n_sites
        matrices, _ = mobile_kernel_matrices(graph)
        norm = matrices.sum(axis=-1, keepdims=True)
        matrices = np.divide(matrices, norm, out=np.zeros_like(matrices), where=norm > 0)
        self.register_buffer("kernels", torch.as_tensor(matrices, dtype=torch.float32))
        self.register_buffer("known_rows", torch.as_tensor(norm[..., 0] > 0))

    def features(self, occupancy, assigned):
        if occupancy.ndim != 2 or occupancy.shape[1] != self.n_sites or assigned.shape != occupancy.shape:
            raise ValueError("context needs occupancy and assigned of shape(batch,mobile_sites)")
        assigned = assigned.to(dtype=torch.bool)
        present = torch.where(assigned, occupancy, torch.zeros_like(occupancy))
        absent = torch.where(assigned, 1 - occupancy, torch.zeros_like(occupancy))
        unknown = (~assigned).to(occupancy.dtype)
        classes = torch.stack((present, absent, unknown), dim=-1)
        kernels = self.kernels.to(dtype=occupancy.dtype)
        return torch.einsum("kij,bjc->bikc", kernels, classes).flatten(-2)

    def teacher_features(self, occupancy):
        if occupancy.ndim != 2 or occupancy.shape[1] != self.n_sites:
            raise ValueError("context mask has wrong mobile-site count")
        kernels = self.kernels.to(dtype=occupancy.dtype)
        past = torch.arange(self.n_sites, device=occupancy.device)[None, :] < torch.arange(self.n_sites, device=occupancy.device)[:, None]
        # Mask before multiplying: even a NaN sentinel in an unassigned future
        # position must not contaminate an earlier site's features via 0*NaN.
        known_present = torch.where(past[None], occupancy[:, None, :], torch.zeros_like(occupancy[:, None, :]))
        known_absent = torch.where(past[None], 1 - occupancy[:, None, :], torch.zeros_like(occupancy[:, None, :]))
        present = torch.einsum("kij,bij->bik", kernels, known_present)
        absent = torch.einsum("kij,bij->bik", kernels, known_absent)
        unknown = kernels.sum(-1).T[None] - present - absent
        return torch.stack((present, absent, unknown), dim=-1).flatten(-2)

    def step_features(self, occupancy, step):
        if not 0 <= step < self.n_sites:
            raise ValueError("context step outside mobile-site range")
        assigned = torch.arange(self.n_sites, device=occupancy.device)[None] < step
        return self.features(occupancy, assigned.expand_as(occupancy))[:, step]


def vacancy_pair_fraction(occupancy, graph, shell_tolerance_A=1e-5):
    """Nearest-shell VV pairs / all eligible mobile pairs (dimensionless).

    Both directions of every periodic image pair appear in numerator and
    denominator, so their ratio is exactly the once-per-undirected-pair ratio.
    A nonzero-shift self-image pair is physical and contributes v_i squared.
    The descriptor is not divided by the number of vacancies or vacancy pairs.
    """
    values = np.asarray(occupancy)
    single = values.ndim == 1
    if single:
        values = values[None]
    if values.ndim != 2 or values.shape[1] != graph.n_sites or not np.isin(values, [0, 1]).all():
        raise ValueError("descriptor requires binary present-ion occupancy")
    matrices, nearest = mobile_kernel_matrices(graph, shell_tolerance_A)
    nearest_matrix = matrices[0]
    total = nearest_matrix.sum()
    if total <= 0:
        raise ValueError("no eligible nearest-shell pair")
    vacancy = 1 - values.astype(float)
    fraction = np.einsum("bi,ij,bj->b", vacancy, nearest_matrix, vacancy, optimize=True) / total
    return float(fraction[0]) if single else fraction


def descriptor_symmetry_audit(graph, tolerance=1e-12):
    """Exact weighted-pair invariance under all retained mobile permutations."""
    matrix, nearest = mobile_kernel_matrices(graph)
    matrix = matrix[0]
    largest = max(float(np.max(np.abs(matrix - matrix[p][:, p]))) for p in graph.parent.permutations)
    if largest > tolerance:
        raise ValueError("nearest-shell descriptor is not invariant under the retained parent symmetry")
    return dict(status="PASS", descriptor="nearest-shell vacancy-vacancy pair fraction",
                units="dimensionless", nearest_shell_A=nearest, directed_pairs=int(matrix.sum()),
                undirected_image_pairs=int(matrix.sum() / 2), operations=len(graph.parent.permutations),
                maximum_kernel_residual=largest, tolerance=tolerance,
                convention="vacancy=1-occupancy; each undirected periodic-image pair counted once")


def fit_descriptor_schema(values, n_mobile, split, *, lower_fraction=.5):
    """One training-fitted low/high split per composition, with ties intact.

    This returns actual support and no promise of support in every joint energy
    class.  The caller must intersect these labels with the energy labels using
    training rows only before issuing a joint request.
    """
    values = np.asarray(values, float)
    counts = np.asarray(n_mobile, int)
    split = np.asarray(split).astype(str)
    if values.ndim != 1 or counts.shape != values.shape or split.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("descriptor schema arrays differ or contain nonfinite values")
    if not 0 < lower_fraction < 1:
        raise ValueError("descriptor quantile must be strictly inside zero and one")
    boundaries, support = {}, {}
    for n in np.unique(counts[split == "train"]):
        selected = np.sort(values[(split == "train") & (counts == n)])
        boundary = float(selected[max(0, int(np.ceil(len(selected) * lower_fraction)) - 1)])
        boundaries[str(int(n))] = boundary
        support[str(int(n))] = dict(low=int(np.sum(selected <= boundary)), high=int(np.sum(selected > boundary)))
    return dict(schema="vacancy-pair-fraction-bins-v1", descriptor="nearest-shell vacancy pair fraction", units="dimensionless",
                lower_fraction=lower_fraction, boundaries=boundaries, training_support=support,
                ties="equal to boundary belongs to low; ties never split", fit_scope="training rows only")


def classify_descriptor(values, counts, schema):
    values, counts = np.asarray(values, float), np.asarray(counts, int)
    if values.shape != counts.shape:
        raise ValueError("descriptor values/counts differ")
    result = np.empty(values.shape, dtype=np.int64)
    for n in np.unique(counts):
        if str(int(n)) not in schema["boundaries"]:
            raise ValueError("unsupported composition in descriptor request")
        result[counts == n] = values[counts == n] > schema["boundaries"][str(int(n))]
    return result


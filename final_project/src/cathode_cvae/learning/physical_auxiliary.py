"""TRAIN-only scaling for a symmetry-invariant vacancy-pair auxiliary target.

The observable is the fraction of directed nearest-shell periodic image edges
whose two endpoints are vacant. It describes fixed-parent ordering, not energy,
relaxation, a transition pathway, or an equilibrium probability.
"""

import hashlib

import numpy as np
import torch
from torch import nn


def vacancy_pair_values(occupancy, graph, shell_tolerance_A=1e-5):
    """Return dimensionless nearest-shell vacancy-pair fractions per mask.

    Distinct periodic images retain their multiplicity; reversing an edge is
    counted in both numerator and denominator. No labels enter edge selection.
    The shell tolerance groups numerically equal parent distances, in angstrom.
    """
    x = np.asarray(occupancy)
    if x.ndim != 2 or x.shape[1] != graph.n_sites or not np.isin(x, [0, 1]).all():
        raise ValueError("Physical target needs binary (rows, mobile sites) occupancy")
    if not np.isfinite(shell_tolerance_A) or shell_tolerance_A <= 0:
        raise ValueError("Shell tolerance must be positive finite angstrom")
    source, destination, images, distance = graph.mobile_edges()
    if len(distance) == 0:
        raise ValueError("Physical pair target requires mobile-site image edges")
    take = distance <= distance.min() + shell_tolerance_A
    source, destination, images = source[take], destination[take], images[take]
    v = 1 - x.astype(np.float64)
    # Chunking avoids an unbounded rows-by-edges allocation for large parents.
    values = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), 1024):
        block = v[start:start + 1024]
        values[start:start + len(block)] = (block[:, source] * block[:, destination]).mean(1)
    payload = np.column_stack((source, destination, images)).astype(np.int64)
    spec = dict(observable="nearest_shell_vacancy_pair_fraction",
        units="dimensionless", directed_periodic_image_edges=len(source),
        nearest_distance_A=float(distance.min()), shell_tolerance_A=float(shell_tolerance_A),
        edge_sha256=hashlib.sha256(payload.tobytes()).hexdigest(),
        graph_hash=graph.to_dict()["graph_hash"],
        definition="mean((1-x_i)*(1-x_j)) over directed nearest-shell periodic image edges",
        scope="Fixed-parent vacancy ordering; not an energy or thermodynamic label")
    return values, spec


def prepare_targets(occupancy, counts, bins, splits, graph, *, mode, seed,
                    shell_tolerance_A=1e-5):
    """Fit target centering/scaling on TRAIN cells; never process TEST masks.

    Cells are composition x measured energy bin. Standardization is unweighted
    and TRAIN-only. Exact zero TRAIN variance uses scale 1 and is disclosed.
    The shuffled control permutes TRAIN targets only within the same cells;
    VAL always retains its genuine target and is never used in normalization.
    """
    if mode not in {"latent", "condition", "shuffled"}:
        raise ValueError("Unknown physical auxiliary mode")
    x, counts, bins, roles = map(np.asarray, (occupancy, counts, bins, splits))
    if counts.shape != (len(x),) or bins.shape != counts.shape or roles.shape != counts.shape:
        raise ValueError("Physical target arrays must have matching rows")
    active = np.flatnonzero(np.isin(roles, ["train", "val"]))
    raw = np.full(len(x), np.nan, dtype=np.float64)
    raw[active], observable = vacancy_pair_values(x[active], graph, shell_tolerance_A)
    genuine = np.zeros(len(x), dtype=np.float32)
    target = np.zeros(len(x), dtype=np.float32)
    cells = {}
    rng = np.random.default_rng(seed)
    permutations = []
    for n, b in sorted({(int(counts[i]), str(bins[i])) for i in active}):
        members = active[(counts[active] == n) & (bins[active] == b)]
        train = members[roles[members] == "train"]
        if len(train) == 0:
            raise ValueError(f"No TRAIN physical-target scale for cell {(n,b)}")
        mean, std = float(raw[train].mean()), float(raw[train].std())
        scale = std if std > 0 else 1.0
        genuine[members] = (raw[members] - mean) / scale
        target[members] = genuine[members]
        if mode == "shuffled":
            perm = rng.permutation(train)
            target[train] = genuine[perm]
            permutations.extend(zip(train.tolist(), perm.tolist()))
        cells[f"{n}:{b}"] = dict(train_rows=len(train), mean=mean, scale=scale,
                                  train_std=std, zero_variance=std == 0)
    metadata = dict(mode=mode, observable=observable, cells=cells,
        normalization="Unweighted TRAIN mean/std separately within composition x measured energy bin; exact zero std uses scale 1",
        target_scope="Only TRAIN/VAL masks are passed to observable; TEST targets are not calculated",
        shuffle_seed=int(seed) if mode == "shuffled" else None,
        shuffle_scope="Fixed permutation of TRAIN targets within composition x measured energy-bin cells; genuine VAL target" if mode == "shuffled" else None,
        shuffle_map_sha256=hashlib.sha256(np.asarray(permutations, dtype=np.int64).tobytes()).hexdigest() if mode == "shuffled" else None,
        head_inputs="condition only" if mode == "condition" else "posterior sampled latent concatenated with condition",
        checkpoint_selection="Auxiliary loss is excluded from validation checkpoint selection; terminal checkpoint is primary for matched updates",
        warm_start="Auxiliary head always initialized anew, including model-weight warm starts")
    return raw, genuine, target, metadata


def make_head(latent_dim, condition_dim, mode, seed, device):
    """Initialize an auxiliary head without consuming the fit's random stream."""
    width = condition_dim if mode == "condition" else latent_dim + condition_dim
    if mode != "condition" and latent_dim < 1:
        raise ValueError("Latent/shuffled physical auxiliary needs latent_dim >= 1")
    devices = [] if torch.device(device).type != "cuda" else [torch.device(device).index or torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        head = nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 1)).to(device)
    return head


def head_prediction(head, z, condition, mode):
    features = condition if mode == "condition" else torch.cat((z, condition), dim=-1)
    return head(features).squeeze(-1)

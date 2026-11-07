"""Versioned, train-fitted energy ranges for fixed-parent vacancy models.

Edges use the unweighted nearest-rank quantile: sorted[ceil(p*m)-1]. Equal
energies never receive different labels merely to fill a requested bin. The
schema describes the energy partition independently of the neural encoding.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "cathode-energy-bins-v1"
BIN_NAMES = ("low", "middle", "high")


def schema_identity(schema):
    body = {k: v for k, v in schema.items() if k != "identity"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _counts(values):
    x = np.asarray(values)
    if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all() or not np.equal(x, np.floor(x)).all():
        raise ValueError("Target counts must be finite integers")
    return x.astype(np.int64)


def _partition(energy, edges):
    lower, upper = map(float, edges)
    # side='left' keeps every exact boundary tie in the lower-energy bin.
    return np.searchsorted([lower, upper], energy, side="left")


def fit_schema(n_target, energy_eV_atom, split, *, n_sites, quantiles=(0.2, 0.8), provenance=None):
    """Fit composition-specific edges using only rows marked 'train'.

    Bin support is empirical training support. Empty bins are explicitly
    unsupported; we do not assign equal-energy observations different labels
    or silently redirect a request to another class.
    """
    n = _counts(n_target)
    e = np.asarray(energy_eV_atom, dtype=float)
    labels = np.asarray(split).astype(str)
    if n.ndim != 1 or e.shape != n.shape or labels.shape != n.shape:
        raise ValueError("Counts, energies and split must be matching vectors")
    if len(quantiles) != 2 or not np.isfinite(e).all() or not 0 < quantiles[0] < quantiles[1] < 1:
        raise ValueError("Finite energies and 0 < lower quantile < upper quantile < 1 are required")
    if int(n_sites) != n_sites or n_sites < 1 or np.any((n < 0) | (n > n_sites)):
        raise ValueError("Counts are outside the parent site range")
    if not len(n):
        raise ValueError("Cannot fit an empty condition reference")
    compositions = {}
    for count in sorted(set(n.tolist())):
        values = np.sort(e[(n == count) & (labels == "train")])
        if not len(values):
            raise ValueError(f"Composition {count} has no training reference energies")
        edges = [float(values[math.ceil(p * len(values)) - 1]) for p in quantiles]
        bins = _partition(values, edges)
        support = {name: int(np.sum(bins == i)) for i, name in enumerate(BIN_NAMES)}
        present = [name for name in BIN_NAMES if support[name]]
        compositions[str(count)] = dict(
            train_rows=len(values), unique_training_energies=int(len(np.unique(values))),
            edges_eV_atom=edges, lower_quantile_eV_atom=edges[0], upper_quantile_eV_atom=edges[1],
            q20_eV_atom=float(values[math.ceil(0.2 * len(values)) - 1]),
            q80_eV_atom=float(values[math.ceil(0.8 * len(values)) - 1]),
            bin_counts=support, supported_bins=present,
            empty_bins=[name for name in BIN_NAMES if not support[name]],
            support_status="full" if len(present) == 3 else "partial",
            sparse=bool(len(values) < 5 or len(present) < 3),
            coincident_edges=bool(edges[0] == edges[1]), merged_bins={},
        )
    schema = dict(schema_version=SCHEMA_VERSION, n_sites=int(n_sites), units="eV/atom",
        bin_names=list(BIN_NAMES), quantiles=list(map(float, quantiles)),
        quantile_estimator="unweighted nearest rank; one-based index ceil(p*m)",
        tie_policy="low: e<=lower; middle: lower<e<=upper; high: e>upper; equal energies never split",
        sparse_policy="retain actual nonempty training bins; unsupported requests fail explicitly; no automatic merging",
        fitting_scope="train rows only, separately within each target count",
        compositions=compositions, provenance={} if provenance is None else dict(provenance))
    schema["identity"] = schema_identity(schema)
    return schema


def validate_schema(schema):
    if schema.get("schema_version") != SCHEMA_VERSION or schema.get("units") != "eV/atom":
        raise ValueError("Unsupported condition schema version or energy units")
    if schema.get("bin_names") != list(BIN_NAMES):
        raise ValueError("Condition bin names/order differ from the implemented encoding")
    if not isinstance(schema.get("n_sites"), int) or schema["n_sites"] < 1:
        raise ValueError("Invalid parent size in condition schema")
    quantiles = schema.get("quantiles", [])
    if len(quantiles) != 2 or not 0 < quantiles[0] < quantiles[1] < 1:
        raise ValueError("Invalid quantiles in condition schema")
    if schema.get("identity") != schema_identity(schema):
        raise ValueError("Condition schema identity does not match its contents")
    for n, entry in schema["compositions"].items():
        if not 0 <= int(n) <= schema["n_sites"] or entry["train_rows"] < 1:
            raise ValueError(f"Invalid training support for composition {n}")
        edges = np.asarray(entry["edges_eV_atom"], dtype=float)
        if edges.shape != (2,) or not np.isfinite(edges).all() or edges[0] > edges[1]:
            raise ValueError(f"Invalid energy edges for composition {n}")
        if (set(entry["bin_counts"]) != set(BIN_NAMES)
                or any(not isinstance(v, int) or v < 0 for v in entry["bin_counts"].values())
                or sum(entry["bin_counts"].values()) != entry["train_rows"]
                or entry["supported_bins"] != [b for b in BIN_NAMES if entry["bin_counts"][b]]):
            raise ValueError(f"Invalid supported bins for composition {n}")
    return schema


def save_schema(schema, path):
    validate_schema(schema)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(json.dumps(schema, indent=2, allow_nan=False) + "\n")


def load_schema(path):
    return validate_schema(json.loads(Path(path).read_text()))


def supported_bins(schema, n_target):
    n = int(_counts(np.asarray(n_target)).item())
    try:
        return list(schema["compositions"][str(n)]["supported_bins"])
    except KeyError:
        raise ValueError(f"Composition {n} is absent from the fitted condition schema") from None


def classify(schema, n_target, energy_eV_atom):
    """Return low/middle/high labels, preserving the broadcast input shape."""
    n, e = np.broadcast_arrays(_counts(n_target), np.asarray(energy_eV_atom, dtype=float))
    if not np.isfinite(e).all():
        raise ValueError("Cannot classify a missing or nonfinite energy")
    result = np.empty(n.shape, dtype="<U6")
    for count in np.unique(n):
        if str(count) not in schema["compositions"]:
            raise ValueError(f"Composition {count} is absent from the fitted condition schema")
        selected = n == count
        values = _partition(e[selected], schema["compositions"][str(count)]["edges_eV_atom"])
        result[selected] = np.asarray(BIN_NAMES)[values]
    return result


def condition_dimension(mode):
    return {"composition": 1, "energy": 2, "multibin": 4}[mode]


def encode_requests(schema, n_target, requested_bins, *, mode="multibin", require_supported=True):
    """Encode requested classes; middle/high map to 'rest' in a binary model.

    require_supported=False is intended for conditional likelihood on observed
    held-out rows. Generation requests must retain the default support check.
    """
    n, requested = np.broadcast_arrays(_counts(n_target), np.asarray(requested_bins).astype(str))
    if np.any((n < 0) | (n > schema["n_sites"])):
        raise ValueError("Requested target count is outside the fitted parent")
    allowed = set(BIN_NAMES) | ({"rest"} if mode == "energy" else set()) | ({"unconditioned"} if mode == "composition" else set())
    if not set(requested.flat) <= allowed:
        raise ValueError(f"Unknown requested energy bin for {mode} encoding")
    for count, label in zip(n.flat, requested.flat):
        if str(count) not in schema["compositions"]:
            raise ValueError(f"Composition {count} is absent from the fitted condition schema")
        support = supported_bins(schema, count)
        if require_supported and label in BIN_NAMES and label not in support:
            raise ValueError(f"Requested bin {label} has no training support at composition {count}; available bins: {support}")
        if require_supported and label == "rest" and not any(b != "low" for b in support):
            raise ValueError(f"Binary rest class has no training support at composition {count}")
    c = np.zeros(n.shape + (condition_dimension(mode),), dtype=np.float32)
    c[..., 0] = n / schema["n_sites"]
    if mode == "energy":
        c[..., 1] = requested == "low"
    elif mode == "multibin":
        for i, name in enumerate(BIN_NAMES):
            c[..., i + 1] = requested == name
    return c


def encode_observations(schema, n_target, energy_eV_atom, *, mode="multibin"):
    return encode_requests(schema, n_target, classify(schema, n_target, energy_eV_atom),
                           mode=mode, require_supported=False)

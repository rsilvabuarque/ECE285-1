"""Train-only joint support for a symmetry-invariant structural condition.

The scalar is nearest-shell vacancy-pair fraction, dimensionless.  It is not an
energy or probability. A composition-specific median partitions low/high pair
fraction; each energy/descriptor combination must have actual training support
before it may be requested. Held-out observations are encoded without deleting
unsupported outcomes from an evaluation denominator.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .conditions import BIN_NAMES, _counts


SCHEMA_VERSION = "cathode-vacancy-pair-condition-v1"
RANGE_NAMES = ("low", "high")


def schema_identity(schema):
    body = {k: v for k, v in schema.items() if k != "identity"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def fit_descriptor_schema(values, n_target, energy_bins, split, *, parent_hash=None,
                          graph_hash=None, lower_fraction=.5):
    """Fit nearest-rank descriptor boundaries on training rows, ties intact."""
    values = np.asarray(values, dtype=float)
    n = _counts(n_target)
    energy = np.asarray(energy_bins).astype(str)
    role = np.asarray(split).astype(str)
    if (values.ndim != 1 or n.shape != values.shape or energy.shape != values.shape
            or role.shape != values.shape or not np.isfinite(values).all()):
        raise ValueError("descriptor values, counts, energy bins and split must be finite matching vectors")
    if np.any((values < -1e-12) | (values > 1 + 1e-12)):
        raise ValueError("vacancy-pair fraction must lie in [0,1]")
    if not set(energy) <= set(BIN_NAMES):
        raise ValueError("unrecognized energy bin in descriptor fit")
    if not 0 < lower_fraction < 1:
        raise ValueError("descriptor quantile must be inside (0,1)")
    compositions = {}
    for count in np.unique(n):
        chosen = (n == count) & (role == "train")
        train = np.sort(values[chosen])
        if not len(train):
            raise ValueError(f"composition {count} has no training descriptor reference")
        boundary = float(train[int(np.ceil(lower_fraction * len(train))) - 1])
        lower = values[chosen] <= boundary
        joint = {b: {"low": int(np.sum((energy[chosen] == b) & lower)),
                     "high": int(np.sum((energy[chosen] == b) & ~lower))} for b in BIN_NAMES}
        compositions[str(int(count))] = dict(train_rows=len(train), boundary=boundary,
            unique_training_values=len(np.unique(train)),
            bin_counts={"low": int(lower.sum()), "high": int((~lower).sum())},
            joint_support=joint,
            supported_requests=[dict(energy_bin=b, descriptor_range=r, training_rows=joint[b][r])
                                for b in BIN_NAMES for r in RANGE_NAMES if joint[b][r] > 0])
    schema = dict(schema_version=SCHEMA_VERSION, descriptor="nearest-shell vacancy-vacancy pair fraction",
                  units="dimensionless", ranges=list(RANGE_NAMES), lower_fraction=float(lower_fraction),
                  quantile_estimator="nearest rank, ceil(p*n), using training rows only",
                  tie_policy="equal boundary values all belong to low; never split ties",
                  sparse_policy="unsupported joint energy/descriptor requests fail explicitly",
                  compositions=compositions,
                  provenance=dict(parent_hash=parent_hash, graph_hash=graph_hash,
                                  occupancy="1=present; vacancy=1-occupancy",
                                  pair_convention="nearest-shell undirected periodic-image pairs, each once; divide by all eligible mobile pairs"))
    schema["identity"] = schema_identity(schema)
    return validate_schema(schema)


def validate_schema(schema):
    if (schema.get("schema_version") != SCHEMA_VERSION or schema.get("units") != "dimensionless"
            or schema.get("ranges") != list(RANGE_NAMES)):
        raise ValueError("unrecognized structural descriptor schema")
    if schema.get("identity") != schema_identity(schema):
        raise ValueError("descriptor schema identity mismatch")
    if not 0 < schema.get("lower_fraction", 0) < 1:
        raise ValueError("invalid descriptor quantile")
    for n, entry in schema["compositions"].items():
        if int(n) < 0 or not np.isfinite(entry["boundary"]) or not 0 <= entry["boundary"] <= 1:
            raise ValueError("invalid descriptor composition/boundary")
        counts = entry["bin_counts"]
        if set(counts) != set(RANGE_NAMES) or sum(counts.values()) != entry["train_rows"]:
            raise ValueError("invalid descriptor training support")
        joint = entry["joint_support"]
        if set(joint) != set(BIN_NAMES) or any(set(v) != set(RANGE_NAMES) for v in joint.values()):
            raise ValueError("invalid joint energy/descriptor support")
        if any(not isinstance(v, int) or v < 0 for d in joint.values() for v in d.values()):
            raise ValueError("joint support must be nonnegative integer counts")
        if any(sum(joint[b][r] for b in BIN_NAMES) != counts[r] for r in RANGE_NAMES):
            raise ValueError("joint support does not match descriptor class counts")
    return schema


def load_schema(path):
    return validate_schema(json.loads(Path(path).read_text()))


def save_schema(schema, path):
    validate_schema(schema)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(schema, indent=2, allow_nan=False) + "\n")


def classify(schema, n_target, values):
    n, values = np.broadcast_arrays(_counts(n_target), np.asarray(values, dtype=float))
    if not np.isfinite(values).all():
        raise ValueError("missing/nonfinite structural descriptor")
    result = np.empty(n.shape, dtype="<U4")
    for count in np.unique(n):
        entry = schema["compositions"].get(str(int(count)))
        if entry is None:
            raise ValueError(f"unsupported descriptor composition {count}")
        chosen = n == count
        result[chosen] = np.where(values[chosen] <= entry["boundary"], "low", "high")
    return result


def encode_requests(schema, n_target, energy_bins, requested_ranges, *, require_supported=True):
    """Append these two one-hot components to an existing energy condition."""
    n, energy, ranges = np.broadcast_arrays(_counts(n_target), np.asarray(energy_bins).astype(str),
                                           np.asarray(requested_ranges).astype(str))
    if not set(energy.flat) <= set(BIN_NAMES) or not set(ranges.flat) <= set(RANGE_NAMES):
        raise ValueError("unrecognized energy/descriptor request")
    for count, e, r in zip(n.flat, energy.flat, ranges.flat):
        entry = schema["compositions"].get(str(int(count)))
        if entry is None:
            raise ValueError(f"unsupported descriptor composition {count}")
        if require_supported and entry["joint_support"][e][r] == 0:
            raise ValueError(f"unsupported joint request: n={count}, energy={e}, descriptor={r}")
    return np.stack((ranges == "low", ranges == "high"), axis=-1).astype(np.float32)


def encode_observations(schema, n_target, energy_bins, values):
    return encode_requests(schema, n_target, energy_bins, classify(schema, n_target, values),
                           require_supported=False)


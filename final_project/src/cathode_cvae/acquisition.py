"""Temperature-dependent acquisition under a static MLIP surrogate.

At one parent and composition, i labels an occupation representative, E_i is
its total single-point energy in eV/cell, and T is physical temperature in K.
For complete representatives the measure g_i counts raw occupation patterns
in the symmetry orbit. For a uniform-raw-mask reference sample the measure is
instead its observed raw frequency f_i: multiplying by g_i again is incorrect.
The dimensionless weight is m_i exp[-(E_i-E_ref)/(k_B T)]. Bin probabilities
sum normalized weights over representatives assigned to each frozen energy bin.
These deletion-geometry weights are an acquisition policy, not basin free
energies, calibrated hydrothermal populations, or configurational entropy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

KB_EV_K = 8.617333262145e-5
DEFAULT_TEMPERATURES_K = (393.15, 433.15, 473.15, 493.15)
BIN_ORDER = ("low", "middle", "high")
REFERENCE_KINDS = ("exhaustive_orbits", "uniform_raw_sample")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def _identity(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _binary_occupancies(occupancy):
    x = np.asarray(occupancy)
    if x.ndim != 2 or not len(x) or not np.isin(x, [0, 1]).all():
        raise ValueError("occupancy must be a nonempty binary matrix; 1 means target present")
    return x.astype(np.uint8)


def canonical_keys(occupancy, permutations):
    """Minimum vacancy strings under a declared parent action, with no bit overflow."""
    x = _binary_occupancies(occupancy)
    perms = np.asarray(permutations, dtype=int)
    if perms.ndim != 2 or perms.shape[1] != x.shape[1] or not np.all(np.sort(perms, 1) == np.arange(x.shape[1])):
        raise ValueError("Parent permutations do not match occupancy sites")
    inv = np.argsort(perms, axis=1)
    keys = []
    for row in x:
        packed = np.packbits((1 - row)[inv], axis=1, bitorder="big")
        smallest = min(p.tobytes() for p in packed)
        bits = np.unpackbits(np.frombuffer(smallest, dtype=np.uint8), bitorder="big")[:x.shape[1]]
        keys.append("".join(map(str, bits.tolist())))
    return np.asarray(keys)


def orbit_multiplicities(occupancy, permutations):
    """Number of distinct raw masks in each orbit, not atom-label permutations.

    This directly counts distinct images. It remains correct if a stored action
    contains duplicate rows; the separately reported group order is deduplicated.
    """
    from .data import validate_permutation_group
    x = _binary_occupancies(occupancy)
    perms = np.asarray(permutations, dtype=int)
    validate_permutation_group(perms)
    if perms.shape[1] != x.shape[1]:
        raise ValueError("Parent and occupancy dimensions differ")
    inverse = np.argsort(perms, axis=1)
    return np.asarray([len(np.unique(np.packbits(row[inverse], axis=1), axis=0)) for row in x], dtype=np.int64)


def validate_exhaustive_support(occupancy, permutations):
    """Prove completeness by disjoint orbit keys and sum(g)=binomial(N,n).

    Valid fixed-count orbits partition the raw-mask space. Thus unique orbit keys
    plus the complete raw count establish finite support without enumerating all
    masks again. Every count 0..N is required, including deterministic endpoints.
    """
    x = _binary_occupancies(occupancy)
    keys = canonical_keys(x, permutations)
    if len(set(keys.tolist())) != len(x):
        raise ValueError("Representatives contain duplicate symmetry orbits")
    multiplicity = orbit_multiplicities(x, permutations)
    n_sites = x.shape[1]
    counts = x.sum(1)
    rows = []
    for n in range(n_sites + 1):
        selected = counts == n
        expected = math.comb(n_sites, n)
        actual = int(multiplicity[selected].sum())
        rows.append(dict(n_target=n, representatives=int(selected.sum()), raw_patterns=actual,
                         expected_raw_patterns=expected, complete=actual == expected))
    if not all(r["complete"] for r in rows):
        raise ValueError(f"Incomplete finite-support reference: {[r for r in rows if not r['complete']]}")
    return dict(status="PASS", n_sites=n_sites, representatives=len(x),
                distinct_group_permutations=len(np.unique(permutations, axis=0)),
                multiplicities=multiplicity.tolist(), counts=rows)


def _logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    maximum = float(values.max())
    return maximum + float(np.log(np.exp(values - maximum).sum()))


def static_weights(energy_eV, measure, temperature_K, *, measure_kind):
    """Stable normalization, concentration and clearly defined ESS diagnostics.

    representative_ess = 1/sum_i p_i^2 for collapsed representative masses p_i.
    microstate_or_particle_ess = Z^2/sum_i m_i a_i^2, a_i=exp[-DeltaE/kBT].
    For exact orbit multiplicity this counts effective physical raw masks. For
    an IID uniform raw reference it is the usual importance-sampling ESS over
    sampled particles, retaining repeated draws. It is not a confidence interval.
    """
    e, m = np.asarray(energy_eV, dtype=float), np.asarray(measure, dtype=float)
    if measure_kind not in REFERENCE_KINDS:
        raise ValueError(f"measure_kind must be one of {REFERENCE_KINDS}")
    if e.ndim != 1 or not len(e) or m.shape != e.shape or not np.isfinite(e).all():
        raise ValueError("Every prespecified reference geometry needs one finite total energy")
    if not np.isfinite(m).all() or np.any(m <= 0) or not np.equal(m, np.floor(m)).all():
        raise ValueError("Measures must be positive integer orbit sizes or raw sampling frequencies")
    if not np.isfinite(temperature_K) or temperature_K <= 0:
        raise ValueError("Physical temperature must be positive and finite, in kelvin")
    reference = float(e.min())
    log_a = -(e - reference) / (KB_EV_K * float(temperature_K))
    if not np.isfinite(log_a).all():
        raise ValueError("Energy differences exceed the supported floating-point range")
    log_mass = np.log(m) + log_a
    log_z = _logsumexp(log_mass)
    p = np.exp(log_mass - log_z)
    # Renormalize the final roundoff without changing the energy reference.
    p /= p.sum()
    particle_ess = float(np.exp(2 * log_z - _logsumexp(np.log(m) + 2 * log_a)))
    return dict(probability=p, E_ref_eV=reference, log_partition_relative=log_z,
        temperature_K=float(temperature_K), measure_kind=measure_kind,
        representative_ess=float(1 / np.dot(p, p)),
        microstate_or_particle_ess=particle_ess, total_measure=float(m.sum()),
        representative_count=len(e), maximum_representative_mass=float(p.max()),
        maximum_single_microstate_or_particle_probability=float(np.exp(log_a.max() - log_z)),
        mean_energy_eV=float(np.dot(p, e)),
        ess_interpretation=("Exact raw occupation microstate ESS" if measure_kind == "exhaustive_orbits"
                            else "IID uniform-raw reference importance-sampling particle ESS; duplicates retained"))


def bin_probabilities(probability, measured_bins, supported_bins):
    """Aggregate to training-supported bins; do not silently discard any mass."""
    p, labels = np.asarray(probability, dtype=float), np.asarray(measured_bins).astype(str)
    supported = list(supported_bins)
    if len(set(supported)) != len(supported) or not supported:
        raise ValueError("Supported bins must be a nonempty unique list")
    if p.ndim != 1 or p.shape != labels.shape or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Probability and measured-bin arrays differ or are invalid")
    if set(labels.tolist()) - set(supported):
        raise ValueError("Reference has mass in a bin unsupported by the frozen training schema")
    total = float(p.sum())
    if total <= 0 or not np.isclose(total, 1., atol=1e-12, rtol=1e-12):
        raise ValueError("Reference probabilities must sum to one")
    return {name: float(p[labels == name].sum() / total) for name in supported}


def exploration_policy(bin_probability, exploration_fraction=0.1):
    """Mix a chosen uniform-over-supported-bin floor into static acquisition."""
    if not np.isfinite(exploration_fraction) or not 0 <= exploration_fraction <= 1:
        raise ValueError("Exploration fraction must lie in [0,1]")
    names = list(bin_probability)
    p = np.asarray([bin_probability[n] for n in names], dtype=float)
    if not len(names) or np.any(p < 0) or not np.isfinite(p).all() or not np.isclose(p.sum(), 1):
        raise ValueError("Bin probabilities must be finite, nonnegative and normalized")
    p = (1 - exploration_fraction) * p + exploration_fraction / len(p)
    return dict(zip(names, p.tolist()))


def allocate_integer(probability, budget):
    """Largest-remainder allocation; insertion order deterministically breaks ties."""
    if isinstance(budget, bool) or int(budget) != budget or budget < 0:
        raise ValueError("Proposal budget must be a nonnegative integer")
    names = list(probability)
    p = np.asarray([probability[n] for n in names], dtype=float)
    if not len(p) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("Allocation probabilities must sum to one")
    p /= p.sum()
    real = p * int(budget)
    counts = np.floor(real).astype(np.int64)
    remainder = int(budget) - int(counts.sum())
    order = np.argsort(-(real - counts), kind="stable")
    counts[order[:remainder]] += 1
    assert counts.sum() == budget
    return {name: int(n) for name, n in zip(names, counts)}


def make_allocation_plan(energy_eV, measured_bins, measure, temperatures_K,
                         supported_bins, budget, *, measure_kind,
                         exploration_fraction=0.1, mixture_weights=None, schema_id=None,
                         composition=None, reference_id=None):
    """Return per-temperature pure/floored plans and joint T/bin mixture plans.

    Each per-temperature plan independently allocates `budget` attempts. Each
    mixture plan allocates a separate total `budget` jointly across T and bin.
    This distinction prevents accidentally multiplying a declared budget by the
    number of temperatures. The caller chooses which demonstration to execute.
    """
    temperatures = np.asarray(temperatures_K, dtype=float)
    if temperatures.ndim != 1 or not len(temperatures) or len(set(temperatures.tolist())) != len(temperatures):
        raise ValueError("Temperature grid must be nonempty and contain no duplicates")
    alpha = np.ones(len(temperatures)) if mixture_weights is None else np.asarray(mixture_weights, dtype=float)
    if alpha.shape != temperatures.shape or not np.isfinite(alpha).all() or np.any(alpha < 0) or alpha.sum() <= 0:
        raise ValueError("Mixture condition weights must be nonnegative and match the temperature grid")
    alpha = alpha / alpha.sum()
    per_temperature, joint = [], {"static": {}, "static_exploration": {}}
    names = list(supported_bins)
    for ti, temperature in enumerate(temperatures):
        weights = static_weights(energy_eV, measure, float(temperature), measure_kind=measure_kind)
        pure = bin_probabilities(weights["probability"], measured_bins, names)
        exploratory = exploration_policy(pure, exploration_fraction)
        diagnostics = {k: v for k, v in weights.items() if k != "probability"}
        for policy, probabilities in [("static", pure), ("static_exploration", exploratory)]:
            integer = allocate_integer(probabilities, budget)
            requests = [dict(requested_bin=b, temperature_K=float(temperature),
                             allocation_policy=policy, probability=probabilities[b], attempts=integer[b],
                             schema_id=schema_id, n_target=composition) for b in names]
            per_temperature.append(dict(temperature_K=float(temperature), allocation_policy=policy,
                exploration_fraction=0. if policy == "static" else exploration_fraction,
                budget=int(budget), requests=requests, reference_diagnostics=diagnostics))
            for b in names:
                joint[policy][(ti, b)] = float(alpha[ti] * probabilities[b])
    mixtures = []
    for policy, probabilities in joint.items():
        integer = allocate_integer(probabilities, budget)
        requests = [dict(requested_bin=b, temperature_K=float(temperatures[ti]),
                         temperature_condition_weight=float(alpha[ti]),
                         allocation_policy="temperature_mixture_" + policy,
                         probability=probabilities[ti, b], attempts=integer[ti, b],
                         schema_id=schema_id, n_target=composition) for ti, b in probabilities]
        mixtures.append(dict(allocation_policy="temperature_mixture_" + policy,
            budget=int(budget), requests=requests,
            exploration_fraction=0. if policy == "static" else exploration_fraction))
    reference_id = reference_id or _identity(dict(energy_eV=np.asarray(energy_eV).tolist(),
        measured_bins=np.asarray(measured_bins).astype(str).tolist(), measure=np.asarray(measure).tolist(),
        measure_kind=measure_kind))
    composition = None if composition is None else int(composition)
    plan = dict(schema_version="static-acquisition-v1", schema_id=schema_id, reference_id=reference_id,
        composition=composition, n_target=composition, reference_measure=measure_kind,
        temperature_units="K", energy_units="eV/cell", k_B_eV_K=KB_EV_K,
        temperatures_K=temperatures.tolist(), mixture_condition_weights=alpha.tolist(),
        supported_bins=names, per_temperature=per_temperature, temperature_mixtures=mixtures,
        interpretation="Temperature-dependent acquisition under a static MLIP surrogate; not a thermodynamic population estimate")
    plan["allocation_id"] = _identity(plan)
    return plan


def reference_pool_sensitivity(energy_eV, measured_bins, raw_frequency, temperature_K,
                               supported_bins, sizes=None, seed=20261107,
                               raw_sample_indices=None):
    """Nested-prefix sensitivity of a prespecified IID raw-mask reference pool.

    If original draw order is absent, a seeded shuffle of the frequency multiset
    provides an energy-independent order. Prefixes are nested, not independent
    replicates or a confidence interval. No score is discarded for cache status.
    """
    e, labels = np.asarray(energy_eV), np.asarray(measured_bins)
    f = np.asarray(raw_frequency)
    if f.shape != e.shape or not np.equal(f, np.floor(f)).all() or np.any(f <= 0):
        raise ValueError("raw_frequency must be positive integer counts")
    n = int(f.sum())
    if raw_sample_indices is None:
        order = np.repeat(np.arange(len(f)), f.astype(np.int64))
        np.random.default_rng(seed).shuffle(order)
        order_source = "Seeded energy-independent permutation of observed raw-frequency multiset"
    else:
        order = np.asarray(raw_sample_indices)
        if order.ndim != 1 or len(order) != n or not np.equal(order, np.floor(order)).all() or np.any((order < 0) | (order >= len(e))):
            raise ValueError("raw_sample_indices does not describe the complete prespecified pool")
        order = order.astype(np.int64)
        if not np.array_equal(np.bincount(order, minlength=len(e)), f):
            raise ValueError("Raw sampling order disagrees with raw_frequency")
        order_source = "Original prespecified raw-sample order"
    sizes = sorted(set([max(1, n // 4), max(1, n // 2), n] if sizes is None else map(int, sizes)))
    if not sizes or sizes[0] < 1 or sizes[-1] > n:
        raise ValueError("Prefix sizes must lie inside the complete raw pool")
    full = static_weights(e, f, temperature_K, measure_kind="uniform_raw_sample")
    full_bins = bin_probabilities(full["probability"], labels, supported_bins)
    rows = []
    for size in sizes:
        counts = np.bincount(order[:size], minlength=len(e))
        keep = counts > 0
        weight = static_weights(e[keep], counts[keep], temperature_K, measure_kind="uniform_raw_sample")
        bins = bin_probabilities(weight["probability"], labels[keep], supported_bins)
        rows.append(dict(raw_pool_size=size, unique_orbits=int(keep.sum()), bin_probability=bins,
            l1_difference_to_full_pool=float(sum(abs(bins[b] - full_bins[b]) for b in supported_bins)),
            particle_ess=weight["microstate_or_particle_ess"],
            maximum_representative_mass=weight["maximum_representative_mass"]))
    return dict(temperature_K=float(temperature_K), sampling_seed=int(seed), order_source=order_source,
                interpretation="Nested-prefix numerical sensitivity, not independent replication or statistical convergence proof", rows=rows)


def load_reference(path, parent_path, reference_kind):
    """Read one completely scored parent/protocol reference; cache misses are fatal."""
    from .data import target_indices
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    with np.load(parent_path, allow_pickle=False) as archive:
        parent = {k: archive[k] for k in archive.files}
    x = _binary_occupancies(data["occupancy"] if "occupancy" in data else 1 - data["masks"])
    indices = target_indices(parent)
    if x.shape[1] != len(indices):
        raise ValueError("Reference and parent dimensions differ")
    if "parent_sha256" in data and str(data["parent_sha256"].item()) != _sha(parent_path):
        raise ValueError("Reference belongs to a different parent")
    n = x.sum(1).astype(int)
    if "n_target" in data and not np.array_equal(data["n_target"], n):
        raise ValueError("n_target disagrees with occupancy")
    if "n_Li" in data and not np.array_equal(data["n_Li"], n):
        raise ValueError("n_Li disagrees with occupancy")
    atoms = len(parent["numbers"]) - len(indices) + n
    if "n_atoms" not in data or not np.array_equal(data["n_atoms"], atoms):
        raise ValueError("Actual atom counts are required for the fixed parent and occupancy")
    if "energy_eV" in data:
        energy = np.asarray(data["energy_eV"], dtype=float)
    elif "energy_eV_atom" in data:
        energy = np.asarray(data["energy_eV_atom"], dtype=float) * atoms
    else:
        raise ValueError("Every reference row requires a measured exact-geometry energy")
    if energy.shape != n.shape or not np.isfinite(energy).all():
        raise ValueError("Incomplete/nonfinite reference scores; do not select only available cache hits")
    if "energy_eV_atom" in data and not np.allclose(data["energy_eV_atom"] * atoms, energy, atol=1e-10, rtol=1e-12):
        raise ValueError("Total and per-atom energies disagree")
    keys = canonical_keys(x, parent["permutations"])
    if len(set(keys.tolist())) != len(x):
        raise ValueError("Reference rows must be unique orbits; preserve raw repeats with raw_frequency")
    support = None
    if reference_kind == "exhaustive_orbits":
        support = validate_exhaustive_support(x, parent["permutations"])
        measure = np.asarray(support["multiplicities"], dtype=np.int64)
        if "raw_frequency" in data:
            raise ValueError("Exact orbit reference must not also supply sampled raw frequencies")
    elif reference_kind == "uniform_raw_sample":
        if "raw_frequency" not in data:
            raise ValueError("Uniform raw reference must retain raw_frequency after deduplicating scores")
        measure = np.asarray(data["raw_frequency"])
        if measure.shape != n.shape or not np.isfinite(measure).all() or np.any(measure <= 0) or not np.equal(measure, np.floor(measure)).all():
            raise ValueError("Invalid raw sampling frequencies")
        if "orbit_multiplicity" in data:
            raise ValueError("Use raw frequencies alone, not an additional orbit multiplicity")
    else:
        raise ValueError(f"Unknown reference measure {reference_kind}")
    return dict(data=data, occupancy=x, n_target=n, n_atoms=atoms, energy_eV=energy,
                measure=measure, canonical_keys=keys, finite_support=support)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--schema", type=Path, help="Training-derived condition schema; omitted only with --validate-exhaustive")
    parser.add_argument("--reference-kind", choices=REFERENCE_KINDS, required=True)
    parser.add_argument("--validate-exhaustive", action="store_true")
    parser.add_argument("--budget", type=int, default=256, help="Per-composition budget for each separately labeled plan")
    parser.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES_K)
    parser.add_argument("--mixture-weights", type=float, nargs="+")
    parser.add_argument("--exploration", type=float, default=.1)
    parser.add_argument("--counts", type=int, nargs="+")
    parser.add_argument("--sensitivity-seed", type=int, default=20261107)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    from .runtime import require_compute_allocation
    require_compute_allocation()
    if args.schema is None and not args.validate_exhaustive:
        parser.error("--schema is required for allocation; use --validate-exhaustive alone for a support audit")
    ref = load_reference(args.reference, args.parent, args.reference_kind)
    if args.validate_exhaustive and ref["finite_support"] is None:
        raise ValueError("--validate-exhaustive requires --reference-kind exhaustive_orbits")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = dict(reference_sha256=_sha(args.reference), parent_sha256=_sha(args.parent),
                  reference_kind=args.reference_kind, reference_rows=len(ref["occupancy"]))
    source["reference_metadata"] = {}
    for field in ["teacher", "head", "checkpoint_sha256", "geometry_protocol", "sampling_measure", "sampling_seed"]:
        if field in ref["data"]:
            value = np.asarray(ref["data"][field])
            unique = np.unique(value)
            if len(unique) != 1:
                raise ValueError(f"Reference mixes incompatible {field} values")
            source["reference_metadata"][field] = unique[0].item()
    if ref["finite_support"] is not None:
        (args.output_dir / "finite_support.json").write_text(json.dumps(ref["finite_support"], indent=2) + "\n")
    plans, sensitivity = [], []
    if args.schema:
        from .conditions import classify, load_schema, schema_identity, supported_bins
        schema = load_schema(args.schema)
        sid = schema_identity(schema)
        source.update(condition_schema_sha256=_sha(args.schema), condition_schema_id=sid)
        counts = sorted(set(ref["n_target"].tolist())) if args.counts is None else args.counts
        for n in counts:
            selected = ref["n_target"] == n
            if not selected.any():
                raise ValueError(f"No complete reference rows at requested count {n}")
            labels = classify(schema, np.full(int(selected.sum()), n), ref["energy_eV"][selected] / ref["n_atoms"][selected])
            supported = supported_bins(schema, n)
            plan = make_allocation_plan(ref["energy_eV"][selected], labels, ref["measure"][selected],
                args.temperatures, supported, args.budget, measure_kind=args.reference_kind,
                exploration_fraction=args.exploration, mixture_weights=args.mixture_weights,
                schema_id=sid, composition=int(n), reference_id=source["reference_sha256"])
            plans.append(plan)
            if args.reference_kind == "uniform_raw_sample":
                # The full-pool indices are first restricted to this composition.
                original_order = ref["data"].get("raw_sample_indices")
                local_order = None
                if original_order is not None:
                    original_order = np.asarray(original_order)
                    if (original_order.ndim != 1 or not np.isfinite(original_order).all()
                            or not np.equal(original_order, np.floor(original_order)).all()
                            or np.any((original_order < 0) | (original_order >= len(selected)))):
                        raise ValueError("Invalid raw_sample_indices in reference")
                    global_to_local = np.full(len(selected), -1, dtype=np.int64)
                    global_to_local[np.flatnonzero(selected)] = np.arange(int(selected.sum()))
                    local_order = global_to_local[original_order.astype(int)]
                    local_order = local_order[local_order >= 0]
                for temperature in args.temperatures:
                    diagnostic = reference_pool_sensitivity(ref["energy_eV"][selected], labels, ref["measure"][selected],
                        temperature, supported, seed=args.sensitivity_seed, raw_sample_indices=local_order)
                    diagnostic["n_target"] = int(n)
                    sensitivity.append(diagnostic)
    result = dict(schema_version="static-acquisition-run-v1", source=source, plans=plans,
        reference_pool_sensitivity=sensitivity, finite_support_verified=ref["finite_support"] is not None,
        interpretation="Temperature-dependent acquisition under a static MLIP surrogate; energies are not basin free energies",
        physical_temperature_is_not_decoder_temperature=True,
        complete_reference_required=True)
    result["run_id"] = _identity(result)
    (args.output_dir / "allocation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    rows = []
    for plan in plans:
        for group in [*plan["per_temperature"], *plan["temperature_mixtures"]]:
            for request in group["requests"]:
                rows.append({**request, "n_target": plan["n_target"], "allocation_id": plan["allocation_id"]})
    if rows:
        with (args.output_dir / "requests.csv").open("w", newline="") as f:
            names = list(dict.fromkeys(k for r in rows for k in r))
            writer = csv.DictWriter(f, fieldnames=names); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(status="PASS", plans=len(plans), finite_support_verified=result["finite_support_verified"],
                         output=str(args.output_dir), run_id=result["run_id"]), indent=2))
    return result


if __name__ == "__main__":
    main()

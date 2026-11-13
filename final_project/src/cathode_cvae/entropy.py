"""Exact finite-lattice configurational thermodynamics and sampling diagnostics.

Occupancy one is a present mobile atom. Energies are TOTAL eV per periodic cell;
symmetry multiplicities count labelled site masks, not additional phases or spins.
The routines describe the supplied fixed lattice and supplied energy model only.
"""
from math import comb
import numpy as np
from scipy.special import logsumexp
from .data import validate_permutation_group

KB_EV_K = 8.617333262145e-5


def boltzmann_training_weights(energy_eV, counts, splits, multiplicity, temperature_K):
    """Reweight TRAIN orbits toward a fixed-T labelled-mask Boltzmann target.

    Energies are TOTAL eV per cell. Each composition preserves its original
    TRAIN multiplicity sum, so only its internal target distribution changes.
    VAL/TEST weights are unchanged and their energies are never consulted.
    """
    e=np.asarray(energy_eV,float);n=np.asarray(counts);roles=np.asarray(splits).astype(str)
    g=np.asarray(multiplicity,float);T=float(temperature_K)
    if (e.ndim!=1 or any(v.shape!=e.shape for v in [n,roles,g]) or not len(e)
            or not np.isfinite(T) or T<=0 or not np.isfinite(g).all() or np.any(g<=0)):
        raise ValueError('Aligned finite positive multiplicities and T>0 required')
    train=roles=='train'
    if not train.any() or not np.isfinite(e[train]).all():
        raise ValueError('Finite TRAIN total-cell energies required')
    result=g.copy();groups=[]
    for count in sorted(set(n[train])):
        ix=np.flatnonzero(train & (n==count));reference=float(e[ix].min())
        total=float(g[ix].sum());logw=np.log(g[ix])-(e[ix]-reference)/(KB_EV_K*T)
        result[ix]=total*np.exp(logw-logsumexp(logw))
        if not np.isfinite(result[ix]).all() or np.any(result[ix]<=0):
            raise ValueError('Boltzmann TRAIN weights exceed floating-point dynamic range; do not silently clip')
        if not np.isclose(result[ix].sum(),total,rtol=1e-12,atol=1e-12):
            raise AssertionError('Per-composition TRAIN mass changed')
        groups.append(dict(n_target=int(count),train_orbits=len(ix),
            train_energy_reference_eV=reference,original_train_weight_sum=total,
            new_train_weight_sum=float(result[ix].sum()),minimum_weight=float(result[ix].min()),
            maximum_weight=float(result[ix].max()),weight_ESS=float(result[ix].sum()**2/np.sum(result[ix]**2))))
    return result,groups


def enumerate_orbits(permutations, maximum_sites=20):
    """Exhaustively partition all binary masks under a verified finite site group."""
    p = np.asarray(permutations, dtype=np.int64)
    validate_permutation_group(p)
    n = p.shape[1]
    if not 1 <= n <= maximum_sites:
        raise ValueError("Exhaustive enumeration exceeds explicit site cap")
    powers = np.left_shift(np.uint64(1), np.arange(n - 1, -1, -1, dtype=np.uint64))
    codes = np.arange(2 ** n, dtype=np.uint64)
    raw = ((codes[:, None] & powers) != 0).astype(np.uint8)
    inv = np.argsort(p, axis=1)
    canonical = np.empty(len(raw), dtype=np.uint64)
    for start in range(0, len(raw), 512):
        vacancy = 1 - raw[start:start + 512]
        transformed = vacancy[:, inv]
        canonical[start:start + 512] = (transformed @ powers).min(axis=1)
    representatives, raw_to_orbit, multiplicity = np.unique(
        canonical, return_inverse=True, return_counts=True)
    masks = ((representatives[:, None] & powers) != 0).astype(np.uint8)
    occupancy = 1 - masks
    n_target = occupancy.sum(axis=1).astype(np.int64)
    counts = []
    for retained in range(n + 1):
        selected = n_target == retained
        expected = comb(n, retained)
        if int(multiplicity[selected].sum()) != expected:
            raise AssertionError("Orbit multiplicities do not sum to binomial count")
        counts.append(dict(n_target=retained, labelled_masks=expected,
                           symmetry_orbits=int(selected.sum())))
    if not np.array_equal(n_target[raw_to_orbit], raw.sum(axis=1)):
        raise AssertionError("Symmetry operation changed composition")
    return dict(occupancy=occupancy, masks=masks,
                canonical_vacancy_codes=representatives,
                raw_occupancy=raw, raw_to_orbit=raw_to_orbit,
                multiplicity=multiplicity.astype(np.int64), n_target=n_target,
                counts=counts)


def thermodynamics(energy_eV, multiplicity, temperature_K):
    """Canonical finite-state observables, using a stable minimum-energy shift.

    S/kB = log(Z_rel) + (<E>-E_ref)/(kB*T); C/kB = Var(E)/(kB*T)^2.
    The reference shift cancels from normalized probabilities and entropy.
    """
    energy = np.asarray(energy_eV, dtype=float)
    g = np.asarray(multiplicity, dtype=float)
    temperature = float(temperature_K)
    if (energy.ndim != 1 or not len(energy) or energy.shape != g.shape
            or not np.isfinite(energy).all() or not np.isfinite(g).all()
            or np.any(g <= 0) or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError("Finite energies, positive multiplicities and T>0 are required")
    reference = energy.min()
    reduced = (energy - reference) / (KB_EV_K * temperature)
    log_weight = np.log(g) - reduced
    logZ = float(logsumexp(log_weight))
    probability = np.exp(log_weight - logZ)
    shifted_mean = float(probability @ (energy - reference))
    variance = float(probability @ (energy - reference - shifted_mean) ** 2)
    entropy = logZ + shifted_mean / (KB_EV_K * temperature)
    # Independent identity checks the multiplicity term and energy units.
    support = probability > 0
    entropy_check = float(-np.sum(probability[support]
                         * (np.log(probability[support]) - np.log(g[support]))))
    if not np.isclose(entropy, entropy_check, atol=1e-10, rtol=1e-10):
        raise AssertionError("Partition-function and probability entropies differ")
    return dict(temperature_K=temperature, reference_energy_eV=float(reference),
                logZ_relative=logZ, mean_energy_eV=float(reference + shifted_mean),
                variance_energy_eV2=variance, entropy_kB=float(entropy),
                heat_capacity_kB=variance / (KB_EV_K * temperature) ** 2,
                orbit_probability=probability)


def energy_bins(energy_eV, n_bins):
    """Benchmark bins over the exactly known support; not a degeneracy tolerance."""
    energy = np.asarray(energy_eV, dtype=float)
    if not np.isfinite(energy).all() or energy.ndim != 1 or n_bins < 1:
        raise ValueError("Finite energy vector and positive bin count required")
    lo, hi = float(energy.min()), float(energy.max())
    if lo == hi:
        return np.zeros(len(energy), int), np.array([lo, np.nextafter(hi, np.inf)])
    edges = np.linspace(lo, np.nextafter(hi, np.inf), n_bins + 1)
    return np.clip(np.searchsorted(edges, energy, side='right') - 1, 0, n_bins - 1), edges


def importance_benchmark(energy_eV, multiplicity, temperature_K, *,
                         draws, seed, tilt_temperature_K=100., exploration=.1):
    """Known biased orbit proposal: finite-budget importance and naive diagnostics.

    The proposal mixes uniform *labelled-mask* probability with a low-energy
    Boltzmann tilt. It is explicitly synthetic, not a fitted CVAE. The positive
    exploration mixture ensures support, so the Z estimator is unbiased; ratios
    and entropy are finite-sample biased estimators and report that limitation.
    """
    e = np.asarray(energy_eV, float); g = np.asarray(multiplicity, float)
    if draws < 1 or not 0 < exploration <= 1:
        raise ValueError("Positive draws and exploration in (0,1] required")
    exact = thermodynamics(e, g, temperature_K)
    tilted = thermodynamics(e, g, tilt_temperature_K)['orbit_probability']
    q = exploration * g / g.sum() + (1 - exploration) * tilted
    rng = np.random.default_rng(seed); chosen = rng.choice(len(e), size=draws, p=q)
    shifted = e - e.min(); beta = 1 / (KB_EV_K * temperature_K)
    log_w = np.log(g[chosen]) - beta * shifted[chosen] - np.log(q[chosen])
    logZ = float(logsumexp(log_w) - np.log(draws))
    normalized = np.exp(log_w - logsumexp(log_w))
    mean_shift = float(normalized @ shifted[chosen])
    entropy = logZ + beta * mean_shift
    ess = float(1 / np.sum(normalized ** 2))
    frequencies = np.bincount(chosen, minlength=len(e)) / draws
    nz = frequencies > 0
    naive_entropy = float(-np.sum(frequencies[nz]
                        * np.log(frequencies[nz] / g[nz])))
    return dict(draws=int(draws), seed=int(seed), temperature_K=float(temperature_K),
                proposal='synthetic known-probability tilted/uniform microstate mixture',
                exact_entropy_kB=exact['entropy_kB'], naive_sample_entropy_kB=naive_entropy,
                importance_entropy_kB=entropy, importance_ESS=ess,
                importance_mean_energy_eV=float(e.min() + mean_shift),
                exact_mean_energy_eV=exact['mean_energy_eV'],
                unique_orbits=int(np.count_nonzero(frequencies)),
                limitation='Self-normalized ratios/entropy are finite-sample biased; ESS is weight concentration, not independent chain length')


def wang_landau_benchmark(raw_energy_eV, raw_occupancy, *, n_target, n_bins,
                         steps, seed, check_every=1000, flatness=.8, minimum_logf=1e-4):
    """Bounded WL benchmark on an exact lookup table at one fixed composition.

    Symmetric occupied/vacant swaps preserve count. Exact occupied-bin support
    is supplied to the flatness check, making this an intentionally favourable
    validation of DOS estimation, not proof of discovery of unknown support.
    Bin midpoint energies introduce a separate, quantifiable discretization.
    """
    raw = np.asarray(raw_occupancy, np.uint8); e = np.asarray(raw_energy_eV, float)
    n = raw.shape[1]
    if len(raw) != 2 ** n or len(e) != len(raw) or not 0 <= n_target <= n:
        raise ValueError("Complete binary lookup table is required")
    if steps < 1 or check_every < 1 or not 0 < flatness <= 1:
        raise ValueError("Invalid WL budget/flatness")
    selected = np.flatnonzero(raw.sum(axis=1) == n_target)
    bins, edges = energy_bins(e[selected], n_bins)
    code_bins = np.full(len(raw), -1, int); code_bins[selected] = bins
    exact_counts = np.bincount(bins, minlength=len(edges) - 1)
    active = exact_counts > 0
    powers = (1 << np.arange(n - 1, -1, -1)).astype(np.int64)
    if not np.array_equal(raw @ powers, np.arange(len(raw))):
        raise ValueError("Raw table must be in ascending binary occupancy-code order")
    rng = np.random.default_rng(seed); code = int(rng.choice(selected))
    state = raw[code].copy(); logg = np.zeros(len(exact_counts)); hist = np.zeros_like(exact_counts)
    logf = 1.; phases = []; converged = False
    for t in range(1, steps + 1):
        if 0 < n_target < n:
            i = int(rng.choice(np.flatnonzero(state))); j = int(rng.choice(np.flatnonzero(1 - state)))
            proposal = code ^ int(powers[i]) ^ int(powers[j])
            old_bin, new_bin = code_bins[code], code_bins[proposal]
            if np.log(rng.random()) < min(0., logg[old_bin] - logg[new_bin]):
                code = proposal; state[i] = 0; state[j] = 1
        b = code_bins[code]; logg[b] += logf; hist[b] += 1
        if t % check_every == 0 and hist[active].min() >= flatness * hist[active].mean():
            phases.append(dict(steps=t, logf=logf, minimum_histogram=int(hist[active].min()),
                               mean_histogram=float(hist[active].mean())))
            logf *= .5; hist[:] = 0
            if logf <= minimum_logf:
                converged = True; break
    normalized = np.zeros(len(logg)); normalized[active] = np.exp(logg[active] - logsumexp(logg[active]))
    truth = exact_counts / exact_counts.sum()
    log_error = logg[active] - np.log(exact_counts[active]); log_error -= log_error.mean()
    return dict(seed=int(seed), n_target=int(n_target), n_bins=int(n_bins), steps=int(t),
                flatness=float(flatness), final_logf=float(logf), converged=converged,
                active_bins=int(active.sum()), visited_bins=int(np.count_nonzero(logg)),
                density_total_variation=float(.5 * np.abs(normalized - truth).sum()),
                log_density_RMSE_up_to_constant=float(np.sqrt(np.mean(log_error ** 2))),
                estimated_probability=normalized, exact_counts=exact_counts,
                edges_eV=edges, phases=phases,
                scope='Exact bin support supplied; fixed lattice and total-energy lookup, not unknown-space convergence')

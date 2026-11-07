# Static-energy temperature acquisition

This module allocates generation requests. It does not change the CVAE decoder's
sampling temperature, compute basin free energies, or measure configurational
entropy. The physical temperature is in kelvin. The default project grid is
393.15, 433.15, 473.15 and 493.15 K; its provenance belongs in the campaign receipt.

At one fixed parent and composition, representative i has total energy E_i in
**eV per cell**. The Boltzmann constant is k_B = 8.617333262145e-5 eV/K and the
common reference E_ref is the lowest energy in that reference pool. For a complete
list of symmetry representatives, g_i counts the different raw occupation masks
in i's orbit. It does not count permutations of labels on identical atoms.
The unnormalized dimensionless weight is

    w_i(T) = g_i exp[-(E_i - E_ref)/(k_B T)].

The fraction assigned to energy bin b is W_b = sum_{i in b} w_i / sum_i w_i.
Subtracting E_ref improves numerical stability and cancels from the probabilities.
The code uses logarithmic normalization. A constant energy offset cancels, but
distorted energy gaps change these fractions even if the energy ranks agree.
Stored energy per atom is multiplied by the **actual** atom count before weighting.
Do not use an energy per atom directly in this exponential for a whole cell.

There are two deliberately different reference measures:

* `exhaustive_orbits`: every orbit in every count 0..N is present. Multiplicities
  are calculated from the declared parent action, and their sums must equal
  binomial(N,n) at every n. The Hu 12-site archive provides 87 representatives;
  six Li has 17 representatives covering all 924 raw fixed-count masks.
* `uniform_raw_sample`: a prespecified IID uniform-raw fixed-count sample is
  completely scored. After canonicalizing and deduplicating energy calls, retain
  each representative's **raw_frequency**. Its weight is f_i exp[-DeltaE/kBT].
  **Do not multiply by g_i as well.** The original sampling frequency already
  carries the occupation degeneracy. A proposal-biased training/CVAE/mutation
  histogram is not such a reference sample.

The discrete finite-support audit is exact for the declared permutation action.
Physical scalar-label reuse additionally assumes that the supplied parent,
geometry protocol and symmetry tolerance justify the equivalence. Forces cannot
be reused by an orbit key without transforming their coordinate frame.

The 48-site pool remains a finite Monte Carlo reference. Its self-normalized
weights do not establish coverage of unseen low-energy states. Score the complete
prespecified pool: selecting only available cache hits or successful calls can
bias the measure. Missing or nonfinite energies cause an error rather than being
dropped. Keep this allocation reference separate from assessment proposals.

## Inputs and commands

The reference NPZ contains one unique orbit per row, `occupancy` (1=present),
`n_atoms`, `n_Li` or `n_target`, and either `energy_eV` or `energy_eV_atom`.
A uniform-raw reference also requires positive integer `raw_frequency`. It can
retain `raw_sample_indices`, one reference-row index per original raw draw.
The parent NPZ supplies numbers, coordinates, removable-site indices and site
permutations. A scalar `parent_sha256` is verified when provided. Include scalar
teacher/head, geometry protocol, sampling seed/measure, and checkpoint identity
in the reference metadata or its accompanying immutable receipt.

From an appropriate compute allocation on an HPC system:

```bash
python -m cathode_cvae.acquisition \
  --reference runs/REFERENCE.npz \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --schema runs/FIT/condition_schema.json \
  --reference-kind uniform_raw_sample --counts 12 24 36 \
  --temperatures 393.15 433.15 473.15 493.15 \
  --mixture-weights 1 1 1 1 --exploration .1 --budget 256 \
  --output-dir runs/NEW_ALLOCATION
```

This produces `allocation.json` and `requests.csv`, with the schema and reference
identities, requested bin, physical temperature, policy, probability and integer
attempt count. Each *per-temperature* plan independently totals the declared
budget. Each *temperature-mixture* plan instead totals that budget **jointly**
over temperature and bin. Choose a labeled plan to execute, rather than accidentally
executing every diagnostic plan as one budget. The mixture weights are a chosen
allocation over operating conditions; they are not calibrated reactor frequencies.

Pure static weights and an exploration variant are both saved. The latter uses

    P_b = (1 - epsilon) W_b + epsilon / number_of_supported_bins.

The default epsilon=0.1 is a deliberate acquisition choice. It is not a
thermodynamic constant. Unsupported training bins cannot be requested. Integer
budgets use largest remainders with deterministic ties in listed bin order.

An exact finite-support audit requires no fitted model:

```bash
python -m cathode_cvae.acquisition \
  --reference artifacts/current/extras/learning_all_parents/learning_Hu_O3_12_uma.npz \
  --parent artifacts/current/extras/learning_all_parents/parents/Hu_O3_12.npz \
  --reference-kind exhaustive_orbits --validate-exhaustive \
  --output-dir runs/NEW_HU_SUPPORT_AUDIT
```

## Concentration and sensitivity diagnostics

For collapsed representative probabilities p_i, `representative_ess` is
1/sum(p_i^2). It tells how many representatives effectively carry the weight.
A second quantity is Z^2 / sum_i m_i a_i^2, where a_i=exp[-DeltaE/kBT] and m_i is
g_i or f_i. For exhaustive support it is the effective number of physical raw
occupation microstates; for an IID raw sample it is the usual importance-sampling
**particle ESS**, counting repeated sampled masks. These quantities answer
different questions. For equal energies, the second equals the total raw count,
while representative ESS can be much smaller when orbit sizes are unequal.
Neither is a confidence interval or proof of global coverage.

The output also reports the largest representative mass and largest single
microstate/particle probability. Uniform-raw references include nested-prefix
pool-size sensitivity at each temperature. Original raw draw order is used when
available. Otherwise the raw-frequency multiset receives a seeded,
energy-independent shuffle. Prefixes are correlated, not independent replicates.
Strong concentration or changing prefix allocations must be reported rather than
hidden by quoting only the normalized bin totals.

Finally, rescore the generated proposals. Requested allocation proportions are
input counts; measured energy-bin confusion, in-bin hits, duplicates, and novelty
are distinct outcomes. Correct allocation totals do not demonstrate that the
CVAE generates the requested energy distribution.

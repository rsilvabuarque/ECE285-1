# Numerical conventions for the same-UMA pilot

This is a numerical and method comparison under UMA-S-1.2.1 with its `omat`
head. It does not establish DFT accuracy or real-LCO free energies. The
calculator uses the project's pinned fairchem-core 2.25.0 checkpoint, FP32,
TF32 off, no compilation, and the same common fixed parent cell for every
candidate. New calculations run only in root-managed compute allocations.

`scripts/overnight_20261107/thermal_dynamics.py` supplies the actual runner.
Every output directory is new; the runner records its source hash, input and
parent hashes, checkpoint/head/package/precision context, and raw structures.

## Relaxation and site identity

Fixed-cell LBFGS uses a provisional maximum force target of 0.001 eV/Å, at most
1,500 steps, and a maximum atomic step of 0.1 Å. This target is not a universal
phonon standard. Convergence status and residual forces are recorded, and
force reproducibility and displacement-size sensitivity must be assessed
before interpreting small Hessian eigenvalues.
Each Hessian shard also performs three fresh predictions at its common
undisplaced endpoint, resetting the ASE result cache before each. Reported
energy/force/stress spans measure inference reproducibility only; even an
exactly zero span does not bound derivative bias or model error.

Each surviving atom retains its original parent index. A periodic rectangular
Hungarian assignment maps the present Li atoms to distinct sites among the
48 parent Li sites. Its objective is the sum of squared minimum-image
distances. Co/O displacements provide a common translation alignment before
the Li assignment. The output also records nearest/second-nearest distance
gaps, nearest-site collisions, assigned distances, host RMS/max displacement,
and relative layer-normal displacement. A changed assigned occupation or a
small distance gap is an observation requiring inspection, not proof of a hop.
Different initial masks that merge into one relaxed basin cannot be counted
as independent thermodynamic states.

## Finite differences and mode counting

For Cartesian coordinates R and positive displacement δ, a raw Hessian column
is H_ij = −[F_i(R+δe_j)−F_i(R−δe_j)]/(2δ), in eV/Å². The initial displacement
is δ = 0.01 Å; one representative basin is also evaluated at 0.02 Å. Positive
and negative displaced forces and energies, the input hash, all raw columns,
and their timing are saved. Shards must cover every column exactly once at
one displacement and share the same geometry, masses, teacher, and precision.

The antisymmetric part H−Hᵀ is measured and saved before forming the symmetric
matrix (H+Hᵀ)/2. Mass weighting gives D = M⁻¹ᐟ² H M⁻¹ᐟ², in
eV/(Å² amu), with one repeated atomic mass per Cartesian coordinate. The three
uniform translations have mass-weighted components √m_a divided by
√(Σ_a m_a). An orthonormal complement projects those three known directions
separately; the residual norm of D acting on the translation basis is reported.
For 180 atoms, 540 Cartesian modes therefore become 537 internal modes.

All signed internal eigenvalues are retained. Frequencies are represented as
sign(λ)√|λ| times the documented unit conversion; a negative number explicitly
denotes an imaginary frequency rather than being converted into a stable
oscillation. Small negative modes must be compared across displacement sizes
and numerical precision. No mode is removed merely because it is inconvenient.
Agreement of two displacements is a diagnostic, not a proof of MLIP accuracy.

These are Γ modes of the entire finite periodic simulation cell, matching the
MD cell. They include the primitive-cell wave vectors folded into that cell,
but do not provide longer-wavelength bulk modes absent from the finite cell.

## Dynamics and saved observables

The initial protocol uses a 0.5 fs integration step, Maxwell–Boltzmann initial
momenta, removal of center-of-mass momentum, and ASE `FixCom` throughout.
Only three translations are constrained, so the kinetic temperature uses
3N−3 degrees of freedom: T = 2K/[(3N−3)k_B]. Periodic bulk rotations are not
removed. The target is 433.15 K. Bussi stochastic velocity rescaling with a
100 fs coupling time equilibrates the system for 5 ps, followed by up to
20 ps of NVE velocity Verlet production. Two independent seeds are planned
only for mechanically stable, distinct basins. Short throughput/stability
trajectories precede long production; a duration cap is not convergence.

Thermostat sampling is distinguished from dynamical correlation analysis.
NVE production avoids continuous thermostat modification of the velocities;
bounded NVT production can provide an explicit sensitivity comparison. No
stepwise velocity rescaling is imposed during NVE. Actual kinetic temperature,
total-energy drift/span, residual center-of-mass velocity, forces, occupation
assignments, and host distortion are reported.

ASE-native velocities have units Å per ASE internal time unit. The conversion
to Å/ps is `v_A_ps = v_ASE * ase.units.fs * 1000`. The arrays and metadata
state this explicitly. Potential, kinetic, and total energies are eV per
simulation cell. The production energy anchor is the mean **total** MD energy
over exactly the saved analyzed frames; the relaxed minimum is stored
separately, so it cannot accidentally be added twice.

Detailed positions/energies/forces/stresses and mapping diagnostics are saved
every 2 fs on `time_fs`. A separate inexpensive observer saves velocities every
0.5 fs on `velocity_time_fs`, with kinetic energy and the current cached
potential/total energy. It asserts that the calculator geometry is current
and adds no MLIP inference. Thus `velocity_energy_total_eV` provides an anchor
on exactly the velocity frames; the downstream analyzer must recompute its
mean after selecting a block or discarding frames. `detailed_velocities_ASE`
and `detailed_velocities_A_ps` remain aligned with the 2 fs data. The ATLAS
py-xPT cage kernel has a saved-step/Einstein-frequency criterion beyond a
Nyquist check; actual frequency and stride sensitivity must therefore be
reported. The py-xPT implementation automatically subtracts periodic
translations, so its `constraints` setting is 0 for the already COM-fixed
system; downstream integrated mode counting must still give 537.

Nested blocks, independent repeats, saved-stride checks, and δ sensitivity
address different uncertainties. They must not be presented as interchangeable
confidence intervals. Thermal corrections and weights are conditional on the
selected, distinct, retained basins rather than the complete equilibrium
configuration space.

## Verification

`test_thermal_dynamics.py` checks an analytically soluble two-mass spring,
mass-weighted translation projection, retained unstable signs, saved raw
antisymmetry, periodic Li assignment, and the short mocked MD array/unit/DOF
contract. These are implementation tests and do not validate UMA forces.

Official ASE interfaces were checked against the installed 3.26.0 source and
[the ASE dynamics documentation](https://wiki.fysik.dtu.dk/ase/ase/md.html).
The separate xPT analysis records its pinned ATLAS commit and conventions.

## Interpreting the kinetic mode-group diagnostic

The kinetic equipartition comparison does **not** require a harmonic potential.
Write the mass-weighted momentum as `p_tilde = sqrt(M) v`, and let the columns
of U be any fixed orthonormal basis of the 537-dimensional subspace remaining
after removing the three translations. The projected momenta are
`a = U.T p_tilde`, and exactly `K = sum_s a_s^2 / 2`. In a canonical ensemble,
the distribution factorizes into a position part `exp[-U_potential(R)/(kB T)]`
and independent momentum factors `exp[-a_s^2/(2 kB T)]`. Consequently
`mean(a_s^2) = kB T` and `mean(K_s) = kB T / 2` for every projected component,
even when the potential energy is strongly anharmonic. A group containing m
components therefore has expected kinetic temperature
`T_group = 2 mean(sum_s K_s)/(m kB) = T`.

Using the relaxed Hessian eigenvectors supplies useful low-to-high frequency
labels for those groups. Interpreting their zero-temperature frequencies as
the actual finite-temperature oscillation frequencies is an approximation;
the canonical kinetic equipartition identity is not. Anharmonic mode coupling
affects energy exchange, frequency interpretation, and correlation times, not
the exact canonical identity.

Measured group differences in a short trajectory remain descriptive. Frames
are correlated, and weakly coupled NVE modes can retain unequal individual
energies over the available trajectory. Global Bussi temperature control does
not demonstrate rapid sampling of every mode-energy distribution. Independent
seeds and time-window comparisons are more informative than treating tens of
thousands of neighboring velocity frames as independent canonical draws.
These observations do not define a new universal pass/fail threshold.

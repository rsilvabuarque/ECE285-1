# Same-UMA finite-cell thermal pilot

This research extension starts after the [completed midterm gate](../../artifacts/overnight_20261107/phase1_gate.json) and is separate
from the class report. It tests implementation, numerical consistency and cost
under one pinned potential. It does not establish real-LCO free energies or an
accurate general anharmonic correction. No new DFT or MLIP fine-tuning is run.

The completed pilot gives three useful findings. Finite-cell harmonic
contributions reverse the static UMA ordering of three selected minima. MACE
also reverses UMA's static ordering on the exact same coordinates, making
matched DFT checks important. Finally, two of four 20 ps UMA trajectories
undergo persistent Li hops. Their full trajectories cannot supply the initial
occupation basin's vibrational free energy. An exploratory common first-10-ps
analysis has 3PT between-basin differences whose observed cross-seed ranges
cross zero under both temperature transports. An accurate thermal ordering is
therefore unresolved. Thirteen exact structures and existing labels are
prepared for a later DFT check; no DFT is executed or scheduled.

## Question, controls and evidence boundary

The hypothesis is that differences in thermal contributions between competing
vacancy basins might change their relative weights compared with relaxed
static-energy weights. A useful negative result is that the correction is small,
unresolved, estimator-dependent, or invalidated by instability. Comparison of two
estimators under the same potential distinguishes their numerical/method choices
from the separate question of whether that potential is physically accurate.

The [prespecified design](DESIGN.md) freezes Li36Co48O96, the parent cell,
UMA-S-1.2.1/`omat`, FP32 and TF32 off. Its 180 atoms imply 540 Cartesian
coordinates. Removing three uniform center-of-mass translations leaves 537
internal modes; periodic fixed-cell crystal calculations do not also remove
three free-molecule rotations. The target is 433.15 K. The starting masks are
existing training examples, not new CVAE discoveries or DFT ground states.

The existing DFT archive contains no 48-Co cell. At x=0.75 the relevant evidence
is ten historical 90-atom rattled frames, ten Hu 45-atom optimized frames and
600 Hu 45-atom DPGEN frames. The following values come from exact matched
geometries in the older FP64 model audit, not the new FP32 thermal trajectory:

| Reference cohort | Model | Frames | Spearman ρ | Pair-gap RMSE (meV/atom) | Median force-component RMSE (eV/Å) | Median stress RMSE (GPa) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Historical O3 rattled | UMA | 10 | 0.9030 | 4.084 | 0.1054 | 2.726 |
| Historical O3 rattled | MACE | 10 | 0.8545 | 4.311 | 0.1178 | 1.760 |
| Hu O3 optimized | UMA | 10 | 0.9605 | 0.806 | 0.1418 | 2.480 |
| Hu O3 optimized | MACE | 10 | 0.6322 | 2.212 | 0.1289 | 4.337 |
| Hu O3 DPGEN | UMA | 600 | 0.9097 | 16.510 | 0.1513 | 2.397 |
| Hu O3 DPGEN | MACE | 600 | 0.8196 | 47.187 | 0.1791 | 4.722 |

Force RMSE here is √mean[(Fmodel−FDFT)²] over Cartesian components in one
frame; the table then takes the median across frames. It is distinct from a
maximum atomic force or a mean vector-error magnitude.

The complete three-model table, frame identities, source hashes and actual
historical DFT parameter audit are in
[`DFT_candidates/`](../../reports/thermal_pilot_20261107/DFT_candidates/).
Rank agreement within a small optimized pool does not validate forces, local
curvatures or a different-sized cell. Historical and Hu electronic protocols
remain separate. The exact physical-validation boundary and next reference
package are described in [DFT_REQUIRED_NEXT.md](DFT_REQUIRED_NEXT.md).

## What is compared

For a basin i, write its minimum energy as Eᵢ⁰ in eV per cell and its basin free
energy as Fᵢ(T,V), also in eV per cell. Here a basin means a region of coordinate
space associated with one local minimum, not simply the identity of the
unrelaxed starting mask. Relaxation and thermal motion can merge or leave it.

At a fixed volume V and temperature T, the comparison is Helmholtz free energy,
F=U−TS. U is mean internal energy in eV/cell and S is entropy in eV/(cell·K).
No pressure term or Li reservoir chemical potential is added. The comparison
is between equal-composition cells; it is not an insertion reaction or a reactor
equilibrium calculation. An xPT atomic-group chemical-potential field is not
used as an insertion chemical potential. That observable would require
consistent whole-cell free-energy differences across Li counts and appropriate
reservoir/reaction references.

Near a stationary point, write the coordinate displacement vector as u in Å
and approximate E(R0+u)≈E0+½uᵀHu. The Hessian H contains second derivatives
of energy, equivalently minus force derivatives, in eV/Å². Mass weighting,
D=M⁻¹ᐟ²HM⁻¹ᐟ², separates this coupled quadratic motion into independent normal
modes. After consistent unit conversion, its eigenvalues are ω². A negative
internal eigenvalue describes downhill curvature rather than a stable oscillator;
replacing its frequency by an absolute value would change that physical problem.

For stable harmonic modes with angular frequencies ωₖ, the quantum approximation
is

$$
F_i^{\mathrm{harm},q}=E_i^0+
\sum_{k=1}^{537}\left[\frac{\hbar\omega_{ik}}2+
k_BT\ln\left(1-e^{-\hbar\omega_{ik}/k_BT}\right)\right].
$$

kB=8.617333262145×10⁻⁵ eV/K is Boltzmann's constant; ℏ is the reduced
Planck constant in eV·s; and k indexes
internal modes. The stored ordinary frequencies ν in THz satisfy ℏω=hν;
angular and ordinary frequencies must not be substituted without the 2π factor. Each quantum mode has partition function
Zk=exp[−ℏωk/(2kBT)]/[1−exp(−ℏωk/kBT)]. Multiplying independent mode partition
functions with exp[−Eᵢ⁰/(kBT)] and applying F=−kBT ln Z gives the sum above,
with Eᵢ⁰ included once.
The corresponding classical expression is
Eᵢ⁰+kBT Σₖ ln(ℏωᵢₖ/kBT). A common phase-space reference cancels from
equal-mode-count differences. These formulas require genuine positive internal
curvatures, not the absolute value of an unstable frequency. The force-constant,
dynamical-matrix and free-energy conventions are summarized in the official
[Phonopy formulations](https://phonopy.github.io/phonopy/formulation.html#thermodynamic-properties).
The pilot uses the finite periodic cell's Γ Hessian so its modes match the
periodic MD cell; a dense primitive-cell wave-vector integral would add a
different finite-size approximation.

The trajectory estimator uses atomic velocities and their mass-weighted
autocorrelation: how similar the velocity at time t is to an earlier velocity,
averaged over available time origins. Its Fourier spectrum assigns weight to
frequencies of motion. ATLAS 1PT treats it as an all-solid spectrum; 2PT separates
diffusive and solid-like contributions; 3PT adds a particular liquid cage term.
The cage term is explicitly gated off in the nondiffusive crystal limit, so its
name does not guarantee an arbitrary crystal anharmonic free-energy correction.
This scope follows the [3PT paper, §II.2 and Eq.8](https://arxiv.org/html/2607.29026v2).

The actual [ATLAS implementation and conventions](ATLAS_CONVENTIONS.md) are
pinned to py-xPT 2.1.0, commit 197bdcf5f57c9d04b5527f9e7d6f4fdb8bf3b82d.
The package automatically removes periodic COM translations, so the adapter
passes zero additional constraints; passing three would incorrectly leave 534
modes. Its version-specific behavior is recorded with the xPT outputs. In particular, the quantum energy uses the MD anchor,

$$
E_q=\langle K+E_{\rm pot}\rangle_{\mathrm{MD}}+(E_{\mathrm{QHO}}-E_{\mathrm{CHO}}),
\qquad F_q=E_q-TS_q.
$$

K and Epot are instantaneous kinetic and potential energy; QHO and CHO denote
quantum and classical oscillator-weighted spectral energies. The subtraction
replaces the spectral classical oscillator contribution instead of adding a
second copy of the minimum energy. A reported thermal correction is Fq−Eᵢ⁰;
adding Eᵢ⁰ to Fq again would be incorrect. As a check, an equilibrated classical harmonic reference has
⟨K+Epot⟩=Eᵢ⁰+537kBT and Sc=kBΣk[1−ln(ℏωk/kBT)]. Subtracting TSc
recovers Eᵢ⁰+kBTΣk ln(ℏωk/kBT). This explains why the minimum is included once
and why the MD anchor needs kinetic plus potential energy, not potential alone.
This is a reference-limit identity, not an assertion that the finite MD data
already satisfy equipartition.

The native classical DOS energy is not MD-anchored; its physical comparison
uses ⟨K+Epot⟩MD−TSc while preserving the original output field. This convention
is explicit for the quantum field in
[Ramji et al., §2.4, Eq.10](https://arxiv.org/html/2610.37586v1).

## Executed selection and numerical protocol

Three n=36 masks were selected sequentially from the first 1,000 lowest-UMA
training deletion energies, requiring at least six sites of symmetry-minimized
Hamming separation from prior selections. Six differing bits mean three Li/
vacancy swaps. This is a declared diversity heuristic, not proof of distinct
relaxed basins. The initial cell-energy differences are approximately
0/0.08070/0.12798 eV. They are deletion differences, not relaxed-minimum gaps.

The design specifies fixed-cell LBFGS relaxation, maximum force 0.001 eV/Å,
maximum step 0.1 Å and a 1,500-step cap. These are provisional numerical settings;
residual forces and finite-difference sensitivity determine what the spectrum
supports. Central ±0.01 Å force differences form the full Hessian, with one
±0.02 Å sensitivity case. Raw Hessian antisymmetry, translation residuals,
symmetrization and all signed internal eigenvalues remain visible.

For mechanically stable distinct endpoints, the executed bounded trajectory plan was
0.5 fs integration, 5 ps Bussi equilibration and 20 ps initial NVE production
with two independent velocity seeds. Velocities are retained at every step;
detailed structure/force/stress frames every 2 fs. A thermostat or timestep
control is interpreted separately. A 20 ps cap is an allocation choice, not
proof of convergence. All four primary trajectories and both controls completed.
The mapping audit below distinguishes successful execution from retention of
the initial occupation basin.

The saved `stress_eV_A3` is calculator/configurational stress. The runner calls
ASE `get_stress()` with its default `include_ideal_gas=False`, verified in the
pinned ASE 3.26.0 implementation and the [official source](https://docs.ase-lib.org/_modules/ase/atoms.html#get_stress).
This is the appropriate quantity for an unchanged-geometry DFT single-point
comparison. It excludes the ionic kinetic term and is not presented as the full
finite-temperature pressure tensor; that term would have to be included once
from saved momenta for a future pressure analysis.

## Completed fixed-cell relaxation and orbit checks

All three starts reached the declared UMA force target. The full force/stress
arrays and exact POSCARs are preserved with source receipts; the handoff copies
them byte for byte rather than generating approximate look-alikes.

| Basin | Steps | Final energy (eV/cell) | Gap to basin1 (eV/cell) | Maximum force (eV/Å) | Three-member static weight at 433.15 K |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 84 | −1032.619389825 | 0 | 0.0009363 | 0.73744 |
| 2 | 80 | −1032.541724904 | 0.0776649 | 0.0009757 | 0.09206 |
| 3 | 83 | −1032.564728070 | 0.0546618 | 0.0008996 | 0.17050 |

Relaxation changes the second/third ordering from the deletion ranking.
The fresh initial energies differ slightly from inherited deletion labels
(up to 0.00151 eV/cell); the relaxed-energy comparison uses fresh final energies,
not a mixture of cached deletion and relaxed values.

All three initial masks have trivial stabilizers under the 192 retained parent
operations, so their orbit multiplicities are 192. Every pair remains eight
sites apart after symmetry minimization. At the relaxed endpoints, Hungarian and
nearest-site assignments agree without collisions or changed occupation sites.
Largest Li assignment distances are 0.175–0.244 Å, while nearest/second-site
margins are 2.354–2.465 Å. Host RMS displacements are 0.03945–0.04085 Å.
Thus the mapped occupation keys remain distinct under the declared parent
symmetry action; no endpoint merge is observed in that description. This is
not evidence that finite-temperature trajectories stay in their initial basins.
Because all three multiplicities are equal, including them changes none of the
selected-subset weight ratios.

Sources: `numerics/relaxation_mapping.csv` and
`DFT_candidates/{selected_occupancy_symmetry,relaxed_basin_symmetry}.json`.
The three-member weights are static context only. Any two-basin thermal
comparison must normalize its static and corrected weights over those same two
members, not compare a three-member baseline with a two-member correction.

## Completed finite-cell harmonic result

The independent raw-force reconstruction and the separately implemented xPT
adapter's harmonic calculation agree. All 537 projected internal modes are
positive in each of the four Hessian cases. At δ=0.01 Å, minimum ordinary
frequencies are 0.9751/0.9991/0.9847 THz for basins 1/2/3. The maximum raw
Hessian antisymmetry is 0.00416–0.00466 eV/Å² and relative Frobenius norm
0.000182–0.000202; symmetrization and translation projection are explicit.
Fresh-reset force-component spans are at most 2.375×10⁻⁶ eV/Å and energy
spans at most 4.307×10⁻⁵ eV/cell across the checked cases.

| Basin | Static ΔE from basin 1 (eV/cell) | Classical harmonic ΔF from basin 1 (eV/cell) | Quantum harmonic ΔF from basin 1 (eV/cell) | Quantum selected-three weight |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 | 0.007768 |
| 2 | +0.077665 | −0.146906 | −0.180619 | 0.981382 |
| 3 | +0.054662 | −0.000576 | −0.012475 | 0.010851 |

These are Eminimum+Fvibrational at 433.15 K, with exactly the same 537-mode
finite-cell convention. They reverse the static order to 2<3<1. This change
already occurs in the harmonic model; it is not evidence of a 3PT cage or an
anharmonic correction. The dominant basin 2 probability is conditional on exactly
these three selected representatives and their equal multiplicities, not the
full vacancy population.

A useful calculation explains the size of this change. In the classical
harmonic model, both basins have the same 537kBT thermal energy. Basin 2's
entropy exceeds basin 1's by 5.1846×10⁻⁴ eV/(cell·K), about 6.016 kB per cell.
At 433.15 K this supplies TΔS=0.22457 eV, outweighing the static 0.077665 eV
penalty. It corresponds to only about a 1.1% smaller geometric-mean frequency
across 537 modes; it does not require every mode to soften equally. These are
calculations within the harmonic model, not measured experimental entropies.

Changing basin 1 central displacement from 0.01 to 0.02 Å changes its quantum
harmonic F by +0.0003836 eV/cell and classical F by +0.0002983 eV/cell; maximum
mode-frequency change is 0.003953 THz. These are two-setting numerical sensitivity
measurements, not statistical confidence intervals or a bound on physical MLIP
error. Positive sampled curvatures support a local same-potential mechanical
stability check, not finite-temperature basin retention.

Using only the saved residual forces and positive Hessians, a local quadratic
Newton estimate gives energy decreases of 4.62–6.74 μeV/cell and largest
predicted atomic shifts of 0.000639–0.000788 Å. These are calculations in the
local quadratic model, not new relaxations or anharmonic bounds. They are
smaller than the tested displacement-size sensitivity and the fresh energy
variation; the source calculation is `numerics/hessian_audit_residual_forces.csv`.

Sources: `numerics/hessian_audit_summary.csv`, the independent raw-column audit,
and `xpt/harmonic_reference_verified/{harmonic_reference.csv,receipt.json}`. The selected
weights and energy contrasts are also supplied in
[`basin_energy_context.csv`](../../reports/thermal_pilot_20261107/DFT_candidates/basin_energy_context.csv)
and its [figure](../../reports/thermal_pilot_20261107/DFT_candidates/basin_energy_context.pdf).

## Exact-geometry teacher disagreement at the new minima

Three new MACE-MH-1/`omat_pbe` single points evaluated the exact UMA-relaxed
coordinates, with no MACE relaxation. The geometry hashes were matched directly
to the completed UMA output NPZs.

| Basin | UMA gap from its lowest member (eV/cell) | MACE gap from its lowest member (eV/cell) | MACE maximum force on the UMA geometry (eV/Å) |
| --- | ---: | ---: | ---: |
| 1 | 0 | 0.258545 | 0.30605 |
| 2 | 0.077665 | 0 | 0.25871 |
| 3 | 0.054662 | 0.114746 | 0.26181 |

UMA orders the three geometries 1→3→2; MACE orders them 2→3→1. In particular,
E2−E1 is +0.077665 eV under UMA and −0.258545 eV under MACE. This is a measured
teacher reversal, not evidence that either is wrong relative to DFT. MACE's
large force residuals also mean these are not MACE stationary points. Force
component differences between teachers have RMS 0.0647–0.0687 eV/Å. The result
makes matched DFT energies and forces at basins 1 and 2 an immediate priority
before any quantitative physical interpretation of the thermal weights.

Source: `DFT_candidates/mace_minima_comparison.csv`, the source-hashed receipt
and per-candidate exact-geometry MACE arrays. Three logical single-point E/F/
stress calls were made; no extra model survey or model relaxation is implied.

## Production-pair amendment recorded before production

The primary production pair is basins 1 and 2. This replaced the initially
prepared 1+3 pair after the completed, independently checked harmonic calculation
showed basin 2 to be the stronger finite-cell harmonic competitor. The old
preparation and the initial 1+3 short stability pilots remain preserved rather
than being discarded or counted as the primary production. The decision and
its timing are recorded in [DESIGN.md](DESIGN.md). The primary comparison uses
two independent initial-velocity seeds per retained basin and matched basin 1
thermostat/timestep controls. Selecting an informative pair does not make it a
representative sample of all vacancy basins.

## Completed trajectories and persistent Li hops

The full-rate production audit independently reconstructs kinetic energy,
temperature, COM motion, total-energy identities and saved-frame alignment.
These are implementation checks. The following 20 ps summaries use all
40,001 velocity/energy frames at 0.5 fs spacing; the geometry audit uses
10,001 detailed frames at 2 fs spacing.

| Initial basin / repeat | Mean kinetic T (K) | Standard deviation (K) | Total-energy span (eV/cell) | Parent-site history |
| --- | ---: | ---: | ---: | --- |
| 1 / 1 | 454.450 | 20.052 | 0.001887 | No assignment change observed |
| 1 / 2 | 425.058 | 18.372 | 0.001811 | Persistent Li hop after 16.702 ps |
| 2 / 1 | 435.777 | 18.793 | 0.001768 | Persistent Li hop after 16.882 ps |
| 2 / 2 | 471.548 | 20.997 | 0.001931 | No assignment change observed |

The hopping events are supported by continuous atom positions, unique final
site assignments and several picoseconds of residence, rather than a single
nearest-site label flicker. In basin 1 / repeat 2, Li local atom 65 (original
parent atom 66) leaves parent Li site 18 and reaches site 31 by 16.736 ps.
In basin 2 / repeat 1, Li local atom 59 (parent atom 59) leaves site 11 and
reaches site 23 by 16.924 ps. All indices are zero based. Both original-to-final
parent-site distances are 2.81126 Å. Final-site residence lasts 3.264 and
3.076 ps, with median nearest/second-site distance margins of 2.298 and
2.213 Å. Hungarian and nearest assignments agree without collisions.
The intervening 32 and 40 fs intervals are ambiguous transit geometries,
not additional established basins.

These are observed hops under UMA, not validated real-LCO diffusion events or
barrier measurements. Two events do not determine a diffusion coefficient.
They are also not numerical explosions: the bounded energy spans remain
small while the atoms access another occupation arrangement. Successful
integration and positive zero-temperature curvatures therefore coexist with
failure of initial-basin retention.

All warmups, both controls and the first 10 ps of all four primary trajectories
have unchanged atom-resolved Li assignments at the saved 2 fs resolution.
After inspecting the motion audit, and before inspecting primary free-energy
results, the analysis scope was amended: full 20 ps outputs are retained as
mixed-occupation spectral/event diagnostics; the common first 10 ps is an
exploratory pre-event comparison. First-half windows were already planned as
a sensitivity, but choosing this one as the reported restricted comparison
is post hoc. Absence of a detected hop does not establish equilibrium within
a basin, and repeated saved frames are not independent realizations.

Sources: `numerics/md_audit_summary.csv`, `numerics/md_audit_events.json`,
the source-bound continuous-geometry event exports and the
[full-trajectory event/spectral figure](../../reports/thermal_pilot_20261107/xpt/full20_diagnostic/full20_assignment_event_scoped.pdf).
The original full-trajectory outputs are preserved; their numerical F fields
are not reported as initial-basin free energies or physical populations.

## Temperature, sampling and selected-subset weights

An NVE trajectory need not have mean kinetic temperature exactly equal to its
equilibration target. Even an equilibrated canonical warmup samples a
distribution of energies; NVE then conserves one sampled energy. A finite cell
can therefore have a different long-run microcanonical temperature without that
difference alone proving inadequate equilibration. Instantaneous fluctuations also have a useful reference scale. For s=537
classical canonical momentum coordinates, 2K/(kBT) has a chi-square distribution,
so σ(Tinst)=T√(2/s), about 26.3 K at 430 K. The NVT production standard deviation
is 27.27 K. A harmonic microcanonical reference instead gives
σ(Tinst)≈T/√(s+1), about 18.9 K at 438 K, compared with 18.90 K in the half-step
NVE control. These scale checks help interpret instantaneous excursions; they
are not universal stability tolerances or proof of ergodic sampling.

Native xPT results use each repeat's measured temperature.
Any common-T0=433.15 K table uses the stated first-order transport
F(T0)≈F(T)−S(T)(T0−T), with the actual temperature shift and a curvature
diagnostic; this is not an exact reweighting of a different ensemble. NVE
total-energy fluctuations are not used as the canonical heat-capacity estimator.

A separate, source-frozen sensitivity transports the known harmonic reference
exactly while transporting only the residual to first order. Write
δF(T)=FxPT(T)−Fharm(T), δS(T)=SxPT(T)−Sharm(T), and ΔT=T0−T. Then

$$
F_{\rm ref}(T_0)=F_{\rm harm}(T_0)+\delta F(T)-\delta S(T)\Delta T.
$$

Its difference from direct first-order transport is the known harmonic curvature,
not a fitted correction. Actual-Hessian zero/linear-residual checks establish
that algebraic property. The anharmonic/estimator residual curvature is still
unbounded. Both transported results are reported side by side; their agreement
or disagreement does not justify selecting a favorable ordering. Native matched-T
comparisons remain essential. Source: `xpt/transport_sensitivity.py` and its
analytic validation receipt.

For selected, deduplicated basins with declared multiplicity mᵢ, static weights
are proportional to mᵢexp[−Eᵢ⁰/(kBT)] and corrected weights to
mᵢexp[−Fᵢ/(kBT)]. Both are normalized over the same selected subset. Equal
representative weights mᵢ=1 are the primary comparison until relaxed basin
equivalence is verified; exact parent-mask multiplicities are a labeled
sensitivity, not automatically physical basin degeneracies. A common energy
offset cancels, but an error in a gap changes the odds exponentially.

The modal kinetic check has a precise reference. In a classical canonical
ensemble, mass-weighted momenta have covariance kBT times the identity on the
COM-constrained internal subspace. Any orthonormal change of basis preserves
that covariance, so every such coordinate has mean kinetic energy ½kBT even
for an anharmonic potential. The zero-temperature Hessian eigenvectors supply
frequency-bin labels; their use does not make that kinetic identity approximate.
Finite-time correlations, the fixed NVE energy and incomplete mode population
sampling affect the observed group averages and their dependence. Quartile
imbalances therefore require block/seed assessment; basis inapplicability is
not an explanation by itself. A normalized 537-mode spectrum can still have an
incorrectly sampled spectral shape, because amplitudes reflect sampled kinetic
mode energies.

The measured full-production quartile/global kinetic-temperature ratios are
listed below, from lowest to highest relaxed-harmonic frequency quarter. Each
group contains 134 or 135 modes. These descriptive primary-production
projections each use 2,001 subsampled detailed frames; the frames are not
additional independent replicas.

| Initial basin / repeat | Lowest quarter | Second quarter | Third quarter | Highest quarter |
| --- | ---: | ---: | ---: | ---: |
| 1 / 1 | 0.9723 | 1.0207 | 1.0391 | 0.9682 |
| 1 / 2 | 0.9903 | 0.9924 | 1.0438 | 0.9736 |
| 2 / 1 | 0.9923 | 1.0151 | 0.9958 | 0.9969 |
| 2 / 2 | 1.0041 | 0.9736 | 0.9884 | 1.0339 |

Group temperatures also change between the first and second halves by up to
25.82 K in these primary runs. Near-unity full-window averages therefore do
not eliminate time dependence or validate spectral equilibration. The initial
Hessian still supplies an orthonormal momentum basis after a hop, but its
frequency labels are then especially limited as a description of the new
environment. Sources: `numerics/md_audit_mode_temperatures.csv`, its
[mode-group figure](../../reports/thermal_pilot_20261107/numerics/md_audit_modes.pdf)
and the phase/block tables.

Independent seeds and sufficiently separated blocks diagnose sampling variation.
Nested prefixes and alternative correlation windows are sensitivity checks,
not extra independent replicas. Mode normalization, low-frequency resolution,
frame stride, fluidicity, COM constraints, drift and basin mapping are assessed
before assigning a difference to anharmonicity. If uncertainty rivals the
between-basin gap, its ordering remains unresolved. If Li changes basin, the
trajectory is not assigned automatically to its initial mask's vibrational
entropy; occupation entropy is not added again to motion already sampled.

## Completed five-picosecond thermostat control

One basin 1 NVT control completed 5 ps after its 5 ps warmup. Its native
VAC-derived temperature is 430.6321 K, and the integrated spectrum is
536.9999949 modes. The independent motion audit found no occupation changes
or atom-resolved Li site transitions at the saved 2 fs resolution. The following
estimators use exactly this same window and energy anchor:

| Estimator | Closure fluidicity f | Quantum F at native T (eV/cell) | Quantum cage entropy (eV/(cell·K)) |
| --- | ---: | ---: | ---: |
| 1PT | 0 by construction | −1025.499068 | 0 |
| 2PT | 0.0122249 | −1026.024835 | 0 |
| 3PT | 0.0122249 | −1026.339707 | 0.000731184 |

The table's f is the solver's closure parameter. After the gas spectrum is
capped against the total spectrum, its integrated area here is 3.89413 modes,
0.0072516 of the 537 internal modes. The cage calculation instead uses
f_cage=Ngas/(3N)=0.00721136 and g=f_cage²/(f_cage²+0.01²)=0.34212.
Closure fluidicity, assigned gas area and cage gate are distinct quantities.
In this window the 3PT−2PT F change is −0.314871 eV/cell, despite
no observed Li site transitions. This is a sizeable estimator contribution to
be compared with duration, lag, 1PT and NVE controls; it is not established
anharmonic crystal entropy. The saved 0.5 fs interval meets the package's cage
resolution recommendation here (dtΩ0=0.045607 versus 0.066), which alone does
not establish thermodynamic accuracy. The optional Debye-gate comparison is
unavailable on this 5 ps frequency grid; it was not forced to produce a result.

The completed 0.25 fs NVE control is another native 5 ps diagnostic. It has
T=438.26394 K, closure f=0.0184817, assigned gas area 6.42336 modes,
f_cage=0.0118951, gate 0.58591 and a quantum cage contribution −0.289858 eV/cell.
Its sampling-resolution product is 0.0240652. Different native temperatures and
warmup paths prevent attributing its difference from the NVT case to timestep
or thermostat alone; matched primary-NVE windows provide the relevant controls.
The exact partition distinctions are saved in
`xpt/control_partition_diagnostics/partition_diagnostics.csv`.

An observation-grid check reuses the exact same 0.25 fs integration trajectory
but retains every second velocity frame, giving 0.5 fs observations over the same
5 ps. This adds no dynamics or force calls. The change F(observation 0.5 fs)−
F(observation 0.25 fs), in eV/cell, is:

| Estimator | Classical ΔF | Quantum ΔF |
| --- | ---: | ---: |
| 1PT | −0.0003281 | −0.0149500 |
| 2PT | −0.00000965 | −0.00000784 |
| 3PT | +0.0011041 | +0.0009669 |

The observed temperature changes by only −0.000831 K and the mean energy anchor
by 9.58×10⁻⁸ eV. This isolates observation-grid sensitivity within one trajectory;
it is not an integrator-error order test. Source: `xpt/half_dt_observation05/`
versus `xpt/half_dt_control_native/`, with exact retained indices and source hashes.

Sources: `xpt/nvt_control_native/native_results.csv`, its receipt and the
[native NVT spectral figure](../../reports/thermal_pilot_20261107/xpt/nvt_control_native/nvt_spectral_diagnostics.pdf).
The completed comparison uses equal 5 ps windows, 0.5 fs observations and
1.25 ps physical maximum lag. Its
[control spectra](../../reports/thermal_pilot_20261107/xpt/matched5ps_controls/matched5ps_spectra_scoped.pdf)
and `matched5ps_control_table.csv` retain native temperatures and both common-T
transports. Matching the observation window does not match native temperature
or equilibration path, so this is a sensitivity comparison rather than an
isolated thermostat/integrator-error experiment. NVT's energy exchange with its
thermostat is expected and is not an NVE integration-drift failure.

## Synthetic control exposes a finite-window artifact

The preserved 8-atom harmonic control has 21 internal modes and no physical
diffusion. Its 20 ps, 2 fs-sampled full trajectory gives zero inferred fluidicity.
A 5 ps block can instead give f≈0.01723 and shift quantum F by about −0.0438 eV,
even with the full-trajectory comparison's maximum lag matched. This is a
synthetic numerical diagnostic, not an LCO result or a transferable error bar.
It shows that a short block's apparent gas contribution and free-energy shift
need not be genuine diffusion or anharmonicity. Consequently block spread is
not automatically pure sampling variance, and it cannot be divided by a square
root of the number of blocks without checking these estimator effects.

The actual production saves velocities at 0.5 fs, rather than this older 2 fs
control. Its own correlation-window, sampling and gas-gate sensitivities are
reported separately; fine timestep resolution alone does not fix low-frequency
finite-window leakage. Source: `xpt/adapter_sensitivity_smoke/` native and
sensitivity tables, with the known harmonic input and package logs preserved.

## Exploratory first-10-ps estimator comparison

All four restricted windows contain 20,001 velocity frames at 0.5 fs spacing.
The primary maximum VAC lag is 2.5 ps. The 12 native analyses cover 1PT, 2PT
and 3PT for each of four windows; 40 further analyses test window and lag
sensitivity. The matched block comparison uses 0.625 ps maximum lag, not the
1.25 ps lag used for the separate 5 ps controls. Integrated mode counts differ
from 537 by at most 1.09×10⁻⁶; the quantum thermodynamic-identity residual is
at most 1.53×10⁻¹³ eV/cell. These checks verify normalization and algebra,
not the accuracy of the sampled spectral shape or its thermodynamics.

The table reports the observed minimum and maximum of
ΔF=F(basin 2)−F(basin 1), after transport to 433.15 K. Each range uses every
combination of the two basin-1 and two basin-2 initial-velocity repeats. Those
four differences reuse four trajectories; they are neither four independent
replicates nor a confidence interval. Units are eV per cell.

| Quantum spectral estimator | Direct first-order transport: observed ΔF range | Harmonic-reference residual transport: observed ΔF range |
| --- | ---: | ---: |
| 1PT | [−0.34971, −0.09381] | [−0.33014, −0.15496] |
| 2PT | [−0.45740, −0.08072] | [−0.43783, −0.14188] |
| 3PT | [−0.66657, +0.08799] | [−0.64700, +0.02684] |

Negative ΔF favors basin 2 within that estimator; positive ΔF favors basin 1.
The 3PT sign changes across observed repeat combinations under both transports.
Its classical counterpart also crosses zero: [−0.63574,+0.12125] for direct
transport and [−0.61278,+0.05034] for harmonic-reference residual transport.
Although the observed 1PT and 2PT ranges remain negative, these short,
post-selected windows do not validate an equilibrium ordering. In particular,
a negative average of the 3PT differences cannot remove the observed sign
ambiguity. No corrected population table is inferred from those averages.

The
[scoped range figure](../../reports/thermal_pilot_20261107/xpt/prefix10_analysis/prefix10_transport_ranges_scoped.pdf)
shows both transports without selecting a favorable one. Sources are
`xpt/prefix10_analysis/{native_results.csv,sensitivity_results.csv,receipt.json}`,
`xpt/prefix10_transport_sensitivity/` and
`xpt/prefix10_cross_seed_ranges/{observed_unpaired_ranges.csv,correlated_cross_combinations.csv}`.
Exact sensitivity filenames and the plotting inputs are bound in
`xpt/scoped_figures_receipt.json`.

The preserved short-block sensitivity is particularly unfavorable. On the two
repeat-1 windows, changing the maximum lag from 2.5 to 0.625 ps shifts native
quantum 3PT F by −0.0178 and −0.1058 eV/cell. At that same 0.625 ps lag,
5 ps blocks differ from their respective full 10 ps reference by
−1.2141 to −0.0489 eV/cell; 2.5 ps blocks differ by −5.4640 to −0.1949 eV/cell.
Within each basin, the block mean temperatures vary by less than 1 K, and no
negative spectral area is reported. These are observed failures of stability
with shorter records even after matching lag, not evidence for enormous
physical anharmonic corrections. Shorter records alter the finite-time VAC
estimate and inferred gas/cage partition. The blocks share trajectories and
their estimator biases differ; treating their spread as independent noise and
dividing by a square root would manufacture an unjustified error bar.
Source: `xpt/prefix10_analysis/sensitivity_shifts.csv`; all native outputs and
warnings are retained alongside it.

The numerical lesson is narrower than an anharmonic correction: the harmonic
reference predicts a meaningful reversal, while this bounded trajectory-based
comparison does not establish a robust 3PT correction to that ordering.
Longer unrestricted trajectories would not restore the original basin identity
after a genuine hop. Additional production, artificial restraints or frozen-Li
simulations would change the scientific problem and are not used to rescue
the result.

## Accuracy stop

The single atomic-group fluidicity does not by itself validate multicomponent
Li-diffusion thermodynamics when Li moves while the Co/O framework remains
solid. Atom-resolved site histories and the species/mode diagnostics must
accompany that interpretation; no separate occupation entropy is added over
motion already represented in the trajectory.

No trusted same-potential thermodynamic-integration reference has been verified
for these basins. Classical integration, if later available, would test the
classical free energy. It would not independently validate the approximate
quantum spectral correction: the nuclear trajectories here are classical,
not quantum path-integral samples.

A same-potential harmonic/xPT discrepancy is an estimator diagnostic. Without
a trusted same-potential free-energy reference, it does not establish accurate
anharmonic corrections. Without matched new DFT at the 48-Co structures, neither
agreement nor discrepancy establishes physical LCO accuracy. The proposed DFT
single points would test sampled static energies, forces, stresses and selected
curvature columns; they would not establish a full DFT free energy. The completed
handoff contains 13 distinct exact geometries, including the original typical
and high-force probes, the observed first transit and a clear post-hop frame.
Its validation checks declared file hashes, species count and POSCAR round trip;
the maximum round-trip displacement is 3.77×10⁻¹⁵ Å. This is serialization
accuracy, not chemical or physical validation. The next physical-reference work
stops at that [prepared handoff](DFT_REQUIRED_NEXT.md); no DFT is submitted or
scheduled.

## Reproducible evidence and retrieval

The small tables, scoped figures, exact DFT candidate structures and their
source-hash receipts are kept with this repository. Native trajectories,
displaced-force columns and ATLAS outputs remain outside Git. The
[raw retrieval manifest](../../reports/thermal_pilot_20261107/raw_manifest.json)
identifies the exact HPC archive and every payload hash. Its separate
[archive verification](../../reports/thermal_pilot_20261107/raw_archive_verification.json)
passed for all 3,371 files, with 4,303,680,575 payload bytes. The uncompressed
PAX archive contains 4,310,732,800 bytes and has SHA256
`142e9d9807635250ba9580c687ab37218757ed087c62a18532d29f7fe1b67d7c`.
This is an integrity check, not a physical-accuracy test.

Executed interfaces and preservation conventions are in
[COMMANDS.md](../../reports/thermal_pilot_20261107/COMMANDS.md),
[the numerical protocol](../../reports/thermal_pilot_20261107/numerics/PROTOCOL.md)
and [ATLAS_CONVENTIONS.md](ATLAS_CONVENTIONS.md). The
[environment package record](../../reports/thermal_pilot_20261107/environment_packages.json)
and source snapshots distinguish the actual run from a later rerun with
different installed code. Foundation model weights and proprietary PAW content
are not redistributed. The raw archive is a separate retrieval requirement;
downloading the class training bundle alone does not download these trajectories.

The final [resource receipt](../../reports/thermal_pilot_20261107/cost_receipt.json)
and [scheduler accounting](../../reports/thermal_pilot_20261107/allocation_accounting.tsv)
record the completed allocation, including startup, idle time, failed attempts,
CPU analysis and archival. Phase 2 used eight allocated GPUs for 8,349 seconds:
18.5533 GPU-device hours, or 4.6383 hours of a four-GPU node. Across both phases,
the total is 30.9544 GPU-device hours, equivalent to 7.7386 four-GPU-node hours.
Actual physical node-hours are 8.2744 because the shared allocation used part
of a node. The measured peak was 18 concurrent GPUs. All owned allocations
are completed and released. These are allocated-resource measures, not neural
backbone forward-call counts or inferred monetary charges. No new DFT job or
MLIP fine-tuning campaign was run.

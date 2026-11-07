# ATLAS conventions for the bounded same-MLIP thermal pilot

This analysis tests numerical consistency under one MLIP. It does not establish
real-LCO free energies, calibrated equilibrium populations, or anharmonic
accuracy. The class midterm and its frozen Phase 1 measurements remain separate.

## Source, environment, and actual implementation

The implementation is [ATLAS py-xPT](https://github.com/atlas-nano/codes/tree/197bdcf5f57c9d04b5527f9e7d6f4fdb8bf3b82d/py-xPT),
version **2.1.0**, pinned to Git commit
`197bdcf5f57c9d04b5527f9e7d6f4fdb8bf3b82d` on November 7, 2026. The ignored checkout
is `runs/thermal_pilot_20261107/vendor/atlas-codes`. Source hashes and the imported
package path are checked at runtime; modified or different source is rejected.

This optional research environment is separate from the class's required core
and MLIP setup. On a fresh checkout, use a compatible Python (3.11 was tested),
fresh environment/output directories, and ordinary dependency installation:

```bash
git clone --filter=blob:none --sparse https://github.com/atlas-nano/codes.git \
  runs/atlas_vendor/codes
git -C runs/atlas_vendor/codes sparse-checkout set py-xPT
git -C runs/atlas_vendor/codes checkout --detach \
  197bdcf5f57c9d04b5527f9e7d6f4fdb8bf3b82d

python3 -m venv .venvs/xpt
.venvs/xpt/bin/python -m pip install -e runs/atlas_vendor/codes/py-xPT \
  matplotlib ase
.venvs/xpt/bin/python -m pip check
.venvs/xpt/bin/python -m pip freeze > runs/atlas_vendor/xpt_packages.txt

.venvs/xpt/bin/python scripts/overnight_20261107/thermal_xpt.py selftest \
  --vendor runs/atlas_vendor/codes --output-dir runs/new_xpt_selftest
```

The editable installation brings py-xPT's declared NumPy, SciPy and MDAnalysis
dependencies; Matplotlib supports the wrapper's figures and ASE supports the
trajectory/unit workflow. No dependency bypass is used. The dedicated successful
installation and self-tests below used these same steps with scratch paths;
this portable block was not a second installation experiment. On a cluster,
perform installation and analysis within the permitted compute allocation when
site rules require it. No account, scheduler, hostname or scratch location is a
portable package default, and ATLAS is not added to required class dependencies.

The dedicated environment is on native scratch:
`/pscratch/sd/r/rsb/ece285_thermal_20261107/venvs/xpt`.
It installs the actual package with all ordinary dependencies, including NumPy,
SciPy and MDAnalysis. The installation log and `xpt_packages.txt` are beside the
vendor checkout. `pip check` passed. This environment installs no MLIP backend.
Version 2.0 introduced quantum-consistent cage weights; 2.1 changed the Volterra
kernel integration to trapezoidal product weights. Older results cannot be
silently described as results from this corrected version.

The adapter in `scripts/overnight_20261107/thermal_xpt.py` constructs the package's
own `System`, `AtomInfo`, `MolInfo`, `GroupInfo`, and `FrameData` objects, then calls
`xPTEngine._setup`, `accumulate`, `compute`, and `write`. Its FFT VACF, spectrum,
fluidicity, gas/solid partition, cage and thermodynamic integration are ATLAS
code. The adapter handles NPZ I/O, units, provenance, comparison and diagnostics;
it does not implement a substitute 3PT estimator or modify the vendor source.

## Atomic analysis, mass weighting, and translations

Use one atomic group containing all present Li, Co and O. The lattice is not
partitioned into fictitious LiCoO2 molecules. For velocities in Å/ps and atomic
masses in amu, the scalar VACF is the trace of the mass-weighted tensor,

\[
C(t)=\sum_{i=1}^{N}m_i\left\langle\mathbf v_i(t_0+t)\cdot
\mathbf v_i(t_0)\right\rangle_{t_0}.
\]

The average is over valid time origins. Mass weighting gives each thermally
excited Cartesian degree of freedom the same mean kinetic contribution despite
different Li/Co/O masses. The DOS is normalized with the VAC-derived kinetic
temperature. Atomic masses come from the MD arrays, never a default unit mass.
The single group represents all internal modes. If Li diffuses while the Co/O
framework remains solid, its hard-sphere closure still uses the group's mean
mass and number density; that is not validated multicomponent LCO thermodynamics.
The bounded comparison therefore concerns nondiffusive basins, or explicitly
reports hopping as a limitation. No species sweep repairs that assumption by
itself, and the reported group chemical potential is not an insertion potential.

MD has zero total linear momentum. The harmonic comparison projects out the
three mass-weighted translations and retains **3N−3** internal modes: 537 when
N=180. ATLAS itself removes three translations for a periodic system in
`engine._compute_postvac`, after subtracting user constraints in `_compute_dof`.
Therefore this adapter uses **`constraints=0`**, because these are *additional*
constraints. Setting it to three would incorrectly leave 534 modes. The MD
metadata still correctly records three physical COM constraints.

`subtract_com_velocity=True` removes any residual per-frame drift before the
VACF. Repeating this subtraction on already COM-fixed velocities does not remove
another three physical modes. The wrapper checks the package's analytic DOF
count and records the integrated DOS, its difference from 3N−3, and negative
spectral area. It does not clip negative DOS to make a normalization check pass.
The finite-Γ Hessian and periodic MD cell share the same modes; this is not a
primitive-cell Brillouin-zone integration or a thermodynamic-limit result.
These are COM-constrained internal free energies. The same omitted translational
partition factors cancel between the present equal-composition, equal-cell
basins. They would need renewed accounting for absolute unconstrained or
cross-composition thermodynamics.

## Time and velocity units

ASE velocities are not numerically in Å/fs. Convert using
`v_A_ps = v_ASE * (1000 * ase.units.fs)`, approximately
`98.22695 * v_ASE`. The wrapper requires explicit `velocities_A_ps` and verifies
their kinetic energy against the producer's recorded kinetic energy. The
independent identity is

\[
K\ [\mathrm{eV}]=\tfrac12(1.0364269656262\times10^{-4})
\sum_i m_i\ [\mathrm{amu}]\,|\mathbf v_i\ [\mathrm{Å/ps}]|^2.
\]

The production observer saves full-rate velocities every 0.5 fs, together with
their timestamps, kinetic energy, cached potential energy, total energy and
kinetic temperature. Detailed positions, forces, stress and occupation mapping
remain on their separate 2-fs axis. Full-rate potential energies are read from
the current calculator cache; recording them does not trigger an extra force
evaluation. Use `velocity_time_fs` for the VACF and exactly matching
`velocity_energy_total_eV` for its energy anchor.
The half-timestep control integrates and saves velocities every 0.25 fs. Its
native analysis is retained, and a separate matched-observation analysis keeps
exactly every second velocity frame, yielding 0.5-fs observations over the same
five-ps window and 1.25-ps physical lag. The original trajectory and integration
metadata stay unchanged; retained indices and the corresponding energy mean are
saved. This isolates observation-grid sensitivity within one trajectory. A
comparison across separately equilibrated timestep/thermostat runs still has
temperature and warmup-path differences and does not establish a causal error
order.

Atomic thermodynamics does not consume positions. Its required `FrameData`
position field is filled by the first detailed geometry and explicitly marked
unused; no interpolated full-rate geometries are claimed. Basin survival and Li
hopping are assessed separately from the actual detailed trajectory.

The [pinned ATLAS README](https://github.com/atlas-nano/codes/blob/197bdcf5f57c9d04b5527f9e7d6f4fdb8bf3b82d/py-xPT/README.md)
recommends saved interval × Einstein angular frequency ≤0.066 for resolved cage
inversion. This is an implementation recommendation, not a universal convergence
test. The wrapper computes this dimensionless product from the second DOS
moment, records whether the recommendation is met, and retains coarser-frame
and correlation-window sensitivity. A high Nyquist frequency alone does not
establish accurate memory-kernel inversion.

## Native temperatures and energy anchors

ATLAS uses the observed VAC-derived kinetic temperature in the harmonic and
hard-sphere weights, even when `temperature=433.15` is supplied. We retain that
temperature for every full trajectory and block and evaluate the harmonic
reference at exactly the same temperature. A nominal thermostat target is not
silently substituted for the observed NVE temperature.
Every result also records the full-rate producer temperature, an independent
COM-cleaned `2<K>/(537 kB)`, their ratios/residuals against native VAC temperature,
and the exact native `VAC(0)/(0.1 R × 537)` identity. Small differences from the
modern-constant kinetic temperature are retained rather than silently corrected.
These checks distinguish constant/precision differences from an erroneous
540-versus-537 normalization.

For quantum thermodynamics the package's energy anchor is

\[
U_q=\langle K+U\rangle_{\mathrm{MD}}+
        (U_{q,\mathrm{DOS}}-U_{c,\mathrm{DOS}}),\qquad
F_q=U_q-T S_q.
\]

The stored MD average belongs to exactly the analyzed frames. The minimum
energy must **not** be added again. This is the convention in
[Ramji et al., §2.4, Eq. 10](https://arxiv.org/html/2610.37586v1#S2.SS4).
Native results and their `F=U−TS` residual are saved in eV per simulation cell.

The pinned source applies this MD anchor to quantum fields only. Its native
`A_classical` and `E_classical` remain DOS-model quantities. We preserve those
fields and additionally report

\[
F_c^{\mathrm{MD\ anchor}}=\langle K+U\rangle_{\mathrm{MD}}-T S_c.
\]

Comparing the unanchored native classical field with `E_min + F_vib` would mix
energy references. The adapter converts its eV anchor using the inverse of the
package's own output conversion so that the round trip is exact; its receipt
records the residual. No NVE total-energy variance is used as a canonical heat
capacity. Package heat-capacity output is not presented as measured canonical Cv.

For a common reporting temperature T0=433.15 K, values are labeled first order:

\[
F(T_0)\simeq F(T_{\rm obs})-S(T_{\rm obs})(T_0-T_{\rm obs}).
\]

This follows from `(∂F/∂T)_V=−S`; it is not exact canonical reweighting of the NVE
trajectory. The output reports ΔT and the harmonic curvature scale
`Cv (ΔT)^2/(2 T_obs)` for both classical and quantum weights. That scale is not a
validated bound on the anharmonic estimator. Repeat/block spread and numerical
sensitivity may be larger than the proposed basin free-energy differences.

When different NVE repeats have appreciably different temperatures, the separate
`reports/thermal_pilot_20261107/xpt/transport_sensitivity.py` also retains the
known harmonic temperature dependence exactly. Define the same-treatment
residual `R(T)=F_xPT(T)−F_harm(T)` and `ΔT=T0−T`. The second transport is

\[
F_{\mathrm{ref}}(T_0)=F_{\mathrm{harm}}(T_0)+R(T)
 -[S_{\mathrm{xPT}}(T)-S_{\mathrm{harm}}(T)]\Delta T.
\]

It transports only the residual to first order. Its difference from the direct
first-order transport is exactly
`F_harm(T0)−F_harm(T)+S_harm(T)ΔT`, the known harmonic curvature contribution.
In the harmonic limit (`R=0` and `S_xPT=S_harm`) it gives the exact target harmonic
free energy. Tests at 390 and 470 K use all three actual positive Hessians and
also a clearly synthetic linear residual to check signs. Their `1e−10 eV`
algebra tolerance addresses floating-point roundoff, not physical accuracy.
Both transports are reported as sensitivity choices. Unknown curvature of the
xPT-minus-harmonic residual is not measured or bounded; this correction is not
a reason to select a favorable ordering or claim exact common-temperature
thermodynamics. The frozen xPT wrapper remains unchanged and hash checked.
Both transports use the reported xPT entropy as a local thermodynamic slope;
this dataset does not independently establish that derivative consistency for
the approximate estimator.

## Crystal limit, estimator comparisons and limitations

Run 1PT (all solid), rigorous 2PT, and 3PT on the same atomic velocities. In the
nondiffusive limit the explicit cage is suppressed by
`g(f)=f²/(f²+0.01²)`, so 3PT approaches the solid baseline. The cage therefore
does not supply an arbitrary correction for every anharmonic crystal. This limit
is part of [Buarque et al., Eq. 8 and Supporting Information S6](https://arxiv.org/html/2607.29026v2).
The default `gas_gate=debye_warn` only records the optional low-frequency
crystalline-gas diagnostic; it does not alter the result. An explicitly labeled
`debye` sensitivity run measures that optional gate's effect.
The reported solver fluidicity is not necessarily the integral of the assigned
gas spectrum divided by `3N`: the partition caps gas DOS by the observed total
DOS. The cage function specifically computes its gate input from that capped
gas integral, divided by `3N` after per-atom normalization. Preserve the solver
value and the integrated assigned-gas value separately. Neither measures a
fraction of Li atoms observed to diffuse.

For 3PT, the source explicitly uses `cage_ws=consistent`,
`cage_quadrature=trapezoid`, `cage_nf_run=3`, and no kernel taper. Its guards can
set an unphysical cage correction to zero; logs and warnings are retained. The
vendor also replaces some non-finite thermodynamic intermediates with zero.
A completed call is consequently not, by itself, numerical validation.

The harmonic comparison sums exactly the positive internal frequencies from the
finite-cell Hessian. Imaginary or zero internal modes block that reference;
they are never abs-valued or silently removed. Classical harmonic weights use
the same h-dependent phase-space convention as the high-temperature xPT weights.
Classical and quantum calculations are compared separately. Their disagreement
with xPT can reflect finite sampling, finite-frequency resolution, finite-T
spectral shifts, imperfect estimator assumptions or basin changes. Without
same-potential thermodynamic integration it cannot be called accurate
anharmonic free energy. Agreement between MLIPs also does not validate the
physical DFT energy surface.
The quantum output reweights spectra from a classical trajectory with harmonic
quantum weights. It is an approximate quantum correction. Even a successful
classical thermodynamic-integration comparison would not independently validate
that quantum approximation.

## Commands and outputs

Inside the authorized compute allocation, with CPU threads bounded to four:

```bash
XPT_PY=/pscratch/sd/r/rsb/ece285_thermal_20261107/venvs/xpt/bin/python
"$XPT_PY" scripts/overnight_20261107/thermal_xpt.py selftest \
  --vendor runs/thermal_pilot_20261107/vendor/atlas-codes \
  --output-dir runs/new_synthetic_xpt_check

"$XPT_PY" scripts/overnight_20261107/thermal_xpt.py analyze \
  --vendor runs/thermal_pilot_20261107/vendor/atlas-codes \
  --trajectory basin_1:repeat1=PATH/production.npz \
               basin_2:repeat1=PATH/production.npz \
  --hessian basin_1=PATH/hessian.npz basin_2=PATH/hessian.npz \
  --minimum basin_1=PATH/relaxed.npz basin_2=PATH/relaxed.npz \
  --target-temperature-K 433.15 --sensitivity \
  --raw-output-dir SCRATCH/new_xpt_native \
  --output-dir reports/thermal_pilot_20261107/xpt/production_analysis
```

The synthetic test uses known harmonic velocities, not LCO MD. It checks actual
package execution, COM/mode counting, units, quantum energy anchoring, the
nondiffusive cage limit and rejection of imaginary harmonic modes. The first
2-fs check is preserved; the finer 0.5-fs check tests the actual planned sampling.

Production output includes `native_results.csv`, `sensitivity_results.csv`,
`basin_differences.csv`, `repeat_summary.csv`, `difference_summary.csv`,
`sensitivity_shifts.csv`, diagnostic figures and a source-hashed `receipt.json`. Each analysis retains
native `.thermo`, `.pwr`, `.vac`, `.3n`, logs, configuration and unrounded arrays.
The primary lag fraction is 0.25 (five ps for a twenty-ps trajectory). Block
comparisons hold the maximum lag at 1/16 of the full trajectory length (1.25 ps
in the twenty-ps pilot) across the full matched-lag control and both ten-ps and
five-ps blocks. Their shifts from that control are saved separately from shifts
relative to the five-ps primary lag. This separates changing observation windows
from changing correlation truncation. Additional stride, lag, Hann-window and
optional crystal-gas-gate runs are explicitly numerical sensitivity tests.
Both 1PT and 3PT are retained for blocks: finite-window low-frequency leakage can
otherwise be mistaken for gas-like motion, even for synthetic harmonic input.
Blocks and sampling/lag variants are correlated diagnostics. Independent repeat
spread is descriptive; two repeats do not establish a precise confidence
interval. No reported chemical-potential field is interpreted as Li insertion
chemical potential, and no occupation entropy is added over hopping already
present in the trajectory.

# Bounded same-potential thermal pilot

This extension starts only after the Git-tracked Phase1 completion gate. It is
separate from the course report. No new DFT or MLIP fine-tuning is authorized.

## Question and controls

Can finite-periodic-cell harmonic phonons and the actual ATLAS py-xPT estimator
be evaluated consistently, at measurable cost, for two competing Li/vacancy
basins under one pinned potential? A successful implementation establishes a
same-potential diagnostic. It does not establish real-LCO free energies or
accurate anharmonic corrections without an independent reference method.

Existing matched DFT/MLIP data motivate the choice of an inexpensive teacher but
also show rank reversals. There is no exact main-parent Li36Co48O96 DFT reference.
Previously unstable delithiated dynamics and uncertain low-frequency spectra are
contrary evidence that the new stability and numerical checks must address.

The selected parent, teacher, head, precision and cell are frozen: the Phase1
48-Li parent, UMA-S-1.2.1/omat, FP32 with TF32 off, unchanged parent cell. Selection
uses existing **training** deletion labels, not new CVAE discoveries. Three
nLi=36 masks are chosen sequentially by lowest UMA deletion energy within the
first 1,000 rows, requiring at least six sites of symmetry-minimized Hamming
separation. This three-swap separation is an illustrative diversity heuristic,
not a thermodynamic criterion. Exact masks, source rows, hashes and initial
structures are in the selection receipt. Their deletion-energy gaps are about
0, 0.081 and 0.128 eV per cell; those are not relaxed-basin energy differences.

## Prespecified bounded sequence

1. Relax all three arrangements at the same fixed cell. A provisional maximum
   atomic force of 0.001 eV/Å, LBFGS maximum displacement 0.1 Å and 1,500-step cap
   define the initial numerical experiment. The force target is not a universal
   phonon standard: finite-difference sensitivity and residual forces determine
   what the resulting spectrum supports. Preserve unconverged endpoints.
2. Map Li to parent sites with periodic rectangular Hungarian assignment;
   inspect distances, competing-site margins and host displacements. Compare
   final basins before treating starting masks as distinct thermodynamic states.
3. For stable distinct endpoints, calculate the complete finite-cell Γ Hessian
   from central force differences at 0.01 Å. Repeat one case at 0.02 Å to assess
   displacement sensitivity; inspect repeated-force precision, antisymmetry,
   acoustic residuals and every signed internal eigenvalue. Symmetrization is
   reported explicitly. Only the three uniform mass-weighted translations are
   projected out, matching the COM-constrained MD's 3×180−3=537 modes. Meaningful
   instabilities are not replaced by absolute frequencies or silently discarded.
4. Begin with a short stability/throughput trajectory. If stable, use two
   independent initial velocity seeds in each of two retained basins, with a
   0.5 fs integration step, 5 ps Bussi equilibration and an initial 20 ps NVE
   production cap. Two prespecified basin1 controls use the repeat1 initial
   velocity seed: 5 ps NVT production at the same timestep, and 5 ps NVE
   production at 0.25 fs. Each has its own 5 ps equilibration. Compare the NVT
   control with matched-length NVE windows; compare timestep drift and spectra
   without treating different time-discretized paths as identical samples.
   Duration is a budget choice, not a convergence claim.
5. Save velocities every integration step and detailed thermodynamic/structural
   frames every 2 fs. This separates inexpensive velocity sampling from force,
   stress and mapping work. Inspect ATLAS's actual dt×Einstein-frequency
   resolution criterion, not just Nyquist frequency. Retain failed trajectories
   and their triggering configurations.
6. Analyze actual ATLAS 1PT/2PT/3PT outputs with its pinned code and units. Check
   spectral mode count, fluidicity, COM handling, window/stride sensitivity and
   independent blocks/repeats. Match harmonic and xPT finite-cell modes and
   measured temperature. Never add a minimum energy twice to xPT's MD anchor.
7. Report free-energy **differences**, not just large absolute energies. Native
   xPT at each repeat's measured VAC temperature is primary. Any common-433.15 K
   table uses explicitly first-order transport F(T0)≈F(T)−S(T)(T0−T), with ΔT,
   an indicative harmonic curvature scale, and limits of that approximation.
   Recompute selected-subset weights from the corrected differences and the
   reviewed multiplicity measure; do not present them as the complete population.

## Decisions, uncertainty and stopping

Before production MD, the two independent harmonic calculations agreed that
basin2 reverses its static ordering relative to basin1: ΔE21=+0.077665 eV/cell
but ΔF21(harmonic, quantum, 433.15 K)=−0.180619 eV/cell. Basin3's corresponding
ΔF31 is only −0.012475 eV/cell. The primary MD pair was therefore changed from
the two closest **static** minima (1 and3) to the static minimum1 and harmonic
minimum2. This is a more discriminating same-potential method comparison. No
production-MD outcome informed the choice, and the old prepared-only task JSON
and stability runs are retained. A separate 0.75-ps stability pilot checks basin2
before its production run. Both pairs remain within the original three selected
arrangements, one composition, one cell and one temperature.

Negative internal eigenvalues, ambiguous site mappings, basin changes, drift,
thermostat/sampling dependence or block spread can prevent a vibrational-basin
interpretation. Report the actual diagnostic and preserve the evidence. Do not
freeze Li, add restraints, or merge distinct trajectories to force convergence.
If estimator uncertainty is comparable to a between-basin difference, report
that the ordering is unresolved. Without a trusted same-potential integration
reference, estimator disagreement does not measure anharmonic accuracy or MLIP
physical error. In the nondiffusive crystalline limit the explicit 3PT cage
correction vanishes; it is not an automatic general anharmonic correction.

The initial reservation is two four-GPU nodes for at most 3.5 hours, at most
28 allocated GPU-device-hours (7 four-GPU-equivalent hours), including startup,
failures and idle time. Root alone submits GPU work. The overall new budget is
250 GPU-node-hours with at most 36 GPUs; it is a ceiling, not a target. All GPU
work must end by 15:00:40 UTC on November 7. Further DFT accuracy questions stop
at `DFT_REQUIRED_NEXT.md`; no DFT job is created or scheduled.

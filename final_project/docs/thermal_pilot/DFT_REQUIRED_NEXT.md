# DFT reference handoff after the thermal pilot

No new DFT calculation is run, submitted, or scheduled in this work. The precise
physical-accuracy stop is the absence of matched DFT references for the
Li36Co48O96 fixed-cell minima, their local curvatures, and their thermal force
environments. A completed same-UMA calculation can test numerical consistency;
it cannot establish real-LCO free energies.

The source tables and protocol audit are in
[`DFT_candidates/`](../../reports/thermal_pilot_20261107/DFT_candidates/).
Thirteen exact candidates have been copied from completed, preserved pilot
outputs and a fully scored deletion pool. Thermal probes retain actual production
coordinates. The numerical validation confirms distinct exact geometries,
composition, hashes and POSCAR serialization; it does not certify equilibrium
sampling or physical accuracy.

## Existing references that can already be reused

The matched archive contains no 48-Co cell. At x=0.75 it contains ten historical
O3 Li18Co24O48 rattled geometries, ten Hu O3 Li9Co12O24 optimized geometries,
and 600 Hu O3 Li9Co12O24 DPGEN geometries. They provide 620 distinct exact
geometries, each with archived DFT and three MLIP predictions. Their source
protocols and geometry cohorts remain separate.

The older FP64 audit's median force-component RMSE is 0.105 eV/Å for UMA on
the historical ten-frame pool and 0.142 eV/Å on Hu's optimized ten-frame pool.
UMA's optimized-pool rank correlation is 0.9605 and selection regret is zero
within those ten members, but good ordering does not remove the force errors
or certify new-cell phonons. No smaller-cell energy is substituted for a
48-Co candidate's energy.

## Protocol facts checked against actual files

The ten historical x=0.75 cases have readable INCAR, KPOINTS, OUTCAR and
vasprun.xml files. The audit verified the following actual setup:

| Item | Verified historical value |
| --- | --- |
| Code | VASP 6.4.3, build June 3, 2025 |
| PAW metadata | PBE Li_sv, September 10, 2004; Co, August 2, 2007; O, April 8, 2002 |
| Plane-wave / electronic controls | ENCUT=680 eV, PREC=Accurate, EDIFF=1e-6 eV, NELM=300, LREAL=False |
| Smearing | ISMEAR=0, SIGMA=0.05 eV |
| Spin initialization | ISPIN=2; 0 on Li/O and 0.4 μB per Co initially |
| Hubbard correction | LDAU=True, LDAUTYPE=2, LDAUL=0/2/0, LDAUU=0/3.32/0 eV, LDAUJ=0/0/0 |
| Additional settings | ISYM=0, LASPH=True, LMAXMIX=4, NSW=0 |
| Historical k mesh | Gamma 5×5×1 in the 90-atom cell |
| Archived energy | Matches the final XML `e_0_energy` in every one of the ten cases |
| Electronic algorithm | Two INCAR variants differ only in ALGO=All versus Normal |

All ten OUTCARs contain the EDIFF stopping message, and their final electronic
steps range from 160 to 299. This checks the available electronic stopping
record; it does not prove a lowest-energy magnetic solution. Alternate Co spin
initializations and resulting metastable spin states have not been certified.
The archive's extrapolated energy, smeared free energy, forces and stresses
should all be retained in a new reference record rather than silently mixing
the two energy conventions.

The new parent doubles the historical second cell vector. Copying 5×5×1 into
it does not preserve reciprocal-space spacing. A single k-mesh choice must be
reviewed and converged for the desired *between-candidate energy differences*
and force/curvature observables. The archived 1e-6 eV electronic stopping value
does not itself establish that accuracy.

Hu's supplied `param_run.json` names `fp_style=vasp` and a missing `INCAR`.
The original frame-level electronic/spin inputs are unavailable in that archive.
Its Hubbard, magnetic and exporter protocol therefore remains unresolved here;
it is not silently combined with the verified historical setup. Source paths,
file hashes and nonproprietary parameter text are in `existing_dft_protocols.json`.
Only PAW titles/hashes are recorded; no POTCAR content is distributed.

The saved MLIP stresses are calculator/configurational stresses, matching the
ionic-static DFT comparison at a fixed geometry. They exclude the ionic kinetic
term and are not a finite-temperature pressure tensor.

The new candidate POSCARs retain **Co/Li/O** order, whereas the historical
inputs use **Li/Co/O**. Species-indexed Hubbard arrays and the concatenated PAW
order must follow each actual POSCAR. If the legacy Hamiltonian is retained,
Co/Li/O implies LDAUL=2/0/0 and LDAUU=3.32/0/0 eV; copying the old arrays would
put the correction on the wrong element. The initial moment list must likewise
follow 48 Co, 36 Li and 96 O atoms. This bookkeeping does not resolve which
magnetic state is physically appropriate.

## Exact candidates already prepared

`candidate_manifest.json` records the exact structures, their hashes,
existing MLIP labels, source receipts and required observables. The initial
static subset is also preserved as `static_candidate_manifest.json`. No DFT input deck
or job is attached.

| Candidate | Priority / geometry | Purpose |
| --- | --- | --- |
| DFT_001 | 0 / basin1 UMA fixed-cell minimum | Lowest of the three tested local minima |
| DFT_002 | 0 / basin2 UMA fixed-cell minimum | Competing arrangement; relaxation changes its order relative to basin3 |
| DFT_003 | 0 / basin3 UMA fixed-cell minimum | Closest static competitor to basin1; gap 0.054662 eV/cell |
| DFT_004 | 2 / `mask48_000304344d88` deletion | Most negative centered MACE−UMA residual in 13,721 scored n=36 candidates |
| DFT_005 | 2 / `mask48_00081c921690` deletion | Most positive centered residual in the same complete scored pool |
| DFT_006/007 | 1 / basin 1 first-Li x±0.01 Å | Already evaluated matched finite-difference force-column spot check |
| DFT_008/009 | 1 / basin 2 first-O x±0.01 Å | Already evaluated oxygen force-column spot check |
| DFT_010 | 1 / basin 1 repeat 1, 19.040 ps | Closest to median potential energy among admissible detailed frames at t≥2 ps |
| DFT_011 | 1 / basin 2 repeat 1, 8.726 ps | Largest atomic force among the same admissible regime; original frozen-rule probe retained |
| DFT_012 | 1 / basin 2 repeat 1, 16.882 ps | First changed Li assignment during a confirmed later-persistent neighbor hop; transit geometry |
| DFT_013 | 2 / basin 2 repeat 1, 17.024 ps | Clear post-hop geometry 0.100 ps after the final site assignment becomes persistent |

The three exact minima now have new MACE single points as well. UMA ranks them
1→3→2, while MACE ranks them 2→3→1. E2−E1 is +0.077665 eV/cell under UMA and
−0.258545 eV/cell under MACE; MACE maximum forces are 0.259–0.306 eV/Å at the
UMA stationary points. Basins 1 and 2 therefore already span a strong teacher
ordering/force disagreement. This is not a DFT error measurement. The two
unrelaxed deletion probes remain lower-priority optional gap checks.

The disagreement selection subtracts the pool's median MACE−UMA difference,
7.208856 eV/cell, from each model difference. The two selected residuals are
−0.336990 and +0.521074 eV/cell. Both teachers favor DFT_005 over DFT_004,
but UMA predicts a −0.896028 eV gap and MACE a −0.037964 eV gap. Thus these
are a gap-distortion test, not an invented rank reversal. The pool contains
proposals from several samplers and is not a density-of-states reference.
These two structures are unrelaxed deletions; large forces there are expected.

The thermal selection policy was frozen before completed production outputs
were inspected. It requires complete 20 ps trajectories and, for DFT_010/011,
unchanged and unambiguous parent-site assignments. The t≥2 ps restriction is
a provisional selection exclusion, not a claim of equilibration. DFT_010 and
011 have maximum atomic forces 3.6486 and 5.8762 eV/Å, respectively; high
instantaneous forces are expected to differ from minimum forces. Neither frame
is described as an independent equilibrium sample.

The event review confirms persistent Li hops in basin 1 / repeat 2 and basin 2 /
repeat 1. In the latter, local atom 59 moves from parent Li site 11 to site 23,
separated by 2.81126 Å. Its intervening nearest-site-40 labels are an ambiguous
transit, not an established third basin. DFT_012 intentionally preserves the
first transit frame: its nearest/second-site distance margin is only 0.00935 Å.
DFT_013 has margin 2.51453 Å and is clearly assigned to site 23. The original
maximum-force DFT_011 is retained rather than silently replaced by the event.
The other observed hop and continuous-geometry evidence remain in the numerical
audit. These are events under UMA; single-point DFT on a transit frame tests
its energy and forces, not a real-LCO migration barrier or diffusion rate.

## Priority and required observables

The final manifest records cell, composition, source geometry,
teacher/head/precision, file hash, existing labels, selection reason and status.
All 13 exact array geometries are distinct. Its SHA256 is
`a661aa31ca05ad2fb3367ac25f6f29f48f8d9ab20e64b6d5513bfb9efade385c`.
The maximum POSCAR round-trip displacement is 3.77×10⁻¹⁵ Å; the stated 1e-10 Å
position tolerance is only a numerical serialization check, not a structural
equivalence criterion. The scientific priority is:

1. **Competing distinct fixed-cell minima.** Evaluate each retained UMA minimum
   at the same cell, first as a single point. Save total energies in both
   extrapolated and smeared conventions, every force component, tensile-positive
   stress, total/site magnetic moments, electronic stopping history and the exact
   evaluated geometry. This tests static relative energies and whether an MLIP
   stationary point is also nearly stationary under DFT. It does not create a
   DFT-relaxed minimum. If starts merged into one basin, retain only one duplicate
   geometry and preserve the merging evidence.
2. **Teacher-disagreement structure.** Select a fully scored same-composition
   geometry with a large *centered* MACE−UMA energy difference or an actual
   reversal among retained minima. A model-dependent absolute offset is removed
   before selection. This discriminates teacher-specific ordering rather than
   choosing an irrelevant absolute energy zero.
3. **Representative thermal force environments.** Select a small number of exact
   production frames from retained, mapped basins. Use a typical frame and a
   high-force or atypical frame, with selection rule and trajectory time saved.
   Request E/F/stress at the unchanged frame. These test forces sampled at the
   stated temperature, not only forces at a minimum. A short MD frame is not
   called an independent equilibrium configuration.
4. **Targeted Cartesian curvature pairs.** Four already MLIP-evaluated frames
   use the first present Li atom of basin 1 (local index 48, parent index 48,
   x column 144) and the first O atom of basin 2 (local index 84, parent index 96,
   x column 252), each at ±0.01 Å. All indices are zero based. Full force arrays
   give Φ[:,column]=−(Fplus−Fminus)/(2δ), in eV/Å². This tests selected force-
   constant columns without requiring all 1,080 central-displacement evaluations
   of a 180-atom Hessian. The raw column input hash matches the exact minimum.
   Individual displaced stresses were not retained, so minimum stresses are
   never substituted. These spot checks do not validate a full DFT spectrum or
   its phonon free energy.
5. **Observed transition environment.** The actual pilot produced persistent
   neighboring-site Li hops. DFT_012 preserves the first meaningful transit and
   DFT_013 a clear post-hop geometry; their source minimum is already included.
   The transit frame is labeled accordingly and is not presented as a stable
   harmonic basin. Its force field can discriminate a problematic teacher
   environment before a separately designed barrier calculation is considered.

No initial numeric force or energy-accuracy threshold is promoted to a universal
DFT standard. The needed precision follows the observable. For example, at
433.15 K, kBT≈0.03733 eV. An error δ(ΔF) in a cell free-energy difference
changes its relative weight by exp[−δ(ΔF)/kBT]. Requiring relative odds within
a chosen factor r>1 would require |δ(ΔF)|<kBT ln(r), in **eV per cell**. The
choice of r must be made explicitly; a meV/atom tolerance cannot be copied into
this formula without multiplying by the actual atom count.

## Manual decisions before any later DFT campaign

Decide whether to extend the verified historical PBE+U reference or establish
another consistent reference. Confirm PAW identities, the main-cell k-point
convergence, magnetic-state strategy, energy convention and force/stress signs.
Apply the same protocol to competing structures. If DFT forces invalidate an
MLIP minimum, a subsequent separately authorized DFT relaxation may be needed
before comparing DFT basin thermodynamics.

No ready-to-submit input deck is supplied while those main-cell and magnetic
choices remain unresolved. Exact structures, verified historical settings and
required observables provide a concrete handoff without pretending those
scientific decisions have been made. Nothing here submits or schedules work.

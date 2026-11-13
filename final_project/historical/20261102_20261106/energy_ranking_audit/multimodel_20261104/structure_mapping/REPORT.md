# Verified duplicates and phase-specific joint-geometry dataset

Prepared 2026-11-04. This extends the completed frozen-MLIP audit; no new DFT,
MLIP inference, relaxation, MD, or CVAE training was performed.

## What was actually duplicated?

Every one of the 4,012 duplicate pairs has exactly equal archived coordinates,
cell, energy, forces, and all nine virial components. Every maximum absolute
difference is zero. Equality is at the stored float32 precision, with no
allclose tolerance. The duplicate records are exclusively Hu's Li-free migration
data: 190 pairs within set.000 and 3,822 pairs between set.000 and set.001.
No historical or Hu bulk frame was removed by this deduplication. The full
source aliases remain in ../raw_frame_map.csv. See duplicate_verification.json.

## Mapping and disposition

All 11,456 unique geometries were assigned a phase-appropriate reference or a
clearly marked diagnostic reference. Raw geometry and labels are never changed.

| Source/family | Unique frames | Localized joint subset | Mapping review | Geometry-only migration |
| --- | ---: | ---: | ---: | ---: |
| Historical O3 | 524 | 524 | 0 | 0 |
| Hu O3 | 4,261 | 4,174 | 87 | 0 |
| Hu spinel | 1,255 | 1,213 | 42 | 0 |
| Hu O1 | 606 | 570 | 36 | 0 |
| Hu migration | 4,810 | 0 | 0 | 4,810 |
| Total | 11,456 | 6,481 | 165 | 4,810 |

All 165 review frames are DPGEN snapshots. All 218 Hu optimized frames pass the
localization check. A review flag does not prove an invalid geometry or an
impossible mapping. It says this bounded alignment did not establish the stated
local-site certificate. Those frames remain in continuous_or_review_dataset.csv
and in the complete original reference_frames.npz.

The 6,481 localized frames contain 165 distinct parent-mask orbits: historical
O3 72, Hu O3 67, spinel 15, and O1 11. These are not 6,481 independent vacancy
patterns. Different displacements and cells within an orbit remain separate
joint-geometry training examples. No extra frame is removed merely for sharing
a canonical mask. Grouping and trajectory/parent lineage still need to be used
when creating the actual training split; this export does not invent one.

## References and symmetry

| Reference | Removable sites | Space group | Full operations / distinct Li permutations |
| --- | ---: | --- | ---: |
| Historical O3 | 24 | R-3m | 288 / 288 |
| Hu O3 | 12 | R-3m | 144 / 144 |
| Completed Hu O1 | 12 | P-3m1 | 144 / 144 |
| Idealized Hu spinel | 8 | Fd-3m | 192 / 192 |

The first two references match the previously documented parent choices.
Hu O1's archived Li3 seed does not provide a complete alkali lattice. Its
derived template completes the twelve interlayer sites supported by the
2x2x3 host registry. Co/O in-plane positions are idealized onto that registry;
the paired oxygen height is fitted from optimized Hu Li3 frame index 6633,
giving 0.066593923462918 in fractional c. This template is NOT a newly calculated
or DFT-relaxed Li12 O1 structure.

The spinel reference uses eight tetrahedral Li sites, not sixteen octahedral
O3-like sites. The derived ideal Fd-3m template uses quarter-grid Li, eighth-grid
Co, a fitted oxygen internal offset of 0.01303488412688991, and a cubic reference
cell. The actual distorted seed is unique frame 6039. Idealization defines a
coordinate reference; the real distortions remain in each displacement array.
Both derived ideal references have species-mapping symmetry residuals below
5e-15 Angstrom at symprec 1e-5 Angstrom.

parents.json contains explicit fractional rotations/translations, all removable
site permutations, reference coordinates/cells and provenance. The old
symmetry_counts.json summarized group/cycle counts rather than storing the
operations themselves. operation_examples.json gives identity, translation,
inversion and threefold examples with source-to-destination site permutations.

For fractional column coordinates an operation is f' = R f + t modulo integer
lattice translations. R is an integer matrix in a nonorthogonal lattice basis;
it need not resemble an orthogonal Cartesian rotation matrix. Internal
supercell translations can permute distinct listed sites while preserving the
infinite crystal. An allowed mask symmetry preserves every species of the
parent, not only the Li sublattice.

A mask orbit is the set of masks obtained from one mask under these operations.
The canonical ID is its lexicographically smallest vacancy bit string. A
stabilizer is the set of operations that leave the chosen mask unchanged;
orbit_size * stabilizer_size = group_size. Equal mask orbit IDs do not imply
equal actual coordinates, strain, electronic labels, energy, force or stress.
The absent frame-level magnetic records also preclude certifying electronic
state equivalence solely from a geometric mask.

All-mask enumeration gives 87 possible Hu O3 orbits, 87 for completed O1 across
all occupancies, and 16 for spinel. In the O1 composition range actually observed
(0 to 3 retained Li), there are only 11 possible orbits, all represented in the
localized subset. Novel-mask claims in that covered range would be incorrect;
geometry generation remains meaningful. See observed_orbit_coverage.pdf.

## Assignment method and exact reconstruction

Each source is registered to a phase reference using candidate cell-compatible
reference orientations, host translation fitting and species-aware Hungarian
assignment with periodic distances. Cached orientations are rechecked with all
orientation candidates and independent Co-anchor origins whenever localization
fails. Host fitting uses at most twelve translation iterations per start; a
1e-9 Angstrom update is an arithmetic optimization stopping rule, not a physical
acceptance threshold. The search does not prove a globally optimal assignment.

For every atom a, the code stores its parent slot j(a), reference orientation Q,
fractional origin shift t, and Cartesian displacement u_a in Angstrom. With
row-vector fractional convention, reconstruction is

    r_a = ((f_parent @ Q.T + t)[j(a)]) @ h + u_a

up to a periodic lattice image. h is the original frame cell, so homogeneous
strain is kept separate from local displacements. Atomic positions, DFT force
vectors and stress components remain in their original coordinate frame.

The geometric certificate requires each displacement to be strictly smaller
than half the nearest same-species reference-site distance in that supplied
cell. Triangle inequalities then guarantee that another same-species site
cannot be closer. The factor one-half is an exact sufficient condition for
local uniqueness, not an empirical acceptable-distortion tolerance. It is
conservative, is not necessary for a useful mapping, and does not establish
physical validity, SCF convergence, low energy or MD stability.

Migration frames use O3 only as a diagnostic coordinate reference. Their Li
mask is fully vacant and supplies no Li-ordering variation. Large Co motion
belongs to the continuous-geometry task; it is not forced into the localized
vacancy-training subset.

## Independent checks

- All 11,456 coordinate roundtrips pass; maximum periodic error 2.474e-14 A.
- 15,936 periodic test vectors, including cell-shape/displacement extremes,
  agree with ASE's general MIC norms within 8.882e-15 A.
- 1,159 distinct observed raw masks independently match integer-encoded orbit
  canonicalization and the orbit-stabilizer relation.
- Combined rigid rotation, origin translation and atom reordering preserve the
  orbit ID and localization for one nontrivial example from each parent.
- All exported species, coordinates, cells and DFT E/F/stress arrays are exactly
  equal to their corresponding original reference arrays.

The 1e-10 A check bound is floating-point implementation consistency only. It
is not used to merge geometries. Check receipts are independent_validation.json
and export_validation.json. Parent group closure/identity are also checked.

## Files and data contract

- joint_training_candidates.npz: 6,481 exact-geometry examples, 313,821 atoms.
- joint_template_dataset.csv: metadata, orbit IDs, mapping diagnostics and source
  references for those examples.
- frame_mapping.csv: all 11,456 mappings and disposition flags.
- continuous_or_review_dataset.csv: all 4,975 other geometries, retained.
- site_displacements.npz: assignments/displacements for all unique frames.
- conditioning_observables.csv and observable_support.csv: exact DFT-derived
  force RMS, maximum force, configurational pressure and deviatoric stress.
- observed_mask_orbits.csv and mapping_summary.csv: counts, observed support,
  source coverage and per-orbit energy ranges.
- mapping_diagnostics.pdf and observed_orbit_coverage.pdf, plus PNG versions.
- ../reference_frames.npz and ../raw_frame_map.csv remain the complete raw-label
  and original-alias sources. geometry_key joins the existing three-MLIP tables.

In the training NPZ, reference_index points to the original unique-frame index;
offsets index flattened atom arrays. vacancy_mask uses 1=absent, 0=present,
and -1 padding beyond mask_length. parent_index refers to parent_names. Arrays
include exact positions_A, cell_A, dft_energy_eV, dft_forces_eV_A and
dft_stress_eV_A3. Stress is ASE tensile-positive xx,yy,zz,yz,xz,xy; Hu conversion
retains the existing conventional -sym(virial)/volume assumption. This work
does not recertify missing original frame-level SCF/spin/exporter metadata.

## Consequences for the joint CVAE

The chosen direction is a shared periodic graph CVAE with supplied phase/parent,
composition, cell and calculator/reference context; scalar occupancy outputs
and equivariant vector displacements. O3/O1/spinel have separate parent
namespaces. A shared graph network does not make their masks interchangeable.
Supply cells initially instead of adding lattice generation. Continuous
migration decoding remains a separate extension or validation use.

Force/stress labels can be used immediately for evaluation. A controlled
conditioning ablation can add component force RMS qF, configurational pressure
p=-tr(stress)/3 and a rotation-invariant deviatoric stress norm. These are
requested observables, not physical guarantees. Do not equate configurational
DFT pressure with imposed external pressure or the full MD pressure including
kinetic contributions. Do not target zero force for every finite-temperature
training example. Current median component-force RMS values are about 0.830
eV/A for historical snapshots, 0.875 for Hu O3 DPGEN and 0.00246 for optimized
Hu O3. Source/cohort support therefore matters when defining conditions.

Raw force vectors require per-node vector conditioning and atom correspondence;
raw stress requires a tensor representation or a consistently aligned frame.
Appending their components as orientation-independent global scalars would be
incorrect. Fit normalization/class cutoffs on training data only, with explicit
source/phase/composition context. No force/stress acceptance threshold or energy
class has been chosen in this preprocessing.

Behler-Parrinello atom-centered symmetry functions can encode invariant local
environments or provide descriptor baselines. They do not enumerate crystal
mask orbits and do not uniquely invert to coordinates. Periodic graph encoders
plus equivariant coordinate decoders are a more direct joint-generation design.
The distinction between an MLIP's structure-to-energy map and a generator's
latent/condition-to-structure map remains essential.

Primary methodological references: [Behler and Parrinello 2007, Eqs. 2,4,5](https://doi.org/10.1103/PhysRevLett.98.146401),
[Behler 2011](https://doi.org/10.1063/1.3553717),
[NequIP](https://arxiv.org/abs/2101.03164), and
[CDVAE, sections 3-4](https://arxiv.org/html/2110.06197).

## Reproduction and resources

Run map_frames.py --workers 8, then validate_mapping.py, summarize_mapping.py,
and check_export.py, in a CPU allocation. Existing raw data are read-only.
Map script/input hashes and software versions are recorded in completion.json.
CPU allocation 60820897 used account m5397, one CPU node reserved for 672 s,
and is COMPLETED/released. Calculated reservation is 0.18667 CPU node-hours;
no GPUs were allocated. Final site-posted charges can differ. See resources.json.

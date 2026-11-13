# Symmetry coverage of the 144-site fixed-parent LCO catalog

[Coverage plot](observed_orbit_coverage.pdf) ([PNG](observed_orbit_coverage.png)) ·
[Exact counts by lithiation](coverage_by_lithiation.csv) ·
[Unique available representatives](unique_representatives.csv) ·
[Mapping of every input candidate](candidate_orbit_mapping.csv)

The 24,145 existing candidates contain **23,912
symmetry-distinct vacancy configurations** under this saved periodic parent's
species-preserving symmetry group. All raw masks are distinct before symmetry;
there are **233 redundant symmetry representations**
(0.96500% of entries), across 83 multi-member groups.
All fifty sampled compositions remain represented. Original data are unchanged.

## Where symmetry reduces the catalog

Only compositions with a nonzero reduction are listed below; every other sampled
composition has equal raw and symmetry-unique counts.

| Retained Li | Lithiation (%) | Raw entries | Unique orbits | Redundant representations |
| ---: | ---: | ---: | ---: | ---: |
| 143 | 99.30556 | 144 | 1 | 143 |
| 141 | 97.91667 | 500 | 415 | 85 |
| 140 | 97.22222 | 500 | 495 | 5 |

The 144 one-vacancy configurations form one orbit. The pristine configuration
forms one orbit. Unsampled integer compositions, including Li142 and Li108,
are retained as zero-availability rows in the counts table. Missing compositions
are not interpolated into apparent observations.

## Meaning of the figure

The first panel compares all possible symmetry-distinct masks with the available
unique masks. Theoretical counts span about forty orders of magnitude, so it uses
a logarithmic vertical axis. The second panel shows raw entries and unique
orbits over the sampled lithiation range on a linear count axis. The third panel
shows the fraction of possible mask orbits actually represented. This is
configurational coverage, not energetic, thermodynamic, or statistical coverage.

At 50% lithiation the possible count is
5,139,628,466,836,063,160,129,497,351,009,120,243,978. The catalog therefore remains
a tiny selected subset even after finding that almost all available masks are
symmetry-distinct. Uniqueness alone does not establish a useful low-energy
distribution or a generative-model advantage over random/search baselines.

## Exact counting and canonicalization

The source is the saved pristine_4x6x2_POSCAR with Li144Co144O288, not the original
CIF's differently ordered Li list. spglib 2.6.0 identifies
R-3m and 288 full operations, inducing
288 distinct permutations of its 144 Li sites. Rotations
and translations preserve the entire Co/O/Li parent and the specified periodic
supercell. Maximum species-mapping residual is
4.0301e-14 Angstrom. Group identity and closure are
checked. Identification is stable over the tested symprec values 1e-6 to 1e-2 A;
the selected value is 1e-5 A. This tolerance identifies the parent symmetry; it is
not an energy tolerance or a criterion for merging distorted structures.

For a permutation p[source]=destination, transformed[p]=mask. A vacancy bit of
one denotes an absent Li. The canonical mask is the lexicographically smallest
144-bit string over all permitted transformations; its hexadecimal form is
exported without losing leading zeros. An orbit is all symmetry-related masks.
A stabilizer is the subset of operations leaving a mask unchanged, and

    full orbit size = 288 / stabilizer size.

The number of members present in the catalog is stored separately from the full
mathematical orbit size. A canonical mask need not itself be an available entry.
Each orbit's representative is the lexicographically first actual candidate path,
not the candidate with the smallest historical score. The recorded witness maps
an input mask to its canonical mask, not necessarily directly to that available
representative.

Theoretical counts use exact-integer Burnside coefficient calculations for every
vacancy count. They exactly reproduce the prior symmetry_counts.json and pass
independent zero-, one-, and two-vacancy enumeration checks. Input hashes and
explicit operations are in parent_symmetry.json and completion.json.

This defines equivalence for the fixed periodic parent. It does not claim
equivalence across different supercell tilings, nor electronic/spin-state
uniqueness beyond the species-and-geometry description.

## Saved-file and independent validation

All 24,145 saved POSCARs were checked, not only a subset.
Species-aware periodic coordinate matching confirms that each is exactly the
saved parent with its listed Li atoms removed, up to floating-point output
precision. Maximum coordinate difference: 3.7051e-15
A. Maximum cell difference: 0 A. The numerical
comparison bound is 256 times float64 machine epsilon times the sum of parent
cell-vector lengths; it is an arithmetic reproduction bound, not a physical
geometry-equivalence tolerance. No distinct structure is merged using that bound.

Independent checks include:

- 400 Python-integer group-orbit checks,
  independent of the vectorized packed-bit implementation.
- All 24,145 exported transformation witnesses.
- All 24,145 saved compositions and atom counts.
- One real representative per orbit, catalog membership counts, and the
  orbit-stabilizer identity.
- 18 independent ASE POSCAR reads as a parser,
  geometry and site-order cross-check.
- Input manifest, parent, old count receipt and score-file hashes unchanged
  throughout execution.

Receipts: completion.json, independent_validation.json and
saved_geometry_verification.csv. Scripts and logs are retained.

## Historical energy labels

These are the existing UMA-only labels, not DFT energies and not a rerun using
the corrected current UMA checkpoint. Scorer status is
torchsim_uma_uma-s-1p2_omat_cuda. Energies were not used to decide symmetry
equivalence. Old energies are preserved in the complete candidate map and in the
deterministic representative table; group minima/maxima/means/spans are
diagnostics only, not newly assigned labels.

The largest cached within-orbit energy spread is
0.000244140625 eV/cell, for a group with
143 retained Li. See orbit_groups.csv for every spread. In particular,
the one-vacancy group's two energy values differ by 0.000244140625 eV/cell,
one float32 spacing at that energy magnitude. This does not invalidate geometric
equivalence or establish a universal acceptable energy tolerance.

Original heuristic rank/distance metadata are retained for traceability. The
previous audit identified a generation-versus-saved-parent ordering mismatch;
those fields must not be treated as validated physical descriptors of the scored
structures. This symmetry analysis uses the actual saved-parent ordering.

## Files and reproduction

- coverage_by_lithiation.csv: all 145 integer Li counts, exact raw binomial and
  symmetry-distinct possible counts, available counts, redundancy and fractions.
  Preserve the large count columns as strings/integers when reading spreadsheets;
  floating-point display can lose exact digits.
- candidate_orbit_mapping.csv: every original entry, canonical ID, stabilizer,
  orbit size, witness, actual representative and preserved old score.
- unique_representatives.csv: 23,912 actual source entries,
  one per orbit. Source files are not deleted or overwritten.
- orbit_groups.csv: available multiplicity and old-score spread for every orbit.
- parent_symmetry.json: full symmetry operations/permutations and exact theory.

Run analyze_catalog.py in a CPU allocation, then independent_checks.py and
write_report.py. No DFT, MLIP prediction, relaxation, MD or training is run.
Resource/accounting evidence is recorded separately in resources.json.

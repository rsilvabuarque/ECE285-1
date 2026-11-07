# Exact structures for a later DFT reference check

These are structure files and existing MLIP labels. No DFT calculation is run,
submitted or scheduled, and no executable VASP input deck or POTCAR is supplied.
The scientific priority and unresolved reference choices are explained in
[DFT_REQUIRED_NEXT.md](../../../docs/thermal_pilot/DFT_REQUIRED_NEXT.md).

Use `candidate_manifest.json` for each structure's source, selection rule,
geometry protocol, teacher/head/precision, observables and SHA256. The flat
`candidate_manifest.csv` and `candidate_validation.json` provide the completed
serialization/hash check for all 13 distinct exact geometries. A POSCAR can
be opened directly in VESTA.

| IDs | Content | Role |
| --- | --- | --- |
| DFT_001–003 | Exact completed UMA fixed-cell local minima | Competing arrangements; minima 1 and 2 also have a strong exact-geometry MACE order reversal |
| DFT_004–005 | Fully scored, unrelaxed parent-deletion structures | Optional broader teacher gap-disagreement probes |
| DFT_006–007 | Basin 1 first-Li x±0.01 Å | Existing central force-difference column; all force vectors retained |
| DFT_008–009 | Basin 2 first-O x±0.01 Å | Existing oxygen force-difference column |
| DFT_010–011 | Frozen-rule frames from completed primary production | Basin 1 typical frame at 19.040 ps and basin 2 high-force frame at 8.726 ps |
| DFT_012 | Basin 2 repeat 1 first transit, 16.882 ps | Diagnostic frame during a subsequently persistent neighbor hop; not a stable-basin assignment |
| DFT_013 | Basin 2 repeat 1 clear post-hop frame, 17.024 ps | Optional occupied-site-change check, 0.100 ps after persistent final assignment begins |

All structures are Li36Co48O96 at one common cell, with 180 atoms. POSCAR order
is **Co / Li / O**, with counts **48 / 36 / 96**. Force arrays in the companion
NPZ retain that exact atom order and Cartesian orientation. Energies are eV per
cell; forces eV/Å; available stresses eV/Å³, tensile positive in ASE order
`xx yy zz yz xz xy`. They are calculator/configurational stresses without the
ionic kinetic contribution, appropriate for an ionic-static DFT comparison.
They are not full finite-temperature pressure tensors. Individual finite-displacement stresses were not saved;
the minimum's stress is not substituted into a displaced structure.

The verified historical VASP cases have a different Li/Co/O ordering. Copying
their species-indexed Hubbard/MAGMOM arrays or concatenated PAW order into these
POSCARs unchanged would be wrong. Main-cell k-point convergence and magnetic
state selection remain manual scientific decisions. No ready-to-submit INCAR
is implied by the preserved historical parameter text.

`archived_x075_EFS.csv`, `archived_x075_summary.csv`,
`existing_reference_receipt.json` and `existing_dft_protocols.json` inventory the
old matched references. They contain no newly evaluated DFT. PAW titles and
hashes are metadata only; no proprietary potential content is distributed.

`selected_occupancy_symmetry.json` and `relaxed_basin_symmetry.json` document
the retained parent action. All three selected masks have orbit size 192;
their pairwise symmetry-minimized Hamming distance is eight. The relaxed
assignments remain distinct. Equal orbit factors cancel from their relative
weights; this does not make a three-member subset the complete population.

`mace_minima_comparison.csv` and its receipt compare both teachers on identical
coordinates. MACE was not used to relax those structures. The energy-context
CSV/PDF separates teacher static predictions from the UMA finite-cell harmonic
calculation. Teacher disagreement is not a measured DFT error.

The helper scripts only audit existing files, reproduce exact candidate
geometries, and verify serialization. Run substantial NPZ/trajectory analysis
inside an authorized compute allocation. The frozen thermal selection policy
retains any diagnostic event for review rather than dropping it to claim stable
sampling. The 2-ps selection exclusion is provisional and does not establish
equilibration. Full trajectories and their original receipts remain on HPC
storage; the [raw retrieval manifest](../raw_manifest.json) identifies them,
with a separate [completed archive check](../raw_archive_verification.json).

The completed review preserves the original high-force probe and the actual
event, rather than selecting only trajectories that stayed in the initial mask.
`candidate_manifest.json` binds each active structure to its exact source;
`thermal_selection_policy.json` preserves the rule frozen before completed
production was inspected. `diagnostic_event_basin_2_production_frame08441.*`
is a preserved alias of the DFT_012 transit export, not a fourteenth candidate.
The source audit also preserves the other primary trajectory's confirmed hop.

The final manifest SHA256 is
`a661aa31ca05ad2fb3367ac25f6f29f48f8d9ab20e64b6d5513bfb9efade385c`.
All candidate POSCAR/NPZ hashes passed, with maximum coordinate round-trip
error 3.77×10⁻¹⁵ Å. That test verifies file representation, not chemical
correctness or a DFT result. Helpers make no new calculator calls.

ECE 285 preliminary dataset and CVAE experiments

Start with collaborator_package/README.txt. That smaller package contains the
learning arrays, frozen splits, four example checkpoints and MLIP-free CPU/CUDA
training and sampling code. collaborator_package.tar.gz is the same portable
core; validation/core_package.json gives its checksum. The thermal export and
full-campaign audit are separate and do not hold up use of the core package.

The scientific handoff is reports/preliminary_report.pdf with its LaTeX source.
While thermal postprocessing is pending, the isolated draft is under
reports/draft_preview/ and is marked DRAFT. See execution.json for current
workflow state and resources.json for allocation accounting; those are the
authoritative changing status records.

What is in the full campaign

  data/parents/                  Five pristine-parent mask catalogs and symmetry
  data/mask_manifest.csv          Parent/mask/composition identities
  data/labels/<teacher>/          Raw static E/F/stress shards
  data/learning_*.npz             Small teacher-specific learning arrays
  splits/lineage_splits.csv       Frozen mask-lineage and composition splits
  data/fixed/, data/variable/     Physical relaxation traces and snapshots
  data/fixed_resume1/             Explicitly linked infrastructure retry attempts
  data/geometry_*_fixed.npz       Aligned, capped fixed-cell learning frames
  data/geometry_*_variable.npz    Aligned variable-cell frames plus quality flags
  data/generated/                Fresh five-model scores for frozen proposals
  data/generated_scalar_*.npz    Compact scalar exports of those scores
  cvae/proposal_attempts.csv      All proposal attempts, including duplicates
  cvae/runs/                     Sixteen frozen mask fits and histories
  cvae/geometry_runs/            CVAE and regressor coordinate prototypes
  data/decoded_score/            Fresh decoded-geometry physical checks
  data/decoded_relax/            Matched relaxation-cost comparison
  data/phase_context*             Actual archived Hu O3/O1/spinel diagnostic
  data/md_prod/                  Short UMA/MACE thermal trajectories
  data/geometry_*_md.npz         Derived thermal frames when export is complete
  reports/                       Figures and per-attempt/grouped metric tables
  validation/                    Provenance, numerical and dataset audits

Conventions that matter

Occupancy=1 means Li present; canonical vacancy-mask strings use 1=vacancy.
The main cell has 48 Li sites, 48 Co and 96 O. Read li_indices/parent_slot from
the files: Li atoms are parent rows 48–95. Total atoms are 144+n_Li. Missing-atom
rows are padding; their zero displacements and forces require a presence mask.
Energy is eV/cell, forces eV/Angstrom, ASE stress eV/Angstrom^3 in
xx yy zz yz xz xy order, tensile positive. 1 eV/Angstrom^3=160.21766208 GPa.

Keep energies separate by teacher, parent, composition and geometry cohort.
Deletion-energy ordering and relaxed-energy ordering are different objectives.
All descendants of one parent mask retain its frozen split. Correlated frames,
symmetry augmentations and multiple teacher labels are not independent samples.
Thermal U/F arrays retain ancestral slots; observed occupancy is a diagnostic
that requires reanchoring before replacing the ancestral mask after a hop.

What the current evidence supports

The initial mask CVAE has a small, seed-dependent average advantage over uniform
sampling and does not beat low-energy-mask swap mutation. Mutation explores
closer to the training set, so quality and nearest-training-orbit distance are
reported together. Geometry prediction improves relaxation starting points;
the deterministic regressor performs similarly to the CVAE. Decoded geometries
still require relaxation. All new large-cell labels are MLIP predictions, not
new DFT truth. Short MD provides preliminary thermal geometries, not established
equilibrium thermodynamics or optimal recycling conditions.

Execution and reuse

Original cvae/train.py and chemistry workers preserve their executed Perlmutter
protocols. Use collaborator_package/cvae/ for portable training. Do not launch
the chemistry dispatchers just to train a CVAE. Raw failed/capped attempts and
retry lineage remain preserved; choose the first valid logical completion,
never the lowest-energy retry. The final scientific delivery check is
validation/requirement_audit.json, followed by root review of the rendered report
and terminal scheduler accounting. No DFT jobs are part of this campaign.

MD-specific applicability is documented in validation/MD_mapping_review.json
and reports/MD_template_applicability.pdf. Of 4,000 capped thermal frames,
1,076 meet the conservative parent-site bound. Other frames remain available;
they should not be fed to an unchanged-site displacement model without review.

data/generated/ is assessment data for the reported frozen fits. Using those
labels for another training round requires a new split/assessment protocol and
an updated canonical novelty cache. Do not train on that pool and continue to
describe its old scores as an untouched test result.

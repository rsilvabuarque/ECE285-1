ECE 285 preliminary learning package — 6 November 2026

Start with the mask sampler. The geometry model is a separate prototype.
This folder contains NumPy arrays, PyTorch code and saved checkpoints. Training
and sampling require no VASP, ASE, UMA, MACE, SevenNet, CHGNet or M3GNet install.
The full campaign contains raw E/F/stress outputs and independent validation
receipts; this package contains the smaller learning inputs.

QUICK START

Use a Python environment with NumPy and PyTorch:
  pip install -r requirements.txt

Run commands from this package directory. Output run directories must be new.
On Perlmutter, run training in an allocation; CPU mode also works on a laptop.

  python cvae/train.py --teacher uma --kind cvae --device auto \
    --seed 20261106 --epochs 150 --output-dir runs/uma_cvae_new

  python cvae/train.py --teacher uma --kind cvae --beta-final 0.1 \
    --device auto --output-dir runs/uma_beta01_new

  python cvae/train.py --teacher uma --kind composition --device auto \
    --output-dir runs/composition_new

  python cvae/train.py --teacher uma --kind vae --device auto \
    --output-dir runs/vae_new

  python cvae/portable_sample.py \
    --checkpoint checkpoints/uma_cvae_main_20261106/best.pt \
    --n-li 24 --count 100 --sampling conditional --device cpu \
    --output samples_Li24.npz

  python cvae/train_geometry.py --kind cvae --device auto \
    --output-dir runs/geometry_new

Use --data-root /path/to/package when the data live elsewhere. Training supports
--device cpu/cuda/auto and --threads. Sampling supports cpu/cuda. Direct scripts
are supported; python -m cvae.train is not currently supported.

WHAT ONE ROW MEANS

The parent has 48 Li sites and 48 Co + 96 O atoms. Total atoms are 144+n_Li.
Occupancy o_i=1 means Li present at site i. The raw campaign vacancy masks use
the opposite convention, v_i=1-o_i. Read li_indices from the parent NPZ: the Li
atoms are parent rows 48–95, not the first 48 rows. Padded missing-atom forces
and displacements are zeros used with a presence mask, never training labels.

data/learning_O3_N48_pilot_<teacher>.npz contains occupancy, structure_ids,
n_Li, n_atoms, energy_eV, stress_eV_A3 and force summaries. Five teachers are
separate energy namespaces. Compare energies within one teacher/composition.
stress uses xx yy zz yz xz xy, eV/Angstrom^3, tensile positive.

splits/lineage_splits.csv groups all frames/models/conditions of a parent mask
together. main split is stratified by composition. composition_split withholds
six interior Li counts; it tests composition interpolation, not extrapolation.
Symmetry operations augment training rows AFTER the split. Validation/test
rows and source counts are never inflated by augmented copies.

THE CONDITION AND THE LOSS

For each teacher and Li count, the trainer uses only training E/n_atoms values.
The low-class threshold q is sorted_value[ceil(0.2*m)-1], m=training-row count.
Ties are retained. A CVAE condition is (n_Li/48, E/n_atoms <= q). The class
label 0 means outside the lowest 20%, not the highest 20%.

The encoder reads occupancy and condition and predicts an 8-dimensional normal
latent distribution. The decoder reads a latent draw and condition and predicts
48 occupancy logits. Training minimizes summed Bernoulli reconstruction loss
plus beta times the Gaussian KL divergence. KL discourages the encoded latent
distribution from moving too far from the standard-normal prior used at sample
time. beta warms over 20 epochs; --beta-final controls its final weight.
All checkpoints, including beta=0.1 fits, are selected by ordinary validation
negative ELBO (beta=1), using fixed validation noise.

The composition-only CVAE receives n_Li/48; the VAE receives no condition.
Their generated masks still use the same exact-count decoding constraint.
Topk gives the constrained most-probable mask for a decoder logit vector.
conditional uses dynamic programming to draw the exact factorized-Bernoulli
distribution conditioned on sum(o)=n_Li; it is not Gumbel-topk sampling.
The sampler retains raw logits/expected count and symmetry keys. It does not
reject duplicates or evaluate energy. Its printed uniqueness is per batch.

GOOD RECONSTRUCTION IS NOT GOOD DISCOVERY

Judge generated masks by novel symmetry-unique low-energy orbits per fixed
attempt and oracle budget. Include uniform fixed-count masks, low-training-mask
swap mutation, composition-only CVAE and VAE controls. Keep duplicate attempts
in the denominator. Uniform raw masks are not uniform over symmetry orbits.
Report three training seeds separately. Do not treat 512 proposals as 512
independent scientific replications. Low class conditioning does not enforce an
energy inequality, follow a physical energy gradient, or prove a ground state.

GEOMETRY PROTOTYPE

data/geometry_uma_fixed.npz contains 1,000 independent parent masks and capped
trajectory frames. Each mask's frames share a frozen split and total row weight
one. Displacements are from pristine fractional parent sites mapped through the
frame cell, after removing global mass-weighted translation, then applying the
minimum-image convention. Cell deformation is stored separately.

The geometry decoder receives occupancy, composition, cohort, supplied cell
deformation, and four standardized properties: energy/atom, log1p(force RMS),
configurational pressure, and log1p(deviatoric stress norm). All transforms are
fitted on training frames only, separately by composition and cohort. Desired
property tuples are attained training endpoints, never held-out oracle values.

This prototype predicts coordinates at a SUPPLIED FIXED cell. It does not
predict cell relaxation or enforce zero force/stress. Its validation coordinate
RMSE is a posterior reconstruction measure. Fresh prior-decoded E/F/stress and
matched relaxations are the necessary additional tests. Regressor and CVAE have
one fit each; three latent draws are not three independently trained models.

Thermal MD exports live in the full campaign. They retain ancestral parent
slots and separately diagnosed current occupancy. A hopped Li cannot simply
be relabeled by changing its occupancy bit; reanchoring U/F rows is required.
Do not apply a geometry-optimization force threshold to thermal frames.

SCOPE AND REPRODUCIBILITY

These dense architectures are specific to this 48-site parent. Auxiliary
12/24/36-site data exist in the full campaign but mixed-size transfer has not
been demonstrated. Cross-parent primitive aliases need checking before a
leakage-free transfer claim. Phase-context diagnostics do not establish phase
stability. New geometries have MLIP labels, not newly computed DFT labels.

The full campaign's validation/portable.json records actual CPU fit, checkpoint
reload and exact-count sampling tests. Those short tests establish operability,
not scientific convergence. MANIFEST.json records package file hashes. Original
executed training scripts/checkpoints remain in the full campaign unchanged.

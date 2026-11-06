# Dependency profiles

`scripts/install.sh` installs the project and **all transitive dependencies**,
then runs `pip check` in every environment. The four MLIP profiles are isolated
because the verified MACE release uses e3nn 0.4.4 while UMA and SevenNet use a
newer e3nn. CHGNet here is the MatGL MatPES checkpoint, not the separate original
`chgnet` Python package. Installing one monolithic environment is unsupported.

These pins match the November 2026 scientific comparison. The installer records
the resolved transitive versions in each environment's `installed-packages.txt`.
They are a tested SDK target, not a claim that every GPU driver supports its
PyTorch wheel. Install a compatible NVIDIA driver or choose the official PyTorch
wheel channel with `--torch-index-url`. PyTorch's ROCm builds expose GPUs through
the same `cuda` API, but upstream backend support must be checked on that hardware;
the completed scientific audit used NVIDIA A100 GPUs. CPU is useful for small
learning checks and backend checks when supported, and does not require Slurm.

The packages do not include every checkpoint. The installer downloads the pinned
upstream weights and checks their SHA256 unless `--no-weights` is supplied. UMA
requires an authorized Hugging Face account and its model license/access process.
Use `hf auth login` in the learning environment after obtaining access; do not
put tokens in scripts, Git, CLI arguments, or shared logs. No gated weights are
redistributed by this repository.

Primary API references checked while preparing these profiles:

- [FAIRChem ASE inference](https://facebookresearch.github.io/fairchem/ase-calculator/): predictor and task selection; fast defaults can change precision.
- [MACE foundation models](https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html): MACE-MH-1 and explicit heads.
- [SevenNet](https://github.com/MDIL-SNU/SevenNet): ASE calculator and Omni model release.
- [MatGL](https://github.com/materialyzeai/matgl): PyG models and ASE PESCalculator.

The local adapters explicitly disable TF32, Nitro, compilation, optional fused
equivariant kernels and added dispersion so that their settings agree with the
stored reference calculation receipts. The scoring output records exact weights,
package versions, task/head and device. An API success is not DFT validation.

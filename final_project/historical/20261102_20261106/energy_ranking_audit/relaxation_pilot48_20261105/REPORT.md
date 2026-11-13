# 100-structure, five-model relaxation performance pilot

500/500 task bundles have terminal records; 0 missing.

[Performance figure](performance_comparison.pdf) · [Lithiation dependence](performance_by_lithiation.pdf) · [Summary CSV](model_summary.csv) · [All results](per_structure_results.csv)

| model    |   Converged / 100 |   EFS mean (s) |   EFS median (s) |   Relax mean (s) |   Relax median (s) |   Mean FIRE steps |
|:---------|------------------:|---------------:|-----------------:|-----------------:|-------------------:|------------------:|
| uma      |               100 |          0.095 |            0.094 |            3.819 |              3.859 |            37.690 |
| mace     |               100 |          0.819 |            0.067 |            7.308 |              2.676 |            34.810 |
| sevennet |               100 |          0.243 |            0.225 |            6.608 |              6.772 |            28.360 |
| chgnet   |               100 |          0.032 |            0.032 |            1.507 |              1.525 |            38.910 |
| m3gnet   |               100 |          0.033 |            0.033 |            1.473 |              1.377 |            32.930 |

Models: UMA-S-1.2.1 (`omat`), MACE-MH-1 (`omat_pbe`), SevenNet-Omni (`mpa`), CHGNet-PES-MatPES-PBE-1M-2026.9, and M3GNet-PES-MatPES-PBE-2025.2. Checkpoint hashes are frozen in checkpoints/receipt.json.

Each task contains three uncached initial E/F/stress evaluations and one fixed-cell ionic relaxation. The initial model warm-up is excluded. CUDA is synchronized for timing. All models use native FP32 with TF32 disabled and explicit conservative calculator paths; these are controlled settings, not each package's maximally optimized throughput. FIRE maximum step=0.15 Å, maximum force target=0.05 eV/Å, step cap=500. This target is a provisional timing criterion, not a DFT accuracy or stability certificate. No random rattling is applied.

The dataset spans all 49 integer Li counts, with one mask at 0, 1, 47, 48; two at every other composition; and a third at 6, 12, 18, 30, 36, 42. Mean size: 168 atoms (range 144–192). Raw fixed-count random masks are symmetry-deduplicated, not sampled uniformly over symmetry orbits. See manifest.csv and parent_symmetry.json for provenance. The witness indexes the sorted permutation list; it is not directly an index into the separate spglib rotation/translation arrays.

Arithmetic means weight the 100 selected structures equally. EFS means and medians describe all 300 timed initial-geometry calls per model. Relaxation summaries describe 100 trajectories per model. composition_summary.csv and the macro-composition mean in model_summary.csv provide alternate composition-balanced summaries. The three individual call times are retained for every structure. MACE's mean times for calls 1, 2, and 3 are 0.114, 2.277, and 0.065 s: excluding only the first call does not remove its timing spikes. The [timing diagnostic](single_point_timing_diagnostic.pdf) shows this effect. Its cause is not established, so none of these values is advertised as a definitive steady-state benchmark. All observed spikes remain in the mean; medians describe typical observed performance.

hardware_summary.csv separates observed A100 40 GB and 80 GB variants. Models received different mixtures of GPUs and compositions through dynamic scheduling, so this is a practical campaign measurement rather than a matched hardware comparison. Model loading and initial warm-up are recorded separately and excluded from these task times. Relaxation wall time includes optimizer and trajectory I/O; calculator-only time is also saved. Shared filesystem and scheduling conditions can affect timings. All 500 tasks converged; no step-limit or exception cases occurred.

All starting cells are identical and remain fixed. Residual stress need not vanish. Force convergence does not prove a stable minimum, especially for symmetry-preserving endpoints. Displacements identify structures for further inspection; they are not a validated O3/other-phase classifier. Relaxed endpoints from different models are different geometries. Their absolute energies cannot establish DFT accuracy or rank agreement with DFT. cross_model_ranks.csv is descriptive only: most composition groups have 2–3 structures.

Per-task initial.npz/final.npz preserve E/F/stress, coordinates and cells; relaxation.traj retains optimizer snapshots; optimizer.log and result.json preserve failures and stopping outcomes. Energies are eV/cell, forces eV/Å, ASE tensile-positive stresses eV/Å³ in xx,yy,zz,yz,xz,xy order. No VASP, training, or MD was run.

Checkpoint identity, package/precision configuration, worker hardware, and load/warm-up times are retained in checkpoints/receipt.json and workers/. validation/ retains numerical derivative and repeated-supercell checks. resources.json records allocations separately from summed task times.

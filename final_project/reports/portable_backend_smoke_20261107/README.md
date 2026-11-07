# Optional-backend CPU forward check, November 7

All three optional providers completed one single-point prediction through the
normal project CLI and its backend-environment dispatcher. The same unrelaxed
O3-derived Li36Co48O96 geometry was used in each case. This is an engineering
check of installed dependencies and the E/F/stress interface, not a new training,
ranking, DFT-accuracy, relaxation, or MD benchmark.

| Model / selected head | Backend package | Result | Total command wall time |
| --- | --- | --- | ---: |
| SevenNet-Omni / mpa | sevenn 0.13.0 | PASS | 33.14 s |
| CHGNet MatPES-PBE 1M / MatPES-PBE | matgl 4.1.4 | PASS | 14.61 s |
| M3GNet MatPES-PBE 2025.2 / MatPES-PBE | matgl 4.1.4 | PASS | 8.72 s |

Each command used CPU, four threads, FP32 and the existing pinned checkpoint.
Times include process dispatch, hash verification, imports/model loading and
the single-point operation; they are not steady-state inference throughput.
There were no new allocations, GPU calls, DFT calculations, installations or
core adapter/requirement changes. A 300-second cap applied to each model.

Each returned one finite energy, 180×3 finite forces and one six-component stress
under the declared CLI contract: eV/cell, eV/Å, and eV/Å³ in ASE tensile-positive
`xx yy zz yz xz xy` order. An independent array comparison confirmed identical
atomic numbers, coordinates and cell before/after every calculation. This does
not independently validate the models' physical stress convention or accuracy.

The tested environments used CPU PyTorch 2.13.0, NumPy 2.3.4, ASE 3.26.0 and
torch-geometric 2.7.0. SevenNet used e3nn 0.6.0; MatGL used pymatgen-core
2026.10.31. Optional compiled PyG extensions were absent but were not required by
these tested forward paths. SevenNet emitted its expected warning that tensor
product accelerators were disabled. No lazy-dependency error occurred. GPU
execution/parity and other structure/element paths are outside this check.

`receipt.json`, model-specific receipts, logs, `summary.csv`, and
`independent_geometry_check.json` preserve the outcomes. Each model directory
contains the normal CLI's raw E/F/stress NPZ and receipts. Checkpoint hashes and
package versions are recorded; model weights are excluded. The input POSCAR is
copied under `input/` with identical bytes to the actual tested file.

On another machine, use the normal installer and interface:

```bash
bash scripts/install.sh --python python3 --prefix .venvs --backends sevennet,matgl
source .venvs/learning/bin/activate
for model in sevennet chgnet m3gnet; do
  cathode-cvae mlip score --model "$model" --device cpu --threads 4 \
    --input reports/portable_backend_smoke_20261107/input/basin_1_deletion.vasp \
    --output "runs/new_backend_smoke/$model"
done
```

Use a permitted compute allocation when the local site requires one. No account,
queue, hostname or GPU scheduler is a package default. The recorded check reused
existing verified weights through explicit `--checkpoint` paths because this
machine's default `.models` cache was empty; it did not retest network downloads.
`run_smoke.py` accepts explicit environment/checkpoint manifests and records the
exact executed commands without embedding a site account or scheduler command.

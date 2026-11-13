# Frozen MLIP comparison — November 4, 2026

Start with [REPORT.md](REPORT.md), [the main comparison](comparison/model_comparison.pdf),
or [the Excel tables](comparison/MLIP_comparison_tables.xlsx).

Every model evaluates the same 524 historical and 10,932 unique Hu geometries.
The 4,012 redundant Hu entries remain mapped in `raw_frame_map.csv`.
No relaxation, training, MD, or new DFT was performed.

`comparison/side_by_side_structure_ranks.csv` has one row per geometry and all
three model ranks/energies. Rank 1 means lowest energy within its DFT source,
phase, and composition. `comparison/ranking_metrics.csv` additionally includes
cohort and historical temperature/pressure breakdowns. The force/stress arrays
are preserved under `results/<model>/shard_*/chunk_*.npz`.

The frozen checkpoints, private environments, numerical checks, and terminal
allocation records are documented in `REPORT.md`, `environment_lock.json`,
`execution.json`, `resources.json`, and the worker receipts.

The inference source actually executed is preserved in `executed_sources/`.
The current `benchmark.py` also caches each input NPZ array once and checks
reference/model signatures before reuse; these changes do not alter predictions.
Do not regenerate the immutable reference export over these results.

To regenerate the comparison from the saved predictions, use an allocated
compute node with the recorded NumPy/Pandas/SciPy/Matplotlib environment:

```bash
python analyze_models.py
python write_report.py
```

Analysis reads the existing reference and worker caches. It verifies complete,
non-overlapping coverage and reference/model signatures, and requires a compute
allocation for the full analysis. Figures are exported as both PDF and PNG.
The small Excel dependency is isolated under `analysis_deps/`.

All four audit allocations are released. The full-node and shared-node fractions
are kept separate in `resources.json`; unrelated user jobs were not modified.

The independent raw-input audit passed 2,515,436 field comparisons across the
16 main/matched result tables, with zero mismatches. The largest aggregate
floating-point difference was 1.11e-16. Integer counts, canonical identities,
measured labels and exact boundary ties agree exactly. The arithmetic comparison
allows at most 8*eps64*max(1,|a|,|b|) for floating-point aggregates; it does not
move a bin boundary or relax a scientific success criterion.

The checker reads the frozen attempt CSVs, raw occupancy arrays, actual parent
operations, exact candidate masks/IDs, teacher scalar score archives, protected
TRAIN identities, frozen history and teacher-specific schemas. It independently
implements exhaustive 192-operation canonicalization, candidate/count identity,
energy division by actual atom count, q20/q80 classification, original-q20 tests,
separate training/history novelty, distinct-success/all-attempt denominators,
confusion matrices, per-seed/macro aggregation, and attempt/query-budget curves.
It imports no production evaluator or condition-classification helpers. Root's
attempt-outcomes table is checked as an output, not trusted as an input.

The full comparison covers 34,560 UMA attempts. The audit subset covers exactly
20,736 attempts for each of UMA and MACE, selected only by the family membership
declared before generation and label access. Both teachers assess the same
structures using their own frozen training-derived bins. All ten full-assessment
families have three independent fit or sampler replicates. The checker validates
the frozen proposal hashes and preserves every attempt. Repeated proposals are
penalized within each composition/request/seed group; group totals must not be
summed and interpreted as globally distinct discoveries across models.

The independent recomputation confirms these final means:

| Teacher/scope | Model | Balanced useful yield | Requested-low original-q20 yield |
|---|---|---:|---:|
| UMA full | Graph encoder with causal context | 62.0370% | 63.5417% |
| UMA full | Frozen mutation | 58.6227% | 52.5174% |
| UMA full | Dense D4 CVAE | 31.0475% | 30.2951% |
| UMA full | Uniform | 34.1435% | 19.9653% |
| MACE matched | Graph encoder with causal context | 63.4549% | 67.6215% |
| MACE matched | Frozen mutation | 57.2917% | 52.9514% |

The retained architecture was selected using DEVELOPMENT before these labels
were opened; this audit changes no model choice. The graph-context latent
diagnostics remain nearly inactive, so the positive result supports its
condition-sensitive spatial autoregressive sampling. It does not establish a
benefit from its variational latent pathway or DFT/relaxed-geometry accuracy.

All 76,016 imported UMA cache IDs and 51,441 imported MACE cache IDs are already
in the 278,382-orbit frozen history. There are no unrecognized IDs or missing
known-history orbits. The separate history-only check accessed no energy arrays
and is retained in `../final_cache_history_check/cache_history.json`.

Actual commands, run on CPUs inside allocation 60985189 with CUDA hidden:

```bash
cd /global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_next_goal_worktree/final_project
srun --jobid=60985189 --overlap --exact -N1 -n1 -c4 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_next_goal_worktree/final_project/src /global/cfs/cdirs/m5397/rsb/2026/classes/ECE285/final_project/energy_ranking_audit/multimodel_20261104/envs/uma/bin/python scripts/conditioning_20261107/check_final_metrics.py --self-test
srun --jobid=60985189 --overlap --exact -N1 -n1 -c4 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_next_goal_worktree/final_project/src /global/cfs/cdirs/m5397/rsb/2026/classes/ECE285/final_project/energy_ranking_audit/multimodel_20261104/envs/uma/bin/python scripts/conditioning_20261107/check_final_metrics.py --history-only
srun --jobid=60985189 --overlap --exact -N1 -n1 -c4 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_next_goal_worktree/final_project/src /global/cfs/cdirs/m5397/rsb/2026/classes/ECE285/final_project/energy_ranking_audit/multimodel_20261104/envs/uma/bin/python scripts/conditioning_20261107/check_final_metrics.py
```

All commands completed successfully. The full audit used step 60985189.106 and
19.72 CPU seconds. It performed no new training or MLIP evaluations. Input hashes
were unchanged. `receipt.json` binds every source and verified output; small
independently recomputed tables are saved alongside it. Existing outputs cannot
be overwritten; a reproduction needs a new explicit output directory.

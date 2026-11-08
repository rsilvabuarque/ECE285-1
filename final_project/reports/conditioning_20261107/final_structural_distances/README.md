This label-free audit preserves all 34,560 frozen final attempts and checks all
192 retained parent symmetries. For each composition, the reference contains
1024 protected TRAIN orbits selected by a fixed SHA256 rule, with identities
written before distance computation. Distances are minima over the actual
symmetry images, rather than differences between canonical representatives.
They are upper bounds on nearest distance to the complete TRAIN set. A positive
distance to this subset alone does not establish novelty or separation from all
training examples. The main evaluator supplies exact training/history novelty.

Mean reference distances are 6.858 differing sites for graph context, 6.850 for
dense D4, 6.891 for uniform and 6.498 for mutation. The similar neural/uniform
values do not show a substantial diversity gain. Mutation's mean distance from
its actual TRAIN seed is 4.037 sites in the preserved raw orientation and 4.034
after minimizing over parent symmetries. All 3456 mutation children implement
exactly the recorded one/two/three swaps: raw Hamming distance equals 2k, and
the actual edited sites agree with the source records. That seed supplies a
better bound on full-TRAIN distance than the 1024-reference bound for some
mutation proposals.

The main audit completed in 11.46 CPU seconds in step 60985189.99, using three
CPU workers and no GPUs, energy fields, MLIP calls or training. Every input hash
remained unchanged. Independent direct-bit checks cover all 256 pairs of four-bit
test vectors, and all final distances have the correct parity and fixed-count
bound. Tables retain 270 composition/bin/seed groups and 30 method/seed summaries;
all ten families have three replicates. The plot shows individual replicate
means without a proposal-level confidence interval. See METHOD.md for the exact
distance definitions and limitations.

```bash
cd /global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_next_goal_worktree/final_project
srun --jobid=60985189 --overlap --exact -N1 -n1 -c4 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_next_goal_worktree/final_project/src /global/cfs/cdirs/m5397/rsb/2026/classes/ECE285/final_project/energy_ranking_audit/multimodel_20261104/envs/uma/bin/python scripts/conditioning_20261107/audit_final_distances.py --reference-size 1024 --max-seconds 300
srun --jobid=60985189 --overlap --exact -N1 -n1 -c4 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_next_goal_worktree/final_project/src /global/cfs/cdirs/m5397/rsb/2026/classes/ECE285/final_project/energy_ranking_audit/multimodel_20261104/envs/uma/bin/python reports/conditioning_20261107/final_structural_distances/plot_distances.py
```

Both commands succeeded and refuse overwriting existing outputs. The first
implementation attempt failed before any distances were calculated because an
integer property was called as a function. Its 2.22-second failure receipt,
unchanged reference identities and executed source are preserved separately in
`../final_structural_distances_initial_failure/`. The repair changed no physical
protocol, reference selection, frozen final proposal, or model recommendation.

The confirmed development comparison favors the graph encoder with causal graph
context, while its latent variables have little measured effect. The context
provides the autoregressive decoder with neighboring occupied, vacant and
unassigned site fractions from the already generated prefix. It must not expose
current or future occupancy. The separate trained causality audit passed all
three retained checkpoints; its receipt is `../trained_causality_audit.json`.

All 13,824 proposals have exact UMA coverage: 10,368 primary proposals and 3,456
diagnostic proposals. There are no unknown energies or invalid masks. The
diagnostics contain 6,912 condition pairs, 2,304 latent pairs and 576 matched
posterior/prior pairs. All 1,152 posterior/prior target distances were verified
against the saved raw masks and exact protected validation identities. The
manifest combines the original spatial generation and the two confirmation
seeds; fit diagnostics cover exactly the 12 retained neural fit folders.

The independent audit in `../../../artifacts/conditioning_20261107/architecture_selection.json`
recomputed distinct novel requested-bin yield from the primary proposals and
reproduced all 12 balanced/low metrics in the main evaluator. Graph context is
the unique maximum under the frozen development selection rule. Its balanced
yield is 60.880%, compared with 58.796% for mutation, 34.491% for the graph encoder
alone, 32.755% for uniform, 32.118% for symmetry augmentation, and 31.829% for the
dense CVAE. Requested-low yield against the original fixed q20 threshold is
61.111% for graph context and 50.347% for mutation. These are DEVELOPMENT
observations; they do not establish an improvement on the untouched final pool.

| Model | Low to middle shift (meV/atom) | Low to high shift (meV/atom) | Positive low to high shifts | Resampling latent: differing sites |
|---|---:|---:|---:|---:|
| Dense | -0.135 | 0.026 | 43.23% | 12.722 |
| Symmetry augmentation | -0.104 | 0.026 | 40.63% | 13.698 |
| Graph encoder | 0.121 | 0.259 | 46.70% | 0.677 |
| Graph encoder and causal context | 2.100 | 5.263 | 97.40% | 0.795 |

Condition comparisons change the requested bin while holding the Gaussian
latent vector and autoregressive uniforms fixed. Latent comparisons resample
the latent while holding the uniforms and condition fixed. The displayed
averages give equal weight to each of the three compositions within each of
three fitted seeds; they do not treat proposals as independently fitted models.
Mutation and uniform controls are paired by attempt index, without the neural
models' shared conditional streams. Their responses remain in the raw tables
and figures as controls.

Graph-context validation KL totals are 0.01099, 0.00745 and 0.00912 nats for the
three selected checkpoints, and all have zero components with pooled posterior
mean variance above 0.01. Resampling their latent retains the same symmetry orbit
in 68.23% of paired draws. The posterior/prior target Hamming distances are
16.778/16.806 sites, respectively; target energy MAEs are 2.101/2.089 meV/atom.
This supports a useful condition-sensitive spatial autoregressive sampler with
little observed latent contribution. It does not support claiming that its
latent representation explains the gain. The variance cutoffs are descriptive
dimensionless diagnostics, not physical accuracy tolerances.

`family_diagnostic_summary.json` retains the full seed values and weighting
definitions. `paired_diagnostics_by_family_seed.csv` retains composition-averaged
paired results for each fitted seed. All composition-resolved results and raw
proposal comparisons remain in the original CSV files. The energy distributions
and paired condition-response figures show the substantial condition response
of the graph-context model. `selected_latent_diagnostics.csv` and
`latent_components.csv` show the nearly inactive graph latent dimensions.

Executed on CPUs inside allocation 60985189, with CUDA hidden:

```bash
cd /global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_next_goal_worktree/final_project
srun --jobid=60985189 --overlap --exact -N1 -n1 -c4 env CUDA_VISIBLE_DEVICES= PYTHONPATH=/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_next_goal_worktree/final_project/src /global/cfs/cdirs/m5397/rsb/2026/classes/ECE285/final_project/energy_ranking_audit/multimodel_20261104/envs/uma/bin/python scripts/conditioning_20261107/analyze_spatial_confirmed.py
srun --jobid=60985189 --overlap --exact -N1 -n1 -c4 env CUDA_VISIBLE_DEVICES= PYTHONPATH=/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_next_goal_worktree/final_project/src /global/cfs/cdirs/m5397/rsb/2026/classes/ECE285/final_project/energy_ranking_audit/multimodel_20261104/envs/uma/bin/python reports/conditioning_20261107/spatial_confirmed_diagnostics/summarize.py
```

Both commands completed successfully. Sources and inputs are bound by the saved
receipts and architecture-selection hashes. The scripts refuse to overwrite
their outputs. No new training, optimizer updates, MLIP evaluations or final
assessment were performed by this analysis.

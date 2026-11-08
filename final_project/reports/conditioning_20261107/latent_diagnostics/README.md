# Latent-dimension development comparison

This stage compares four genuine CVAEs (latent dimensions 4, 8, 16 and 32) and
a separate autoregressive model without latent variables. Each method has three
fit seeds and 4,000 optimizer updates. Dimension-8 fits are reused from the
selected training policy; twelve new trajectories supply the other four arms.
All arms use the selected class-balancing policy and validation beta 0.1.

`receipt.json` binds 16,416 completely scored attempts: 12,096 primary and 4,320
diagnostic proposals. The paired diagnostics contain 8,064 condition pairs,
2,880 latent pairs and 720 posterior/prior pairs, with 1,440 raw validation-target
distances independently checked. Fit aggregation is restricted to the frozen
manifest's fifteen directories. Exact allocated commands and the executed
analyzer source are preserved here.

`latent_selection_audit.json` independently reproduces all fourteen balanced and
original-q20 core summary values and the frozen dimension-selection rule.
Dimensions 4 and 32 satisfy its joint competitive margins; the smaller dimension
4 is retained. These margins are a declared practical heuristic, not confidence
intervals or physical tolerances.

| Generator | Balanced useful yield | Requested-low/original-q20 yield |
| --- | ---: | ---: |
| AR without latent | 34.32% | 22.05% |
| CVAE, dimension 4 | 32.64% | 33.16% |
| CVAE, dimension 8 | 29.98% | 29.34% |
| CVAE, dimension 16 | 28.82% | 27.43% |
| CVAE, dimension 32 | 32.47% | 31.42% |
| Uniform fixed-count | 33.04% | 19.62% |
| Requested-bin mutation | 59.72% | 52.08% |

The no-latent AR control has higher observed balanced performance than these
CVAEs, while dimension-4 CVAE generation has better low-tail performance. Do not
conflate those outcomes. Mutation remains substantially stronger for both.
This is development evidence, not an untouched final assessment.

The latent vectors affect generation, but a larger dimension does not give
monotonically better useful yield. Validation reconstruction improves from
approximately 17 nats for dimension 4 to 14.9 for dimension 32, without a useful
yield improvement. The KL term was summed across components throughout training;
per-component KL in the diagnostic table is a reported statistic, not a changed
training normalization. Active-component cutoffs are dimensionless descriptive
variance thresholds, with pooled and within-composition statistics kept separate.

The no-latent model is not a zero-dimensional CVAE. Its engineered null probes
behave exactly as expected: all nine latent-resampling groups have zero mask and
energy differences, and all 144 matched posterior/prior pairs are identical.
An ordinary stochastic autoregressive decoder can generate varied masks without
a latent variable.

`latent_dimension_summary.pdf`/`.png` show individual fit seeds and sampler mean
guides. `latent_fit_diagnostic_summary.csv` preserves individual fit measurements;
`selected_latent_diagnostics.csv` and `latent_components.csv` contain both
checkpoint rules and complete component diagnostics. `summary_figure_receipt.json`
records the visually checked compact figure. No confidence interval is inferred
from proposal draws, and no DFT accuracy or physical convergence is established.

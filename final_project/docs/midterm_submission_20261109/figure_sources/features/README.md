This self-contained figure uses saved, frozen November 8 results only. Run
`python make_feature_figure.py` in a core environment with NumPy and Matplotlib.

The vector PDF is exactly 5.5 inches wide and 2.15 inches tall. The PNG provides
an image preview at 240 dpi. Panel (a) reconstructs three independent-fit macro
yields from seed_metrics.csv, then checks the means and sample SD against the
saved mechanism_first3.csv. Each fit macro averages the nine composition/request
cells equally. Points show fits 2026110801–03. Panel (b) uses five fit/sampling
replicate means and sample SD for each requested energy class, averaging three
compositions. Mutation is stronger in the middle class; the graph CVAE and
spatial AR controls have similar macro yield.

All error bars are descriptive sample SD, not confidence intervals. Mutation's
replicates are sampling replicates, not trained models. D's scrambled association
preserves global degrees/weights and parameter count but changes causal-prefix
exposure; do not call it a pure locality/equivariance ablation. The graph and
dense posterior encoders have different parameter counts. Shell features also
add 576 GRU parameters relative to the six-feature baseline.

full_fivefit_table.csv additionally contains six methods evaluated with both
teachers and separately named original fixed-q20 LOW results. Never interchange
that LOW metric with the request-bin LOW metric in panel (b). Every teacher has
its own frozen class schema. Labels are teacher energies, not DFT validation.

source_registry.json binds the copied inputs to their original paths and SHA256
hashes. receipt.json binds the outputs and plotting script. No model, MLIP, DFT,
geometry relaxation or new proposal was run. The copied inputs make plotting
reproducible away from the original filesystem.

The historical reference fits use `training_round0.npz`, `kind=ar_cvae`, energy
conditioning, canonical orientation, latent dimension 8, GRU hidden dimension 64,
batch size 512, Adam learning rate 0.001, beta ramping to 0.1 over 20 epochs,
maximum 100 epochs/600 seconds, minimum 20 epochs, and patience 15. Checkpoint
selection minimizes validation reconstruction + KL (validation beta 1), with two
fixed validation noise samples. The three fit seeds are 20261106–20261108.
Use the immutable run configuration in the data bundle as the full receipt.

A controlled adaptation test uses the same starting checkpoint, seed, beta,
validation beta, original protected validation rows and optimizer-update budget
for every arm. Compare one change at a time: expanded development training rows,
then class balancing. For example, set `--max-updates 500 --epochs 1000
--max-seconds 3600 --beta-final .1 --beta-warmup 20 --validation-beta 1`, using a new
output directory for every arm. Confirm the update cap is actually reached in
`summary.json`; do not compare an early-stopped arm with a budget-capped arm as an
isolated equal-compute test. The exposed trainer retains epoch-based beta warmup
and validation, so equal update counts alone do not hold warmup exposure fixed
when dataset sizes differ. To isolate extra data, use `--beta-warmup 1` after the
warm start, keep beta fixed in all arms, and evaluate the same protected validation
subset at common update intervals (this last scheduling modification is a
collaborator task, not an implemented promise).

The previously completed round1 update coupled additional data, changed train-only
energy thresholds, class balancing and warm starts. Its decline is not evidence
that extra data alone is harmful. The existing final oracle pool is frozen
historical evidence; after looking at it, do not tune on it and call it untouched.
A new model can be compared on a held-out labelled mask set without new SCFs,
but generation outside that set needs fresh physical labels for energy-yield claims.

The displacement prototype uses `geometry_uma_fixed.npz` with fixed parent splits,
CVAE latent dimension 16, or a matching deterministic regressor, batch size 128,
Adam 0.001, beta ramping to 1 in 20 epochs, maximum 150 epochs, checkpoint eligibility
from epoch 20, early stopping after epoch 50/patience 25. For a short installation
smoke fit, `--epochs 2` makes the last smoke epoch checkpoint eligible and records
that deviation. Such a smoke fit is not a scientific reproduction.

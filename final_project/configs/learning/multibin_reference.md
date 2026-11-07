# Multibin reference fit and matched controls

The multibin model extends the existing energy-conditioned AR CVAE. Its input
condition is `[n_target/n_sites, onehot(low,middle,high)]`. The encoder and decoder
both receive these four components. The exact-count recurrent likelihood and
sampling rule are unchanged. The binary control keeps two components
`[n_target/n_sites, is_low]`; its middle/high requests both map to `rest`. The
composition-only control keeps one component and explicitly ignores energy class.

Default edges are the unweighted nearest-rank training 20th and 80th percentiles,
separately at each target count. Equal boundary energies stay together. Empty
training bins are unsupported requests, rather than being forced nonempty or
silently redirected. A condition schema contains empirical support, source hashes,
teacher/head/checkpoint identity, geometry protocol and a content SHA256 identity.
One JSON schema is shared by training, generation, acquisition and evaluation.

For the controlled three-composition comparison use `training_round0.npz`, counts
12/24/36, canonical orientation, latent dimension 8, GRU hidden dimension 64,
batch size 512, Adam learning rate .001, beta .1 after the 20-epoch warmup,
validation beta 1 and two fixed validation-noise draws. Use the same epoch,
optimizer-update and stopping policy across the compared fits; do not attribute
data/weighting/schedule changes to the condition representation alone. Three
independent fit seeds are 20261107, 20261108 and 20261109 unless the actual campaign
configuration declares different ones. Root's campaign receipts identify which
settings actually ran; this file does not assert an unexecuted fit is complete.

Example from `final_project`, inside a permitted compute allocation:

```bash
cathode-cvae train-mask \
  --data artifacts/current/improvement_v1/training_round0.npz \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --kind ar_cvae --condition-mode multibin --counts 12 24 36 \
  --teacher uma --teacher-head omat --orientation canonical \
  --schema-quantiles .2 .8 --beta-final .1 --validation-beta 1 \
  --epochs 100 --minimum-epochs 20 --patience 15 --max-seconds 900 \
  --seed 20261107 --output-dir runs/multibin_seed20261107

cathode-cvae generate-mask \
  --checkpoint runs/multibin_seed20261107/best.pt \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --n-li 24 --requested-bin middle --count 128 --seed 20261117 \
  --temperature-K 433.15 --allocation-policy static-MLIP-exploration-0.1 \
  --output runs/multibin_seed20261107/middle_433K.npz
```

The temperature annotates the acquisition policy. It does not change the latent
normal prior or introduce a decoder softmax-temperature parameter. To freeze one
partition across fits, pass `--condition-schema-in FIRST_RUN/condition_schema.json`.
A new checkpoint embeds the schema and checks its encoding dimension. Historical
binary and composition checkpoints remain loadable; incompatible binary weights
cannot silently be loaded into a four-component model.

Each `.attempts.csv` row records requested class, effective class, schema identity,
composition, fit seed, sampling seed, physical temperature, allocation policy,
checkpoint hash, parent hash and geometry protocol. Duplicates remain attempts.
Unsupported requests preserve failed-attempt rows and return a failure status.
The `.unique.npz` holds canonical geometries for deduplicated scalar scoring;
forces belong only to the evaluated orientation. Root's condition-aware evaluator
must retain these request groups rather than merging them into historical groups.

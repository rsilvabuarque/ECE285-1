# Fresh development comparison

This adds a key-only boundary and exact source/checkpoint manifests around the
existing proposal generator, oracle registry and scalar evaluator. It does not
train a model, score a structure, submit a job or read historical FINAL energy
members. Run the commands on an allocated compute node with the repository's
matching environment, and use an unused output location. The CLI intentionally
honors the project's compute-allocation guard.

`known_history.json` is the union of the frozen historical novelty set, the
receipt-certified earlier candidate pools and registered physical identities.
`protected.json` additionally identifies existing H0, round-one and paired
validation/test masks, editor validation/held-out masks, all earlier FINAL
candidates and prior latent-development candidates. Original H0 TRAIN rows are
unchanged. If earlier protected candidates coincide with H0 TRAIN, the boundary
receipt reports the inherited overlaps; they remain known training identities,
not novelty successes. H0 VAL may still be used under its original checkpoint
selection role, but protected identities cannot enter a **new** scoring pool or
new training/acquisition additions. This is a development experiment and no old
FINAL scores are used to choose it.

The plan is JSON with this shape (replace checkpoint paths, seeds, dimensions,
replicate IDs and contrast definitions with the frozen comparison). Each model
checkpoint's adjacent `configuration.json` must bind the original H0 data,
parent and request schema; checkpoints with format versions 3, 4 or 5 require
the matching strictly loading runtime source.

```json
{
  "status": "DECLARED_DEVELOPMENT_COMPARISON",
  "namespace": "fresh_development_20261110",
  "counts": [12, 24, 36],
  "requested_bins": ["low", "middle", "high"],
  "count_per_group": 64,
  "latent_stream_width": 32,
  "diagnostics": true,
  "models": [
    {"method": "uniform", "replicate_id": 1, "sampler": "uniform"},
    {"method": "mutation", "replicate_id": 1, "sampler": "mutation"},
    {"method": "cvae", "replicate_id": 1, "fit_seed": 2026110801,
     "checkpoint": "/absolute/path/to/best_validation_beta_0p1.pt"}
  ],
  "contrasts": [{"method": "cvae", "reference": "mutation"}]
}
```

Every model/reference arm in a paired contrast needs identical replicate IDs.
Repeat model entries for all independent fits. Counts are **remaining Li atoms**,
not vacancies. Freeze a plan before drawing or scoring. The source hashes are
from the worktree running `manifest`; generate from that same unchanged
worktree. The whole set of attempts is kept, including duplicates, failed
requests and protected identities, so filtering cannot inflate yields.

Example shell setup (`PY` must name an environment with the installed project,
NumPy and Torch; `OUT` must not contain prior evidence):

```bash
export PROJECT=/absolute/path/to/final_project
export PY=/absolute/path/to/python
export OLD=/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project
export PRIOR=/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_next_goal_worktree/final_project
export OUT=/absolute/path/to/new-comparison
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
"$PY" "$PROJECT/scripts/angular_20261110/development_boundary.py" freeze \
  --overnight-root "$OLD" --prior-root "$PRIOR" --output-dir "$OUT/boundary"
"$PY" "$PROJECT/scripts/angular_20261110/development_boundary.py" manifest \
  --plan "$OUT/plan.json" --boundary-dir "$OUT/boundary" \
  --thresholds "$OLD/artifacts/current/reports/generation_thresholds.csv" \
  --teacher-schema "uma=$OLD/artifacts/overnight_20261107/data/condition_schema_uma.json" \
  --teacher-schema "mace=$OLD/artifacts/overnight_20261107/data/condition_schema_mace.json" \
  --output "$OUT/manifest.json"
"$PY" "$PROJECT/scripts/angular_20261110/development_boundary.py" verify --manifest "$OUT/manifest.json"
"$PY" "$PROJECT/scripts/overnight_20261108/generate_controls.py" \
  --manifest "$OUT/manifest.json" \
  --parent "$OLD/artifacts/current/core/data/parents/O3_N48_pilot.npz" \
  --data "$OLD/artifacts/current/improvement_v1/training_round0.npz" \
  --condition-schema "$OLD/artifacts/overnight_20261107/data/condition_schema_uma.json" \
  --history "$OUT/boundary/known_history.json" --output-dir "$OUT/generated_raw" \
  --count 64 --diagnostics --device cuda
"$PY" "$PROJECT/scripts/angular_20261110/development_boundary.py" filter \
  --generated-dir "$OUT/generated_raw" --boundary-dir "$OUT/boundary" \
  --output-dir "$OUT/generated_eligible"
```

The count/diagnostics options must match the frozen plan. A combined manifest
runs generation sequentially; its common streams are checked across arms and
requests. There is no receipt-preserving shard merger. Parallelize oracle
workers rather than independently regenerating or concatenating rows.

Register `generated_eligible/candidates.npz` with the existing
`python -m cathode_cvae.pool_scoring register`, run one existing `worker` per
assigned GPU with the approved UMA/MACE checkpoints, then use `collect` to
produce complete teacher score files. Use the teacher-specific score/schema
pairs and the manifest's threshold file with `evaluate_controls.py`; pass the
eligible bundle, bound parent, expanded history and original H0 data. The
resulting `paired_attempts.csv` is accepted by
`summarize_paired_sensitivity.py`. That script requires complete paired fits;
if protected-boundary exclusions make pairs unavailable, report coverage and
incomplete fits, rather than silently dropping excluded attempts or claiming a
representative paired result. Query cost is the count of distinct physical
model/structure keys in the oracle registry, separately reporting cache hits,
failures and attempts. A registry key shared by arms counts once globally; GPU
node-hours come from the allocation accounting, not from the number of masks.

The included unit tests use only six synthetic masks and two simple symmetry
permutations: `python -m unittest discover -s final_project/tests -p
"test_angular_development_boundary.py" -v`. Passing them validates the small
bookkeeping example; it does not validate trained models or physical energies.

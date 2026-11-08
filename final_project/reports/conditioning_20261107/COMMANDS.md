# Reproduce the required packet

Run training and substantial analysis inside your site's compute allocation.
These commands do not choose a scheduler, account, or GPU. On other GPU hosts,
use the visible device assigned by that host. Installation resolves dependencies;
cached replay needs no MLIP installation or foundation weights.

```bash
git clone --branch midterm/conditioning-graph-edit-20261107 \
  https://github.com/rsilvabuarque/ECE285-1.git ECE285-midterm
cd ECE285-midterm/final_project
bash scripts/install.sh --python python3 --prefix .venvs --core-only
source .venvs/learning/bin/activate
python -m pip check
cathode-cvae data verify --snapshot artifacts/current
python - <<'PY'
import json, subprocess
from pathlib import Path
r = json.loads(Path('artifacts/conditioning_20261107/download.json').read_text())
subprocess.run(['cathode-cvae', 'fetch-artifact', '--url', r['url'],
                '--sha256', r['sha256'], '--output-dir', r['extract_to']], check=True)
PY
python scripts/conditioning_20261107/replay_cached_results.py \
  --packet artifacts/conditioning_20261107 \
  --output-dir runs/varad_required_replay --plots
```

Expected: `runs/varad_required_replay/replay_receipt.json` has `status: PASS`,
two jobs and 24 compared tables, zero MLIP calls and zero training updates.
The integrating run reproduced all tables byte-for-byte in a clean extraction.
The separate independent raw-input checker made 2,515,436 field comparisons;
its algorithm does not import the production evaluator's bin/metric helpers.

Print an exact, portable measured training command, inspect it, then run it in a
new directory on your assigned GPU:

```bash
python scripts/conditioning_20261107/replay_cached_results.py \
  --packet artifacts/conditioning_20261107 \
  --print-training-command graph_context_20261106 \
  --training-output runs/beihao_reproduction --device cuda
```

`execution/portable_training_commands.json` contains all 28 required fits and
their original task-manifest provenance. Each fit must actually complete 4000
updates; a zero process exit alone is insufficient. Both validation-selection
checkpoints come from one trajectory. Do not replace protected data or reuse an
already inspected assessment as an untouched test.

Build the report from the repository:

```bash
bash docs/midterm1/build.sh
```

This requires a LaTeX installation with the packages named in `main.tex` and
BibTeX/latexmk or pdflatex. NERSC users can first run `module load texlive`.
The result is five body pages plus references. The frozen required report is
also self-contained in `artifacts/conditioning_20261107/required_packet`; copy
that directory elsewhere before building, so the verified payload stays intact.

The original execution commands, complete arguments, source snapshots and
checkpoint hashes are in the release's `execution/` and `source/` directories.
Raw E/F/stress shard hashes are in `raw_evidence/manifest.json`; arrays remain in
the preserved HPC archive. The public source map translates local execution
paths into Git or release-payload locations without changing their hashes.

The required cost receipt is an as-of snapshot with one allocation still live.
The final completion receipt supplies post-release totals, including startup,
idle allocation, failures and optional work. No new DFT or MLIP fine-tuning is
part of this run. Collaborator responsibilities are in
[`COLLABORATOR_TASKS.md`](../../docs/COLLABORATOR_TASKS.md).

## Replay the optional companion without calculators

Download both ZIPs with the exact names and SHA256 values in
`artifacts/conditioning_20261107/download.json` and
`artifacts/conditioning_20261107/extensions_v2/download_ext.json`. Keep the ZIPs:
the optional replay authenticates both full member inventories before analysis.
From the installed learning environment, on your site's analysis allocation:

```bash
python scripts/conditioning_20261107/replay_extensions.py \
  --core-archive ECE285_conditioning_20261108_required.zip \
  --extensions-archive ECE285_conditioning_20261108_extensions_v2.zip \
  --download-ext artifacts/conditioning_20261107/extensions_v2/download_ext.json \
  --output-dir runs/varad_optional_replay
```

This recomputes the editor, posthoc latent proposal, temperature allocation,
best-of-budget and NEB summary tables from archived scalar labels/bands. It
performs zero MLIP calls and zero training updates. Original historical paths
remain in preserved protocols; a separate, explicit path-only replay view points
to authenticated files in the fresh mirror. Raw global force/stress shard bytes
remain on HPC and are bound by the archive inventory rather than downloaded.

The editor's complete generation/training commands are retained in its protocol,
fit tasks and checkpoint configurations in the optional ZIP. The exact matched
seeds, swap counts and sampling uniforms remain assessment inputs. A reproduction
must use a new output directory and preserve the protected orbit roles.

## Execution audit and post-release resource closure

The historical commands below refer to this run's owned Slurm allocation IDs;
they are receipts, not commands to acquire or reuse these allocations elsewhere.
The heavy audit requires an active compute allocation. Its original failed
preclosure audit, narrow filename-binding repair and independent exact checkpoint
alias proof are preserved. No checkpoint or training summary was rewritten.

```bash
env TZ=UTC sacct -X -P \
  -j 60985189,60985612,60985847,60986556,60988553,60989957,60990329 \
  --format=JobIDRaw,Account,AllocNodes,AllocTRES,State,Start,End,ElapsedRaw \
  > reports/conditioning_20261107/allocation_final.tsv

python reports/conditioning_20261107/audit_execution.py --root "$PWD" \
  --accounting reports/conditioning_20261107/allocation_final.tsv \
  --owned-jobs 60985189 60985612 60985847 60986556 60988553 60989957 60990329 \
  --finalize-from reports/conditioning_20261107/execution_heavy.json \
  --output reports/conditioning_20261107/execution_final.json
```

The finalizer checks metadata hashes, retained raw-payload stat signatures and
terminal allocation accounting after a separately recorded full raw-byte audit.
It loads no arrays or models. Final allocation totals include startup, idle time,
analysis and failures. NEB calculator calls have their own pilot receipt and are
not silently counted as pristine-deletion static labels.

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

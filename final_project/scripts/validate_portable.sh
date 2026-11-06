#!/usr/bin/env bash
# Installation and small operability checks only; no new MLIP labels or DFT/MD.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
validation_dir="${1:-$project_dir/runs/infrastructure_validation/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$validation_dir"
exec > >(tee "$validation_dir/validation.log") 2>&1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MAX_JOBS=4
printf 'Validation directory: %s\n' "$validation_dir"
validation_prefix="${CATHODE_CVAE_TEST_PREFIX:-$project_dir/.venvs}"
validation_python="${CATHODE_CVAE_TEST_PYTHON:-$(command -v python)}"
bash scripts/install.sh --python "$validation_python" --prefix "$validation_prefix" --no-weights \
  --torch-index-url https://download.pytorch.org/whl/cpu
export CATHODE_CVAE_TEST_PREFIX="$validation_prefix"
py="$validation_prefix/learning/bin/python"
cli="$validation_prefix/learning/bin/cathode-cvae"
"$py" -m pytest -q
"$cli" data verify --snapshot artifacts/current
"$cli" evaluate --snapshot artifacts/current --output-dir "$validation_dir/cached_evaluation"
"$cli" evaluate-geometry --snapshot artifacts/current --output-dir "$validation_dir/cached_geometry"
"$cli" data prepare-learning --labels artifacts/current/core/data/learning_O3_N48_pilot_uma.npz \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --splits artifacts/current/core/splits/lineage_splits.csv --weighting equal-composition \
  --output "$validation_dir/main_all_counts_uma.npz"
"$cli" data prepare-learning --labels artifacts/current/extras/learning_all_parents/learning_Hu_O3_12_uma.npz \
  --parent artifacts/current/extras/learning_all_parents/parents/Hu_O3_12.npz \
  --splits artifacts/current/core/splits/lineage_splits.csv \
  --output "$validation_dir/small12_uma.npz"
"$py" - "$validation_dir" <<'PY'
import csv, pathlib, sys
p=pathlib.Path(sys.argv[1])
for name in ['metrics.csv','seed_metrics.csv','summary.csv','teacher_agreement.csv']:
    reference=list(csv.DictReader((pathlib.Path('artifacts/current/improvement_v1/reports/final')/name).open()))
    actual=list(csv.DictReader((p/'cached_evaluation'/name).open()))
    assert reference==actual, f'Frozen {name} differs; investigate before publishing'
print('CACHED_METRICS_EXACT_MATCH')
PY
"$cli" train-mask --data artifacts/current/improvement_v1/training_original.npz \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz --kind ar_cvae \
  --device cpu --epochs 2 --output-dir "$validation_dir/mask_smoke"
"$cli" generate-mask --checkpoint "$validation_dir/mask_smoke/best.pt" \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --n-li 24 --count 4 --device cpu --output "$validation_dir/smoke_proposals.npz"
"$cli" generate-mask --checkpoint artifacts/current/improvement_v1/runs/ar_adapt_cvae_20261106/best.pt \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz \
  --n-li 24 --count 4 --device cpu --output "$validation_dir/frozen_proposals.npz"
"$cli" generate-geometry --checkpoint artifacts/current/core/checkpoints/cvae_20261400/best.pt \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz --input "$validation_dir/frozen_proposals.unique.npz" \
  --draws 1 --device cpu --output "$validation_dir/geometry_cvae.npz"
"$cli" generate-geometry --checkpoint artifacts/current/core/checkpoints/regressor_20261400/best.pt \
  --parent artifacts/current/core/data/parents/O3_N48_pilot.npz --input "$validation_dir/frozen_proposals.unique.npz" \
  --device cpu --output "$validation_dir/geometry_regressor.npz"
"$py" - "$validation_dir" <<'PY'
import importlib.metadata as md, json, os, pathlib, subprocess, sys, time
root=pathlib.Path.cwd(); out=pathlib.Path(sys.argv[1])
checks={}
for profile, module in [('uma','fairchem.core'),('mace','mace.calculators'),('sevennet','sevenn.calculator'),('matgl','matgl')]:
    interpreter=pathlib.Path(os.environ['CATHODE_CVAE_TEST_PREFIX'])/profile/'bin/python'
    result=subprocess.run([str(interpreter),'-c',f'import {module}; print("IMPORT_PASS")'],capture_output=True,text=True)
    (out/f'import_{profile}.log').write_text(result.stdout+result.stderr)
    checks[profile]={'import_ok':result.returncode==0}
    assert result.returncode==0, f'{profile} import failed; inspect saved log'
receipt=dict(status='PASS',time_unix=time.time(),backend_imports=checks,
             torch=md.version('torch'),numpy=md.version('numpy'),
             scope='Fresh CPU SDK installation/pip check, exact cached metric reproduction, numerical tests, short fit/reload and frozen checkpoint decoding. No new MLIP inference/DFT/MD.',
             gpu_prediction_parity='Not tested by this CPU validation')
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
PY

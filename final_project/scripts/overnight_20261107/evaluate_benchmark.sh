#!/usr/bin/env bash
set -euo pipefail
project_dir="$(pwd)"
py=/pscratch/sd/r/rsb/ece285_midterm_20261106/venvs/learning/bin/python
export PYTHONPATH="$project_dir/src" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"$py" - <<'PY'
import json,time
from pathlib import Path
r=Path('runs/overnight_20261107/oracle');tasks=json.loads((r/'registry.json').read_text())['tasks'];start=time.time()
while True:
 missing=[t['task_id'] for t in tasks if not (r/'scores'/t['model']/(t['task_id']+'.json')).exists()]
 if not missing:break
 if time.time()-start>900:raise RuntimeError(f'Bounded wait exceeded; {len(missing)} specific registered tasks still need inspection')
 print('WAIT_REGISTERED_TASKS',len(missing),flush=True);time.sleep(5)
print('REGISTERED_BENCHMARK_TASKS_COMPLETE',len(tasks),flush=True)
PY
"$py" -m cathode_cvae.pool_scoring collect --root runs/overnight_20261107/oracle --output runs/overnight_20261107/collected_benchmark
"$py" - <<'PY'
from pathlib import Path
import numpy as np,json
r=Path('runs/overnight_20261107/collected_benchmark')
assert json.loads((r/'receipt.json').read_text())['status']=='PASS'
for m in ['uma','mace']:
 p=r/f'{m}.npz'
 with np.load(p) as z:d={k:z[k] for k in z.files}
 d['provenance']=np.full(len(d['structure_ids']),'fresh')
 np.savez_compressed(r/f'{m}_with_provenance.npz',**d)
PY
"$py" -m cathode_cvae.condition_evaluation --attempts runs/overnight_20261107/benchmark/attempts.csv --candidates runs/overnight_20261107/benchmark/candidates.npz --scores uma=runs/overnight_20261107/collected_benchmark/uma_with_provenance.npz mace=runs/overnight_20261107/collected_benchmark/mace_with_provenance.npz --schemas uma=runs/overnight_20261107/data/condition_schema_uma.json mace=runs/overnight_20261107/data/condition_schema_mace.json --training-keys runs/overnight_20261107/data/training_keys.json --history runs/overnight_20261107/data/history_assessment.json --historical-thresholds artifacts/current/reports/generation_thresholds.csv --parent artifacts/current/core/data/parents/O3_N48_pilot.npz --output-dir runs/overnight_20261107/evaluation/benchmark --plots

#!/usr/bin/env bash
# Install in fresh, isolated environments; never alter the caller's environment.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_command="python3"
prefix="$project_dir/.venvs"
model_dir="${CATHODE_CVAE_MODEL_DIR:-$project_dir/.models}"
core_only=0
download_weights=1
torch_index_url=""

usage() {
  cat <<'EOF'
Usage: bash scripts/install.sh [options]

Default: learning + UMA + MACE + SevenNet + MatGL (M3GNet/CHGNet), all dependencies,
pip check, pinned model download and SHA256 verification. No jobs are submitted.

  --python PATH             Python 3.11–3.13 executable (3.12 recommended)
  --prefix DIR              Environment directory (default .venvs)
  --model-dir DIR           Model cache (default .models)
  --core-only               Install learning/report workflow only; no MLIP envs
  --no-weights              Install all packages but defer checkpoint download
  --torch-index-url URL     Official PyTorch wheel channel matching your driver
  --help                    Show this text

The installer fails on missing/incompatible dependencies. On a partial failure,
rerun it after correcting the reported issue. No --no-deps or system-site-packages
shortcuts are used. UMA access must be obtained by the actual model user.
EOF
}

while (($#)); do
  case "$1" in
    --python) python_command="$2"; shift 2 ;;
    --prefix) prefix="$2"; shift 2 ;;
    --model-dir) model_dir="$2"; shift 2 ;;
    --torch-index-url) torch_index_url="$2"; shift 2 ;;
    --core-only) core_only=1; shift ;;
    --no-weights) download_weights=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

"$python_command" - <<'PY'
import sys
if not (3, 11) <= sys.version_info[:2] <= (3, 13):
    raise SystemExit('Use Python 3.11–3.13; Python 3.12 is the verified campaign version.')
PY
mkdir -p "$prefix" "$model_dir"
prefix="$(cd -- "$prefix" && pwd)"
model_dir="$(cd -- "$model_dir" && pwd)"
profiles=(learning)
if (( ! core_only )); then profiles+=(uma mace sevennet matgl); fi

for profile in "${profiles[@]}"; do
  env_dir="$prefix/$profile"
  if [[ ! -x "$env_dir/bin/python" ]]; then
    "$python_command" -m venv "$env_dir"
  fi
  py="$env_dir/bin/python"
  "$py" -m pip install --upgrade 'pip>=24.2' 'setuptools>=68,<81' wheel
  if [[ -n "$torch_index_url" ]]; then
    "$py" -m pip install --index-url "$torch_index_url" 'torch==2.13.0'
  fi
  if [[ "$profile" != learning ]]; then
    "$py" -m pip install -r "$project_dir/requirements/$profile.txt"
  else
    "$py" -m pip install -r "$project_dir/requirements/learning.txt"
  fi
  "$py" -m pip install -e "$project_dir[dev]"
  "$py" -m pip check
  "$py" -m pip freeze > "$env_dir/installed-packages.txt"
done

"$prefix/learning/bin/python" - "$project_dir" "$prefix" "$model_dir" "${profiles[@]}" <<'PY'
import json,pathlib,sys
project,prefix,models=map(pathlib.Path,sys.argv[1:4])
profiles=sys.argv[4:]
config={'schema_version':1,'interpreters':{p:str(prefix/p/'bin/python') for p in profiles},
        'model_dir':str(models),'note':'Machine-local paths; regenerate after moving this checkout.'}
(project/'environments.json').write_text(json.dumps(config,indent=2)+'\n')
PY

if (( ! core_only && download_weights )); then
  "$prefix/learning/bin/python" -m cathode_cvae.mlip.cli fetch-models \
    --models uma mace sevennet chgnet m3gnet --model-dir "$model_dir"
fi

printf '\nInstalled environments under %s\nActivate: source "%s/learning/bin/activate"\n' "$prefix" "$prefix"
printf 'Backend paths recorded in %s/environments.json\n' "$project_dir"
if (( ! core_only && ! download_weights )); then
  printf 'Weights deferred. Next: cathode-cvae mlip fetch-models --models uma mace sevennet chgnet m3gnet\n'
fi

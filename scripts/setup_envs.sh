#!/usr/bin/env bash
# Create one isolated .venv per feature-extraction package (+ the TPOT benchmark)
# using uv.
#
# Prerequisite: uv  (https://docs.astral.sh/uv/). Install with:
#     curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Usage:
#     ./scripts/setup_envs.sh                # create all environments
#     ./scripts/setup_envs.sh tsfresh benchmark   # create only the listed ones
#
# Kats note: kats requires Python 3.7. uv will try to fetch a managed CPython 3.7.
# If that fails, install Python 3.7 and run:
#     uv venv .venvs/kats --python /path/to/python3.7
#     uv pip install -r requirements/kats.txt --python .venvs/kats
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_root="$root/.venvs"
mkdir -p "$venv_root"

declare -A py_versions=(
    [tsfresh]=3.12 [tsfel]=3.12 [pycatch22]=3.10 [seglearn]=3.12
    [tsfeatures]=3.12 [benchmark]=3.12 [kats]=3.7
)

if [ "$#" -gt 0 ]; then
    names=("$@")
else
    names=(tsfresh tsfel pycatch22 seglearn tsfeatures kats benchmark)
fi

for name in "${names[@]}"; do
    py="${py_versions[$name]:-}"
    if [ -z "$py" ]; then
        echo "Unknown environment name: '$name'. Valid: ${!py_versions[*]}" >&2
        exit 1
    fi
    venv="$venv_root/$name"
    req="$root/requirements/$name.txt"
    echo
    echo "==> [$name] creating venv (Python $py) at $venv"
    uv venv "$venv" --python "$py"
    echo "==> [$name] installing dependencies from $req"
    uv pip install -r "$req" --python "$venv"
done

echo
echo "Done. Environments created under $venv_root"
echo "Activate a venv with:  source .venvs/<name>/bin/activate"

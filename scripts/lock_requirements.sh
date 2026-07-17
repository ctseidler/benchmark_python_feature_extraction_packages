#!/usr/bin/env bash
# Freeze per-environment requirements into fully pinned lock files using
# `uv pip compile` (no interpreter/venv required to resolve).
#
# Produces requirements/<name>.lock.txt with all transitive dependencies pinned,
# resolved for the Python version each environment uses. Commit the lock files
# for full reproducibility.
#
# Usage:
#     ./scripts/lock_requirements.sh
#     ./scripts/lock_requirements.sh tsfresh benchmark
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
        echo "Unknown environment name: '$name'." >&2
        exit 1
    fi
    req="$root/requirements/$name.txt"
    out="$root/requirements/$name.lock.txt"
    echo "==> [$name] compiling lock for Python $py -> $out"
    uv pip compile --python-version "$py" "$req" -o "$out"
done

echo
echo "Done. Lock files written to requirements/*.lock.txt"

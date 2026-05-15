#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python -m pip install --upgrade build twine

rm -rf dist build *.egg-info
python -m build

REPOSITORY="${REPOSITORY:-pypi}"

if [[ "$REPOSITORY" != "pypi" && "$REPOSITORY" != "testpypi" ]]; then
  echo "Invalid REPOSITORY: $REPOSITORY (use pypi or testpypi)" >&2
  exit 1
fi

twine upload --repository "$REPOSITORY" dist/*

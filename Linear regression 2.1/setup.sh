#!/usr/bin/env bash
# One-time setup: creates .venv, installs packages, registers a Jupyter kernel, builds the dataset.
# Usage from the repository root: bash "Linear regression 2.1/setup.sh"
set -euo pipefail
cd "$(dirname "$0")"

PY=$(command -v python3 || true)
if [ -z "$PY" ]; then echo "python3 not found - install it with: brew install python"; exit 1; fi
echo "Using $($PY --version) at $PY"

$PY -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q
python -m ipykernel install --user --name hit140-lr21 --display-name "Python (hit140-lr21)" >/dev/null

python src/build_lr21_refined.py
echo
echo "Done. Next time just run:  source .venv/bin/activate"
echo "Open notebooks with:       jupyter lab"

#!/usr/bin/env bash
set -euo pipefail

# Optional venv
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt

# Train + evaluate + uncertainty + importance
python src/ald_rebuild_quickstart.py --config src/ald_rebuild_config.yaml

# Quick plots
python src/plot_quicklook.py --artifacts outputs

echo "Done. Artifacts in ./outputs"
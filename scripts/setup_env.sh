#!/usr/bin/env bash
# Create / update the project venv on macOS (M1, MPS) or Linux (lab, CUDA).
# Usage:  bash scripts/setup_env.sh
# Optional: TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash scripts/setup_env.sh
#   (only needed if check_env.py warns that the torch build has no kernels for the GPU)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3.12}"
command -v "$PY" >/dev/null || { echo "Need Python 3.12 (set PYTHON=/path/to/python3.12)"; exit 1; }

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip

if [ -n "${TORCH_INDEX_URL:-}" ]; then
  # install the CUDA build first; requirements.txt then sees torch==X as already satisfied
  TORCH_PIN=$(grep -E '^torch==' requirements.txt)
  VISION_PIN=$(grep -E '^torchvision==' requirements.txt)
  .venv/bin/python -m pip install "$TORCH_PIN" "$VISION_PIN" --index-url "$TORCH_INDEX_URL"
fi

.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/check_env.py
echo "Done. Activate with:  source .venv/bin/activate"

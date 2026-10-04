#!/usr/bin/env bash
# Install the lightweight runtime; --model enables the optional CPU research backend.
set -euo pipefail
INFERENCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_VENV="${VENV:-$INFERENCE_DIR/.venv}"
INFERENCE_PYTHON="${PYTHON:-python3}"
WITH_MODEL=0
WITH_DEV=0
for argument in "$@"; do
  case "$argument" in
    --model) WITH_MODEL=1 ;;
    --skip-model) WITH_MODEL=0 ;;
    --dev) WITH_DEV=1 ;;
    --recreate) echo 'Create a separate VENV path to preserve existing environments.' >&2; exit 2 ;;
    -h|--help) echo 'venv.sh [--dev] [--model] (default: lightweight runtime)'; exit 0 ;;
    *) echo "Unknown argument: $argument" >&2; exit 2 ;;
  esac
done
if [ ! -x "$INFERENCE_VENV/bin/python" ]; then
  "$INFERENCE_PYTHON" -m venv "$INFERENCE_VENV"
fi
"$INFERENCE_VENV/bin/python" -m pip install -r "$INFERENCE_DIR/requirements.txt"
if [ "$WITH_DEV" = 1 ]; then
  "$INFERENCE_VENV/bin/python" -m pip install -r "$INFERENCE_DIR/requirements-dev.txt"
fi
if [ "$WITH_MODEL" = 1 ]; then
  "$INFERENCE_VENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.6'
  "$INFERENCE_VENV/bin/python" -m pip install -r "$INFERENCE_DIR/requirements-model.txt"
fi
(cd "$INFERENCE_DIR" && "$INFERENCE_VENV/bin/python" -c 'import app.config, app.candles, model; print("Runtime imports OK; no weights loaded.")')

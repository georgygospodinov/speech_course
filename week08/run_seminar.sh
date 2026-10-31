#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

export JUPYTER_CONFIG_DIR="$PWD/.jupyter"
export JUPYTER_RUNTIME_DIR="$PWD/.jupyter/runtime"
export IPYTHONDIR="$PWD/.ipython"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
mkdir -p "$JUPYTER_RUNTIME_DIR"

.venv/bin/python -m ipykernel install --prefix "$PWD/.venv" \
  --name week08 --display-name "Speech week08"
if compgen -G 'logs/*/events.out.tfevents.*' > /dev/null; then
  .venv/bin/tensorboard --logdir logs --host 127.0.0.1 --port 6008 > logs/tensorboard.log 2>&1 &
  tensorboard_pid=$!
  trap 'kill "$tensorboard_pid" 2>/dev/null || true' EXIT
fi
.venv/bin/jupyter lab --ip 127.0.0.1 --port 8888 --no-browser visualize.ipynb

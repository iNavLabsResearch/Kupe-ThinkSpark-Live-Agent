#!/usr/bin/env bash
# Kupe ThinkSpark Live Agent — one-shot setup into the current Python.
#   ./setup.sh          pip install deps (Colab / Kaggle / system — no venv)
#   ./setup.sh --docker build and run the container instead
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--docker" ]]; then
  docker compose up --build
  exit 0
fi

if command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=${PYTHON:-python3}
fi
echo "==> python: $($PY --version)"

$PY -m pip install -q --upgrade pip

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> CUDA GPU detected — installing CUDA torch"
  $PY -m pip install -q "torch>=2.5" --index-url https://download.pytorch.org/whl/cu124 || \
    $PY -m pip install -q "torch>=2.5"
else
  echo "==> no CUDA — installing default torch (CPU / Apple MPS)"
  $PY -m pip install -q "torch>=2.5"
fi

$PY -m pip install -q -r requirements.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "==> created .env — fill in your keys before running"
fi

$PY - <<'PY'
import torch
print(f"==> torch {torch.__version__} | cuda={torch.cuda.is_available()} "
      f"| mps={torch.backends.mps.is_available()}")
PY

cat <<'MSG'

setup complete.

  ./start.sh                # deps + server + ngrok URL
  ./start.sh --tmux         # same, detached

  python main.py            # terminal-only agent
  cd web && npm install && npm run dev
MSG

#!/usr/bin/env bash
# Kupe ThinkSpark Live Agent — one-shot setup.
#   ./setup.sh          install into a local venv
#   ./setup.sh --docker build and run the container instead
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--docker" ]]; then
  docker compose up --build
  exit 0
fi

PY=${PYTHON:-python3}
echo "==> python: $($PY --version)"

if [[ ! -d .venv ]]; then
  $PY -m venv .venv
fi
source .venv/bin/activate
python -m pip install -q --upgrade pip

# torch first, matched to the hardware, so pip does not resolve a CPU wheel over CUDA
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> CUDA GPU detected — installing CUDA torch"
  pip install -q "torch>=2.5" --index-url https://download.pytorch.org/whl/cu124
else
  echo "==> no CUDA — installing default torch (CPU / Apple MPS)"
  pip install -q "torch>=2.5"
fi

pip install -q -r requirements.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "==> created .env — fill in your keys before running"
fi

python - <<'PY'
import torch
print(f"==> torch {torch.__version__} | cuda={torch.cuda.is_available()} "
      f"| mps={torch.backends.mps.is_available()}")
PY

cat <<'MSG'

setup complete.

  ./start.sh                # venv + deps + preflight + server (use this)
  ./start.sh --tmux         # same, detached on the pod

  python main.py            # terminal-only agent
  cd web && npm install && npm run dev
MSG

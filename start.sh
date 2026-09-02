#!/usr/bin/env bash
# One command: deps + preflight + server (ngrok opens from server.py).
# Uses the current Python (Colab / Kaggle / system). No venv.
#   ./start.sh
#   ./start.sh --tmux
set -euo pipefail
cd "$(dirname "$0")"

if command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=python3
fi

if [[ "${1:-}" == "--tmux" ]]; then
  shift
  command -v tmux >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq tmux; }
  tmux has-session -t agent 2>/dev/null && tmux kill-session -t agent
  tmux new-session -d -s agent "cd $(pwd) && ./start.sh; echo; echo FAILED — scroll up; sleep 120"
  echo "==> tmux session 'agent'  (tmux attach -t agent)"
  echo "    paste the wss://….ngrok…/ws URL the server prints"
  exit 0
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

echo "==> python: $(command -v $PY) ($($PY --version))"
$PY -m pip install -q --upgrade pip

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> CUDA torch"
  $PY -m pip install -q "torch>=2.5" --index-url https://download.pytorch.org/whl/cu124 || \
    $PY -m pip install -q "torch>=2.5"
else
  $PY -m pip install -q "torch>=2.5"
fi

echo "==> deps"
$PY -m pip install -q -r requirements.txt
$PY -m pip install -q -U "transformers>=4.49,<5" "accelerate>=0.34" "huggingface_hub>=0.24" pyngrok

echo "==> preflight"
$PY -c "from agent.preflight import check; check()"

echo "==> server"
exec $PY server.py --host 0.0.0.0 --port 8000 "$@"

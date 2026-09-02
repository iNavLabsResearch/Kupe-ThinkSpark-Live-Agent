#!/usr/bin/env bash
# Atomic boot: venv + pinned deps + preflight + server.
#   ./start.sh          foreground (keep this terminal open)
#   ./start.sh --tmux   detach so a dropped SSH session does not kill it
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--tmux" ]]; then
  shift
  if ! command -v tmux >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -qq tmux
  fi
  tmux has-session -t agent 2>/dev/null && tmux kill-session -t agent
  # re-exec this script inside tmux without --tmux
  tmux new-session -d -s agent "cd $(pwd) && ./start.sh; echo; echo FAILED — scroll up; sleep 120"
  echo "==> tmux session 'agent'  (tmux attach -t agent)"
  echo "    UI: paste the ws:// URL printed by the server"
  echo "    direct IP if the host publishes 8000; tunnel only if it does not"
  exit 0
fi

# System python on the pod has a mixed site-packages. Always the venv.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  [[ -d .venv ]] || python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "==> python: $(command -v python) ($(python --version))"
python -m pip install -q --upgrade pip

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> CUDA torch"
  pip install -q "torch>=2.5" --index-url https://download.pytorch.org/whl/cu124
else
  pip install -q "torch>=2.5"
fi

echo "==> deps"
pip install -q -r requirements.txt
# Force the versions that actually import MimiModel. Must run AFTER requirements
# so a stale transformers left by an old kupe extra cannot survive.
pip install -q -U \
  "transformers>=4.49,<5" \
  "accelerate>=0.34" \
  "huggingface_hub>=0.24"

echo "==> preflight"
python -c "from agent.preflight import check; check()"

echo "==> server  (binds 0.0.0.0:8000 — banner lists direct IP / nginx / optional tunnel)"
exec python server.py --host 0.0.0.0 --port 8000 "$@"

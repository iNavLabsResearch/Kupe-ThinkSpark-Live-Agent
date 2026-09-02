#!/usr/bin/env bash
#   ./start.sh --gradio   Colab/Kaggle/Vast voice UI (Gradio share, no ngrok)
#   ./start.sh --server   FastAPI + tunnel
#   SKIP_PIP=1 ./start.sh --gradio   skip pip entirely
set -euo pipefail
cd "$(dirname "$0")"

if command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=python3
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODE=auto
if [[ "${1:-}" == "--gradio" ]]; then MODE=gradio; shift; fi
if [[ "${1:-}" == "--server" ]]; then MODE=server; shift; fi
if [[ "$MODE" == "auto" ]]; then
  if [[ -d /kaggle ]] || [[ -n "${COLAB_RELEASE_TAG:-}" ]]; then
    MODE=gradio
  else
    MODE=server
  fi
fi

echo "==> python: $(command -v $PY) ($($PY --version))  mode=$MODE"

_have_stack() {
  $PY - <<'PY' >/dev/null 2>&1
import fastrtc, gradio, kupe, torch, transformers
from kupe import ThinkSpark
PY
}

_have_mimi() {
  $PY - <<'PY' >/dev/null 2>&1
from transformers import MimiModel
PY
}

_have_cuda_torch() {
  $PY - <<'PY' >/dev/null 2>&1
import torch
assert torch.cuda.is_available()
PY
}

_install_torchvision() {
  echo "==> torchvision (MimiModel needs torchvision::nms, must match this torch)"
  local idx=""
  if $PY -c "import torch; v=torch.__version__; raise SystemExit(0 if ('cu124' in v or str(torch.version.cuda or '').startswith('12.4')) else 1)" 2>/dev/null; then
    idx="https://download.pytorch.org/whl/cu124"
  elif $PY -c "import torch; v=torch.__version__; raise SystemExit(0 if ('cu128' in v or str(torch.version.cuda or '').startswith('12.8')) else 1)" 2>/dev/null; then
    idx="https://download.pytorch.org/whl/cu128"
  elif $PY -c "import torch; v=torch.__version__; raise SystemExit(0 if ('cu121' in v or str(torch.version.cuda or '').startswith('12.1')) else 1)" 2>/dev/null; then
    idx="https://download.pytorch.org/whl/cu121"
  fi
  if [[ -n "$idx" ]]; then
    echo "==> torchvision from $idx"
    $PY -m pip install -q --prefer-binary --upgrade torchvision --index-url "$idx"
  else
    $PY -m pip install -q --prefer-binary --upgrade torchvision
  fi
}

if [[ "${SKIP_PIP:-}" == "1" ]]; then
  echo "==> SKIP_PIP=1 — not installing"
elif _have_stack; then
  echo "==> deps already present — skipping pip"
  $PY -c "import torch; print('==> torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
else
  # binary wheels only — tokenizers 0.23+ tries to compile Rust and looks frozen
  export PIP_ONLY_BINARY="${PIP_ONLY_BINARY:-tokenizers}"
  $PY -m pip install -q --upgrade pip || true

  if _have_cuda_torch; then
    echo "==> CUDA torch already present — not reinstalling"
    $PY -c "import torch; print('==> torch', torch.__version__, torch.cuda.get_device_name(0))"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    echo "==> CUDA torch"
    $PY -m pip install -q --prefer-binary "torch>=2.5" --index-url https://download.pytorch.org/whl/cu124 || \
      $PY -m pip install -q --prefer-binary "torch>=2.5"
  else
    $PY -m pip install -q --prefer-binary "torch>=2.5"
  fi

  echo "==> pinning tokenizers wheel (no source build)"
  $PY -m pip install -q --prefer-binary --only-binary=:all: \
    "tokenizers>=0.21,<0.22" \
    "transformers>=4.49,<4.58" \
    "huggingface_hub>=0.34,<1"

  echo "==> deps"
  $PY -m pip install -q --prefer-binary -r requirements.txt
fi

if [[ "${SKIP_PIP:-}" != "1" ]] && ! _have_mimi; then
  _install_torchvision
fi

if [[ "$MODE" == "gradio" ]]; then
  if [[ "${SKIP_PIP:-}" != "1" ]] && ! $PY -c "import fastrtc, gradio" >/dev/null 2>&1; then
    $PY -m pip install -q --prefer-binary "gradio>=4.44,<6" "fastrtc>=0.0.19" "huggingface_hub>=0.34,<1"
  fi
  echo "==> Gradio voice UI — ThinkSpark referee, no VAD, no ngrok. Use the *.gradio.live link."
  exec $PY gradio_app.py
fi

if [[ "${SKIP_PIP:-}" != "1" ]]; then
  $PY -m pip install -q --prefer-binary -U "transformers>=4.49,<4.58" "accelerate>=0.34" "huggingface_hub>=0.34,<1" pyngrok
fi
echo "==> preflight"
$PY -c "from agent.preflight import check; check()"
echo "==> FastAPI + tunnel"
exec $PY server.py --host 0.0.0.0 --port 8000 "$@"

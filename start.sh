#!/usr/bin/env bash
#   ./start.sh --gradio   Colab/Kaggle voice UI (Gradio share, no ngrok)
#   ./start.sh --server   FastAPI + ngrok
#   ./start.sh            auto: Gradio on Colab/Kaggle, else FastAPI+ngrok
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

if [[ "$MODE" == "gradio" ]]; then
  $PY -m pip install -q "gradio>=4.44,<6" "fastrtc[vad]>=0.0.19" "huggingface_hub>=0.34,<1" onnxruntime
  echo "==> Gradio voice UI — no ngrok. Use the *.gradio.live link."
  exec $PY gradio_app.py
fi

$PY -m pip install -q -U "transformers>=4.49,<5" "accelerate>=0.34" "huggingface_hub>=0.34,<1" pyngrok
echo "==> preflight"
$PY -c "from agent.preflight import check; check()"
echo "==> FastAPI + ngrok"
exec $PY server.py --host 0.0.0.0 --port 8000 "$@"

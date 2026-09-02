"""Provider config. Keys come from agent/keys.py (gitignored), .env, or env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass

# ThinkSpark checkpoint — the finetuned weights, not the base model
TS_REPO = "anuj-inavlabs/kupe-thinkspark-audio-270m"
TS_SUBFOLDER = "phase2/runs/20260902-103400/step5500"

LLM_MODEL = "gemma-4-31b-it"
LLM_BASE_URL = "https://cloud.olakrutrim.com/v1"

# STT — AssemblyAI streaming v3 (PCM16le). Docs:
# https://www.assemblyai.com/docs/streaming/getting-started/transcribe-streaming-audio
STT_MODEL = "universal-3-5-pro"
STT_WS_URL = "wss://streaming.assemblyai.com/v3/ws"

# TTS — Soniox realtime, Mina voice
TTS_MODEL = "tts-rt-v2"
TTS_VOICE = "Mina"
TTS_WS_URL = "wss://tts-rt.soniox.com/tts-websocket"
TTS_SAMPLE_RATE = 24_000


@dataclass
class Keys:
    llm: str
    stt: str
    tts: str          # Soniox TTS (Mina). STT is AssemblyAI.
    hf: str = ""


def load_thinkspark(device: str = "auto"):
    """Load ThinkSpark after proving kupe + transformers can actually import Mimi."""
    from agent.preflight import check
    from kupe import ThinkSpark

    check()
    keys = load_keys()
    if keys.hf:
        os.environ.setdefault("HF_TOKEN", keys.hf)
    return ThinkSpark(TS_REPO, device=device, subfolder=TS_SUBFOLDER, hf_token=keys.hf or None)


def load_keys() -> Keys:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    try:
        from agent import keys as _k
    except ImportError:
        _k = None

    def _get(name: str) -> str:
        return os.environ.get(name) or (getattr(_k, name, "") if _k else "") or ""

    soniox = _get("SONIOX_API_KEY")
    stt = _get("ASSEMBLYAI_API_KEY")
    llm = _get("KRUTRIM_API_KEY")
    hf = _get("HF_TOKEN")
    if hf:
        os.environ.setdefault("HF_TOKEN", hf)
    if not llm or not stt or not soniox:
        raise SystemExit(
            "Need KRUTRIM_API_KEY, ASSEMBLYAI_API_KEY, and SONIOX_API_KEY (TTS).\n"
            "  export ASSEMBLYAI_API_KEY='...'\n"
            "  export KRUTRIM_API_KEY='...'\n"
            "  export SONIOX_API_KEY='...'\n"
            "  export HF_TOKEN='...'\n"
            "or write agent/keys.py / .env"
        )
    return Keys(llm=llm, stt=stt, tts=soniox, hf=hf)

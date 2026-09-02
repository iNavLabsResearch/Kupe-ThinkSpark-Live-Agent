"""Provider config. Keys come from agent/keys.py (gitignored), .env, or env vars."""

from __future__ import annotations

import os
from dataclasses import dataclass

# ThinkSpark checkpoint — the finetuned weights, not the base model
TS_REPO = "anuj-inavlabs/kupe-thinkspark-audio-270m"
TS_SUBFOLDER = "phase2/runs/20260902-103400/step5500"

LLM_MODEL = "gemma-4-31b-it"
LLM_BASE_URL = "https://cloud.olakrutrim.com/v1"
STT_MODEL = "stt-rt-v5"
STT_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"

# TTS — Soniox realtime, Mina voice
TTS_MODEL = "tts-rt-v2"
TTS_VOICE = "Mina"
TTS_WS_URL = "wss://tts-rt.soniox.com/tts-websocket"
TTS_SAMPLE_RATE = 24_000


@dataclass
class Keys:
    llm: str
    stt: str
    tts: str          # Soniox key drives both STT and TTS
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
    llm = _get("KRUTRIM_API_KEY")
    hf = _get("HF_TOKEN")
    if hf:
        os.environ.setdefault("HF_TOKEN", hf)
    if not llm or not soniox:
        raise SystemExit(
            "Need KRUTRIM_API_KEY and SONIOX_API_KEY. In Colab/Kaggle use a Python cell "
            "(not !export):\n"
            "  import os\n"
            "  os.environ['KRUTRIM_API_KEY'] = '...'\n"
            "  os.environ['SONIOX_API_KEY'] = '...'\n"
            "  os.environ['HF_TOKEN'] = '...'\n"
            "or write agent/keys.py / .env"
        )
    return Keys(llm=llm, stt=soniox, tts=soniox, hf=hf)

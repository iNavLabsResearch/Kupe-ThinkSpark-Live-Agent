"""Provider config. Keys come from agent/keys.py (gitignored), env vars override."""

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
    return ThinkSpark(TS_REPO, device=device, subfolder=TS_SUBFOLDER)


def load_keys() -> Keys:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    try:
        from agent import keys as _k
    except ImportError:
        raise SystemExit(
            "agent/keys.py is missing (it is gitignored, so a fresh clone has none).\n"
            "Create it with KRUTRIM_API_KEY / SONIOX_API_KEY / SARVAM_API_KEY, "
            "or export those three as env vars."
        )

    soniox = os.environ.get("SONIOX_API_KEY") or _k.SONIOX_API_KEY
    return Keys(
        llm=os.environ.get("KRUTRIM_API_KEY") or _k.KRUTRIM_API_KEY,
        stt=soniox,
        tts=soniox,
        hf=os.environ.get("HF_TOKEN") or getattr(_k, "HF_TOKEN", ""),
    )

"""Provider config. Keys come from agent/keys.py (gitignored), env vars override."""

from __future__ import annotations

import os
from dataclasses import dataclass

LLM_MODEL = "gemma-4-31b-it"
LLM_BASE_URL = "https://cloud.olakrutrim.com/v1"
STT_MODEL = "stt-rt-v5"
STT_WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"
TTS_MODEL = "bulbul:v3"
TTS_VOICE = "ritu"
TTS_URL = "https://api.sarvam.ai/text-to-speech"


@dataclass
class Keys:
    llm: str
    stt: str
    tts: str


def load_keys() -> Keys:
    try:
        from agent import keys as _k
    except ImportError:
        raise SystemExit(
            "agent/keys.py is missing (it is gitignored, so a fresh clone has none).\n"
            "Create it with KRUTRIM_API_KEY / SONIOX_API_KEY / SARVAM_API_KEY, "
            "or export those three as env vars."
        )

    return Keys(
        llm=os.environ.get("KRUTRIM_API_KEY") or _k.KRUTRIM_API_KEY,
        stt=os.environ.get("SONIOX_API_KEY") or _k.SONIOX_API_KEY,
        tts=os.environ.get("SARVAM_API_KEY") or _k.SARVAM_API_KEY,
    )

"""Provider config — pulls API keys from the Kupe Supabase, or falls back to env vars.

Keys are never written to disk. Set SUPABASE_DB_URL (the Postgres connection string) to
fetch them live, or export the three keys directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Resolved from provider_catalog on 2026-09-02.
LLM_PROVIDER_ID = "197f618a-475f-4433-9078-e5a9897f277a"   # krutrim gemma-4-31b-it
STT_PROVIDER_ID = "aaa128c9-4ac7-4d5c-b831-78c8c19082e0"   # soniox stt-rt-v5
TTS_PROVIDER_ID = "594bb74f-d56c-4dfa-9401-82b7c776916a"   # sarvam bulbul:v3

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


def _from_supabase() -> dict[str, str]:
    dsn = os.environ.get("SUPABASE_DB_URL")
    if not dsn:
        return {}
    try:
        import psycopg
    except ImportError:
        return {}
    ids = (LLM_PROVIDER_ID, STT_PROVIDER_ID, TTS_PROVIDER_ID)
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "select provider_id::text, api_key from provider_api_keys "
            "where provider_id::text = any(%s)",
            (list(ids),),
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def load_keys() -> Keys:
    db = _from_supabase()
    llm = os.environ.get("KRUTRIM_API_KEY") or db.get(LLM_PROVIDER_ID, "")
    stt = os.environ.get("SONIOX_API_KEY") or db.get(STT_PROVIDER_ID, "")
    tts = os.environ.get("SARVAM_API_KEY") or db.get(TTS_PROVIDER_ID, "")

    missing = [n for n, v in (("KRUTRIM", llm), ("SONIOX", stt), ("SARVAM", tts)) if not v]
    if missing:
        raise SystemExit(
            f"missing API keys: {', '.join(missing)}\n"
            "Either set SUPABASE_DB_URL (and pip install psycopg[binary]) to pull them "
            "from the Kupe provider table, or export "
            "KRUTRIM_API_KEY / SONIOX_API_KEY / SARVAM_API_KEY directly."
        )
    return Keys(llm=llm, stt=stt, tts=tts)

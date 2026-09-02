"""Streaming provider clients: Soniox STT, Krutrim LLM, Sarvam TTS.

All three stream. The agent needs partial transcripts to speculate on, token-by-token LLM
output to start speaking early, and TTS audio in chunks so a barge-in can cut it mid-word.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections.abc import AsyncIterator, Iterator

from agent import config


# --------------------------------------------------------------------------- #
# STT — Soniox realtime websocket
# --------------------------------------------------------------------------- #
class SonioxSTT:
    """Streaming STT. Feed PCM16 bytes, read partial + final transcripts."""

    def __init__(self, api_key: str, language_hints: list[str] | None = None,
                 sample_rate: int = 16_000):
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.language_hints = language_hints or ["en", "hi", "gu"]
        self._ws = None
        self.partial = ""
        self.final = ""

    async def connect(self):
        import websockets

        self._ws = await websockets.connect(config.STT_WS_URL, max_size=None)
        await self._ws.send(json.dumps({
            "api_key": self.api_key,
            "model": config.STT_MODEL,
            "audio_format": "pcm_s16le",
            "sample_rate": self.sample_rate,
            "num_channels": 1,
            "language_hints": self.language_hints,
            "enable_endpoint_detection": True,
        }))
        return self

    async def send_audio(self, pcm16: bytes) -> None:
        if self._ws:
            await self._ws.send(pcm16)

    async def transcripts(self) -> AsyncIterator[tuple[str, bool]]:
        """Yields (text, is_final) as Soniox emits tokens."""
        if not self._ws:
            return
        async for raw in self._ws:
            if isinstance(raw, bytes):
                continue
            msg = json.loads(raw)
            if msg.get("error_code"):
                raise RuntimeError(f"soniox stt: {msg.get('error_message')}")

            final_txt, partial_txt = "", ""
            for tok in msg.get("tokens", []):
                if tok.get("is_final"):
                    final_txt += tok.get("text", "")
                else:
                    partial_txt += tok.get("text", "")

            if final_txt:
                self.final += final_txt
                yield final_txt, True
            if partial_txt:
                self.partial = partial_txt
                yield partial_txt, False

    def reset_turn(self) -> None:
        self.partial = ""
        self.final = ""

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()


# --------------------------------------------------------------------------- #
# LLM — Krutrim (OpenAI-compatible), streaming
# --------------------------------------------------------------------------- #
class KrutrimLLM:
    """Streaming chat completions. Cancellable mid-generation for CANCEL/BARGE."""

    def __init__(self, api_key: str, model: str = config.LLM_MODEL,
                 system: str = "You are a concise, warm voice assistant. "
                               "Reply in one or two short spoken sentences."):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=config.LLM_BASE_URL)
        self.model = model
        self.system = system
        self.history: list[dict] = []

    async def stream(self, user_text: str) -> AsyncIterator[str]:
        """Yields text deltas. Cancel the task to abort generation."""
        messages = [{"role": "system", "content": self.system}, *self.history,
                    {"role": "user", "content": user_text}]
        chunks: list[str] = []
        stream = await self._client.chat.completions.create(
            model=self.model, messages=messages, stream=True,
            temperature=0.6, max_tokens=160,
        )
        try:
            async for event in stream:
                delta = event.choices[0].delta.content or ""
                if delta:
                    chunks.append(delta)
                    yield delta
        finally:
            said = "".join(chunks)
            if said:
                self.history += [{"role": "user", "content": user_text},
                                 {"role": "assistant", "content": said}]
                self.history = self.history[-8:]


# --------------------------------------------------------------------------- #
# TTS — Sarvam bulbul, chunked so playback can be cut mid-utterance
# --------------------------------------------------------------------------- #
class SarvamTTS:
    """Synthesizes short spans so BARGE_HARD can stop playback within one chunk."""

    def __init__(self, api_key: str, voice: str = config.TTS_VOICE,
                 model: str = config.TTS_MODEL, language: str = "en-IN"):
        self.api_key = api_key
        self.voice = voice
        self.model = model
        self.language = language

    async def synth(self, text: str) -> bytes:
        """Returns WAV bytes for one span of text."""
        import httpx

        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                config.TTS_URL,
                headers={"api-subscription-key": self.api_key},
                json={
                    "text": text,
                    "target_language_code": self.language,
                    "speaker": self.voice,
                    "model": self.model,
                },
            )
            r.raise_for_status()
            audios = r.json().get("audios") or []
            if not audios:
                return b""
            return base64.b64decode(audios[0])

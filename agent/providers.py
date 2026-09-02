"""Streaming provider clients: AssemblyAI STT, Krutrim LLM, Soniox TTS (voice Mina)."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from collections.abc import AsyncIterator, Iterator

from agent import config


# --------------------------------------------------------------------------- #
# STT — AssemblyAI streaming v3 websocket (PCM16le)
# https://www.assemblyai.com/docs/streaming/getting-started/transcribe-streaming-audio
# --------------------------------------------------------------------------- #
class AssemblyAISTT:
    """Streaming STT. Feed PCM16 bytes, read partial + final turns."""

    def __init__(self, api_key: str, language_hints: list[str] | None = None,
                 sample_rate: int = 16_000):
        self.api_key = api_key
        self.sample_rate = sample_rate
        self.language_hints = language_hints or ["en", "hi", "gu"]
        self._ws = None
        self.partial = ""
        self.final = ""

    async def connect(self):
        from urllib.parse import urlencode

        import websockets

        langs = [c for c in (self.language_hints or []) if c in {"en", "hi"}] or ["en", "hi"]
        qs = urlencode({
            "sample_rate": str(self.sample_rate),
            "encoding": "pcm_s16le",
            "speech_model": config.STT_MODEL,
            "format_turns": "true",
            "include_partial_turns": "true",
            "language_codes": json.dumps(langs, separators=(",", ":")),
        })
        url = f"{config.STT_WS_URL}?{qs}"
        headers = {"Authorization": self.api_key}
        try:
            self._ws = await websockets.connect(
                url, additional_headers=headers, max_size=None,
            )
        except TypeError:
            self._ws = await websockets.connect(
                url, extra_headers=headers, max_size=None,
            )
        raw = await self._ws.recv()
        msg = json.loads(raw) if isinstance(raw, str) else {}
        if msg.get("type") == "Error" or msg.get("error"):
            raise RuntimeError(f"assemblyai stt: {msg.get('error') or msg}")
        return self

    async def send_audio(self, pcm16: bytes) -> None:
        if self._ws:
            await self._ws.send(pcm16)

    async def transcripts(self) -> AsyncIterator[tuple[str, bool]]:
        """Yields (text, is_final) from AssemblyAI Turn events."""
        if not self._ws:
            return
        async for raw in self._ws:
            if isinstance(raw, bytes):
                continue
            msg = json.loads(raw)
            typ = msg.get("type")
            if typ in ("Begin", "Termination", "SessionInformation"):
                continue
            if typ == "Error" or msg.get("error"):
                raise RuntimeError(f"assemblyai stt: {msg.get('error') or msg}")
            if typ != "Turn":
                continue
            text = (msg.get("transcript") or msg.get("utterance") or "").strip()
            if not text:
                continue
            self.partial = text
            if msg.get("end_of_turn"):
                self.final = text
                yield text, True
            else:
                yield text, False

    async def transcribe_clip(self, pcm16: bytes, timeout: float = 12.0) -> str:
        await self.connect()
        silence = b"\x00\x00" * (self.sample_rate // 4)
        await self.send_audio(pcm16 + silence)
        try:
            await self._ws.send(json.dumps({"type": "Terminate"}))
        except Exception:
            pass
        text = ""
        try:
            async with asyncio.timeout(timeout):
                async for tok, is_final in self.transcripts():
                    text = tok
                    if is_final:
                        break
        except TimeoutError:
            pass
        await self.close()
        return (text or self.final).strip()

    def reset_turn(self) -> None:
        self.partial = ""
        self.final = ""

    async def close(self) -> None:
        if self._ws:
            try:
                await self._ws.send(json.dumps({"type": "Terminate"}))
            except Exception:
                pass
            await self._ws.close()
            self._ws = None


SonioxSTT = AssemblyAISTT


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
                choices = getattr(event, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0].delta, "content", None) or ""
                if delta:
                    chunks.append(delta)
                    yield delta
        finally:
            said = "".join(chunks)
            if said:
                self.history += [{"role": "user", "content": user_text},
                                 {"role": "assistant", "content": said}]
                self.history = self.history[-8:]

    async def one_shot(self, user_text: str) -> str:
        """One completion that does not enter the conversation history."""
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": user_text},
        ]
        stream = await self._client.chat.completions.create(
            model=self.model, messages=messages, stream=True,
            temperature=0.6, max_tokens=60,
        )
        chunks: list[str] = []
        async for event in stream:
            choices = getattr(event, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0].delta, "content", None) or ""
            if delta:
                chunks.append(delta)
        return "".join(chunks)


# --------------------------------------------------------------------------- #
# TTS — Soniox realtime websocket, voice "Mina", streamed chunk-by-chunk
# --------------------------------------------------------------------------- #
class SonioxTTS:
    """Streaming TTS over the Soniox realtime websocket.

    Protocol (verified, not guessed): connect, send a config frame, then a text frame
    with text_end=true. The server streams back base64 WAV chunks keyed by stream_id
    until it sends `terminated`. We yield each chunk the moment it lands so playback
    starts before synthesis finishes and BARGE_HARD can cut it mid-word.
    """

    def __init__(self, api_key: str, voice: str = config.TTS_VOICE,
                 model: str = config.TTS_MODEL, language: str = "en",
                 sample_rate: int = config.TTS_SAMPLE_RATE):
        self.api_key = api_key
        self.voice = voice
        self.model = model
        self.language = language
        self.sample_rate = sample_rate

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        """Yields raw PCM16 chunks at self.sample_rate as they arrive."""
        import uuid

        import websockets

        stream_id = f"tts-{uuid.uuid4().hex[:12]}"
        async with websockets.connect(config.TTS_WS_URL, max_size=None) as ws:
            await ws.send(json.dumps({
                "api_key": self.api_key,
                "model": self.model,
                "language": self.language,
                "voice": self.voice,
                "audio_format": "pcm_s16le",
                "sample_rate": self.sample_rate,
                "stream_id": stream_id,
                "return_timestamps": False,
            }))
            await ws.send(json.dumps({
                "text": text, "text_end": True, "stream_id": stream_id,
            }))

            async for raw in ws:
                if isinstance(raw, bytes):
                    yield raw
                    continue
                data = json.loads(raw)
                if data.get("error_code") is not None:
                    raise RuntimeError(
                        f"soniox tts {data.get('error_code')}: {data.get('error_message')}"
                    )
                audio_b64 = data.get("audio")
                if audio_b64:
                    yield base64.b64decode(audio_b64)
                if data.get("terminated"):
                    return

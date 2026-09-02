"""ThinkSpark floor-control loop shared by Gradio, websocket, and the terminal.

Audio is framed at 80 ms. ThinkSpark is the only VAD/endpoint/barge referee.
AssemblyAI streaming STT runs in parallel (not as a gate). LLM/TTS fire only
when Policy says so. TTS audio is never fed to STT (echo mute + hangover).
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from agent import config
from agent.denoise import Denoiser
from agent.policy import Action, AgentState, Policy
from agent.providers import AssemblyAISTT, KrutrimLLM, SonioxTTS
from agent.smoothing import FlagSmoother

MIC_RATE = 24_000
STT_RATE = 16_000
FRAME_MS = 80
FRAME_SAMPLES = MIC_RATE * FRAME_MS // 1000
TABLE_MAX = 80


def _f32_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or x.size == 0:
        return x
    n = max(1, int(len(x) * dst / src))
    return np.interp(np.linspace(0, len(x), n, endpoint=False),
                     np.arange(len(x)), x).astype(np.float32)


def _mono_f32(data) -> np.ndarray:
    x = np.asarray(data)
    if x.size == 0:
        return x.astype(np.float32)
    if x.ndim == 2:
        x = x[0] if x.shape[0] <= 8 else x.mean(axis=-1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / max(1, np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float32)
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 1.5:
            x = x / 32768.0
    return np.clip(x, -1.0, 1.0)


@dataclass
class FrameRow:
    time: str
    flag: str
    spoken: str
    stt: str
    context: str
    output: str
    ms: float

    def as_list(self) -> list[str]:
        return [
            self.time, self.flag, self.spoken, self.stt,
            self.context, self.output, f"{self.ms:.2f}",
        ]


class PrintUI:
    def log(self, kind: str, detail: str = "", style: str = "") -> None:
        print(f"[{kind}] {detail}", flush=True)

    def frame(self, flag: str, latency_ms: float, raw: bool = False) -> None:
        pass


class FloorAgent:
    """One conversation: 80 ms frames -> ThinkSpark + STT -> Policy -> TTS queue."""

    def __init__(self, referee, keys, ui=None, window: int = 3, denoise: bool = True):
        self.referee = referee
        self.ui = ui or PrintUI()
        self.stt = AssemblyAISTT(keys.stt, sample_rate=STT_RATE)
        self.llm = KrutrimLLM(keys.llm)
        self.tts = SonioxTTS(keys.tts)
        self.denoiser = Denoiser(sample_rate=MIC_RATE, enabled=denoise)
        self.smoother = FlagSmoother(window=window)
        self.policy = Policy(self)

        self.agent_text = ""
        self.last_output = ""
        self.frames = 0
        self.decode_ms: list[float] = []
        self.rows: deque[FrameRow] = deque(maxlen=TABLE_MAX)

        self._in_buf = np.zeros(0, dtype=np.float32)
        self._audio_q: queue.Queue = queue.Queue(maxsize=200)
        self._stt_q: queue.Queue = queue.Queue(maxsize=200)
        self.playback_q: queue.Queue = queue.Queue(maxsize=400)

        self._speaking = False
        self._volume = 1.0
        self._stop_playback = threading.Event()
        self._closed = False
        self._turn_busy = False
        self._tasks: list[asyncio.Task] = []
        self._dirty = True
        self._spoken_latch: dict[str, str] = {}
        self._spoken_claimed = False
        self._echo_until = 0.0

    # -- context -------------------------------------------------------- #
    def _refresh_context(self) -> None:
        self.referee.set_context(
            agent_text=self.agent_text,
            stt_partial=self.stt.partial or self.stt.final,
        )

    def _context_str(self) -> str:
        state = self.policy.state.value
        text = (self.agent_text or "").strip()
        if text:
            return f"{state} | {text[:80]}"
        return state

    def _stt_str(self) -> str:
        return (self.stt.partial or self.stt.final or "").strip()

    # -- audio out ------------------------------------------------------ #
    def duck(self) -> None:
        self._volume = 0.35

    def stop_speaking(self) -> None:
        self._stop_playback.set()
        self._speaking = False
        self._volume = 1.0
        while True:
            try:
                self.playback_q.get_nowait()
            except queue.Empty:
                break

    def _enqueue_pcm(self, pcm: bytes) -> None:
        a = np.frombuffer(pcm, dtype=np.int16)
        if self._volume != 1.0:
            a = (a.astype(np.float32) * self._volume).astype(np.int16)
        try:
            self.playback_q.put_nowait(a)
        except queue.Full:
            try:
                self.playback_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self.playback_q.put_nowait(a)
            except queue.Full:
                pass

    async def speak(self, text: str, filler: bool = False) -> None:
        if not text.strip() or self._closed:
            return
        self.policy.state = AgentState.TTS_SPEAKING
        self._speaking = True
        self._stop_playback.clear()
        self._volume = 1.0
        self.agent_text = text
        self._refresh_context()
        self.ui.log("tts", f"speaking: {text!r}")
        self.last_output = f"TTS: {text}"
        self._dirty = True
        try:
            async for pcm in self.tts.stream(text):
                if self._stop_playback.is_set() or self._closed:
                    self.ui.log("tts", "playback cut")
                    break
                self._enqueue_pcm(pcm)
        except Exception as e:
            self.ui.log("error", f"tts failed: {e}")
        self._speaking = False
        self._echo_until = time.time() + 0.60
        self.agent_text = ""
        self.policy.state = AgentState.TTS_DONE if not filler else AgentState.IDLE
        self._refresh_context()
        self._dirty = True

    # -- audio in ------------------------------------------------------- #
    def push_audio(self, pcm, sample_rate: int) -> None:
        if self._closed:
            return
        x = _mono_f32(pcm)
        x = _resample(x, int(sample_rate), MIC_RATE)
        if x.size == 0:
            return
        self._in_buf = np.concatenate([self._in_buf, x]) if self._in_buf.size else x
        while len(self._in_buf) >= FRAME_SAMPLES:
            frame = self._in_buf[:FRAME_SAMPLES]
            self._in_buf = self._in_buf[FRAME_SAMPLES:]
            try:
                self._audio_q.put_nowait(frame)
            except queue.Full:
                try:
                    self._audio_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._audio_q.put_nowait(frame)
                except queue.Full:
                    pass

    def _echo_active(self) -> bool:
        """TTS (and a short hangover) must not land in STT. ThinkSpark still hears the mic."""
        return (
            self._speaking
            or self.policy.state is AgentState.TTS_SPEAKING
            or time.time() < self._echo_until
        )

    def _step_chunk(self, chunk: np.ndarray) -> list:
        chunk = self.denoiser(chunk)
        pcm = _f32_to_pcm16(_resample(chunk, MIC_RATE, STT_RATE))
        if self._echo_active():
            pcm = b"\x00\x00" * (len(pcm) // 2)
        try:
            self._stt_q.put_nowait(pcm)
        except queue.Full:
            pass
        return list(self.referee.stream(
            source=[chunk],
            sample_rate=MIC_RATE,
            agent_state=self.policy.state.value,
        ))

    def _record(self, flag: str, spoken: str, ms: float, output: str = "") -> FrameRow:
        row = FrameRow(
            time=time.strftime("%H:%M:%S"),
            flag=flag,
            spoken=spoken or "",
            stt=self._stt_str(),
            context=self._context_str(),
            output=output,
            ms=float(ms),
        )
        self.rows.appendleft(row)
        self._dirty = True
        return row

    def _claim_backchannel(self) -> bool:
        """Sync latch so LISTEN frames cannot queue parallel spoken TTS."""
        if self._speaking or self._spoken_claimed:
            return False
        if self.policy.state is AgentState.TTS_SPEAKING:
            return False
        now = time.time()
        if now - self.policy._last_backchannel < 2.0:
            return False
        self._spoken_claimed = True
        self.policy._last_backchannel = now
        return True

    def _should_play_spoken(self, flag: str, spoken: str) -> bool:
        """Guide: LISTEN is pass; spoken TTS only for real back-channel / thinking."""
        if not spoken:
            return False
        if self._speaking or self.policy.state is AgentState.TTS_SPEAKING:
            return False
        if flag == "INCOMPLETE":
            return True
        if flag == "LISTEN" and self._stt_str():
            return True
        return False

    async def _on_decision(self, decision) -> None:
        self.frames += 1
        self.decode_ms.append(decision.latency_ms)
        spoken = (getattr(decision, "spoken", "") or "").strip()
        speaking = self._speaking or self.policy.state is AgentState.TTS_SPEAKING
        playable = (not speaking) and (
            decision.flag == "SILENCE_BREAK"
            or self._should_play_spoken(decision.flag, spoken)
        )
        self.ui.frame(decision.flag, decision.latency_ms, raw=True)
        self._record(decision.flag, spoken if playable else "", decision.latency_ms)

        if speaking:
            spoken = ""
        elif spoken:
            self._spoken_latch[decision.flag] = spoken
            if decision.flag != "SILENCE_BREAK" and playable and self._claim_backchannel():
                asyncio.create_task(self._spoken_task(spoken, decision.flag))

        smoothed = self.smoother.push(decision.flag)
        if smoothed is None:
            return
        self.ui.frame(smoothed, decision.latency_ms, raw=False)
        spoken_for_flag = "" if speaking else self._spoken_latch.get(smoothed, "")
        asyncio.create_task(self._policy_task(smoothed, spoken_for_flag))

    async def _policy_task(self, flag: str, spoken: str = "") -> None:
        urgent = flag in {"BARGE_HARD", "BARGE_SOFT", "CANCEL_LLM"}
        idle = flag in {"LISTEN", "HOLD", "CONTINUE", "INCOMPLETE"}
        if not urgent and not idle and self._turn_busy:
            return
        if not urgent and not idle:
            self._turn_busy = True
        try:
            if flag == "SILENCE_BREAK":
                spoken = self._spoken_latch.pop("SILENCE_BREAK", spoken)
            action = await self.policy.handle(flag, spoken=spoken)
        except Exception as e:
            self.ui.log("error", f"policy {flag}: {e}")
            return
        finally:
            if not urgent and not idle:
                self._turn_busy = False
        if action:
            self._note_action(action)

    async def _spoken_task(self, text: str, flag: str = "") -> None:
        try:
            action = await self.policy.on_spoken(text, flag=flag)
        except Exception as e:
            self.ui.log("error", f"spoken: {e}")
            return
        finally:
            self._spoken_claimed = False
        if action:
            self._note_action(action)

    def _note_action(self, action: Action) -> None:
        self.last_output = f"{action.kind}: {action.detail}"
        self.ui.log(action.kind, action.detail)
        self._record(action.kind, "", 0.0, output=self.last_output)

    async def _referee_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._closed:
            chunk = await loop.run_in_executor(None, self._audio_q.get)
            if chunk is None:
                return
            try:
                decisions = await loop.run_in_executor(None, self._step_chunk, chunk)
            except Exception as e:
                self.ui.log("error", f"ThinkSpark: {e}")
                continue
            for d in decisions:
                await self._on_decision(d)

    async def _stt_send_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._closed:
            pcm = await loop.run_in_executor(None, self._stt_q.get)
            if pcm is None:
                return
            try:
                await self.stt.send_audio(pcm)
            except Exception as e:
                self.ui.log("error", f"stt send: {e}")

    async def _stt_recv_loop(self) -> None:
        try:
            async for text, is_final in self.stt.transcripts():
                if self._closed:
                    return
                if self._echo_active():
                    continue
                self._refresh_context()
                self._dirty = True
                kind = "stt-final" if is_final else "stt"
                self.ui.log(kind, text)
                if is_final:
                    if self._turn_busy:
                        continue
                    self._turn_busy = True
                    try:
                        action = await self.policy.commit_from_stt()
                    finally:
                        self._turn_busy = False
                    if action:
                        self._note_action(action)
        except Exception as e:
            if not self._closed:
                self.ui.log("error", f"stt recv: {e}")

    async def start(self) -> None:
        await self.stt.connect()
        self.ui.log(
            "boot",
            f"ready · {getattr(self.referee, 'device', '?')} · "
            f"AssemblyAI {config.STT_MODEL} · {config.LLM_MODEL} · "
            f"{config.TTS_MODEL}/{config.TTS_VOICE}",
        )
        self._tasks = [
            asyncio.create_task(self._referee_loop()),
            asyncio.create_task(self._stt_send_loop()),
            asyncio.create_task(self._stt_recv_loop()),
        ]

    async def run_until_closed(self) -> None:
        if not self._tasks:
            await self.start()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def close(self) -> None:
        self._closed = True
        self._stop_playback.set()
        for q in (self._audio_q, self._stt_q, self.playback_q):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        for t in self._tasks:
            t.cancel()
        try:
            await self.stt.close()
        except Exception:
            pass

    # -- UI snapshot ---------------------------------------------------- #
    def snapshot(self) -> tuple[str, str, str, str, list[list[str]]]:
        p50 = 0.0
        head = 80.0
        if self.decode_ms:
            w = sorted(self.decode_ms[-200:])
            p50 = w[len(w) // 2]
            head = 80.0 - (sum(w) / len(w))
        flag = self.rows[0].flag if self.rows else "—"
        stats = (
            f"{self.frames} frames · p50 {p50:.1f} ms · "
            f"headroom {head:.1f} ms (of 80 ms)"
        )
        return flag, stats, self._stt_str() or "…", self._context_str(), [
            r.as_list() for r in self.rows
        ]

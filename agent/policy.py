"""The flag -> action policy. This is where ThinkSpark actually controls the pipeline.

ThinkSpark does not transcribe, think, or speak. It decides *when* the STT, LLM, and TTS
are allowed to act. That mapping is the whole point of the model, so it lives in one file:

    LISTEN        user has the floor            -> nothing; keep feeding STT
    HOLD          user paused mid-thought       -> do NOT commit; suppress endpointing
    INCOMPLETE    utterance unfinished          -> same as HOLD, keep buffering
    TURN_END      user is done                  -> commit STT, run LLM, speak
    PREFETCH_LLM  safe to speculate             -> start LLM on the partial, buffer output
    COMMIT_LLM    speculation was right         -> play the buffered reply immediately
    CANCEL_LLM    speculation was wrong         -> abort the in-flight LLM call
    BARGE_SOFT    user talking over agent       -> duck TTS volume, keep speaking
    BARGE_HARD    user is interrupting          -> stop TTS now, truncate agent turn
    CONTINUE      agent may keep speaking       -> nothing
    SILENCE_BREAK dead air                      -> TTS ThinkSpark spoken (or LLM reopen)

The agent state fed *back* into the model matters: barge-in only means anything while
TTS_SPEAKING. Getting this wrong silently degrades the model, so `AgentState` is the
single source of truth and is pushed into the referee every frame.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum


class AgentState(str, Enum):
    IDLE = "IDLE"                    # waiting on the user
    LLM_GEN = "LLM_GEN"              # thinking
    TTS_SPEAKING = "TTS_SPEAKING"    # talking
    TTS_DONE = "TTS_DONE"            # just finished talking


@dataclass
class Action:
    """One thing the agent did, for the terminal log."""

    kind: str
    detail: str = ""
    at: float = field(default_factory=time.time)


class Policy:
    """Turns smoothed ThinkSpark flags into pipeline actions.

    The caller owns the actual STT/LLM/TTS objects; this object owns the *decisions*
    and the agent state machine.
    """

    def __init__(self, agent):
        self.agent = agent
        self.state = AgentState.IDLE
        self._speculation: asyncio.Task | None = None
        self._speculated_text: str = ""
        self._last_silence_break = 0.0
        self._last_backchannel = 0.0

    def _user_stt(self) -> str:
        if hasattr(self.agent, "_stt_str"):
            return (self.agent._stt_str() or "").strip()
        return (self.agent.stt.final or self.agent.stt.partial or "").strip()

    def _reset_stt(self) -> None:
        if hasattr(self.agent, "reset_user_stt"):
            self.agent.reset_user_stt()
        else:
            self.agent.stt.reset_turn()

    # ------------------------------------------------------------------ #
    async def handle(self, flag: str) -> Action | None:
        fn = getattr(self, f"_on_{flag.lower()}", None)
        if fn is None:
            return None
        return await fn()

    # --- user holds the floor ----------------------------------------- #
    async def _on_listen(self) -> None:
        return None

    async def _on_hold(self) -> Action | None:
        # A pause is not an ending. Explicitly refuse to commit here — this is the
        # flag that stops an agent interrupting someone who is still thinking.
        return None

    async def _on_incomplete(self) -> Action | None:
        # A pause is not an ending — never commit here. But when the user is clearly
        # mid-thought, a short thinking-sound ("haan haan...", "hmm") keeps the floor
        # warm and signals we're still listening (Section 5.3, B6). Heavily debounced.
        now = time.time()
        if self.state is not AgentState.IDLE:
            return None
        if now - self._last_backchannel < 2.5:
            return None
        stt = self._user_stt()
        if not stt or _is_placeholder(stt):
            return None
        if getattr(self.agent, "_speaking", False):
            return None
        text = await self.agent.gen_spoken()
        if not text or _is_placeholder(text) or _is_junk_spoken(text):
            return None
        self._last_backchannel = now
        await self.agent.speak(text, filler=True)
        return Action("SPOKEN", text)

    # --- speculation --------------------------------------------------- #
    async def _on_prefetch_llm(self) -> Action | None:
        partial = self._user_stt()
        if not partial or _is_placeholder(partial):
            return None
        if self._speculation and not self._speculation.done():
            return None
        self.state = AgentState.LLM_GEN
        self._speculated_text = ""

        async def _speculate():
            buf = []
            async for delta in self.agent.llm.stream(partial):
                buf.append(delta)
            self._speculated_text = "".join(buf)

        self._speculation = asyncio.create_task(_speculate())
        return Action("PREFETCH", f"speculating on {partial!r}")

    async def _on_cancel_llm(self) -> Action | None:
        if self._speculation and not self._speculation.done():
            self._speculation.cancel()
            self._speculation = None
            self._speculated_text = ""
            self.state = AgentState.IDLE
            return Action("CANCEL", "aborted speculative reply")
        return None

    async def _on_commit_llm(self) -> Action | None:
        if self._speculated_text:
            text = self._speculated_text
            self._speculated_text = ""
            await self.agent.speak(text)
            return Action("COMMIT", f"played prefetched reply ({len(text)} chars)")
        return None

    # --- the user finished --------------------------------------------- #
    async def _on_turn_end(self) -> Action | None:
        text = self._user_stt()
        if not text or _is_placeholder(text):
            return None

        # a correct speculation is already in hand — no LLM round trip
        if self._speculated_text:
            reply = self._speculated_text
            self._speculated_text = ""
            self._reset_stt()
            await self.agent.speak(reply)
            return Action("TURN_END", "used speculative reply (0 ms LLM wait)")

        if self._speculation and not self._speculation.done():
            self._speculation.cancel()

        self._reset_stt()
        await self._run_turn(text)
        return Action("TURN_END", f"commit -> LLM: {text!r}")

    async def _run_turn(self, text: str) -> None:
        """Actually run the turn: LLM -> TTS. Speaks sentence-by-sentence so the
        first audio starts before the LLM has finished writing."""
        self.state = AgentState.LLM_GEN
        buf, spoken_any = "", False
        try:
            async for delta in self.agent.llm.stream(text):
                buf += delta
                # flush on sentence boundaries -> first audio lands sooner
                while True:
                    cut = _sentence_cut(buf)
                    if cut is None:
                        break
                    span, buf = buf[:cut].strip(), buf[cut:]
                    if span:
                        spoken_any = True
                        await self.agent.speak(span)
                        if self.state is AgentState.IDLE:   # barged mid-reply
                            return
            if buf.strip():
                spoken_any = True
                await self.agent.speak(buf.strip())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.agent.ui.log("error", f"turn failed: {e}", style="bold red")
        finally:
            if not spoken_any:
                self.state = AgentState.IDLE

    async def commit_from_stt(self) -> Action | None:
        """AssemblyAI signalled end_of_turn. ThinkSpark decides *when* to speak, but
        the STT endpoint is a hard ground truth: if the model has not produced a
        TURN_END by the time the transcript is final, commit anyway rather than
        leaving the user hanging."""
        if self.state is not AgentState.IDLE:
            return None
        text = self._user_stt()
        if not text or _is_placeholder(text):
            return None

        if self._speculated_text:
            reply, self._speculated_text = self._speculated_text, ""
            self._reset_stt()
            await self.agent.speak(reply)
            return Action("STT_END", "used speculative reply (0 ms LLM wait)")

        self._reset_stt()
        await self._run_turn(text)
        return Action("STT_END", f"commit -> LLM: {text!r}")

    # --- the user interrupted ------------------------------------------ #
    async def _on_barge_soft(self) -> Action | None:
        if self.state is not AgentState.TTS_SPEAKING:
            return None
        self.agent.duck()
        return Action("BARGE_SOFT", "ducked TTS")

    async def _on_barge_hard(self) -> Action | None:
        if self.state is not AgentState.TTS_SPEAKING:
            return None
        self.agent.stop_speaking()
        self.state = AgentState.IDLE
        return Action("BARGE_HARD", "stopped TTS mid-utterance")

    async def _on_continue(self) -> None:
        return None

    # --- dead air ------------------------------------------------------- #
    async def _llm_reopen(self) -> str:
        """Guide: SILENCE_BREAK -> tts_stream(spoken or llm_reopen()).

        Only used when the spoken head returned empty. One short context-aware
        sentence — never a hardcoded filler.
        """
        stt = self._user_stt()
        prompt = (
            "The caller went silent. Re-open the conversation in one short spoken "
            "sentence. Match their language (Hindi, English, or Gujarati). "
            "No quotes, no stage directions. Never say 'please wait' or 'I see'."
        )
        if stt:
            prompt += f" They last said: {stt}"
        llm = self.agent.llm
        if hasattr(llm, "one_shot"):
            return (await llm.one_shot(prompt)).strip()
        parts: list[str] = []
        async for delta in llm.stream(prompt):
            parts.append(delta)
        return "".join(parts).strip()

    async def _on_silence_break(self) -> Action | None:
        now = time.time()
        if self.state is not AgentState.IDLE or now - self._last_silence_break < 6.0:
            return None
        if now - self._last_backchannel < 2.0:
            return None
        if getattr(self.agent, "_speaking", False):
            return None
        # ask the spoken head for a context-aware re-open; fall back to a one-shot LLM
        # sentence only if it stays silent (guide: SILENCE_BREAK -> spoken or llm_reopen).
        text = await self.agent.gen_spoken()
        if not text or _is_placeholder(text) or _is_junk_spoken(text):
            text = await self._llm_reopen()
        if not text or _is_placeholder(text):
            return None
        self._last_silence_break = now
        self._last_backchannel = now
        await self.agent.speak(text, filler=True)
        return Action("SILENCE_BREAK", text)


_ENDERS = ".?!\u0964"
_PLACEHOLDERS = frozenset({
    "please wait", "please wait.", "pls wait",
    "i see", "i see.", "i see...",
    "mm-hmm", "mm hmm", "mhm", "uh huh", "uh-huh",
})


def _is_placeholder(text: str) -> bool:
    n = " ".join((text or "").lower().strip().strip(".!?,;:").split())
    return (not n) or n in _PLACEHOLDERS or n.startswith("please wait")


def _is_junk_spoken(text: str) -> bool:
    """Reject a spoken-head output that is too short / mostly punctuation to say aloud
    (e.g. the checkpoint emitting "I'." or "-,"). Needs >=2 real letters (Latin, Devanagari
    or Gujarati) to count as a real back-channel."""
    import re
    letters = re.sub(r"[^A-Za-zऀ-ॿ઀-૿]", "", text or "")
    return len(letters) < 2


def _sentence_cut(buf: str) -> int | None:
    """Index just past the first sentence end, if the span is long enough to be worth
    synthesizing on its own (short fragments make speech choppy)."""
    for i, ch in enumerate(buf):
        if ch in _ENDERS and i >= 12:
            return i + 1
    return None

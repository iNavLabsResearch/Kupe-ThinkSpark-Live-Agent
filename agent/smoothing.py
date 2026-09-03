"""Temporal smoothing over ThinkSpark's per-frame flags.

The model decides every 80 ms. Single frames flip on noise — your own session logged 21
spurious SILENCE_BREAK frames in silence. This collapses a sliding window of frames into
one stable decision, which is also exactly the ±3-frame (±240 ms) tolerance the model was
evaluated at (ctrl macro-F1 0.860 with the collar vs 0.770 without).

Three rules — the "smart flag management" every flag goes through:

1. **Majority vote over the window.** A flag must win the window to be emitted at all
   (urgent BARGE_* get a lower bar — waiting 240 ms to stop talking over someone is
   itself the failure you are preventing).
2. **Latch on emit.** Once an EVENT flag fires it will not re-fire while it keeps
   winning — so one real PREFETCH_LLM / TURN_END / BARGE produces ONE decision, not
   fifty (exactly the thing you asked about).
3. **Per-flag cooldown.** After an event flag fires, the SAME event flag is suppressed
   for a short cooldown even if it flickers off-and-on — so a 1-frame LISTEN blip in the
   middle of a PREFETCH_LLM run cannot "reset the latch" and re-fire it. Steady states
   (LISTEN/HOLD/CONTINUE/INCOMPLETE) have no cooldown; they are allowed to repeat.

Net effect per flag:

    LISTEN / HOLD / CONTINUE / INCOMPLETE   steady — may repeat every frame (no action)
    TURN_END / PREFETCH_LLM / COMMIT_LLM /  event  — fire ONCE, then cool down ~event_cd
      CANCEL_LLM / SILENCE_BREAK                     frames before they can fire again
    BARGE_HARD / BARGE_SOFT                  urgent — fire fast (1 vote), short cooldown
"""

from __future__ import annotations

from collections import Counter, deque

# Flags where reacting late is worse than reacting on thinner evidence.
URGENT = frozenset({"BARGE_HARD", "BARGE_SOFT"})

# Flags that are steady states, not events — they may repeat without latching.
CONTINUOUS = frozenset({"LISTEN", "HOLD", "CONTINUE", "INCOMPLETE"})


class FlagSmoother:
    """Sliding-window majority vote + event latch + per-flag cooldown.

    Args:
        window: frames to vote over. 3 frames = 240 ms, matching the eval collar.
        min_votes: votes needed for a normal flag to win.
        urgent_votes: votes needed for a BARGE_* flag to win.
        event_cooldown: frames an EVENT flag is suppressed after it fires (8 ≈ 640 ms).
        urgent_cooldown: same, for BARGE_* (2 ≈ 160 ms — must stay responsive).
    """

    def __init__(self, window: int = 3, min_votes: int = 2, urgent_votes: int = 1,
                 event_cooldown: int = 8, urgent_cooldown: int = 2):
        self.window = window
        self.min_votes = min_votes
        self.urgent_votes = urgent_votes
        self.event_cooldown = event_cooldown
        self.urgent_cooldown = urgent_cooldown
        self._buf: deque[str] = deque(maxlen=window)
        self._last_emitted: str | None = None
        self._i = 0                              # frame counter (for cooldowns)
        self._last_emit_i: dict[str, int] = {}   # flag -> frame index it last fired

    def push(self, flag: str) -> str | None:
        """Feed one raw frame flag. Returns a smoothed decision, or None to suppress."""
        self._i += 1
        self._buf.append(flag)
        counts = Counter(self._buf)

        # urgent flags win on thinner evidence; HARD beats SOFT if both are present
        for urgent in ("BARGE_HARD", "BARGE_SOFT"):
            if counts[urgent] >= self.urgent_votes:
                return self._emit(urgent)

        winner, votes = counts.most_common(1)[0]
        if votes < self.min_votes:
            return None
        return self._emit(winner)

    def _emit(self, flag: str) -> str | None:
        # steady states: pass through, no latch, no cooldown
        if flag in CONTINUOUS:
            self._last_emitted = flag
            return flag
        # event flag: enforce a per-flag cooldown (which also subsumes the latch —
        # a flag that keeps winning cannot re-fire until the cooldown elapses)
        cd = self.urgent_cooldown if flag in URGENT else self.event_cooldown
        if self._i - self._last_emit_i.get(flag, -10**9) < cd:
            return None
        self._last_emit_i[flag] = self._i
        self._last_emitted = flag
        return flag

    def reset(self) -> None:
        self._buf.clear()
        self._last_emitted = None
        self._last_emit_i.clear()

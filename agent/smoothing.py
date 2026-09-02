"""Temporal smoothing over ThinkSpark's per-frame flags.

The model decides every 80 ms. Single frames flip on noise — your own session logged 21
spurious SILENCE_BREAK frames in silence. This collapses a sliding window of frames into
one stable decision, which is also exactly the ±3-frame (±240 ms) tolerance the model was
evaluated at (ctrl macro-F1 0.860 with the collar vs 0.770 without).

Two rules:

1. **Majority vote over the window.** A flag must win the window to be emitted at all.
2. **Latch on emit.** Once a flag fires, it will not re-fire until the window has moved
   to a different decision — so one real barge-in produces one BARGE_HARD, not fourteen.

Urgent flags get a lower bar: BARGE_HARD only needs `urgent_votes` of the window, because
waiting 240 ms to stop talking over someone is itself the failure you are preventing.
"""

from __future__ import annotations

from collections import Counter, deque

# Flags where reacting late is worse than reacting on thinner evidence.
URGENT = frozenset({"BARGE_HARD", "BARGE_SOFT"})

# Flags that are steady states, not events — they may repeat without latching.
CONTINUOUS = frozenset({"LISTEN", "HOLD", "CONTINUE", "INCOMPLETE"})


class FlagSmoother:
    """Sliding-window majority vote with event latching.

    Args:
        window: frames to vote over. 3 frames = 240 ms, matching the eval collar.
        min_votes: votes needed for a normal flag to win.
        urgent_votes: votes needed for a BARGE_* flag to win.
    """

    def __init__(self, window: int = 3, min_votes: int = 2, urgent_votes: int = 1):
        self.window = window
        self.min_votes = min_votes
        self.urgent_votes = urgent_votes
        self._buf: deque[str] = deque(maxlen=window)
        self._last_emitted: str | None = None

    def push(self, flag: str) -> str | None:
        """Feed one raw frame flag. Returns a smoothed flag, or None to suppress."""
        self._buf.append(flag)

        counts = Counter(self._buf)

        # urgent flags win on thinner evidence
        for urgent in URGENT:
            if counts[urgent] >= self.urgent_votes:
                return self._emit(urgent)

        winner, votes = counts.most_common(1)[0]
        if votes < self.min_votes:
            return None
        return self._emit(winner)

    def _emit(self, flag: str) -> str | None:
        if flag in CONTINUOUS:
            self._last_emitted = flag
            return flag
        # event flag: latch so one real event fires once
        if flag == self._last_emitted:
            return None
        self._last_emitted = flag
        return flag

    def reset(self) -> None:
        self._buf.clear()
        self._last_emitted = None

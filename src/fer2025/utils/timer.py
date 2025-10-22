from __future__ import annotations

import time
from collections import deque
from typing import Deque


class FPSTracker:
    """Track frames per second using a sliding window."""

    def __init__(self, window: int = 60) -> None:
        self.window = window
        self._timestamps: Deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        total_time = self._timestamps[-1] - self._timestamps[0]
        if total_time <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / total_time

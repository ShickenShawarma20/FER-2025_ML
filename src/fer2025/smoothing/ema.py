from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


class LogitEMA:
    """Exponential moving average for stabilising logits per track."""

    def __init__(self, alpha: float = 0.6) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self._state: Dict[int, np.ndarray] = {}

    def update(self, track_id: int, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=np.float32)
        if track_id not in self._state:
            self._state[track_id] = values
        else:
            previous = self._state[track_id]
            self._state[track_id] = self.alpha * values + (1.0 - self.alpha) * previous
        return self._state[track_id]

    def get(self, track_id: int) -> Optional[np.ndarray]:
        if track_id in self._state:
            return self._state[track_id]
        return None

    def prune(self, active_ids: Iterable[int]) -> None:
        active = set(active_ids)
        to_remove = [track_id for track_id in self._state if track_id not in active]
        for track_id in to_remove:
            self._state.pop(track_id, None)

    def reset(self) -> None:
        self._state.clear()

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from fer2025.models.emotion.ferplus_onnx import EmotionResult
from fer2025.tracking.naive_tracker import Track

_COLOR_PALETTE = (
    (255, 87, 34),
    (76, 175, 80),
    (33, 150, 243),
    (156, 39, 176),
    (255, 193, 7),
    (3, 169, 244),
    (233, 30, 99),
)


def _color_for_track(track_id: int) -> Tuple[int, int, int]:
    return _COLOR_PALETTE[track_id % len(_COLOR_PALETTE)]


def draw_overlays(
    frame: np.ndarray,
    track_results: Sequence[Tuple[Track, Optional[EmotionResult]]],
    fps: float,
    display_fps: bool = True,
) -> np.ndarray:
    output = frame.copy()
    for track, emotion in track_results:
        x1, y1, x2, y2 = track.bbox
        color = _color_for_track(track.track_id)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = "detecting"
        if emotion is not None:
            label = f"{emotion.label}: {emotion.confidence:.2f}"
        text = f"ID {track.track_id} | {label}"
        cv2.putText(
            output,
            text,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            lineType=cv2.LINE_AA,
        )
    if display_fps:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            output,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
    return output


__all__: Sequence[str] = ["draw_overlays"]

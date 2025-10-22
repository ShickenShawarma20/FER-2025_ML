from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from fer2025.detectors.face_detector import Detection


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    score: float
    age: int = 0
    misses: int = 0


class NaiveTracker:
    """Greedy IoU-based tracker to maintain stable face IDs."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 5) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self._next_id = 0
        self._tracks: Dict[int, Track] = {}

    @staticmethod
    def _iou(box_a: Sequence[int], box_b: Sequence[int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def update(self, detections: Optional[List[Detection]]) -> List[Track]:
        # Age existing tracks
        for track in self._tracks.values():
            track.age += 1

        if detections is None:
            for track in self._tracks.values():
                track.misses += 1
            self._prune()
            return list(self._tracks.values())

        unmatched_tracks = set(self._tracks.keys())
        unmatched_detections = list(range(len(detections)))
        assignments: Dict[int, int] = {}

        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            best_iou = 0.0
            best_track_id: Optional[int] = None
            for track_id in list(unmatched_tracks):
                track = self._tracks[track_id]
                iou = self._iou(track.bbox, detection.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            if best_track_id is not None and best_iou >= self.iou_threshold:
                assignments[det_idx] = best_track_id
                unmatched_tracks.discard(best_track_id)

        # Update matched tracks
        for det_idx, track_id in assignments.items():
            detection = detections[det_idx]
            track = self._tracks[track_id]
            track.bbox = detection.bbox
            track.score = detection.score
            track.misses = 0

        # Create new tracks for unmatched detections
        for det_idx in range(len(detections)):
            if det_idx in assignments:
                continue
            detection = detections[det_idx]
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = Track(track_id=track_id, bbox=detection.bbox, score=detection.score)

        # Increment miss counter for unmatched tracks
        for track_id in unmatched_tracks:
            track = self._tracks.get(track_id)
            if track:
                track.misses += 1

        self._prune()
        return list(self._tracks.values())

    def _prune(self) -> None:
        to_remove = [track_id for track_id, track in self._tracks.items() if track.misses > self.max_missed]
        for track_id in to_remove:
            self._tracks.pop(track_id, None)

    def active_track_ids(self) -> Iterable[int]:
        return list(self._tracks.keys())

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 0

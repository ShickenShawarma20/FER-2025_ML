from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from fer2025.utils.io import RemoteFile, ensure_file
from fer2025.utils.logits import merge_classes, softmax

FERPLUS_REMOTE = RemoteFile(
    url="https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
    checksum="0d654fbd8208bcf72732c8830fb8438623b955993ec43768a3f219281015f93e",
)

FERPLUS_LABELS_8: Tuple[str, ...] = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
)

FERPLUS_LABELS_7: Tuple[str, ...] = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
)

CLASS_MAPPING_8_TO_7 = (0, 1, 2, 3, 4, 5, 6, 0)


@dataclass(frozen=True)
class EmotionResult:
    logits: np.ndarray
    probabilities: np.ndarray
    label: str
    confidence: float


class FERPlusONNX:
    """ONNXRuntime wrapper for the FER+ emotion classification model."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        min_face_size: int = 64,
    ) -> None:
        self.cache_dir = cache_dir
        self.min_face_size = int(min_face_size)
        model_path = ensure_file(FERPLUS_REMOTE, cache_dir)
        self.model_path = Path(model_path)
        sess_options = ort.SessionOptions()
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"], sess_options=sess_options)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @property
    def labels_8(self) -> Tuple[str, ...]:
        return FERPLUS_LABELS_8

    @property
    def labels_7(self) -> Tuple[str, ...]:
        return FERPLUS_LABELS_7

    def _preprocess(self, frame: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, List[int]]:
        prepared: List[np.ndarray] = []
        valid_indices: List[int] = []
        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            width = x2 - x1
            height = y2 - y1
            if min(width, height) < self.min_face_size:
                continue
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LINEAR)
            normalized = resized.astype(np.float32) / 255.0
            normalized = (normalized - 0.5) / 0.5
            prepared.append(normalized)
            valid_indices.append(idx)

        if not prepared:
            return np.empty((0, 1, 64, 64), dtype=np.float32), []
        batch = np.stack(prepared, axis=0)[:, None, :, :]
        return batch, valid_indices

    def infer(
        self,
        frame: np.ndarray,
        boxes: Sequence[Tuple[int, int, int, int]],
        class_set: int = 7,
        temperature: float = 1.0,
    ) -> List[Optional[EmotionResult]]:
        batch, valid_indices = self._preprocess(frame, boxes)
        results: List[Optional[EmotionResult]] = [None] * len(boxes)
        if not valid_indices:
            return results

        logits = self.session.run([self.output_name], {self.input_name: batch})[0]
        if logits.ndim != 2:
            raise ValueError("Unexpected logits shape")

        probs_8 = softmax(logits, temperature=temperature)
        if class_set == 8:
            labels = FERPLUS_LABELS_8
            probs = probs_8
        elif class_set == 7:
            labels = FERPLUS_LABELS_7
            probs = merge_classes(probs_8, CLASS_MAPPING_8_TO_7)
        else:
            raise ValueError("class_set must be 7 or 8")

        for local_idx, original_idx in enumerate(valid_indices):
            sample_logits = logits[local_idx]
            sample_probs = probs[local_idx]
            top_idx = int(np.argmax(sample_probs))
            label = labels[top_idx]
            confidence = float(sample_probs[top_idx])
            results[original_idx] = EmotionResult(
                logits=sample_logits.astype(np.float32),
                probabilities=sample_probs.astype(np.float32),
                label=label,
                confidence=confidence,
            )
        return results


__all__: Sequence[str] = [
    "FERPlusONNX",
    "EmotionResult",
    "FERPLUS_LABELS_7",
    "FERPLUS_LABELS_8",
    "CLASS_MAPPING_8_TO_7",
]

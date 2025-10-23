from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from fer2025.utils.io import RemoteFile, ensure_file


@dataclass(frozen=True)
class Detection:
    bbox: Tuple[int, int, int, int]
    score: float


_PROTOTXT = RemoteFile(
    url="https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
    checksum="dcd661dc48fc9de0a341db1f666a2164ea63a67265c7f779bc12d6b3f2fa67e9",
)
_MODEL = RemoteFile(
    url="https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel",
    checksum="510ffd2471bd81e3fcc88a5beb4eae4fb445ccf8333ebc54e7302b83f4158a76",
)


class OpenCVSSDFaceDetector:
    """Single-shot face detector using OpenCV's Caffe SSD."""

    def __init__(self, min_confidence: float = 0.5, cache_dir: Optional[str] = None) -> None:
        self.min_confidence = float(min_confidence)
        self.cache_dir = cache_dir
        prototxt = ensure_file(_PROTOTXT, cache_dir)
        model = ensure_file(_MODEL, cache_dir)
        self.config_path = Path(prototxt)
        self.model_path = Path(model)
        self.net = cv2.dnn.readNetFromCaffe(str(self.config_path), str(self.model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect_faces(self, frame: np.ndarray, max_faces: Optional[int] = None) -> List[Detection]:
        if frame.ndim != 3:
            raise ValueError("Frame must be a color image")

        blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        h, w = frame.shape[:2]

        results: List[Detection] = []
        for i in range(detections.shape[2]):
            score = float(detections[0, 0, i, 2])
            if score < self.min_confidence:
                continue
            box = detections[0, 0, i, 3:7]
            x1 = int(max(0, min(w, box[0] * w)))
            y1 = int(max(0, min(h, box[1] * h)))
            x2 = int(max(0, min(w, box[2] * w)))
            y2 = int(max(0, min(h, box[3] * h)))
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(Detection(bbox=(x1, y1, x2, y2), score=score))

        results.sort(key=lambda det: det.score, reverse=True)
        if max_faces is not None:
            return results[:max_faces]
        return results


__all__: Sequence[str] = ["Detection", "OpenCVSSDFaceDetector"]

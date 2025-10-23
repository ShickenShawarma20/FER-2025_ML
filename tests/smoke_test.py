from __future__ import annotations

import unittest

import numpy as np

from fer2025.detectors.face_detector import OpenCVSSDFaceDetector
from fer2025.models.emotion.ferplus_onnx import FERPlusONNX


class SmokeTest(unittest.TestCase):
    def test_models_initialize_and_run(self) -> None:
        detector = OpenCVSSDFaceDetector(min_confidence=0.5)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect_faces(frame)
        self.assertIsInstance(detections, list)

        model = FERPlusONNX(min_face_size=32)
        synthetic_frame = np.full((200, 200, 3), 127, dtype=np.uint8)
        results = model.infer(synthetic_frame, [(10, 10, 150, 150)], class_set=7)
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0])
        assert results[0] is not None
        self.assertEqual(results[0].logits.shape[0], 8)
        self.assertEqual(results[0].probabilities.shape[0], 7)


if __name__ == "__main__":
    unittest.main()

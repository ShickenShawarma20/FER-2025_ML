#!/usr/bin/env python3
from __future__ import annotations

import argparse

from fer2025.detectors.face_detector import OpenCVSSDFaceDetector
from fer2025.models.emotion.ferplus_onnx import FERPlusONNX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-download FER2025 model assets")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override cache directory path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir

    detector = OpenCVSSDFaceDetector(cache_dir=cache_dir)
    emotion_model = FERPlusONNX(cache_dir=cache_dir)

    print("Face detector config:", detector.config_path)
    print("Face detector weights:", detector.model_path)
    print("Emotion model located at:", emotion_model.model_path)
    print("Assets cached under:", cache_dir or "default cache")


if __name__ == "__main__":
    main()

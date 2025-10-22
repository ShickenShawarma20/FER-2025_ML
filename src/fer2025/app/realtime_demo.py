from __future__ import annotations

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from fer2025 import get_config_path, load_default_config
from fer2025.detectors.face_detector import Detection, OpenCVSSDFaceDetector
from fer2025.models.emotion.ferplus_onnx import (
    CLASS_MAPPING_8_TO_7,
    EmotionResult,
    FERPLUS_LABELS_7,
    FERPLUS_LABELS_8,
    FERPlusONNX,
)
from fer2025.smoothing.ema import LogitEMA
from fer2025.tracking.naive_tracker import NaiveTracker, Track
from fer2025.utils.io import dump_metrics
from fer2025.utils.logits import merge_classes, softmax
from fer2025.utils.timer import FPSTracker
from fer2025.viz.overlay import draw_overlays

DEFAULT_DETECTION_INTERVAL = 3


@dataclass
class RuntimeConfig:
    camera_index: int
    frame_width: int
    frame_height: int
    min_confidence: float
    max_faces: int
    smoothing_alpha: float
    class_set: int
    detection_interval: int
    temperature: float
    min_face_size: int
    cache_dir: Optional[str]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time FER+ demo")
    parser.add_argument("--camera", type=int, help="Camera index", default=None)
    parser.add_argument("--min-conf", type=float, default=None, help="Minimum detection confidence")
    parser.add_argument("--max-faces", type=int, default=None, help="Maximum number of faces to process")
    parser.add_argument("--classes", type=int, choices=(7, 8), default=None, help="Number of emotion classes")
    parser.add_argument("--frame-width", type=int, default=None, help="Frame width override")
    parser.add_argument("--frame-height", type=int, default=None, help="Frame height override")
    parser.add_argument("--smoothing-alpha", type=float, default=None, help="EMA smoothing alpha")
    parser.add_argument("--temperature", type=float, default=None, help="Softmax temperature")
    parser.add_argument("--detection-interval", type=int, default=None, help="Run face detection every N frames")
    parser.add_argument("--display", action="store_true", dest="display", help="Enable display window")
    parser.add_argument("--no-display", action="store_false", dest="display", help="Disable display window")
    parser.set_defaults(display=True)
    parser.add_argument("--save-metrics", nargs="?", const="metrics.json", help="Optional path to save JSON metrics")
    parser.add_argument("--config", type=Path, default=get_config_path(), help="Path to runtime YAML config")
    return parser.parse_args(argv)


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.expanduser().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_runtime_config(args: argparse.Namespace, base_config: Dict[str, Any]) -> RuntimeConfig:
    config = base_config.copy()
    overrides: Dict[str, Any] = {
        "camera_index": args.camera,
        "min_confidence": args.min_conf,
        "max_faces": args.max_faces,
        "class_set": args.classes,
        "frame_width": args.frame_width,
        "frame_height": args.frame_height,
        "smoothing_alpha": args.smoothing_alpha,
        "temperature": args.temperature,
        "detection_interval": args.detection_interval,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    detection_interval = int(config.get("detection_interval", DEFAULT_DETECTION_INTERVAL))
    if detection_interval < 1:
        detection_interval = DEFAULT_DETECTION_INTERVAL

    return RuntimeConfig(
        camera_index=int(config.get("camera_index", 0)),
        frame_width=int(config.get("frame_width", 640)),
        frame_height=int(config.get("frame_height", 480)),
        min_confidence=float(config.get("min_confidence", 0.5)),
        max_faces=int(config.get("max_faces", 5)),
        smoothing_alpha=float(config.get("smoothing_alpha", 0.6)),
        class_set=int(config.get("class_set", 7)),
        detection_interval=detection_interval,
        temperature=float(config.get("temperature", 1.0)),
        min_face_size=int(config.get("min_face_size", 64)),
        cache_dir=config.get("cache_dir"),
    )


def clamp_tracks(tracks: List[Track], max_faces: int) -> List[Track]:
    if len(tracks) <= max_faces:
        return tracks
    ordered = sorted(tracks, key=lambda t: t.score, reverse=True)
    return ordered[:max_faces]


def run_demo(args: argparse.Namespace) -> None:
    base_config = load_config(args.config) if args.config else load_default_config()
    runtime = build_runtime_config(args, base_config)

    detector = OpenCVSSDFaceDetector(min_confidence=runtime.min_confidence, cache_dir=runtime.cache_dir)
    tracker = NaiveTracker()
    emotion_model = FERPlusONNX(cache_dir=runtime.cache_dir, min_face_size=runtime.min_face_size)
    ema = LogitEMA(alpha=runtime.smoothing_alpha)
    fps_tracker = FPSTracker()

    capture = cv2.VideoCapture(runtime.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {runtime.camera_index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, runtime.frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, runtime.frame_height)

    frame_count = 0
    metrics_frames: List[Dict[str, Any]] = []
    window_name = "FER2025 Emotion Demo"

    # Graceful shutdown on SIGINT when running without display
    interrupted = False

    def _signal_handler(sig: int, frame: Any) -> None:  # type: ignore[unused-argument]
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, _signal_handler)

    try:
        while not interrupted:
            ret, frame = capture.read()
            if not ret or frame is None:
                break

            if frame.shape[1] != runtime.frame_width or frame.shape[0] != runtime.frame_height:
                frame = cv2.resize(frame, (runtime.frame_width, runtime.frame_height))

            detections: Optional[List[Detection]] = None
            if frame_count % runtime.detection_interval == 0:
                detected = detector.detect_faces(frame, max_faces=runtime.max_faces)
                detections = detected

            tracks = tracker.update(detections)
            tracks = clamp_tracks(tracks, runtime.max_faces)

            boxes = [track.bbox for track in tracks]
            raw_results = emotion_model.infer(frame, boxes, class_set=runtime.class_set, temperature=runtime.temperature)

            track_results: List[Tuple[Track, Optional[EmotionResult]]] = []
            probabilities_for_metrics: List[Dict[str, Any]] = []

            for track, raw in zip(tracks, raw_results):
                if raw is None:
                    track_results.append((track, None))
                    probabilities_for_metrics.append(
                        {
                            "track_id": track.track_id,
                            "bbox": track.bbox,
                            "score": track.score,
                            "label": None,
                            "confidence": None,
                            "probabilities": None,
                        }
                    )
                    continue
                smoothed_logits = ema.update(track.track_id, raw.logits)
                probs_8 = softmax(smoothed_logits[None, :], temperature=runtime.temperature)[0]
                if runtime.class_set == 7:
                    probs = merge_classes(probs_8[None, :], CLASS_MAPPING_8_TO_7)[0]
                    labels = FERPLUS_LABELS_7
                else:
                    probs = probs_8
                    labels = FERPLUS_LABELS_8
                top_idx = int(np.argmax(probs))
                label = labels[top_idx]
                confidence = float(probs[top_idx])
                result = EmotionResult(
                    logits=smoothed_logits.astype(np.float32, copy=False),
                    probabilities=probs.astype(np.float32, copy=False),
                    label=label,
                    confidence=confidence,
                )
                track_results.append((track, result))
                probabilities_for_metrics.append(
                    {
                        "track_id": track.track_id,
                        "bbox": track.bbox,
                        "score": track.score,
                        "label": label,
                        "confidence": confidence,
                        "probabilities": probs.tolist(),
                    }
                )

            ema.prune([track.track_id for track in tracks])

            fps_tracker.tick()
            fps_value = fps_tracker.fps

            if args.display:
                frame_with_overlay = draw_overlays(frame, track_results, fps=fps_value)
                cv2.imshow(window_name, frame_with_overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.save_metrics:
                metrics_frames.append(
                    {
                        "timestamp": time.time(),
                        "fps": fps_value,
                        "tracks": probabilities_for_metrics,
                    }
                )

            frame_count += 1
    finally:
        capture.release()
        if args.display:
            cv2.destroyAllWindows()
        if args.save_metrics and metrics_frames:
            dump_metrics(Path(args.save_metrics).expanduser(), metrics_frames)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_demo(args)


if __name__ == "__main__":
    main()

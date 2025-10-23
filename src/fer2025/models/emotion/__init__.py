"""Emotion classification models."""

from .ferplus_onnx import (
    CLASS_MAPPING_8_TO_7,
    EmotionResult,
    FERPLUS_LABELS_7,
    FERPLUS_LABELS_8,
    FERPlusONNX,
)

__all__ = [
    "CLASS_MAPPING_8_TO_7",
    "EmotionResult",
    "FERPLUS_LABELS_7",
    "FERPLUS_LABELS_8",
    "FERPlusONNX",
]

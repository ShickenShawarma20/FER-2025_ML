from __future__ import annotations

from typing import Iterable

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    scaled = logits / temperature
    max_per_row = np.max(scaled, axis=1, keepdims=True)
    exps = np.exp(scaled - max_per_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums


def merge_classes(probabilities: np.ndarray, mapping: Iterable[int]) -> np.ndarray:
    mapping_array = np.array(list(mapping))
    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be a 2D array")
    merged = np.zeros((probabilities.shape[0], mapping_array.max() + 1), dtype=probabilities.dtype)
    for src_idx, dst_idx in enumerate(mapping_array):
        merged[:, dst_idx] += probabilities[:, src_idx]
    row_sums = merged.sum(axis=1, keepdims=True)
    np.divide(merged, row_sums, out=merged, where=row_sums > 0)
    return merged

"""Inference-only normalization preserved from Prometheus training."""

from __future__ import annotations

import numpy as np


def normalize_image(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Apply per-channel percentile clipping and z-score on valid pixels."""
    normalized = np.zeros_like(image, dtype=np.float32)
    for channel_index in range(image.shape[0]):
        valid_values = image[channel_index, valid_mask]
        if valid_values.size == 0:
            continue
        low, high = np.percentile(valid_values, (2, 98))
        valid_values = np.clip(valid_values, low, high)
        valid_values = (valid_values - valid_values.mean()) / (valid_values.std() + 1e-8)
        normalized[channel_index, valid_mask] = valid_values
    return normalized

"""Spatial metadata required to restore model outputs to source coordinates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageMeta:
    sample_id: str
    original_size: tuple[int, int]
    model_size: tuple[int, int]
    resized_size: tuple[int, int]
    scale_xy: tuple[float, float]
    pad_xy: tuple[int, int]

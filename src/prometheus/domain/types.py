"""Framework-neutral prediction structures."""

from __future__ import annotations

from dataclasses import dataclass

from .labels import NucleusClass


@dataclass(frozen=True)
class Detection:
    centroid: tuple[float, float]
    label: NucleusClass | str
    confidence: float = 1.0
    box_xyxy: tuple[float, float, float, float] | None = None

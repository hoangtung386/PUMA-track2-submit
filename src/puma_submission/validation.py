"""Challenge-specific validation beyond the generic Prometheus serializers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import tifffile

from prometheus.domain import NucleusClass
from prometheus.submission import validate_submission_outputs

TRACK_2_NAMES = {f"nuclei_{nucleus_class.value}" for nucleus_class in NucleusClass}


def validate_outputs(
    input_path: str | Path,
    tissue_path: str | Path,
    nuclei_path: str | Path,
) -> None:
    """Validate output structure, labels, coordinates, and source dimensions."""
    input_path = Path(input_path)
    tissue_path = Path(tissue_path)
    nuclei_path = Path(nuclei_path)
    validate_submission_outputs(tissue_path, nuclei_path)

    with tifffile.TiffFile(input_path) as source_tiff:
        source_shape = source_tiff.pages[0].shape[:2]
    with tifffile.TiffFile(tissue_path) as output_tiff:
        output_shape = output_tiff.pages[0].shape
    if output_shape != source_shape:
        raise ValueError(f"Tissue shape {output_shape} does not match input shape {source_shape}")

    with nuclei_path.open(encoding="utf-8") as file_obj:
        polygons = json.load(file_obj)["polygons"]
    source_height, source_width = source_shape
    for index, polygon in enumerate(polygons):
        name = polygon["name"]
        if name not in TRACK_2_NAMES:
            raise ValueError(f"Polygon {index} has invalid Track 2 class: {name}")
        score = float(polygon.get("score", 1.0))
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"Polygon {index} has score outside [0, 1]: {score}")
        for point_index, point in enumerate(polygon["path_points"]):
            if len(point) < 2:
                raise ValueError(f"Polygon {index}, point {point_index} must contain x and y")
            x_coord, y_coord = float(point[0]), float(point[1])
            if not math.isfinite(x_coord) or not math.isfinite(y_coord):
                raise ValueError(f"Polygon {index} contains a non-finite coordinate")
            if not (0.0 <= x_coord <= source_width - 1):
                raise ValueError(f"Polygon {index} x-coordinate is out of bounds")
            if not (0.0 <= y_coord <= source_height - 1):
                raise ValueError(f"Polygon {index} y-coordinate is out of bounds")

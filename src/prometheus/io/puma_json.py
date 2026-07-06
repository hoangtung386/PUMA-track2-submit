"""Serialize detections using the official PUMA "Multiple polygons" schema.

The Grand Challenge output interface for ``melanoma-10-class-nuclei-segmentation``
validates against the platform's "Multiple polygons" JSON schema *before* the
challenge evaluator runs. That schema requires a top-level ``type`` and
``version`` plus per-polygon ``seed_point``/``sub_type``/``groups``/``probability``
fields, with 3-element ``[x, y, z]`` path points. The evaluator itself reads
``polygons[].name``, an optional ``polygons[].score`` and ``polygons[].path_points``
(taking only the first two coordinates), so ``score`` is emitted alongside the
schema-required ``probability``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..domain import Detection, NucleusClass, Track, nucleus_class_for_track

# Grand Challenge "Multiple polygons" payloads carry 3D coordinates; the PUMA
# ROIs are single-plane, so the z value is a constant placeholder.
_Z_PLANE = 0.0


def _class_name(label: NucleusClass | str, track: Track) -> str:
    if isinstance(label, NucleusClass):
        canonical_name = nucleus_class_for_track(label, track)
    else:
        canonical_name = str(label).removeprefix("nuclei_")
    return f"nuclei_{canonical_name}"


def _path_points(detection: Detection) -> list[list[float]]:
    x_coord, y_coord = detection.centroid
    if detection.box_xyxy is None:
        x_min, y_min = x_coord - 0.5, y_coord - 0.5
        x_max, y_max = x_coord + 0.5, y_coord + 0.5
    else:
        x_min, y_min, x_max, y_max = detection.box_xyxy
    # Do not repeat the first point: the official evaluator computes the
    # arithmetic mean of path_points rather than a geometric polygon centroid.
    return [
        [x_min, y_min, _Z_PLANE],
        [x_max, y_min, _Z_PLANE],
        [x_max, y_max, _Z_PLANE],
        [x_min, y_max, _Z_PLANE],
    ]


def write_nuclei_json(
    detections: list[Detection],
    path: str | Path,
    track: Track = Track.TRACK_2,
) -> None:
    polygons = []
    for detection in detections:
        points = _path_points(detection)
        score = float(detection.confidence)
        polygons.append(
            {
                "name": _class_name(detection.label, track),
                "seed_point": points[0],
                "path_points": points,
                "sub_type": "",
                "groups": [],
                "probability": score,
                "score": score,
            }
        )
    payload = {
        "type": "Multiple polygons",
        "version": {"major": 1, "minor": 0},
        "polygons": polygons,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj)

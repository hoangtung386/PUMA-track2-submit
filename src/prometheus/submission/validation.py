"""Local structural validation for generated PUMA output files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile


def validate_submission_outputs(tissue_path: str | Path, nuclei_path: str | Path) -> None:
    tissue = Path(tissue_path)
    nuclei = Path(nuclei_path)
    if not tissue.is_file() or tissue.stat().st_size == 0:
        raise ValueError(f"Missing or empty tissue output: {tissue}")
    with tifffile.TiffFile(tissue) as tif:
        page = tif.pages[0]
        tissue_mask = page.asarray()
        if tissue_mask.ndim != 2:
            raise ValueError("Tissue output must be a two-dimensional class-index mask")
        if not set(np.unique(tissue_mask)).issubset(set(range(6))):
            raise ValueError("Tissue output contains labels outside the range [0, 5]")
        required_tags = {"XResolution", "YResolution", "SMinSampleValue", "SMaxSampleValue"}
        missing_tags = {tag for tag in required_tags if tag not in page.tags}
        if missing_tags:
            raise ValueError(f"Tissue output is missing TIFF tags: {sorted(missing_tags)}")
    if not nuclei.is_file():
        raise ValueError(f"Missing nuclei output: {nuclei}")
    with nuclei.open(encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    # The Grand Challenge "Multiple polygons" interface schema validates the
    # output before evaluation. It is strict (additionalProperties: false), so a
    # missing type/version OR any unexpected key silently fails there. Mirror it.
    if data.get("type") != "Multiple polygons":
        raise ValueError('Nuclei output must set type == "Multiple polygons"')
    version = data.get("version")
    if not isinstance(version, dict) or "major" not in version or "minor" not in version:
        raise ValueError("Nuclei output must contain a version object with major/minor")
    polygons = data.get("polygons")
    if not isinstance(polygons, list):
        raise ValueError("Nuclei output must contain a polygons list")
    allowed_top_keys = {"type", "version", "polygons", "name"}
    extra_top = set(data) - allowed_top_keys
    if extra_top:
        raise ValueError(f"Nuclei output has unexpected top-level keys: {sorted(extra_top)}")
    allowed_polygon_keys = {"name", "seed_point", "path_points", "sub_type", "groups", "probability"}
    for index, polygon in enumerate(polygons):
        extra_keys = set(polygon) - allowed_polygon_keys
        if extra_keys:
            raise ValueError(f"Polygon {index} has schema-forbidden keys: {sorted(extra_keys)}")
        if not isinstance(polygon.get("name"), str):
            raise ValueError(f"Polygon {index} has no class name")
        path_points = polygon.get("path_points", [])
        if len(path_points) < 3:
            raise ValueError(f"Polygon {index} must contain at least three path points")
        if any(len(point) < 3 for point in path_points):
            raise ValueError(f"Polygon {index} path points must be 3-element [x, y, z]")
        if len(polygon.get("seed_point", [])) < 3:
            raise ValueError(f"Polygon {index} must contain a 3-element seed_point")
        if not isinstance(polygon.get("probability", 1), (int, float)):
            raise ValueError(f"Polygon {index} has an invalid probability")

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from prometheus.domain import Detection, NucleusClass
from prometheus.io import write_nuclei_json, write_tissue_tiff
from puma_submission.validation import validate_outputs


def _write_input(path: Path) -> None:
    tifffile.imwrite(path, np.zeros((8, 12, 3), dtype=np.uint8))


def test_validate_outputs_accepts_prometheus_writers(tmp_path: Path) -> None:
    input_path = tmp_path / "input.tif"
    tissue_path = tmp_path / "tissue.tif"
    nuclei_path = tmp_path / "nuclei.json"
    _write_input(input_path)
    write_tissue_tiff(np.zeros((8, 12), dtype=np.uint8), tissue_path)
    write_nuclei_json(
        [Detection((5.0, 4.0), NucleusClass.TUMOR, 0.8, (4.0, 3.0, 6.0, 5.0))],
        nuclei_path,
    )

    validate_outputs(input_path, tissue_path, nuclei_path)


def test_validate_outputs_rejects_wrong_tissue_shape(tmp_path: Path) -> None:
    input_path = tmp_path / "input.tif"
    tissue_path = tmp_path / "tissue.tif"
    nuclei_path = tmp_path / "nuclei.json"
    _write_input(input_path)
    write_tissue_tiff(np.zeros((4, 4), dtype=np.uint8), tissue_path)
    write_nuclei_json([], nuclei_path)

    with pytest.raises(ValueError, match="does not match"):
        validate_outputs(input_path, tissue_path, nuclei_path)


def test_validate_outputs_rejects_invalid_track_class(tmp_path: Path) -> None:
    input_path = tmp_path / "input.tif"
    tissue_path = tmp_path / "tissue.tif"
    nuclei_path = tmp_path / "nuclei.json"
    _write_input(input_path)
    write_tissue_tiff(np.zeros((8, 12), dtype=np.uint8), tissue_path)
    nuclei_path.write_text(
        json.dumps(
            {
                "type": "Multiple polygons",
                "version": {"major": 1, "minor": 0},
                "polygons": [
                    {
                        "name": "nuclei_other",
                        "seed_point": [1, 1, 0],
                        "probability": 1.0,
                        "sub_type": "",
                        "groups": [],
                        "path_points": [[1, 1, 0], [2, 1, 0], [2, 2, 0], [1, 2, 0]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid Track 2 class"):
        validate_outputs(input_path, tissue_path, nuclei_path)

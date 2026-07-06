from pathlib import Path

import numpy as np
import tifffile

from puma_submission.prepare import prepare_image


def test_prepare_image_preserves_source_metadata(tmp_path: Path) -> None:
    path = tmp_path / "wide.tif"
    tifffile.imwrite(path, np.full((20, 40, 3), 255, dtype=np.uint8))

    image, metadata = prepare_image(path, 32)

    assert image.shape == (1, 3, 32, 32)
    assert metadata.original_size == (20, 40)
    assert metadata.resized_size == (16, 32)
    assert metadata.pad_xy == (0, 8)
    assert np.isfinite(image.numpy()).all()

"""Native TIFF input reader used by deployment preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def read_image(path: str | Path) -> np.ndarray:
    """Read a grayscale/RGB/RGBA TIFF as normalized float32 RGB."""
    with tifffile.TiffFile(path) as tif:
        if not tif.pages:
            raise ValueError(f"TIFF has no image pages: {path}")
        image = tif.pages[0].asarray()
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim != 3:
        raise ValueError(f"Expected a 2D or HWC image, got shape {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 1, 3, or 4 channels, got shape {image.shape}")
    image = image.astype(np.float32)
    maximum = float(image.max(initial=0.0))
    if maximum > 1.0:
        image /= 255.0 if maximum <= 255.0 else 65535.0
    return image

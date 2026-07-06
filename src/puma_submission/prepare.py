"""Image preparation shared by the container entry point and tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from prometheus.domain import ImageMeta
from prometheus.inference import letterbox_image, normalize_image
from prometheus.io import read_image


def prepare_image(path: str | Path, image_size: int) -> tuple[torch.Tensor, ImageMeta]:
    """Read and transform one TIFF exactly as Prometheus training inference."""
    image, metadata = letterbox_image(
        read_image(path),
        (image_size, image_size),
        Path(path).stem,
    )
    channel_first = image.transpose(2, 0, 1)
    valid_mask = np.zeros((image_size, image_size), dtype=bool)
    pad_x, pad_y = metadata.pad_xy
    resized_height, resized_width = metadata.resized_size
    valid_mask[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = True
    normalized = normalize_image(channel_first, valid_mask)
    tensor = torch.from_numpy(np.ascontiguousarray(normalized)).float()
    return tensor.unsqueeze(0), metadata
